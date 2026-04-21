"""Sharpness-measured engine: full attention at coarse layers,
measure which heads actually fire, use only those for remaining layers.

Finding 04: 80% of heads prunable with 100% token match, measured by
attention sharpness. This engine applies that directly:

1. Run first N layers at FULL width (all 40 Q heads, all 8 KV)
2. Measure attention sharpness per head at layer N
3. Keep only the heads with sharp (peaked) attention
4. Run remaining layers with sparse Q (only sharp heads)
5. Full KV always (for cache completeness)
6. Full MLP always

This should give GOOD TEXT QUALITY because the head selection is
measured from actual attention patterns, not proxied from entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("SHARPNESS-MEASURED ENGINE — Finding 04 applied directly")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.config.hidden_size // N_HEADS
H = model.config.hidden_size
GQA_RATIO = N_HEADS // N_KV

lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def full_layer_with_sharpness(h, layer, cos, sin, kv_cache, layer_idx):
    """Run one layer at FULL width and measure per-head attention sharpness.

    Returns: output h, updated kv_cache, sharpness per head [N_HEADS]
    """
    B, T, D = h.shape
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    # Full Q, K, V
    q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)  # [B, N_HEADS, T, HD]
    k = k.transpose(1, 2)  # [B, N_KV, T, HD]
    v = v.transpose(1, 2)

    # Rotary
    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    # Update KV cache
    if layer_idx in kv_cache:
        k_cached, v_cached = kv_cache[layer_idx]
        k_full = torch.cat([k_cached, k], dim=2)
        v_full = torch.cat([v_cached, v], dim=2)
    else:
        k_full = k
        v_full = v
    kv_cache[layer_idx] = (k_full, v_full)

    # GQA expand
    k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
    v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

    # Compute attention weights manually to measure sharpness
    scale = HEAD_DIM ** -0.5
    attn_weights = (q @ k_exp.transpose(-2, -1)) * scale  # [B, N_HEADS, T, cache_len]
    attn_weights = F.softmax(attn_weights.float(), dim=-1)

    # Sharpness per head: 1 - normalized entropy
    # For single-token generation (T=1), attn_weights is [B, N_HEADS, 1, cache_len]
    ent = -(attn_weights * (attn_weights + 1e-10).log()).sum(-1)  # [B, N_HEADS, T]
    max_ent = torch.log(torch.tensor(float(attn_weights.shape[-1]), device=device))
    sharpness = 1.0 - (ent / max_ent)  # [B, N_HEADS, T]
    sharpness = sharpness[0, :, 0]  # [N_HEADS] for first batch, first token

    # Attention output
    attn_out = (attn_weights.to(v_exp.dtype) @ v_exp)  # [B, N_HEADS, T, HD]
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
    attn_out = attn.o_proj(attn_out)

    h = residual + attn_out

    # Full MLP
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, kv_cache, sharpness


def sparse_layer_cached(h, layer, active_q_heads, cos, sin, kv_cache, layer_idx):
    """Run one layer with sparse Q heads but full cached KV. No sharpness measurement."""
    B, T, D = h.shape
    n_active = len(active_q_heads)
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    # Sparse Q
    q_weight = attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
    q = (h_norm @ q_weight[active_q_heads].reshape(-1, H).T).view(B, T, n_active, HEAD_DIM)

    # Full KV (for cache)
    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Rotary
    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    # Update KV cache (full)
    if layer_idx in kv_cache:
        k_cached, v_cached = kv_cache[layer_idx]
        k_full = torch.cat([k_cached, k], dim=2)
        v_full = torch.cat([v_cached, v], dim=2)
    else:
        k_full = k
        v_full = v
    kv_cache[layer_idx] = (k_full, v_full)

    # GQA for active Q heads
    kv_indices = [qh // GQA_RATIO for qh in active_q_heads]
    k_exp = k_full[:, kv_indices]
    v_exp = v_full[:, kv_indices]

    # Attention
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    # Sparse O
    o_weight = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
    o_active = o_weight[:, active_q_heads, :].reshape(H, n_active * HEAD_DIM)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, n_active * HEAD_DIM)
    attn_proj = (attn_flat @ o_active.T) * (N_HEADS / n_active)

    if attn.o_proj.bias is not None:
        attn_proj = attn_proj + attn.o_proj.bias

    h = residual + attn_proj

    # Full MLP
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, kv_cache


def sharpness_generate(model, input_ids, max_new_tokens=64,
                       measure_layer=5, top_frac=0.5, min_heads=8):
    """Generate with sharpness-measured head selection.

    1. Layers 0 to measure_layer: full width, measure sharpness
    2. Select top_frac heads by sharpness
    3. Layers measure_layer+1 to N_LAYERS: sparse Q (only sharp heads)
    """
    B = input_ids.shape[0]

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    kv_cache = {}
    for layer_idx in range(N_LAYERS):
        kv_cache[layer_idx] = (
            past.layers[layer_idx].keys.clone(),
            past.layers[layer_idx].values.clone(),
        )

    logits = out.logits[0, -1]
    next_tok = logits.argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]

    telem = {"q_heads": [], "sharpness_vals": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1

        h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
        pos_ids = torch.tensor([[seq_len - 1]], device=device)
        cos, sin = model.model.rotary_emb(h, pos_ids)

        with torch.no_grad():
            # Phase 1: full width through measure layers, track sharpness
            accumulated_sharpness = torch.zeros(N_HEADS, device=device)
            n_measured = 0

            for i in range(min(measure_layer + 1, N_LAYERS)):
                h, kv_cache, sharpness = full_layer_with_sharpness(
                    h, model.model.layers[i], cos, sin, kv_cache, i
                )
                accumulated_sharpness += sharpness
                n_measured += 1

            # Average sharpness across measured layers
            avg_sharpness = accumulated_sharpness / max(n_measured, 1)

            # Select top heads by measured sharpness
            n_keep = max(min_heads, int(N_HEADS * top_frac))
            sharp_heads = avg_sharpness.topk(n_keep).indices.sort().values.tolist()

            telem["q_heads"].append(len(sharp_heads))
            telem["sharpness_vals"].append(avg_sharpness.mean().item())

            # Phase 2: sparse Q with only the sharp heads
            for i in range(measure_layer + 1, N_LAYERS):
                h, kv_cache = sparse_layer_cached(
                    h, model.model.layers[i], sharp_heads, cos, sin, kv_cache, i
                )

        # Final
        h_out = final_norm(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        next_tok = logits.argmax(-1).item()
        gen_tokens.append(next_tok)

        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "In computer science, the most fundamental data structure is",
    "The history of mathematics spans thousands of years and includes",
]

N_GEN = 64

# Baseline
print(f"\nBASELINE:")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

base_tps_list = []
base_texts = {}
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps_list.append(tps)
    base_texts[prompt] = (out[0][ids.shape[1]:].tolist(), text)
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

avg_base = sum(base_tps_list) / len(base_tps_list)
print(f"  Average: {avg_base:.1f} tok/s")

# Sharpness-measured configs
print(f"\nSHARPNESS-MEASURED ENGINE:")
configs = [
    {"name": "m5_t100%", "measure": 5, "frac": 1.0, "min": 40},   # control: all heads
    {"name": "m5_t75%",  "measure": 5, "frac": 0.75, "min": 8},
    {"name": "m5_t50%",  "measure": 5, "frac": 0.50, "min": 8},
    {"name": "m5_t25%",  "measure": 5, "frac": 0.25, "min": 4},
    {"name": "m5_t20%",  "measure": 5, "frac": 0.20, "min": 4},
    {"name": "m10_t50%", "measure": 10, "frac": 0.50, "min": 8},
    {"name": "m3_t50%",  "measure": 3, "frac": 0.50, "min": 8},
]

for cfg in configs:
    print(f"\n  Config: {cfg['name']} (measure@L{cfg['measure']}, keep {cfg['frac']*100:.0f}%)")

    all_tps = []
    all_match = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            tokens, telem = sharpness_generate(
                model, ids, max_new_tokens=N_GEN,
                measure_layer=cfg["measure"],
                top_frac=cfg["frac"],
                min_heads=cfg["min"],
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tps = len(tokens) / elapsed
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        # Token match vs baseline
        base_ids = base_texts[prompt][0][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        match_pct = match / max(len(base_ids), 1) * 100

        all_tps.append(tps)
        all_match.append(match_pct)

        avg_q = sum(telem["q_heads"]) / max(len(telem["q_heads"]), 1)
        print(f"    {tps:>5.1f} tok/s ({tps/avg_base:.2f}x) Q={avg_q:.0f}/{N_HEADS} "
              f"match={match_pct:.0f}% [{text[:50]}]")

    avg_tps = sum(all_tps) / len(all_tps)
    avg_match = sum(all_match) / len(all_match)
    print(f"  Average: {avg_tps:.1f} tok/s ({avg_tps/avg_base:.2f}x), match={avg_match:.0f}%")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/sharpness_engine.json", "w") as f:
    json.dump({"baseline": avg_base}, f, indent=2)
print("Saved.", flush=True)
