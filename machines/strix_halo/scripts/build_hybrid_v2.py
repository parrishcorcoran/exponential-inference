"""Hybrid v2: measure head importance by output norm contribution.

v1 used attention sharpness — peaked attention ≠ important for output.
v2 measures which heads actually change the residual stream the most.

At the measurement layer, compute full attention, then measure each
head's contribution to the O-projection output. Keep heads with
largest contributions.

Also test: no scaling (the missing heads might not need compensation
if we keep the right ones).
"""
import torch
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("HYBRID V2 — output norm head selection")
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
final_norm_layer = model.model.norm

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def full_layer_measure_contribution(h, layer, cos, sin, k_cached, v_cached):
    """Run full-width layer AND measure per-head output contribution.

    Returns h, k_full, v_full, per_head_contrib [N_HEADS]
    """
    B, T, D = h.shape
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    k_full = torch.cat([k_cached, k], dim=2) if k_cached is not None else k
    v_full = torch.cat([v_cached, v], dim=2) if v_cached is not None else v

    k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
    v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

    # Per-head attention output
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    # [B, N_HEADS, T, HD]

    # Measure per-head contribution through O-projection
    o_weight = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
    per_head_contrib = torch.zeros(N_HEADS, device=device)
    for hd in range(N_HEADS):
        # This head's contribution to the output
        head_out = attn_out[0, hd, 0, :]  # [HD]
        head_proj = o_weight[:, hd, :] @ head_out  # [H]
        per_head_contrib[hd] = head_proj.float().norm()

    # Full attention output
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
    attn_out = attn.o_proj(attn_out)

    h = residual + attn_out
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, k_full, v_full, per_head_contrib


def sparse_layer_no_scale(h, layer, active_q_heads, cos, sin, k_cached, v_cached, scale_mode="scale"):
    """Sparse Q layer. scale_mode: 'scale', 'noscale', or 'softscale'."""
    B, T, D = h.shape
    n_active = len(active_q_heads)
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    q_weight = attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
    q = (h_norm @ q_weight[active_q_heads].reshape(-1, H).T).view(B, T, n_active, HEAD_DIM)

    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    k_full = torch.cat([k_cached, k], dim=2) if k_cached is not None else k
    v_full = torch.cat([v_cached, v], dim=2) if v_cached is not None else v

    kv_indices = [qh // GQA_RATIO for qh in active_q_heads]
    k_exp = k_full[:, kv_indices]
    v_exp = v_full[:, kv_indices]

    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    o_weight = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
    o_active = o_weight[:, active_q_heads, :].reshape(H, n_active * HEAD_DIM)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, n_active * HEAD_DIM)

    if scale_mode == "scale":
        attn_proj = (attn_flat @ o_active.T) * (N_HEADS / n_active)
    elif scale_mode == "noscale":
        attn_proj = attn_flat @ o_active.T
    elif scale_mode == "softscale":
        # Geometric mean scaling: sqrt(N_HEADS / n_active)
        attn_proj = (attn_flat @ o_active.T) * ((N_HEADS / n_active) ** 0.5)

    if attn.o_proj.bias is not None:
        attn_proj = attn_proj + attn.o_proj.bias

    h = residual + attn_proj
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, k_full, v_full


def hybrid_v2_generate(model, input_ids, max_new_tokens=64,
                       split_layer=20, top_frac=0.5, min_heads=8,
                       selection="contribution", scale_mode="scale"):
    """selection: 'contribution' (O-norm) or 'sharpness' (attention entropy)"""
    B = input_ids.shape[0]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]
    telem = {"q_heads": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1

        with torch.no_grad():
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
            pos_ids = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos_ids)

            # Phase 1: full layers, measure at each
            accumulated_contrib = torch.zeros(N_HEADS, device=device)
            n_measured = 0

            for i in range(split_layer):
                k_cached = past.layers[i].keys
                v_cached = past.layers[i].values

                h, k_new, v_new, contrib = full_layer_measure_contribution(
                    h, model.model.layers[i], cos, sin, k_cached, v_cached
                )
                past.layers[i].keys = k_new
                past.layers[i].values = v_new

                accumulated_contrib += contrib
                n_measured += 1

            # Select heads by accumulated contribution
            avg_contrib = accumulated_contrib / max(n_measured, 1)
            n_keep = max(min_heads, int(N_HEADS * top_frac))
            sharp_heads = avg_contrib.topk(n_keep).indices.sort().values.tolist()
            telem["q_heads"].append(len(sharp_heads))

            # Phase 2: sparse layers
            for i in range(split_layer, N_LAYERS):
                k_cached = past.layers[i].keys
                v_cached = past.layers[i].values

                h, k_new, v_new = sparse_layer_no_scale(
                    h, model.model.layers[i], sharp_heads,
                    cos, sin, k_cached, v_cached, scale_mode=scale_mode
                )
                past.layers[i].keys = k_new
                past.layers[i].values = v_new

        h_out = final_norm_layer(h)
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

print(f"\nBASELINE:")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

base_texts = {}
base_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps.append(tps)
    base_texts[prompt] = out[0][ids.shape[1]:].tolist()
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

# Test configs: selection method × scale mode × head fraction
print(f"\nHYBRID V2 — contribution-based selection:")
configs = [
    # Selection method, scale mode, split, frac, min, name
    ("contrib", "scale",     20, 1.0,  40, "control_100%"),
    ("contrib", "scale",     20, 0.75, 8,  "contrib_75%_scale"),
    ("contrib", "noscale",   20, 0.75, 8,  "contrib_75%_noscale"),
    ("contrib", "softscale", 20, 0.75, 8,  "contrib_75%_sqrt"),
    ("contrib", "scale",     20, 0.50, 8,  "contrib_50%_scale"),
    ("contrib", "noscale",   20, 0.50, 8,  "contrib_50%_noscale"),
    ("contrib", "softscale", 20, 0.50, 8,  "contrib_50%_sqrt"),
    ("contrib", "scale",     30, 0.50, 8,  "contrib_50%_s30_scale"),
    ("contrib", "noscale",   30, 0.50, 8,  "contrib_50%_s30_noscale"),
]

for sel, smode, split, frac, minh, name in configs:
    all_tps = []
    all_match = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            tokens, telem = hybrid_v2_generate(
                model, ids, max_new_tokens=N_GEN,
                split_layer=split, top_frac=frac, min_heads=minh,
                selection=sel, scale_mode=smode,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tps = len(tokens) / elapsed
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        base_ids = base_texts[prompt][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        match_pct = match / max(len(base_ids), 1) * 100

        all_tps.append(tps)
        all_match.append(match_pct)

    avg_tps = sum(all_tps) / len(all_tps)
    avg_match = sum(all_match) / len(all_match)
    print(f"  {name:>30}: {avg_tps:.1f} ({avg_tps/avg_base:.2f}x) match={avg_match:.0f}%")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/hybrid_v2.json", "w") as f:
    json.dump({"baseline": avg_base}, f, indent=2)
print("Saved.", flush=True)
