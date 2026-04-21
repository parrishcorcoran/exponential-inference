"""Novelty-based head selection: keep heads that add new KV angles.

Previous approaches measured importance WITHIN one step:
- Sharpness: which heads have peaked attention → 14% match
- Contribution: which heads have large O-norm → 17% match

New approach: measure which heads add NOVEL information to the KV cache.
Heads whose KV is most different from what's already cached are extending
the manifold. Heads whose KV is redundant with existing cache are not needed.

This is a direct manifold measurement: the KV cache IS the reconstruction.
Heads that add new projection angles ARE the important ones.
"""
import torch
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("NOVELTY ENGINE — select heads by KV information gain")
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


def full_layer_manual(h, layer, cos, sin, k_cached, v_cached):
    B, T, D = h.shape
    attn = layer.self_attn
    residual = h
    h_norm = layer.input_layernorm(h)

    q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None: q = attn.q_norm(q)
    if attn.k_norm is not None: k = attn.k_norm(k)

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

    k_full = torch.cat([k_cached, k], dim=2)
    v_full = torch.cat([v_cached, v], dim=2)

    k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
    v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
    attn_out = attn.o_proj(attn_out)

    h = residual + attn_out
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, k_full, v_full


def masked_layer(h, layer, head_mask, cos, sin, k_cached, v_cached):
    """Full-width computation but MASK the attention output per head.

    head_mask: [N_HEADS] tensor of 0/1 with scaling built in.
    This uses the CORRECT O-projection (all weights) but zeros inactive heads.
    No sparse slicing = no entanglement problems.
    """
    B, T, D = h.shape
    attn = layer.self_attn
    residual = h
    h_norm = layer.input_layernorm(h)

    q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
    k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
    v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None: q = attn.q_norm(q)
    if attn.k_norm is not None: k = attn.k_norm(k)

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

    k_full = torch.cat([k_cached, k], dim=2)
    v_full = torch.cat([v_cached, v], dim=2)

    k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
    v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    # [B, N_HEADS, T, HD]

    # MASK: zero inactive heads, scale active ones
    attn_out = attn_out * head_mask.view(1, N_HEADS, 1, 1)

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
    attn_out = attn.o_proj(attn_out)

    h = residual + attn_out
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, k_full, v_full


def novelty_generate(model, input_ids, max_new_tokens=64,
                     split_layer=20, n_active=20, selection="novelty"):
    """
    Phase 1 (layers 0-split): full width, measure per-head KV novelty
    Phase 2 (layers split-40): masked heads (zero+scale) based on novelty
    """
    B = input_ids.shape[0]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    kv = {}
    for i in range(N_LAYERS):
        kv[i] = (past.layers[i].keys.clone(), past.layers[i].values.clone())

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]
    telem = {"heads_selected": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1

        with torch.no_grad():
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
            pos_ids = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos_ids)

            # Phase 1: full width coarse layers
            for i in range(split_layer):
                h, k_new, v_new = full_layer_manual(
                    h, model.model.layers[i], cos, sin, kv[i][0], kv[i][1])
                kv[i] = (k_new, v_new)

            # Measure novelty at the split layer
            # For each Q head: compute Q, see how it relates to cached KV
            # Heads that attend to SPECIFIC (non-uniform) positions add more info
            layer_m = model.model.layers[split_layer]
            attn_m = layer_m.self_attn
            h_norm_m = layer_m.input_layernorm(h)

            q_m = attn_m.q_proj(h_norm_m).view(B, 1, N_HEADS, HEAD_DIM)
            if attn_m.q_norm is not None: q_m = attn_m.q_norm(q_m)
            q_m = q_m.transpose(1, 2)  # [B, N_HEADS, 1, HD]

            # Rotary on Q
            rd = HEAD_DIM // 2
            cos_r = cos.unsqueeze(1)
            sin_r = sin.unsqueeze(1)
            q1, q2 = q_m[..., :rd], q_m[..., rd:]
            q_m = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                             q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)

            # Get cached KV at split layer
            k_cached_m = kv[split_layer][0]  # [B, N_KV, cache, HD]
            k_exp_m = k_cached_m.repeat_interleave(GQA_RATIO, dim=1)

            # Per-head attention weights
            scale = HEAD_DIM ** -0.5
            attn_w = (q_m @ k_exp_m.transpose(-2, -1)) * scale  # [B, N_HEADS, 1, cache]
            attn_w = F.softmax(attn_w.float(), dim=-1)

            if selection == "novelty":
                # Novelty: entropy of attention distribution
                # LOW entropy = head attends to specific positions = high novelty
                # HIGH entropy = uniform attention = low novelty (redundant)
                ent = -(attn_w * (attn_w + 1e-10).log()).sum(-1)[0, :, 0]  # [N_HEADS]
                # Lower entropy = more novel. Select heads with LOWEST entropy.
                selected = ent.topk(n_active, largest=False).indices.sort().values.tolist()
            elif selection == "variance":
                # Variance of attention weights — high variance = peaked = informative
                var = attn_w.var(-1)[0, :, 0]  # [N_HEADS]
                selected = var.topk(n_active).indices.sort().values.tolist()
            else:
                selected = list(range(n_active))

            telem["heads_selected"].append(selected)

            # Build head mask: active heads get scale, inactive get 0
            scale_factor = N_HEADS / n_active
            head_mask = torch.zeros(N_HEADS, device=device, dtype=h.dtype)
            for hd in selected:
                head_mask[hd] = scale_factor

            # Phase 2: masked layers
            for i in range(split_layer, N_LAYERS):
                h, k_new, v_new = masked_layer(
                    h, model.model.layers[i], head_mask,
                    cos, sin, kv[i][0], kv[i][1])
                kv[i] = (k_new, v_new)

        h_out = final_norm_layer(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        next_tok = logits.argmax(-1).item()
        gen_tokens.append(next_tok)

        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


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
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)

base_texts = {}
base_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps.append(tps)
    base_texts[prompt] = out[0][ids.shape[1]:].tolist()
    print(f"  {tps:.1f} tok/s [{text[:60]}]")
avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

# Test: masking vs slicing, novelty vs variance, different split/active
print(f"\nNOVELTY + MASKING ENGINE:")
configs = [
    ("novelty",  20, 40, "control_100%"),      # all heads = control
    ("novelty",  20, 30, "novelty_75%_s20"),
    ("novelty",  20, 20, "novelty_50%_s20"),
    ("novelty",  20, 10, "novelty_25%_s20"),
    ("novelty",  30, 20, "novelty_50%_s30"),
    ("variance", 20, 20, "variance_50%_s20"),
    ("first_n",  20, 20, "first20_50%_s20"),    # just first 20 heads (no selection)
]

for sel, split, n_act, name in configs:
    all_match = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            tokens, telem = novelty_generate(
                model, ids, max_new_tokens=N_GEN,
                split_layer=split, n_active=n_act, selection=sel
            )
        base_ids = base_texts[prompt][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        match_pct = match / max(len(base_ids), 1) * 100
        all_match.append(match_pct)

    avg_match = sum(all_match) / len(all_match)
    print(f"  {name:>25}: match={avg_match:>4.0f}%")

print(f"\nDone.", flush=True)
