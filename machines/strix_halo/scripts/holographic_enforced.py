"""Holographic Transformer — ENFORCED routing.

Actually skip layers and mask heads based on SAE read.
Measures REAL wall-clock speedup and text quality.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda"

print("=" * 70)
print("HOLOGRAPHIC TRANSFORMER — enforced routing, real speedup")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS
GQA = N_HEADS // N_KV

SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"
sae_0 = torch.load(f"{SAE_PATH}/ae_0.pt", map_location=device, weights_only=False)
sae_enc_w = sae_0["encoder.weight"].float().to(device)
sae_enc_b = sae_0["encoder.bias"].float().to(device)

DEFINED_FEATURES = {1143, 10032, 13093, 4273, 2351, 5963, 3897, 6775, 2523, 8120}
BRANCHING_FEATURES = {3666, 11005, 15246, 9873, 8983, 7071, 6431, 15393, 6452}
HEAD_RANK = [4, 1, 13, 12, 8, 0, 2, 3, 10, 5, 15, 9, 11, 6, 14, 7]

print(f"Model: H={H} L={N_LAYERS} heads={N_HEADS} kv={N_KV} hd={HEAD_DIM}")


def sae_route(h_emb):
    """SAE manifold read → (n_layers, n_heads, head_mask)"""
    acts = F.relu(h_emb.float() @ sae_enc_w.T + sae_enc_b)
    active = set((acts > 0).nonzero(as_tuple=True)[0].cpu().tolist())
    n_def = len(active & DEFINED_FEATURES)
    n_br = len(active & BRANCHING_FEATURES)

    if n_def > n_br + 2:
        n_layers = max(8, N_LAYERS // 2)
        n_heads = max(4, N_HEADS // 4)
    elif n_def > n_br:
        n_layers = max(14, int(N_LAYERS * 0.7))
        n_heads = max(8, N_HEADS // 2)
    elif n_br > n_def + 2:
        n_layers = N_LAYERS
        n_heads = N_HEADS
    else:
        n_layers = max(20, int(N_LAYERS * 0.85))
        n_heads = N_HEADS

    # Build head mask with scaling
    active_heads = HEAD_RANK[:n_heads]
    scale = N_HEADS / n_heads
    mask = torch.zeros(N_HEADS, device=device, dtype=torch.bfloat16)
    for h in active_heads:
        mask[h] = scale

    return n_layers, n_heads, mask


def enforced_layer(h, layer, head_mask, pos_emb):
    """Run one layer with head masking enforced."""
    B, T, D = h.shape
    attn = layer.self_attn
    residual = h
    h_norm = layer.input_layernorm(h)

    # Full Q, K, V (needed for correct computation)
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

    # Rotary
    cos, sin = pos_emb
    rd = HEAD_DIM // 2
    cos_r = cos[..., :rd].unsqueeze(1)
    sin_r = sin[..., :rd].unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r - q2*sin_r, q2*cos_r + q1*sin_r], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r - k2*sin_r, k2*cos_r + k1*sin_r], -1)

    # GQA expand
    k = k.repeat_interleave(GQA, dim=1)
    v = v.repeat_interleave(GQA, dim=1)

    # Attention
    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))

    # HEAD MASKING: zero inactive, scale active
    attn_out = attn_out * head_mask.view(1, N_HEADS, 1, 1)

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
    attn_out = attn.o_proj(attn_out)
    h = residual + attn_out

    # Full MLP
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))
    return h


def enforced_generate(model, input_ids, max_new_tokens=64):
    """Generate with ENFORCED routing — actually skip layers."""
    # Prefill: full model
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]
    telem = {"layers": [], "heads": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1
        with torch.no_grad():
            # Embed
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))

            # SAE ROUTE
            n_layers, n_heads, head_mask = sae_route(h[0, 0])
            telem["layers"].append(n_layers)
            telem["heads"].append(n_heads)

            # Position embeddings
            pos = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos)
            pos_emb = (cos, sin)

            # Run ONLY n_layers (enforced skip)
            for i in range(n_layers):
                h = enforced_layer(h, model.model.layers[i], head_mask, pos_emb)

            # Skip remaining layers — just don't run them

            # Final norm + lm_head
            h = model.model.norm(h)
            logits = model.lm_head(h)
            next_tok = logits[0, -1].argmax(-1).item()

        gen_tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark: baseline vs enforced routing
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
    "Once upon a time in a kingdom far away there lived",
    "The most fundamental concept in quantum mechanics is",
    "To solve a quadratic equation you can use the",
]
N_GEN = 64

# Baseline
print(f"\nBASELINE (full model):")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)

base_tps = []
base_texts = {}
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps.append(tps)
    base_texts[prompt] = (out[0][ids.shape[1]:].tolist(), text)
avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

# Enforced holographic
print(f"\nHOLOGRAPHIC (enforced routing, real speedup):")
holo_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        tokens, telem = enforced_generate(model, ids, N_GEN)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = len(tokens) / elapsed
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Token match vs baseline
    base_ids = base_texts[prompt][0][:len(tokens)]
    match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
    match_pct = match / max(len(base_ids), 1) * 100

    avg_l = sum(telem["layers"]) / len(telem["layers"])
    avg_h = sum(telem["heads"]) / len(telem["heads"])
    speedup = tps / avg_base

    holo_tps.append(tps)
    print(f"  {tps:.1f} tok/s ({speedup:.2f}x) L={avg_l:.0f} H={avg_h:.0f} match={match_pct:.0f}%")
    print(f"    Base: [{base_texts[prompt][1][:60]}]")
    print(f"    Holo: [{text[:60]}]")

avg_holo = sum(holo_tps) / len(holo_tps)
speedup = avg_holo / avg_base

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"  Baseline: {avg_base:.1f} tok/s")
print(f"  Holographic: {avg_holo:.1f} tok/s")
print(f"  REAL speedup: {speedup:.2f}x")
print(f"\nDone.", flush=True)
