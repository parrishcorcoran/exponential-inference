"""Stage 140 — KV cache geometry on 14B.

Measure per-layer cache structure on Qwen3-14B to match Mac's 0.6B
measurements (stages 132, 133, 138, 139). Six axes:

1. K rank (PR + EVR-95) per layer — wormhole shape in cache space
2. V rank (PR + EVR-95) per layer — uniform or shaped?
3. K/V quantization error at Q8/Q6/Q4/Q2 per layer
4. Per-token novelty curve (how fast cache saturates)
5. Attention Gini per layer (concentration → eviction tolerance)
6. Certainty growth (entropy over sequence position)

This gives us the cache compression topography for 14B.
"""
import torch
import torch.nn.functional as F
import math
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 256

print("=" * 60)
print("STAGE 140 — KV CACHE GEOMETRY (14B)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 5, split="validation")

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True, attn_implementation="eager"
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = d // model.config.num_attention_heads
d_kv = n_kv_heads * head_dim
print(f"  L={L} d={d} n_kv_heads={n_kv_heads} head_dim={head_dim} d_kv={d_kv}", flush=True)

results = {"model": MODEL, "L": L, "d": d, "d_kv": d_kv, "seq_len": SEQ_LEN}

# ═══════════════════════════════════════════════════════
# Run forward pass and capture KV cache + attention weights
# ═══════════════════════════════════════════════════════
print("\nRunning forward pass with KV cache capture...", flush=True)

# We need attention weights, so use eager attention
inp = torch.tensor([val_tokens[:SEQ_LEN]], dtype=torch.long, device=device)

# Hook to capture K, V, and attention weights per layer
kv_cache = {}
attn_weights = {}

def make_attn_hook(layer_idx):
    def hook(module, args, kwargs, output):
        # For Qwen3, we need to intercept the attention computation
        # The KV cache is accessible via the past_key_value
        pass
    return hook

# Use model's built-in cache mechanism
with torch.no_grad():
    out = model(inp, use_cache=True, output_attentions=True)

# Extract attention weights
attns = out.attentions  # tuple of [batch, heads, seq, seq] per layer
past_kv_raw = out.past_key_values  # DynamicCache or tuple

# Handle DynamicCache vs tuple
if hasattr(past_kv_raw, 'key_cache'):
    # DynamicCache: .key_cache[layer], .value_cache[layer]
    past_kv = [(past_kv_raw.key_cache[i], past_kv_raw.value_cache[i]) for i in range(L)]
else:
    past_kv = list(past_kv_raw)

print(f"  Got {len(attns)} attention layers, {len(past_kv)} KV pairs", flush=True)
print(f"  K shape: {past_kv[0][0].shape}, V shape: {past_kv[0][1].shape}")

# ═══════════════════════════════════════════════════════
# 1. K and V rank (PR + EVR-95) per layer
# ═══════════════════════════════════════════════════════
print("\n--- K/V rank per layer ---", flush=True)

rank_data = []
for i in range(L):
    K = past_kv[i][0][0].float()  # [n_kv_heads, seq, head_dim]
    V = past_kv[i][1][0].float()

    # Reshape to [seq, d_kv] by concatenating heads
    K_flat = K.permute(1, 0, 2).reshape(SEQ_LEN, -1)  # [seq, d_kv]
    V_flat = V.permute(1, 0, 2).reshape(SEQ_LEN, -1)

    # Center
    K_c = K_flat - K_flat.mean(0, keepdim=True)
    V_c = V_flat - V_flat.mean(0, keepdim=True)

    # SVD
    _, S_k, _ = torch.linalg.svd(K_c, full_matrices=False)
    _, S_v, _ = torch.linalg.svd(V_c, full_matrices=False)

    # PR
    s2_k = S_k ** 2; pr_k = (s2_k.sum() ** 2 / (s2_k ** 2).sum()).item()
    s2_v = S_v ** 2; pr_v = (s2_v.sum() ** 2 / (s2_v ** 2).sum()).item()

    # EVR-95
    cum_k = torch.cumsum(s2_k, 0) / s2_k.sum()
    cum_v = torch.cumsum(s2_v, 0) / s2_v.sum()
    evr95_k = (cum_k < 0.95).sum().item() + 1
    evr95_v = (cum_v < 0.95).sum().item() + 1
    evr99_k = (cum_k < 0.99).sum().item() + 1
    evr99_v = (cum_v < 0.99).sum().item() + 1

    rank_data.append({
        "layer": i,
        "pr_k": round(pr_k, 1), "pr_v": round(pr_v, 1),
        "evr95_k": evr95_k, "evr95_v": evr95_v,
        "evr99_k": evr99_k, "evr99_v": evr99_v,
    })

    if i % 4 == 0 or i == L - 1:
        print(f"  L{i:>2}: K pr={pr_k:.1f} evr95={evr95_k:>3} evr99={evr99_k:>3}  |  V pr={pr_v:.1f} evr95={evr95_v:>3} evr99={evr99_v:>3}")

results["rank_per_layer"] = rank_data

# ═══════════════════════════════════════════════════════
# 2. Quantization error per layer (Q8, Q6, Q4, Q2)
# ═══════════════════════════════════════════════════════
print("\n--- Quantization error per layer ---", flush=True)

quant_data = []
for i in range(L):
    K = past_kv[i][0][0].float().permute(1, 0, 2).reshape(SEQ_LEN, -1)
    V = past_kv[i][1][0].float().permute(1, 0, 2).reshape(SEQ_LEN, -1)

    layer_q = {"layer": i}
    for bits in [8, 6, 4, 2]:
        half = 2 ** (bits - 1)
        for name, T in [("K", K), ("V", V)]:
            scale = T.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
            T_q = torch.round(T / scale).clamp(-(half-1), half-1) * scale
            err = (T - T_q).norm() / T.norm()
            layer_q[f"{name}_Q{bits}_err"] = round(err.item(), 4)

    quant_data.append(layer_q)
    if i % 8 == 0 or i == L - 1:
        print(f"  L{i:>2}: K_Q4={layer_q['K_Q4_err']:.3f} V_Q4={layer_q['V_Q4_err']:.3f} K_Q2={layer_q['K_Q2_err']:.3f} V_Q2={layer_q['V_Q2_err']:.3f}")

results["quant_per_layer"] = quant_data

# ═══════════════════════════════════════════════════════
# 3. Per-token novelty (how fast K cache saturates)
# ═══════════════════════════════════════════════════════
print("\n--- Per-token novelty (K cache saturation) ---", flush=True)

novelty_data = []
for i in range(L):
    K = past_kv[i][0][0].float()  # [n_kv_heads, seq, head_dim]
    K_flat = K.permute(1, 0, 2).reshape(SEQ_LEN, -1)  # [seq, d_kv]

    # Incremental PR: how much does each new token add to the subspace?
    novelty = []
    for t in range(1, SEQ_LEN):
        K_so_far = K_flat[:t]
        K_c = K_so_far - K_so_far.mean(0, keepdim=True)
        if t < 3:
            novelty.append(1.0)
            continue
        _, S, _ = torch.linalg.svd(K_c, full_matrices=False)
        s2 = S ** 2
        # New token's projection onto existing subspace
        new_vec = K_flat[t] - K_so_far.mean(0)
        # Residual after projecting onto top-k components
        _, _, Vt = torch.linalg.svd(K_c, full_matrices=False)
        k = min(10, t - 1)
        proj = new_vec @ Vt[:k].T @ Vt[:k]
        residual = (new_vec - proj).norm() / new_vec.norm()
        novelty.append(residual.item())

        if t >= 50:  # sample every 10 after 50
            if t % 10 != 0:
                continue

    # Summarize: early vs late novelty
    early = np.mean(novelty[:20]) if len(novelty) > 20 else 0
    late = np.mean(novelty[-20:]) if len(novelty) > 20 else 0
    novelty_data.append({"layer": i, "early_novelty": round(early, 4), "late_novelty": round(late, 4)})

    if i % 8 == 0 or i == L - 1:
        print(f"  L{i:>2}: early={early:.3f} late={late:.3f} ratio={early/max(late,1e-6):.1f}x")

results["novelty_per_layer"] = novelty_data

# ═══════════════════════════════════════════════════════
# 4. Attention Gini per layer (eviction tolerance)
# ═══════════════════════════════════════════════════════
print("\n--- Attention Gini per layer ---", flush=True)

gini_data = []
for i in range(L):
    A = attns[i][0].float()  # [n_heads, seq, seq]
    # Average Gini across heads and query positions
    ginis = []
    for h in range(A.shape[0]):
        for q in range(A.shape[1]):
            row = A[h, q].sort()[0]
            n = len(row)
            idx = torch.arange(1, n + 1, device=row.device, dtype=torch.float)
            gini = (2 * (idx * row).sum() / (n * row.sum()) - (n + 1) / n).item()
            ginis.append(gini)
    avg_gini = np.mean(ginis)
    gini_data.append({"layer": i, "gini": round(avg_gini, 4)})

    if i % 8 == 0 or i == L - 1:
        print(f"  L{i:>2}: gini={avg_gini:.3f}")

results["gini_per_layer"] = gini_data

# ═══════════════════════════════════════════════════════
# 5. Certainty growth (entropy over sequence)
# ═══════════════════════════════════════════════════════
print("\n--- Certainty growth over sequence ---", flush=True)

logits = out.logits[0].float()  # [seq, vocab]
probs = F.softmax(logits, dim=-1)
entropy = -(probs * probs.clamp(min=1e-10).log()).sum(-1)  # [seq]
top1_conf = probs.max(-1)[0]  # [seq]

# Bin by position
bins = [(0, 10), (10, 50), (50, 128), (128, 200), (200, SEQ_LEN)]
certainty_data = []
for start, end in bins:
    end = min(end, SEQ_LEN)
    if start >= end: continue
    avg_ent = entropy[start:end].mean().item()
    avg_conf = top1_conf[start:end].mean().item()
    certainty_data.append({
        "pos_range": f"{start}-{end}",
        "avg_entropy": round(avg_ent, 3),
        "avg_top1_conf": round(avg_conf, 3),
    })
    print(f"  pos {start:>3}-{end:<3}: entropy={avg_ent:.3f} top1_conf={avg_conf:.3f}")

results["certainty_growth"] = certainty_data

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CACHE GEOMETRY SUMMARY (14B)")
print(f"{'='*60}")

# K rank shape
k_ranks = [r["evr95_k"] for r in rank_data]
v_ranks = [r["evr95_v"] for r in rank_data]
print(f"\n  K rank (EVR-95):")
print(f"    Min: {min(k_ranks)} Max: {max(k_ranks)} Mean: {np.mean(k_ranks):.0f}")
print(f"    Wormhole shape? {'YES' if max(k_ranks)/min(k_ranks) > 5 else 'NO'} (ratio {max(k_ranks)/min(k_ranks):.1f}x)")

print(f"\n  V rank (EVR-95):")
print(f"    Min: {min(v_ranks)} Max: {max(v_ranks)} Mean: {np.mean(v_ranks):.0f}")
print(f"    Uniform? {'YES' if max(v_ranks)/min(v_ranks) < 3 else 'NO'} (ratio {max(v_ranks)/min(v_ranks):.1f}x)")

# Quantization
k_q4 = [q["K_Q4_err"] for q in quant_data]
v_q4 = [q["V_Q4_err"] for q in quant_data]
print(f"\n  Quantization (Q4 error):")
print(f"    K: {np.mean(k_q4):.3f} mean, {min(k_q4):.3f}-{max(k_q4):.3f} range")
print(f"    V: {np.mean(v_q4):.3f} mean, {min(v_q4):.3f}-{max(v_q4):.3f} range")

# Gini
ginis = [g["gini"] for g in gini_data]
print(f"\n  Attention Gini: {np.mean(ginis):.3f} mean ({min(ginis):.3f}-{max(ginis):.3f})")

# Certainty
print(f"\n  Certainty growth:")
for c in certainty_data:
    print(f"    {c['pos_range']}: entropy={c['avg_entropy']:.3f} conf={c['avg_top1_conf']:.3f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/stage140_cache_geometry_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/stage140_cache_geometry_14b.json", flush=True)
