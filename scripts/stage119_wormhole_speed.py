"""Stage 119 — Wormhole speed: factorize throat for actual wall-clock speedup.

The wormhole-compressed model has:
  Throat (L7-14): KV rank 32, Q4, MLP 70% (rows zeroed)
  Passage (L15-27): KV rank 128-256, Q5, MLP 80-90%
  Mouths (L0-6, L28-39): KV rank 512, Q6, MLP 100%

But the matrices are still full-size — they just have low effective rank
or zeroed rows. No speed gain until we physically reshape them.

This script:
  1. SVD-factorize low-rank weight matrices into thin pairs
     W ≈ A × B where A is [out × r] and B is [r × in]
     One big matmul → two small matmuls = r/min(in,out) × fewer FLOPs
  2. Physically remove zeroed MLP rows (shrink the matrices)
  3. Benchmark: tokens/sec before and after

For throat KV at rank 32:
  Original: [1280 × 5120] = 6.5M multiply-adds
  Factored: [1280 × 32] + [32 × 5120] = 41K + 164K = 205K multiply-adds
  Speedup: 32× fewer FLOPs for KV projection in throat layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import gc
from pathlib import Path

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


class FactoredLinear(nn.Module):
    """Two thin matmuls replacing one fat one. W ≈ A @ B."""
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)  # [out, rank]
        self.B = nn.Parameter(B)  # [rank, in]
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        # x: [..., in] → [..., out]
        out = F.linear(F.linear(x, self.B), self.A)
        if self.bias is not None:
            out = out + self.bias
        return out


def factorize_linear(linear, rank):
    """Replace a nn.Linear with a FactoredLinear at given rank."""
    W = linear.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))

    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).to(linear.weight.dtype)  # [out, k]
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).to(linear.weight.dtype)  # [k, in]

    bias = linear.bias.data if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


class PrunedMLP(nn.Module):
    """MLP with physically removed rows instead of zeroed rows."""
    def __init__(self, gate_proj, up_proj, down_proj, keep_rows):
        super().__init__()
        # Only keep the active rows
        self.gate_proj = nn.Linear(gate_proj.in_features, keep_rows, bias=False)
        self.up_proj = nn.Linear(up_proj.in_features, keep_rows, bias=False)
        self.down_proj = nn.Linear(keep_rows, down_proj.out_features, bias=False)

        self.gate_proj.weight.data = gate_proj.weight.data[:keep_rows].clone()
        self.up_proj.weight.data = up_proj.weight.data[:keep_rows].clone()
        self.down_proj.weight.data = down_proj.weight.data[:, :keep_rows].clone()

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# Wormhole shape from stage 117
WORMHOLE = {
    # layer: (region, kv_target_rank, mlp_keep_pct)
    **{i: ("mouth", 512, 100) for i in range(0, 7)},
    **{i: ("throat", 32, 70) for i in range(7, 15)},
    **{i: ("passage", 128, 85) for i in range(15, 22)},
    **{i: ("reopen", 256, 90) for i in range(22, 28)},
    **{i: ("mouth", 512, 100) for i in range(28, 40)},
}

MODEL_PATH = "checkpoints/qwen_halo/wormhole_compressed"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"

print("=" * 60)
print("STAGE 119 — WORMHOLE SPEED")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load wormhole-compressed model
print("Loading wormhole-compressed model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size

# ── Benchmark BEFORE factorization ──
print("\n--- BEFORE factorization ---", flush=True)
ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(device)

# Warmup
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

N_TOKENS = 100
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N_TOKENS, do_sample=False)
torch.cuda.synchronize()
before_time = time.time() - t0
before_tps = N_TOKENS / before_time
before_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
before_mem = torch.cuda.memory_allocated() / 1e9

print(f"  Speed: {before_tps:.1f} tok/s")
print(f"  VRAM:  {before_mem:.1f} GB")
print(f"  Text:  {before_text[:80]}")

# Count params before
total_params_before = sum(p.numel() for p in model.parameters())
print(f"  Params: {total_params_before/1e9:.2f}B")

# ── Factorize throat and passage layers ──
print("\n--- Factorizing throat/passage layers ---", flush=True)

factored_count = 0
pruned_count = 0

for i in range(L):
    region, kv_rank, mlp_pct = WORMHOLE[i]
    layer = model.model.layers[i]

    # Factorize KV projections if rank < full
    if kv_rank < 512:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            factored = factorize_linear(proj, kv_rank)
            setattr(layer.self_attn, name, factored)
            factored_count += 1

    # Factorize Q and O projections in throat (they're Q4 quantized, low effective rank)
    if region == "throat":
        for name in ("q_proj", "o_proj"):
            proj = getattr(layer.self_attn, name)
            # Find effective rank (where singular values drop off)
            W = proj.weight.data.float()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            S_sq = S ** 2
            cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum()
            eff_rank = max((cumvar < 0.99).sum().item() + 1, 32)
            if eff_rank < min(W.shape) * 0.5:  # only factorize if significant savings
                factored = factorize_linear(proj, eff_rank)
                setattr(layer.self_attn, name, factored)
                factored_count += 1
                if i == 10:
                    print(f"  L{i} {name}: rank {min(W.shape)} → {eff_rank}")

    # Physically prune MLP
    if mlp_pct < 100:
        full_dim = layer.mlp.gate_proj.weight.shape[0]
        keep = int(full_dim * mlp_pct / 100)
        # Check if rows are actually zeroed (from compression)
        gate_norms = layer.mlp.gate_proj.weight.data.float().norm(dim=1)
        active = (gate_norms > 1e-6).sum().item()
        keep = min(keep, active) if active > 0 else keep

        if keep < full_dim:
            pruned_mlp = PrunedMLP(layer.mlp.gate_proj, layer.mlp.up_proj,
                                    layer.mlp.down_proj, keep)
            layer.mlp = pruned_mlp.to(device).to(torch.bfloat16)
            pruned_count += 1
            if i == 10:
                print(f"  L{i} MLP: {full_dim} → {keep} ({keep/full_dim*100:.0f}%)")

torch.cuda.empty_cache()
print(f"  Factored {factored_count} projections, pruned {pruned_count} MLPs")

# Count params after
total_params_after = sum(p.numel() for p in model.parameters())
print(f"  Params: {total_params_before/1e9:.2f}B → {total_params_after/1e9:.2f}B "
      f"({total_params_after/total_params_before*100:.1f}%)")

# ── Verify quality ──
print("\n--- Quality check ---", flush=True)
ids_check = tokenizer(PROMPT, return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    out_check = model.generate(ids_check, max_new_tokens=40, do_sample=False)
check_text = tokenizer.decode(out_check[0][ids_check.shape[1]:], skip_special_tokens=True)
print(f"  Text: {check_text[:80]}")

# ── Benchmark AFTER factorization ──
print("\n--- AFTER factorization ---", flush=True)

# Warmup
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=N_TOKENS, do_sample=False)
torch.cuda.synchronize()
after_time = time.time() - t0
after_tps = N_TOKENS / after_time
after_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
after_mem = torch.cuda.memory_allocated() / 1e9

print(f"  Speed: {after_tps:.1f} tok/s")
print(f"  VRAM:  {after_mem:.1f} GB")
print(f"  Text:  {after_text[:80]}")

# ── Summary ──
print(f"\n{'='*60}")
print("WORMHOLE SPEED RESULTS")
print(f"{'='*60}")
print(f"  Before: {before_tps:.1f} tok/s, {before_mem:.1f} GB, {total_params_before/1e9:.2f}B params")
print(f"  After:  {after_tps:.1f} tok/s, {after_mem:.1f} GB, {total_params_after/1e9:.2f}B params")
speedup = after_tps / before_tps
mem_save = 1 - after_mem / before_mem
param_save = 1 - total_params_after / total_params_before
print(f"  Speedup:     {speedup:.2f}x")
print(f"  Memory save: {mem_save*100:.1f}%")
print(f"  Param save:  {param_save*100:.1f}%")

# Multiple benchmark runs for stability
print(f"\n  Running 5 benchmark passes...", flush=True)
speeds = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=N_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    speeds.append(N_TOKENS / (time.time() - t0))

import numpy as np
print(f"  Stable speed: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} tok/s")

Path("results").mkdir(exist_ok=True)
with open("results/stage119_wormhole_speed.json", "w") as f:
    json.dump({
        "before_tps": before_tps, "after_tps": after_tps,
        "speedup": speedup,
        "before_mem_gb": before_mem, "after_mem_gb": after_mem,
        "params_before": total_params_before, "params_after": total_params_after,
        "factored_projections": factored_count, "pruned_mlps": pruned_count,
        "before_text": before_text[:100], "after_text": after_text[:100],
        "stable_speeds": speeds,
    }, f, indent=2)
print(f"\nSaved results/stage119_wormhole_speed.json", flush=True)
