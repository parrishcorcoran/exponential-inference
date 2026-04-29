"""Diagnostic on BitNet b1.58 master weights.

Tests our 'natural attractor' thesis: did BitNet training (which uses ternary
forward) produce master weights that drift toward unit norm? Or do they
look like a normal pretrained transformer (CV ~0.32)?

If BitNet's training selected for low CV (uniform row norms), that's strong
evidence that:
  1. Hyperspherical geometry IS the natural attractor of transformer training
  2. Quantization-aware training drove the master weights even closer to it
  3. Our nGPT-conversion recipe is doing what BitNet's natural drift does

If BitNet's master weights have HIGH CV (lots of magnitude variation),
that's evidence the per-channel gamma_w bridge is doing important work
that ours-with-alpha would also need.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModelForCausalLM


CHECKPOINT = "1bitLLM/bitnet_b1_58-large"
RESULTS_PATH = Path("results/diag_bitnet_row_norms.json")


print(f"Loading {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval()
print(f"Loaded. Total params: {sum(p.numel() for p in model.parameters()):,}")

# Inspect what kind of layers BitNet uses
linear_types = defaultdict(int)
for name, mod in model.named_modules():
    cls = type(mod).__name__
    if "Linear" in cls or "Bit" in cls:
        linear_types[cls] += 1
print(f"\nLayer types found: {dict(linear_types)}")

# Collect row norms from all weight matrices in the body
all_norms = []
per_type = defaultdict(list)
per_layer = []

for name, p in model.named_parameters():
    if "weight" not in name:
        continue
    if p.ndim != 2:
        continue
    if "embed" in name.lower() or "lm_head" in name.lower():
        continue

    W = p.detach().float()
    norms = W.norm(dim=-1).cpu().numpy()
    all_norms.extend(norms.tolist())

    # Tag by which projection
    proj_type = "other"
    for marker in ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"):
        if marker in name:
            proj_type = marker
            break
    per_type[proj_type].extend(norms.tolist())

    per_layer.append({
        "name": name,
        "shape": list(W.shape),
        "n_rows": int(norms.shape[0]),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
    })

all_arr = np.array(all_norms)
print(f"\nTotal rows: {len(all_norms):,} across {len(per_layer)} matrices")
print(f"\n{'='*70}\nMASTER WEIGHTS — row L2 norm distribution\n{'='*70}")
print(f"  mean:          {all_arr.mean():.4f}")
print(f"  std:           {all_arr.std():.4f}")
print(f"  min:           {all_arr.min():.4f}")
print(f"  max:           {all_arr.max():.4f}")
print(f"  median:        {np.median(all_arr):.4f}")
print(f"  p1:            {np.percentile(all_arr, 1):.4f}")
print(f"  p99:           {np.percentile(all_arr, 99):.4f}")
print(f"  CV (std/mean): {all_arr.std() / all_arr.mean():.4f}")

print(f"\nPer projection type:")
print(f"  {'type':<12} {'n':>8} {'mean':>8} {'std':>8} {'cv':>8}")
type_summary = {}
for t, vals in per_type.items():
    arr = np.array(vals)
    cv = arr.std() / arr.mean()
    print(f"  {t:<12} {len(arr):>8} {arr.mean():>8.3f} {arr.std():>8.3f} {cv:>8.4f}")
    type_summary[t] = {
        "count": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(cv),
    }

# Now compute what the ternary projection looks like
# BitNet's projection: w_q = clip(round(w / gamma), -1, 1) where gamma = mean(|W|)
# The effective forward weight is gamma * w_q
print(f"\n{'='*70}\nTERNARY PROJECTION — effective forward weights\n{'='*70}")

ternary_norms = []
for name, p in model.named_parameters():
    if "weight" not in name:
        continue
    if p.ndim != 2:
        continue
    if "embed" in name.lower() or "lm_head" in name.lower():
        continue
    W = p.detach().float()
    # BitNet b1.58 ternary: scale = mean(|W|), then round to {-1, 0, +1}
    gamma = W.abs().mean()
    W_q = torch.clamp(torch.round(W / gamma), -1, 1)  # ternary
    W_eff = gamma * W_q  # effective forward weight
    norms = W_eff.norm(dim=-1).cpu().numpy()
    ternary_norms.extend(norms.tolist())

tern_arr = np.array(ternary_norms)
print(f"  mean:          {tern_arr.mean():.4f}")
print(f"  std:           {tern_arr.std():.4f}")
print(f"  min:            {tern_arr.min():.4f}")
print(f"  max:            {tern_arr.max():.4f}")
print(f"  CV (std/mean): {tern_arr.std() / tern_arr.mean():.4f}")
print(f"  fraction-zero rows: {(tern_arr == 0).sum() / len(tern_arr):.4f}")

# Save
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_rows_total": len(all_norms),
        "master": {
            "mean": float(all_arr.mean()),
            "std": float(all_arr.std()),
            "cv": float(all_arr.std() / all_arr.mean()),
            "p1": float(np.percentile(all_arr, 1)),
            "p99": float(np.percentile(all_arr, 99)),
            "min": float(all_arr.min()),
            "max": float(all_arr.max()),
        },
        "ternary_effective": {
            "mean": float(tern_arr.mean()),
            "std": float(tern_arr.std()),
            "cv": float(tern_arr.std() / tern_arr.mean()),
            "min": float(tern_arr.min()),
            "max": float(tern_arr.max()),
        },
        "by_type": type_summary,
        "per_layer": per_layer,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
