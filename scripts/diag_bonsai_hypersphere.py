"""Diagnostic: is Bonsai-8B-mlx-1bit (PrismML's true binary Qwen3-8B) on
the unit hypersphere? Or does it have CV like a normal pretrained model?

Bonsai stores each linear as:
  weight:  [out, in/32] uint32   — packed binary bits (1 bit per weight)
  scales:  [out, in/128] fp16    — per-group of 128 weights
  biases:  [out, in/128] fp16    — per-group of 128 weights

Effective forward weight: w = bit * scale + bias (per group).
For bit=0: w = bias    For bit=1: w = scale + bias.
For binary {-x, +x}: bias = -x, scale = 2x.

Row norm² for row r:
  sum over groups g of:
    n1_g * (scale_g + bias_g)²  +  n0_g * bias_g²
  where n1_g = count of 1-bits in group g, n0_g = 128 - n1_g.

If row L2 norms cluster tightly (low CV), Bonsai accidentally landed on
a hypersphere structure. If CV is large, magnitude variation lives in
the per-group scales (closer to standard PTQ pattern).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


CHECKPOINT_PATH = Path(
    "/Users/abundancemachine/.cache/huggingface/hub/"
    "models--prism-ml--Bonsai-8B-mlx-1bit/snapshots/"
    "019934f87a61a654e3960ea22f53688e0d2c49ba/model.safetensors"
)
RESULTS_PATH = Path("results/diag_bonsai_hypersphere.json")
GROUP_SIZE = 128
BITS_PER_UINT32 = 32

TARGET_NAME_MARKERS = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


def popcount_uint32_torch(x):
    """popcount for torch uint32 tensors using bitwise tricks. Returns int32 tensor."""
    x = x.to(torch.int64)  # avoid overflow
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    return ((x * 0x01010101) >> 24) & 0xff


print(f"Reading {CHECKPOINT_PATH.name}...")
all_norms = []
per_type = defaultdict(list)
per_layer_records = []

with safe_open(str(CHECKPOINT_PATH), framework="pt") as f:
    keys = list(f.keys())
    # Find linears: anything with .weight + .scales + .biases triplet
    weight_keys = [k for k in keys if k.endswith(".weight") and k.replace(".weight", ".scales") in keys]
    weight_keys = [k for k in weight_keys if any(m in k for m in TARGET_NAME_MARKERS)]
    print(f"Found {len(weight_keys)} body linears")

    for k in weight_keys:
        prefix = k[:-len(".weight")]
        weights_packed = f.get_tensor(k)        # [out, in/32]  uint32
        scales = f.get_tensor(prefix + ".scales").float()    # [out, in/128]
        biases = f.get_tensor(prefix + ".biases").float()    # [out, in/128]

        out_features, in_div_32 = weights_packed.shape
        in_features = in_div_32 * BITS_PER_UINT32
        n_groups = in_features // GROUP_SIZE
        uints_per_group = GROUP_SIZE // BITS_PER_UINT32   # 4

        if scales.shape != (out_features, n_groups):
            print(f"  skip {k}: shape mismatch scales={scales.shape}, expected ({out_features}, {n_groups})")
            continue

        # Reshape weights to [out, n_groups, uints_per_group]
        w = weights_packed.reshape(out_features, n_groups, uints_per_group)
        # Popcount each uint32 → bit count per uint32, then sum over uints_per_group → bits per group
        n1 = popcount_uint32_torch(w).sum(dim=-1).float()        # [out, n_groups]
        n0 = GROUP_SIZE - n1

        # Sum of squares contribution per (row, group)
        # = n1 * (scale + bias)² + n0 * bias²
        sum_sq_per_group = n1 * (scales + biases).pow(2) + n0 * biases.pow(2)   # [out, n_groups]
        row_norm_sq = sum_sq_per_group.sum(dim=-1)   # [out]
        row_norms = row_norm_sq.sqrt().cpu().numpy()

        all_norms.extend(row_norms.tolist())

        proj_type = next((m for m in TARGET_NAME_MARKERS if m in k), "other")
        per_type[proj_type].extend(row_norms.tolist())

        per_layer_records.append({
            "name": prefix,
            "shape": [out_features, in_features],
            "mean": float(row_norms.mean()),
            "std": float(row_norms.std()),
            "min": float(row_norms.min()),
            "max": float(row_norms.max()),
            "cv": float(row_norms.std() / row_norms.mean()),
        })

all_arr = np.array(all_norms)
print(f"\nTotal rows: {len(all_norms):,} across {len(per_layer_records)} matrices")

print(f"\n{'='*70}\nBONSAI-8B 1-bit — effective forward row L2 norms\n{'='*70}")
print(f"  mean:          {all_arr.mean():.4f}")
print(f"  std:           {all_arr.std():.4f}")
print(f"  min:           {all_arr.min():.4f}")
print(f"  max:           {all_arr.max():.4f}")
print(f"  median:        {np.median(all_arr):.4f}")
print(f"  p1:            {np.percentile(all_arr, 1):.4f}")
print(f"  p99:           {np.percentile(all_arr, 99):.4f}")
print(f"  CV (std/mean): {all_arr.std() / all_arr.mean():.4f}")

print(f"\nPer projection type:")
print(f"  {'type':<12} {'count':>8} {'mean':>8} {'std':>8} {'cv':>8}")
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

# Compare to:
#   Qwen3-0.6B base:    CV 0.32 (mean ~0.97)
#   nGPT τ=1.0:         CV 0.00 (mean = 1.0)
#   BitNet b1.58:       CV 0.38 master, 0.31 ternary effective
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
print(f"  Qwen3-0.6B base FP:           mean=0.97, CV=0.32")
print(f"  nGPT-converted τ=1.0:         mean=1.00, CV=0.00 (perfect sphere)")
print(f"  BitNet b1.58 master:          mean=3.85, CV=0.38")
print(f"  BitNet b1.58 ternary effective: mean=2.32, CV=0.31")
print(f"  Bonsai-8B 1-bit effective:    mean={all_arr.mean():.2f}, CV={all_arr.std()/all_arr.mean():.4f}")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": "prism-ml/Bonsai-8B-mlx-1bit",
        "n_rows": len(all_norms),
        "n_matrices": len(per_layer_records),
        "stats": {
            "mean": float(all_arr.mean()),
            "std": float(all_arr.std()),
            "min": float(all_arr.min()),
            "max": float(all_arr.max()),
            "median": float(np.median(all_arr)),
            "p1": float(np.percentile(all_arr, 1)),
            "p99": float(np.percentile(all_arr, 99)),
            "cv": float(all_arr.std() / all_arr.mean()),
        },
        "by_type": type_summary,
        "per_layer": per_layer_records,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
