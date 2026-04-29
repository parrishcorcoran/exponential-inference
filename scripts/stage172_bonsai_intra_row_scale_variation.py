"""Stage 172: How much do Bonsai's per-group scales vary WITHIN each row?

Bonsai stores 32 scales per row (one per 128-weight group, with d_in=4096).
If those 32 scales are roughly constant per row, then a single per-row α
captures the same information — our recipe is sufficient.

If those 32 scales vary 5-10× within a row, Bonsai's per-group structure
is doing real work that our 1-α-per-row would miss. Either we need:
  (a) Block-α (multiple scales per row, like Bonsai)
  (b) Master-weight QAT during anneal (to train the model into a state
      where 1 scale per row suffices)

Diagnostic: for each row of each linear, compute CV of its 32 per-group
scales. Aggregate distribution. Mean intra-row CV ≈ 0 → our α suffices.
Mean intra-row CV >> 0 → Bonsai's per-group is load-bearing.
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
RESULTS_PATH = Path("results/stage172_bonsai_intra_row_scale.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


per_type_intra_cv = defaultdict(list)   # CV of scales within each row
per_type_inter_cv = defaultdict(list)   # CV of (mean per-row scale) across rows
all_intra = []
all_inter = []
type_summary = {}

with safe_open(str(CHECKPOINT_PATH), framework="pt") as f:
    keys = list(f.keys())
    weight_keys = [k for k in keys if k.endswith(".weight") and k.replace(".weight", ".scales") in keys]
    weight_keys = [k for k in weight_keys if any(m in k for m in TARGET_NAMES)]

    for k in weight_keys:
        prefix = k[:-len(".weight")]
        scales = f.get_tensor(prefix + ".scales").float()  # [out, n_groups]
        biases = f.get_tensor(prefix + ".biases").float()  # [out, n_groups]

        # Effective magnitude per group ≈ |scale + bias| ~= scale magnitude in symmetric case
        # For binary {-x, +x}: bias = -x, scale = 2x → "effective per-group magnitude" = scale/2
        # Use abs(scale) as proxy for per-group magnitude
        group_mags = scales.abs()  # [out, n_groups]

        # Intra-row CV: for each row, how much do its n_groups magnitudes vary?
        intra_cv = (group_mags.std(dim=-1) / group_mags.mean(dim=-1).clamp(min=1e-8))   # [out]
        # Inter-row variation: how much does the per-row mean magnitude vary across rows?
        per_row_mean = group_mags.mean(dim=-1)  # [out]
        inter_cv_layer = (per_row_mean.std() / per_row_mean.mean()).item()

        proj_type = next((m for m in TARGET_NAMES if m in k), "other")
        per_type_intra_cv[proj_type].extend(intra_cv.cpu().numpy().tolist())
        per_type_inter_cv[proj_type].append(inter_cv_layer)
        all_intra.extend(intra_cv.cpu().numpy().tolist())
        all_inter.append(inter_cv_layer)

print(f"\n{'='*78}")
print("Bonsai per-row scale variation analysis")
print(f"{'='*78}")

print(f"\nOVERALL")
all_intra_arr = np.array(all_intra)
all_inter_arr = np.array(all_inter)
print(f"  Intra-row CV (within each row, across its 32 groups):")
print(f"    mean: {all_intra_arr.mean():.4f}")
print(f"    p1:   {np.percentile(all_intra_arr, 1):.4f}")
print(f"    p99:  {np.percentile(all_intra_arr, 99):.4f}")
print(f"  Inter-row CV (across rows, mean-per-row magnitudes):")
print(f"    mean: {all_inter_arr.mean():.4f}")
print(f"    range: {all_inter_arr.min():.4f} to {all_inter_arr.max():.4f}")

print(f"\nBY PROJECTION TYPE")
print(f"  {'type':<14} {'intra_CV_mean':<16} {'intra_CV_p99':<16} {'inter_CV_mean':<16}")
for t in TARGET_NAMES:
    if t not in per_type_intra_cv: continue
    intra = np.array(per_type_intra_cv[t])
    inter = np.array(per_type_inter_cv[t])
    print(f"  {t:<14} {intra.mean():<16.4f} {np.percentile(intra,99):<16.4f} {inter.mean():<16.4f}")
    type_summary[t] = {
        "intra_cv_mean": float(intra.mean()),
        "intra_cv_p99": float(np.percentile(intra, 99)),
        "intra_cv_max": float(intra.max()),
        "inter_cv_mean": float(inter.mean()),
        "n_rows": len(intra),
    }

print(f"\n{'='*78}")
print("INTERPRETATION")
print(f"{'='*78}")
mean_intra = float(all_intra_arr.mean())
print(f"  Mean intra-row CV = {mean_intra:.4f}")
print(f"  If close to 0:    a single per-row α captures the magnitude info Bonsai uses")
print(f"  If much > 0:      Bonsai's 32 scales per row carry load that 1 α cannot")
print(f"")
if mean_intra < 0.05:
    print(f"  VERDICT: 1-α-per-row is SUFFICIENT — Bonsai's per-group is mostly redundant")
    print(f"           Our recipe should work at binary with QAT (no need for block-α)")
elif mean_intra < 0.20:
    print(f"  VERDICT: 1-α-per-row is MARGINAL — works but loses some info")
    print(f"           Block-α (2-4 scales per row) might help recover the gap")
else:
    print(f"  VERDICT: per-group scales DO real work — need block-α or QAT can compensate")
    print(f"           Pure α-per-row + post-hoc binary will probably not match Bonsai")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": "prism-ml/Bonsai-8B-mlx-1bit",
        "n_rows_total": len(all_intra),
        "intra_row_cv_overall": {
            "mean": float(all_intra_arr.mean()),
            "p1": float(np.percentile(all_intra_arr, 1)),
            "p99": float(np.percentile(all_intra_arr, 99)),
            "min": float(all_intra_arr.min()),
            "max": float(all_intra_arr.max()),
        },
        "inter_row_cv_overall_mean": float(all_inter_arr.mean()),
        "by_type": type_summary,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
