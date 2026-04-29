"""Stage 162: per-layer CV profile on Qwen3-0.6B base.

We have overall CV (0.32) and per-projection-type CV from the diag_row_norms
script. This breaks it down further: CV across LAYERS (0-27 for Qwen3-0.6B).

Question: do early layers have different CV than late layers? If yes, the
nGPT conversion will pay differently across layers. Identifies which
layers are "easy" vs "hard" to convert.

Hypothesis: late layers (closer to output) tend to have higher CV because
they encode more idiosyncratic / specialized patterns. Early layers have
more uniform behavior (basic feature extraction). If true, late layers
would dominate the conversion cost.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM


CHECKPOINT = "Qwen/Qwen3-0.6B"
RESULTS_PATH = Path("results/stage162_per_layer_cv.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


print(f"Loading {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).eval()

# Bucket by layer index extracted from name (e.g. "model.layers.14.self_attn.q_proj")
per_layer = defaultdict(list)        # {layer_idx: [row norms across all linears in that layer]}
per_layer_per_type = defaultdict(lambda: defaultdict(list))  # {layer_idx: {type: norms}}

for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear):
        continue
    if not any(m in name for m in TARGET_NAMES):
        continue
    match = re.search(r"layers\.(\d+)\.", name)
    if not match:
        continue
    layer_idx = int(match.group(1))
    proj_type = next(m for m in TARGET_NAMES if m in name)

    norms = mod.weight.detach().norm(dim=-1).cpu().numpy().tolist()
    per_layer[layer_idx].extend(norms)
    per_layer_per_type[layer_idx][proj_type].extend(norms)

n_layers = max(per_layer.keys()) + 1
print(f"Found {n_layers} layers.")

print(f"\n{'='*78}")
print(f"{'layer':>5} {'count':>6} {'mean':>8} {'std':>8} {'CV':>8} {'p1':>8} {'p99':>8}")
print(f"{'='*78}")
records = []
for L in range(n_layers):
    arr = np.array(per_layer[L])
    cv = arr.std() / arr.mean()
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)
    print(f"{L:>5} {len(arr):>6} {arr.mean():>8.3f} {arr.std():>8.3f} {cv:>8.4f} {p1:>8.3f} {p99:>8.3f}")
    type_cvs = {}
    for t in TARGET_NAMES:
        if t in per_layer_per_type[L]:
            ta = np.array(per_layer_per_type[L][t])
            type_cvs[t] = float(ta.std() / ta.mean())
    records.append({
        "layer": L,
        "count": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(cv),
        "p1": float(p1),
        "p99": float(p99),
        "by_type_cv": type_cvs,
    })

# Compare early vs late layers
all_cv = np.array([r["cv"] for r in records])
mid = len(all_cv) // 2
early_cv = all_cv[:mid].mean()
late_cv = all_cv[mid:].mean()
print(f"\nFirst {mid} layers (early): mean CV = {early_cv:.4f}")
print(f"Last {n_layers - mid} layers (late): mean CV = {late_cv:.4f}")
print(f"Late/Early ratio: {late_cv/early_cv:.3f}x")

# Find peak CV layer
peak_idx = int(np.argmax(all_cv))
trough_idx = int(np.argmin(all_cv))
print(f"\nPeak CV at layer {peak_idx}: {all_cv[peak_idx]:.4f}")
print(f"Trough CV at layer {trough_idx}: {all_cv[trough_idx]:.4f}")
print(f"Spread: {all_cv.max() / all_cv.min():.2f}x between most/least uniform layer")

# Try plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), all_cv, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("layer index")
    ax.set_ylabel("CV (std/mean of row L2 norms)")
    ax.set_title(f"{CHECKPOINT} per-layer CV profile (lower = more spherical)")
    ax.axhline(all_cv.mean(), color="gray", linestyle="--", alpha=0.5,
               label=f"mean = {all_cv.mean():.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_path = Path("results/stage162_per_layer_cv.png")
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot saved: {plot_path}")
except Exception as e:
    print(f"(plot skipped: {e})")

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_layers": n_layers,
        "early_late_ratio": float(late_cv / early_cv),
        "peak_layer": peak_idx,
        "peak_cv": float(all_cv[peak_idx]),
        "trough_layer": trough_idx,
        "trough_cv": float(all_cv[trough_idx]),
        "spread": float(all_cv.max() / all_cv.min()),
        "per_layer": records,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}")
