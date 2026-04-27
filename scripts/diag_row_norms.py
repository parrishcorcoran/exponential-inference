"""Measure base Qwen3-0.6B's per-row L2 norm distribution across every
nn.Linear in the body. Tells us: how 'spherical' is the model already?

If row norms are tightly clustered around one value, the model has
already-uniform geometry from pretraining alone — partial nGPT-shape
is somewhat baked in. If norms span orders of magnitude, ordinary
pretraining doesn't select for unit-norm and the anneal is forcing
something the model wasn't doing.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM


CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
TAG = os.environ.get("RUN_TAG", CHECKPOINT.split("/")[-1].replace(".", ""))
RESULTS_PATH = Path(f"results/diag_row_norms_{TAG}.json")
PLOT_PATH = Path(f"results/diag_row_norms_{TAG}.png")
TARGET_NAME_MARKERS = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")


print(f"Loading {CHECKPOINT} on CPU for inspection...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).eval()

# Collect row norms by layer-type
by_type = defaultdict(list)
all_norms = []
per_layer = []

for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if not any(m in name for m in TARGET_NAME_MARKERS):
        continue
    W = module.weight.data
    norms = W.norm(dim=-1).cpu().numpy()  # one norm per row
    proj_type = next(m for m in TARGET_NAME_MARKERS if m in name)
    by_type[proj_type].extend(norms.tolist())
    all_norms.extend(norms.tolist())
    per_layer.append({
        "name": name,
        "shape": list(W.shape),
        "n_rows": int(norms.shape[0]),
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "median": float(sorted(norms.tolist())[len(norms)//2]),
    })

import numpy as np
all_arr = np.array(all_norms)
print(f"\nTotal rows inspected: {len(all_norms):,} across {len(per_layer)} linears")

print(f"\n{'='*70}")
print("OVERALL ROW-NORM DISTRIBUTION")
print(f"{'='*70}")
print(f"  count:    {len(all_arr):,}")
print(f"  mean:     {all_arr.mean():.4f}")
print(f"  std:      {all_arr.std():.4f}")
print(f"  min:      {all_arr.min():.4f}")
print(f"  max:      {all_arr.max():.4f}")
print(f"  median:   {np.median(all_arr):.4f}")
print(f"  p1:       {np.percentile(all_arr, 1):.4f}")
print(f"  p5:       {np.percentile(all_arr, 5):.4f}")
print(f"  p95:      {np.percentile(all_arr, 95):.4f}")
print(f"  p99:      {np.percentile(all_arr, 99):.4f}")
print(f"  spread (max/min):  {all_arr.max() / all_arr.min():.2f}x")
print(f"  cv (std/mean):     {all_arr.std() / all_arr.mean():.4f}")

print(f"\n{'='*70}")
print("PER PROJECTION TYPE")
print(f"{'='*70}")
print(f"  {'type':<12} {'count':>8} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'spread':>8}")
type_summary = {}
for t in TARGET_NAME_MARKERS:
    arr = np.array(by_type[t])
    if len(arr) == 0: continue
    spread = arr.max() / arr.min()
    type_summary[t] = {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "spread_ratio": float(spread),
    }
    print(f"  {t:<12} {len(arr):>8} {arr.mean():>8.3f} {arr.std():>8.3f} "
          f"{arr.min():>8.3f} {arr.max():>8.3f} {spread:>7.2f}x")

# Histogram (terminal-style)
print(f"\n{'='*70}")
print("HISTOGRAM (overall row norms)")
print(f"{'='*70}")
bins = np.linspace(all_arr.min(), all_arr.max(), 30)
hist, edges = np.histogram(all_arr, bins=bins)
max_h = hist.max()
for i, h in enumerate(hist):
    bar = "#" * int(40 * h / max_h)
    print(f"  {edges[i]:>6.3f} - {edges[i+1]:>6.3f}  {h:>6} | {bar}")

# Try a matplotlib plot too
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in TARGET_NAME_MARKERS:
        arr = np.array(by_type[t])
        if len(arr) == 0: continue
        ax.hist(arr, bins=60, alpha=0.5, label=t, density=True)
    ax.axvline(1.0, color="red", linestyle="--", label="unit norm (target)")
    ax.set_xlabel("row L2 norm")
    ax.set_ylabel("density")
    ax.set_title(f"Qwen3-0.6B per-row L2 norm distribution (base, no fine-tune)")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"\nPlot saved: {PLOT_PATH}")
except Exception as e:
    print(f"\n(plot skipped: {e})")

# Save JSON
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_linears": len(per_layer),
        "n_rows_total": len(all_norms),
        "overall": {
            "mean": float(all_arr.mean()),
            "std": float(all_arr.std()),
            "min": float(all_arr.min()),
            "max": float(all_arr.max()),
            "median": float(np.median(all_arr)),
            "p1": float(np.percentile(all_arr, 1)),
            "p99": float(np.percentile(all_arr, 99)),
            "spread_ratio": float(all_arr.max() / all_arr.min()),
            "cv": float(all_arr.std() / all_arr.mean()),
        },
        "by_type": type_summary,
        "per_layer": per_layer,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}")
