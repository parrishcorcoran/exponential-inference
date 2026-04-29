"""Stage 164: per-layer CV on Bonsai-8B 1-bit (effective weights).

We have overall CV 0.275 from diag_bonsai_hypersphere. This breaks it
down by layer.

Question: does per-group binary quantization flatten ALL layers uniformly
toward the sphere? Or does it preserve some of the per-layer CV variation
that base FP models have?

If layer profile is uniformly flattened: per-group quantization is a
strong implicit normalizer. Predicts our explicit anneal will work
similarly across layers.

If some layers stay high-CV: those layers resist quantization-driven
sphericalization. Our anneal would need to push harder there.
"""
import json
import re
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
RESULTS_PATH = Path("results/stage164_per_layer_cv_bonsai.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def popcount_uint32_torch(x):
    x = x.to(torch.int64)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    return ((x * 0x01010101) >> 24) & 0xff


per_layer = defaultdict(list)
per_layer_per_type = defaultdict(lambda: defaultdict(list))

with safe_open(str(CHECKPOINT_PATH), framework="pt") as f:
    keys = list(f.keys())
    weight_keys = [k for k in keys if k.endswith(".weight") and k.replace(".weight", ".scales") in keys]
    weight_keys = [k for k in weight_keys if any(m in k for m in TARGET_NAMES)]

    for k in weight_keys:
        match = re.search(r"layers\.(\d+)\.", k)
        if not match:
            continue
        layer_idx = int(match.group(1))
        proj_type = next(m for m in TARGET_NAMES if m in k)

        prefix = k[:-len(".weight")]
        weights_packed = f.get_tensor(k)
        scales = f.get_tensor(prefix + ".scales").float()
        biases = f.get_tensor(prefix + ".biases").float()

        out_features, in_div_32 = weights_packed.shape
        in_features = in_div_32 * 32
        n_groups = in_features // GROUP_SIZE
        uints_per_group = GROUP_SIZE // 32

        if scales.shape != (out_features, n_groups):
            continue

        w = weights_packed.reshape(out_features, n_groups, uints_per_group)
        n1 = popcount_uint32_torch(w).sum(dim=-1).float()
        n0 = GROUP_SIZE - n1
        sum_sq = n1 * (scales + biases).pow(2) + n0 * biases.pow(2)
        row_norms = sum_sq.sum(dim=-1).sqrt().cpu().numpy()

        per_layer[layer_idx].extend(row_norms.tolist())
        per_layer_per_type[layer_idx][proj_type].extend(row_norms.tolist())

n_layers = max(per_layer.keys()) + 1
print(f"Bonsai-8B has {n_layers} layers.")

print(f"\n{'='*78}")
print(f"{'layer':>5} {'count':>6} {'mean':>8} {'std':>8} {'CV':>8} {'p1':>8} {'p99':>8}")
print(f"{'='*78}")
records = []
for L in range(n_layers):
    arr = np.array(per_layer[L])
    cv = arr.std() / arr.mean()
    print(f"{L:>5} {len(arr):>6} {arr.mean():>8.3f} {arr.std():>8.3f} {cv:>8.4f} {np.percentile(arr,1):>8.3f} {np.percentile(arr,99):>8.3f}")
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
        "by_type_cv": type_cvs,
    })

all_cv = np.array([r["cv"] for r in records])
mid = len(all_cv) // 2
print(f"\nFirst half (early): mean CV = {all_cv[:mid].mean():.4f}")
print(f"Last half (late):  mean CV = {all_cv[mid:].mean():.4f}")
print(f"Late/Early ratio: {all_cv[mid:].mean()/all_cv[:mid].mean():.3f}x")
peak = int(np.argmax(all_cv))
trough = int(np.argmin(all_cv))
print(f"\nPeak CV at layer {peak}: {all_cv[peak]:.4f}")
print(f"Trough CV at layer {trough}: {all_cv[trough]:.4f}")
print(f"Spread: {all_cv.max()/all_cv.min():.2f}x")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(n_layers), all_cv, "o-", linewidth=2, markersize=6, label="Bonsai-8B 1-bit effective")
    # Overlay 0.6B and 4B for comparison
    try:
        with open("results/stage162_per_layer_cv.json") as f: d06 = json.load(f)
        cv06 = [r["cv"] for r in d06["per_layer"]]
        # Stretch 28-layer 0.6B to 36-layer scale for visual comparison
        x06 = np.linspace(0, n_layers-1, len(cv06))
        ax.plot(x06, cv06, "s--", alpha=0.5, label="Qwen3-0.6B base FP (rescaled x)")
    except: pass
    try:
        with open("results/stage163_per_layer_cv_4b.json") as f: d4 = json.load(f)
        cv4 = [r["cv"] for r in d4["per_layer"]]
        x4 = np.linspace(0, n_layers-1, len(cv4))
        ax.plot(x4, cv4, "^--", alpha=0.5, label="Qwen3-4B base FP (rescaled x)")
    except: pass
    ax.set_xlabel("layer index"); ax.set_ylabel("CV")
    ax.set_title("Per-layer CV: Bonsai 1-bit vs FP base models")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = Path("results/stage164_per_layer_cv_bonsai.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Plot: {plot_path}")
except Exception as e:
    print(f"(plot skipped: {e})")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": "prism-ml/Bonsai-8B-mlx-1bit",
        "n_layers": n_layers,
        "early_late_ratio": float(all_cv[mid:].mean()/all_cv[:mid].mean()),
        "peak_layer": peak, "peak_cv": float(all_cv[peak]),
        "trough_layer": trough, "trough_cv": float(all_cv[trough]),
        "spread": float(all_cv.max()/all_cv.min()),
        "per_layer": records,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}")
