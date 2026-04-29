"""
Measure how close each Qwen3 model is to nGPT hypersphere geometry.
Run across 0.6B, 1.7B, 4B, 8B, 14B, 32B.

For each model, measure:
1. Per-row L2 norm distribution across all linear projections
2. Coefficient of variation (CV = std/mean) — lower = more spherical
3. Spread ratio (max/min) — lower = more uniform
4. How close mean is to 1.0 (nGPT target)
5. Per-projection-type breakdown

Hypothesis: bigger models are MORE spherical (closer to nGPT geometry)
because more pretraining steps push toward uniform row norms.
"""

import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


TARGET_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


def measure_model(model_name):
    """Load model and measure row norm distribution."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.0f}s", flush=True)

    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())

    by_type = defaultdict(list)
    all_norms = []
    per_layer_stats = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(m in name for m in TARGET_PROJS):
            continue
        W = module.weight.data
        norms = W.norm(dim=-1).cpu().numpy()
        proj_type = next(m for m in TARGET_PROJS if m in name)
        by_type[proj_type].extend(norms.tolist())
        all_norms.extend(norms.tolist())

    all_arr = np.array(all_norms)

    # Key metrics
    mean = float(all_arr.mean())
    std = float(all_arr.std())
    cv = std / mean  # coefficient of variation
    median = float(np.median(all_arr))
    p1 = float(np.percentile(all_arr, 1))
    p99 = float(np.percentile(all_arr, 99))
    spread = float(all_arr.max() / all_arr.min())
    distance_to_unit = abs(mean - 1.0)

    # Per-type
    type_stats = {}
    for t in TARGET_PROJS:
        arr = np.array(by_type[t])
        if len(arr) == 0:
            continue
        type_stats[t] = {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "cv": float(arr.std() / arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "spread": float(arr.max() / arr.min()),
        }

    print(f"  L={L}, d={d}, params={n_params/1e9:.1f}B")
    print(f"  Rows measured: {len(all_arr):,}")
    print(f"  Mean norm:     {mean:.4f}")
    print(f"  CV (std/mean): {cv:.4f}  {'<-- spherical!' if cv < 0.3 else ''}")
    print(f"  Spread (max/min): {spread:.1f}x")
    print(f"  Distance to unit: {distance_to_unit:.4f}")
    print(f"  p1={p1:.3f}  median={median:.3f}  p99={p99:.3f}")
    print(f"\n  Per type:", flush=True)
    for t in TARGET_PROJS:
        if t in type_stats:
            s = type_stats[t]
            print(f"    {t:<10}: mean={s['mean']:.3f} cv={s['cv']:.3f} spread={s['spread']:.1f}x")

    result = {
        "model": model_name,
        "layers": L,
        "hidden_size": d,
        "params_B": round(n_params / 1e9, 2),
        "n_rows": len(all_arr),
        "overall": {
            "mean": mean, "std": std, "cv": cv,
            "median": median, "p1": p1, "p99": p99,
            "min": float(all_arr.min()), "max": float(all_arr.max()),
            "spread": spread,
            "distance_to_unit": distance_to_unit,
        },
        "by_type": type_stats,
    }

    del model
    gc.collect()
    return result


def main():
    torch.set_num_threads(32)

    print("=" * 60)
    print("nGPT GEOMETRY MEASUREMENT: Qwen3 0.6B → 32B")
    print("  How spherical is each model? (closer to unit norm = more nGPT)")
    print("=" * 60, flush=True)

    results = []
    for model_name in MODELS:
        result = measure_model(model_name)
        results.append(result)

        # Clear RAM between models
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY: nGPT Geometry Across Scale")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Params':>7} {'Mean':>7} {'CV':>7} {'Spread':>8} {'Dist→1':>7} {'Spherical?'}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")
    for r in results:
        o = r["overall"]
        spherical = "YES" if o["cv"] < 0.25 else "partial" if o["cv"] < 0.4 else "no"
        print(f"  {r['model']:<20} {r['params_B']:>6.1f}B {o['mean']:>7.3f} {o['cv']:>7.4f} "
              f"{o['spread']:>7.1f}x {o['distance_to_unit']:>7.4f} {spherical:>10}")

    # Trend
    print(f"\n  Trend: does CV decrease with scale?")
    for i in range(1, len(results)):
        prev_cv = results[i-1]["overall"]["cv"]
        curr_cv = results[i]["overall"]["cv"]
        change = (curr_cv - prev_cv) / prev_cv * 100
        arrow = "↓" if change < 0 else "↑"
        print(f"    {results[i-1]['model'].split('/')[-1]} → {results[i]['model'].split('/')[-1]}: "
              f"CV {prev_cv:.4f} → {curr_cv:.4f} ({change:+.1f}% {arrow})")

    # Save
    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)
    rpath = Path(save_dir) / "ngpt_geometry_scale.json"
    with open(rpath, "w") as f:
        json.dump({"models": results}, f, indent=2)
    print(f"\n  Results: {rpath}")


if __name__ == "__main__":
    main()
