"""
Stage 32 — Minimal essential feature subset via greedy forward selection.

Stage 31 used all 47 features to reach LOPO linear R² = 0.341. But
Finding 07 and stage 25 showed that many of those features are
redundant. For deployment we want the SMALLEST set that keeps
routing accuracy near the full-feature LOPO R².

Greedy forward selection:
  Start with empty set. At each step, pick the feature that, when
  added, gives the largest LOPO linear R² gain. Stop when gain < ε
  or size exceeds max_k.

Reuses the 35-prompt records from stage 31 (recollects them).
Reports the top-k essential subset and its LOPO R² curve.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage29_structural_features import (
    CALIB_TEXTS, SUMMARY_FEATURES, CURVATURE_FEATURES, QUANTUM_FEATURES,
    STRUCTURAL_FEATURES, collect_calibration, collect,
)
from scripts.stage31_expanded_lopo import PROMPTS

ALL_FEATURES = SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES + STRUCTURAL_FEATURES


def linear_regression_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    f = X_train.shape[1]
    XtX = X_train.T @ X_train + ridge * torch.eye(f, dtype=X_train.dtype)
    Xty = X_train.T @ y_train
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_test @ beta
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def lopo_r2(X, y, prompt_ids, ridge=1e-3):
    """Mean LOPO linear R² across all unique prompts."""
    scores = []
    for p in torch.unique(prompt_ids):
        tr_mask = prompt_ids != p
        te_mask = prompt_ids == p
        if te_mask.sum() < 10:
            continue
        X_tr = X[tr_mask]; X_te = X[te_mask]
        mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp_min(1e-8)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        X_tr = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1)], dim=1)
        X_te = torch.cat([X_te, torch.ones(X_te.shape[0], 1)], dim=1)
        scores.append(linear_regression_r2(X_tr, y[tr_mask], X_te, y[te_mask], ridge))
    return sum(scores) / max(len(scores), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--max-k", type=int, default=15)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage32_minimal_subset.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    calib_hidden = collect_calibration(model, tokenizer, CALIB_TEXTS, device)
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample); pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())

    print(f"\n=== collecting records ===")
    all_records = []
    prompt_ids = []
    t0 = time.perf_counter()
    for pid, (cat, prompt) in enumerate(PROMPTS):
        recs = collect(model, tokenizer, prompt, args.max_new_tokens,
                        device, calib_hidden, kde_sigma, knn_k=10)
        all_records.extend(recs)
        prompt_ids.extend([pid] * len(recs))
        if (pid + 1) % 5 == 0:
            print(f"  {pid+1}/{len(PROMPTS)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    N = len(all_records)
    print(f"  {N} records in {time.perf_counter()-t0:.0f}s")

    # Feature matrix
    X = torch.tensor([[r[f] for f in ALL_FEATURES] for r in all_records], dtype=torch.float32)
    y = torch.tensor([r["output_entropy"] for r in all_records], dtype=torch.float32)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)

    print(f"\n=== greedy forward selection (LOPO linear R²) ===")
    selected = []
    remaining = list(range(len(ALL_FEATURES)))
    history = []
    prev_r2 = 0.0

    while selected and len(selected) >= args.max_k:
        break
    while len(selected) < args.max_k and remaining:
        best_gain = -float("inf")
        best_feat = None
        best_r2 = None
        for j in remaining:
            candidate = selected + [j]
            r2 = lopo_r2(X[:, candidate], y, prompt_ids_t)
            gain = r2 - prev_r2
            if gain > best_gain:
                best_gain = gain; best_feat = j; best_r2 = r2
        selected.append(best_feat)
        remaining.remove(best_feat)
        feat_name = ALL_FEATURES[best_feat]
        history.append({"feature": feat_name, "lopo_r2": best_r2, "gain": best_gain})
        print(f"  [{len(selected):2d}] add {feat_name:<32} -> LOPO R² = {best_r2:.3f}  "
              f"(gain +{best_gain:+.3f})", flush=True)
        prev_r2 = best_r2
        if best_gain < 0.003 and len(selected) >= 4:
            # Allow a few more in case there's late jumps
            pass

    # Full-feature LOPO for reference
    full_r2 = lopo_r2(X, y, prompt_ids_t)
    print(f"\n  full (47) LOPO R² = {full_r2:.3f}")

    # Find k at 90% of full
    target = 0.9 * full_r2
    reach_k = None
    for i, h in enumerate(history):
        if h["lopo_r2"] >= target:
            reach_k = i + 1
            break
    if reach_k:
        print(f"\n  minimal subset reaching 90% of full R²: k = {reach_k}")
        print(f"  features: {[h['feature'] for h in history[:reach_k]]}")
    else:
        print(f"\n  90% of full R² not reached within k={args.max_k}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "n_records": N,
            "full_lopo_r2": full_r2,
            "greedy_history": history,
            "min_k_for_90pct": reach_k,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
