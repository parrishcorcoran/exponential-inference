"""
Stage 122 — Nested mouth test: is small model's mouth a subspace of large's?

Hypothesis (user's): each model's mouth is like a hula hoop. Larger
models don't STRETCH a small mouth to full size — they ADD additional
rings around the smaller mouth. So 0.6B's mouth state space should be
CONTAINED in 1.7B's mouth state space.

Stage 121 showed low R² (0.08) at early mouth via direct linear
regression. That was the wrong test — linear regression projects onto
ALL target dimensions, but the small model only uses SOME of them.
The right test is CCA (or equivalent) which finds the top-k aligned
directions.

Procedure:
  1. Load cached mouth states from stage 121 (0.6B [N, 1024] and 1.7B [N, 2048])
  2. Run CCA between them at the same normalized depth position
  3. Top canonical correlations tell us how many directions ARE shared
  4. If top 150 correlations are all ~1.0 → 0.6B mouth fully nested in 1.7B mouth

Predictions:
  - At mouth: many (~150) canonical correlations at ~1.0, then drops → NESTED
  - At throat: all canonical correlations ~1.0 → already aligned (per stage 121)
  - At exit mouth: again many ~1.0 then drops → NESTED

Opposite outcome (each model has truly private mouth space):
  - Correlations decay smoothly with no plateau at high corr
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def cca(X, Y, k=None):
    """Canonical Correlation Analysis between X [N, d1] and Y [N, d2].
       Returns sorted canonical correlations (descending)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # QR for numerical stability
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    # Cross-covariance in the Q-bases
    C = Qx.T @ Qy
    # SVD gives canonical correlations
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    if k:
        S = S[:k]
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", default="Qwen/Qwen3-0.6B")
    p.add_argument("--large", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="cache/stage121_states")
    p.add_argument("--out", default="results/stage122_nested_mouths.json")
    args = p.parse_args()

    safe_s = args.small.replace("/", "_")
    safe_l = args.large.replace("/", "_")
    cache_s = Path(args.cache_dir) / f"{safe_s}.pt"
    cache_l = Path(args.cache_dir) / f"{safe_l}.pt"

    if not cache_s.exists() or not cache_l.exists():
        print(f"ERROR: cache files not found. Run stage 121 first.", flush=True)
        print(f"  expected: {cache_s}  and  {cache_l}")
        return

    s_states = torch.load(cache_s, map_location="cpu")
    l_states = torch.load(cache_l, map_location="cpu")
    print(f"loaded cached states from {args.small} and {args.large}")
    print(f"positions: {list(s_states.keys())}")

    results = {}
    for pos in s_states:
        if pos not in l_states:
            continue
        X = s_states[pos].float().numpy()   # [N, d_small]
        Y = l_states[pos].float().numpy()   # [N, d_large]
        N, d_small = X.shape
        _, d_large = Y.shape
        # CCA limited by N (can only have min(N, d_small, d_large) correlations)
        max_corr = min(N, d_small, d_large)
        corrs = cca(X, Y, k=max_corr)
        results[pos] = {
            "d_small": d_small,
            "d_large": d_large,
            "n_samples": N,
            "n_corrs": len(corrs),
            "corrs": corrs.tolist(),
            "top_1": float(corrs[0]),
            "top_5_avg": float(np.mean(corrs[:min(5, len(corrs))])),
            "top_10_avg": float(np.mean(corrs[:min(10, len(corrs))])),
            "mean_all": float(np.mean(corrs)),
            "frac_above_0_9": float(np.mean(corrs > 0.9)),
            "frac_above_0_7": float(np.mean(corrs > 0.7)),
        }
        print(f"\n=== position {pos} ===")
        print(f"  d_small={d_small}  d_large={d_large}  n_samples={N}  n_corrs={len(corrs)}")
        print(f"  top 1: {corrs[0]:.4f}")
        print(f"  top 5 avg: {np.mean(corrs[:5]):.4f}")
        print(f"  top 10 avg: {np.mean(corrs[:10]):.4f}")
        print(f"  mean all: {np.mean(corrs):.4f}")
        print(f"  fraction > 0.9: {np.mean(corrs > 0.9):.3f}")
        print(f"  fraction > 0.7: {np.mean(corrs > 0.7):.3f}")
        # Show first several and last several correlation values
        print(f"  first 10: " + " ".join(f"{c:.3f}" for c in corrs[:10]))
        if len(corrs) > 10:
            print(f"  last  10: " + " ".join(f"{c:.3f}" for c in corrs[-10:]))

    # Interpretation
    print(f"\n=== interpretation ===")
    for pos, r in results.items():
        if r["mean_all"] > 0.85:
            v = "FULLY ALIGNED — small model's space is essentially identical to large's projected subspace"
        elif r["frac_above_0_9"] > 0.5:
            v = "NESTED — small's dimensions mostly match a subspace of large"
        elif r["frac_above_0_7"] > 0.5:
            v = "PARTIAL NESTING — roughly half of small's space aligns with large's"
        elif r["top_5_avg"] > 0.8:
            v = "TOP-K ALIGNED — top few directions match, rest private"
        else:
            v = "PRIVATE — minimal shared structure"
        print(f"  pos {pos}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"small": args.small, "large": args.large,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
