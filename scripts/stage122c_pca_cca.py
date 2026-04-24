"""
Stage 122c — PCA-reduced CCA between Qwen3-0.6B and Qwen3-1.7B.

Stage 122b was still degenerate: N=2031 < d_large=2048, so CCA gave
perfect correlations even on shuffled data (shuffle baseline=1.0 too).

Fix: PCA both X and Y down to k ≪ N before CCA. This tests the REAL
question: "do the top-k principal directions of small overlap with
the top-k principal directions of large?"

We use the cached per-token states from stage 122b — no re-extraction
needed.

Predictions:
  - Nested mouths: top-k PCA CCA should show a plateau at ~1.0 then
    drop (small's dirs matched, extra large dirs unshared)
  - Private mouths (option 3): smooth decay from top, no plateau
  - Universal throat: high correlations at pos 0.25 and 0.50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def pca_reduce(X, k):
    """Center, then project to top-k PCs. Returns [N, k]."""
    X = X - X.mean(axis=0, keepdims=True)
    # SVD of X gives U S V^T where cols of V are PCs
    # Project: X @ V[:, :k]
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    Z = X @ V[:, :k]
    # Also return explained variance fraction
    evr = (S[:k] ** 2).sum() / (S ** 2).sum()
    return Z, float(evr)


def cca(X, Y, k=None):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    C = Qx.T @ Qy
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    if k:
        S = S[:k]
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", default="Qwen/Qwen3-0.6B")
    p.add_argument("--large", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="cache/stage122b_states")
    p.add_argument("--out", default="results/stage122c_pca_cca.json")
    p.add_argument("--k", type=int, default=200,
                   help="PCA dims per model (must be << N)")
    args = p.parse_args()

    safe_s = args.small.replace("/", "_")
    safe_l = args.large.replace("/", "_")
    cache_s = Path(args.cache_dir) / f"{safe_s}.pt"
    cache_l = Path(args.cache_dir) / f"{safe_l}.pt"
    s_states = torch.load(cache_s, map_location="cpu")
    l_states = torch.load(cache_l, map_location="cpu")

    print(f"k (PCA dims per model) = {args.k}")
    results = {}
    for pos in s_states:
        if pos not in l_states: continue
        X = s_states[pos].float().numpy()
        Y = l_states[pos].float().numpy()
        N = min(X.shape[0], Y.shape[0])
        X = X[:N]; Y = Y[:N]
        Xr, evr_s = pca_reduce(X, args.k)
        Yr, evr_l = pca_reduce(Y, args.k)
        corrs = cca(Xr, Yr, k=args.k)

        rng = np.random.default_rng(0)
        perm = rng.permutation(N)
        corrs_shuf = cca(Xr, Yr[perm], k=args.k)

        def first_below(vals, thresh):
            b = np.where(vals < thresh)[0]
            return int(b[0]) if len(b) else int(len(vals))

        results[pos] = {
            "N": N, "k": args.k,
            "evr_small": evr_s, "evr_large": evr_l,
            "corrs": corrs.tolist(),
            "corrs_shuf": corrs_shuf.tolist(),
            "top_1": float(corrs[0]),
            "top_10_avg": float(np.mean(corrs[:10])),
            "mean_all": float(np.mean(corrs)),
            "mean_all_shuf": float(np.mean(corrs_shuf)),
            "cliff_0.9": first_below(corrs, 0.9),
            "cliff_0.5": first_below(corrs, 0.5),
            "cliff_0.5_shuf": first_below(corrs_shuf, 0.5),
            "frac_above_0_9": float(np.mean(corrs > 0.9)),
            "frac_above_0_7": float(np.mean(corrs > 0.7)),
        }
        print(f"\n=== pos {pos} ===  (small EVR@k={evr_s:.3f}  large EVR@k={evr_l:.3f})")
        print(f"  top1:{corrs[0]:.3f}  top10:{np.mean(corrs[:10]):.3f}  mean:{np.mean(corrs):.3f}"
              f"  (shuf mean:{np.mean(corrs_shuf):.3f})")
        print(f"  cliff@0.9: rank {results[pos]['cliff_0.9']}    "
              f"cliff@0.5: rank {results[pos]['cliff_0.5']}  (shuf @0.5: {results[pos]['cliff_0.5_shuf']})")
        print(f"  frac>0.9: {np.mean(corrs > 0.9):.2f}   frac>0.7: {np.mean(corrs > 0.7):.2f}")
        qs = [0, 10, 25, 50, 100, 150, args.k-1]
        print(f"  corr @ ranks {qs}: " + " ".join(f"{corrs[q]:.3f}" for q in qs))
        print(f"  shuf @ ranks {qs}: " + " ".join(f"{corrs_shuf[q]:.3f}" for q in qs))

    print("\n=== verdict ===")
    for pos, r in results.items():
        real_cliff = r["cliff_0.5"]; shuf_cliff = r["cliff_0.5_shuf"]
        if real_cliff <= shuf_cliff + 5:
            v = "NO REAL SIGNAL — real ≈ shuffle"
        elif r["frac_above_0_9"] > 0.5:
            v = f"HIGHLY NESTED — {r['frac_above_0_9']*100:.0f}% of PCs match >0.9"
        elif r["frac_above_0_7"] > 0.3:
            v = f"PARTIALLY NESTED — {r['frac_above_0_7']*100:.0f}% of PCs match >0.7"
        elif r["top_10_avg"] > 0.7:
            v = "TOP-K ALIGNED — a few shared PCs, rest private"
        else:
            v = "PRIVATE — no significant alignment"
        print(f"  pos {pos}: {v}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"small": args.small, "large": args.large,
                   "k": args.k, "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
