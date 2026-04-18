"""
Stage 16 — Does the embedding matrix carry the manifold directly?

If the ~9-11 intrinsic dim we measure from activations is a property of
the tokenizer+embeddings (static substrate) rather than the transformer
dynamics, then we should be able to measure it from the embedding matrix
alone — no forward pass needed.

Test:
    1. Load Qwen3-0.6B (and any other models available).
    2. Extract embed_tokens.weight ([vocab, hidden]).
    3. Run TwoNN directly on the embedding matrix.
    4. Compare to previously-measured activation TwoNN at final layer.

If they match: the manifold is in the embeddings. We can derive the
factored-weight basis from embed_tokens.weight SVD, skipping calibration
entirely.

Also computes PCA rank coverage on the embedding matrix for comparison
to stage 1 activation measurements.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


MODELS_TO_MEASURE = [
    "Qwen/Qwen3-0.6B",
]


def twonn_dimension(X, sample_limit=3000):
    X = X.to(torch.float64)
    if X.shape[0] > sample_limit:
        idx = torch.randperm(X.shape[0])[:sample_limit]
        X = X[idx]
    N = X.shape[0]
    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    top2, _ = dists.topk(2, dim=1, largest=False)
    r1 = top2[:, 0]
    r2 = top2[:, 1]
    mask = r1 > 1e-10
    if mask.sum() < 10:
        return float("nan")
    mu = (r2[mask] / r1[mask]).clamp_min(1.0 + 1e-10)
    return float(1.0 / torch.log(mu).mean().item())


def rank_coverage(X, fractions=(0.5, 0.9, 0.95, 0.99)):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    eigvals = eigvals.flip(0)
    total = eigvals.sum().clamp_min(1e-12)
    cum = torch.cumsum(eigvals, dim=0) / total
    out = {}
    for f in fractions:
        idx = int((cum >= f).nonzero()[0].item()) + 1 if (cum >= f).any() else len(cum)
        out[f"r{int(f*100)}"] = idx
    return out


def participation_ratio(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    num = eigvals.sum().pow(2)
    den = eigvals.pow(2).sum().clamp_min(1e-12)
    return float((num / den).item())


def measure_model_embeddings(model_id, sample_limit=3000):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n=== {model_id} ===", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    # Qwen3 / Llama / most LMs have embed_tokens at model.model.embed_tokens
    embed = None
    for attr_path in ("model.model.embed_tokens.weight",
                      "model.embed_tokens.weight",
                      "transformer.wte.weight"):
        try:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            embed = obj
            break
        except AttributeError:
            continue
    if embed is None:
        print("  ! couldn't locate embedding matrix")
        return None

    vocab, hidden = embed.shape
    print(f"  embed shape: [{vocab}, {hidden}]")

    X = embed.detach().cpu().to(torch.float32)

    t0 = time.perf_counter()
    twonn = twonn_dimension(X, sample_limit=sample_limit)
    pr = participation_ratio(X)
    rcov = rank_coverage(X)
    print(f"  measured in {time.perf_counter()-t0:.1f}s")
    print(f"  TwoNN (intrinsic dim):  {twonn:.2f}")
    print(f"  PR (effective rank):    {pr:.1f}")
    print(f"  rank coverage: r50={rcov['r50']}  r90={rcov['r90']}  "
          f"r95={rcov['r95']}  r99={rcov['r99']}")

    return {
        "model_id": model_id,
        "vocab": int(vocab),
        "hidden": int(hidden),
        "embedding_twonn": twonn,
        "embedding_pr": pr,
        "embedding_rank_coverage": rcov,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None,
                   help="Comma-separated HF model ids. Falls back to built-in list.")
    p.add_argument("--sample-limit", type=int, default=3000)
    p.add_argument("--out", default="results/stage16_embedding_manifold.json")
    args = p.parse_args()

    models = args.models.split(",") if args.models else MODELS_TO_MEASURE
    print(f"Measuring {len(models)} model(s):  {models}")

    results = []
    for m in models:
        try:
            r = measure_model_embeddings(m, sample_limit=args.sample_limit)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ! failed for {m}: {e}")
            results.append({"model_id": m, "error": str(e)})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"models": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    # Print comparison table
    print(f"\n=== summary ===")
    print(f"  {'model':>30} {'vocab':>8} {'hidden':>8} {'TwoNN':>8} {'PR':>8}")
    for r in results:
        if "error" in r:
            continue
        print(f"  {r['model_id']:>30} {r['vocab']:>8} {r['hidden']:>8} "
              f"{r['embedding_twonn']:>8.2f} {r['embedding_pr']:>8.1f}")


if __name__ == "__main__":
    main()
