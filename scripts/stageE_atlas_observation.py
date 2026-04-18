"""
Stage E — Atlas / chart-transition observational experiment.

Question: per-layer activations distribute on a curved manifold. If the
manifold has locally-distinct regions (charts), a SINGLE global PCA
basis wastes rank capturing geometric embedding curvature. An ATLAS
(per-chart local bases) would give tighter rank-k coverage at each
chart.

Test: for each layer, cluster activations into K clusters, measure:

    1. Silhouette score — cluster quality. >0.2 = real structure.
    2. Per-cluster PCA rank to cover 90% variance — how low-dim each
       chart is locally.
    3. Global PCA rank to cover 90% variance — for comparison.
    4. Cluster compression ratio: (global_rank) / (mean per-cluster rank).
       If >> 1, atlas helps; ≈ 1, one chart is enough.

No training. Pure observation of whether the boundary layer has
multi-chart structure at this model scale.

Usage:
    python scripts/stageE_atlas_observation.py --model Qwen/Qwen3-0.6B --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


CALIBRATION_TEXTS = [
    "The cell is the basic structural unit of life.",
    "Quantum entanglement occurs when particles share correlated states.",
    "Compilers translate source code into machine code through multiple passes.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into sugars.",
    "The Roman Empire fell due to economic, military, and political factors.",
    "Neural networks learn patterns from data through gradient descent.",
    "Black holes have event horizons from which light cannot escape.",
    "Cryptography uses mathematical transformations to protect information.",
    "Evolution proceeds through heritable variation and differential reproduction.",
    "Graph theory studies relationships represented as vertices connected by edges.",
    "The industrial revolution transformed economies through mechanization.",
    "DNA encodes genetic information in a double-helix structure.",
    "Plate tectonics causes earthquakes at boundaries between crustal plates.",
    "Probability theory quantifies uncertainty through mathematical models.",
    "The standard model unifies three of the four fundamental forces.",
    "Bacteria reproduce by binary fission while viruses hijack host cells.",
]


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def capture_layer_inputs(model, tokenizer, texts, device, max_len=256):
    n_layers = model.config.num_hidden_layers
    all_inputs = [[] for _ in range(n_layers)]
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for i in range(n_layers):
                all_inputs[i].append(hs[i][0].to(torch.float32).cpu())
    return [torch.cat(xs, dim=0) for xs in all_inputs]


def pca_rank_for_variance(X, threshold=0.9):
    """Return minimum rank needed to cover `threshold` fraction of variance."""
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64))
    eigvals = eigvals.flip(0).clamp_min(0)  # descending
    total = eigvals.sum()
    cum = torch.cumsum(eigvals, dim=0)
    idx = int((cum / total >= threshold).nonzero()[0].item()) + 1
    return idx


def kmeans_fit(X, K, n_iter=20, seed=0):
    """Simple K-means. X: [N, d]. Returns labels [N] and centroids [K, d]."""
    torch.manual_seed(seed)
    N, d = X.shape
    # Init: k-means++-style. Random for simplicity.
    perm = torch.randperm(N)[:K]
    centroids = X[perm].clone()
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        # Assign
        d2 = torch.cdist(X, centroids)
        labels = d2.argmin(dim=1)
        # Update
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
            else:
                new_centroids[k] = centroids[k]
        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    return labels, centroids


def silhouette_score(X, labels, sample_limit=500):
    """Cheap silhouette score on a sample of points."""
    N = X.shape[0]
    if N > sample_limit:
        idx = torch.randperm(N)[:sample_limit]
        X = X[idx]
        labels = labels[idx]
    N = X.shape[0]
    d = torch.cdist(X, X)
    unique = labels.unique()
    if len(unique) < 2:
        return float("nan")
    scores = []
    for i in range(N):
        li = labels[i].item()
        # Mean distance to own cluster (excluding self)
        own_mask = (labels == li) & (torch.arange(N) != i)
        if own_mask.sum() == 0:
            continue
        a = d[i, own_mask].mean().item()
        # Min mean distance to other clusters
        b = float("inf")
        for lj in unique:
            if lj.item() == li:
                continue
            other_mask = labels == lj
            if other_mask.sum() == 0:
                continue
            bj = d[i, other_mask].mean().item()
            if bj < b:
                b = bj
        if b == float("inf"):
            continue
        s = (b - a) / max(a, b)
        scores.append(s)
    return sum(scores) / max(len(scores), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default=None)
    p.add_argument("--k-clusters", type=int, default=4,
                   help="Number of charts to test per layer")
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\ndevice={device}  clusters={args.k_clusters}")

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers

    print(f"\n=== capturing layer inputs ===", flush=True)
    inputs_per_layer = capture_layer_inputs(
        model, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    print(f"  {inputs_per_layer[0].shape[0]} tokens per layer")

    print(f"\n=== cluster analysis per layer ===", flush=True)
    print(f"  {'layer':>5} {'N':>5} {'global_r90':>10} {'mean_cluster_r90':>16} "
          f"{'silhouette':>11} {'compression':>11}")
    results = []
    K = args.k_clusters
    for i in range(n_layers):
        X = inputs_per_layer[i]
        N = X.shape[0]
        global_r = pca_rank_for_variance(X, threshold=0.9)
        labels, _ = kmeans_fit(X, K)
        per_cluster_ranks = []
        for c in range(K):
            mask = labels == c
            if mask.sum() < 4:
                continue
            r = pca_rank_for_variance(X[mask], threshold=0.9)
            per_cluster_ranks.append(r)
        mean_cluster_r = sum(per_cluster_ranks) / max(len(per_cluster_ranks), 1)
        sil = silhouette_score(X, labels)
        compression = global_r / max(mean_cluster_r, 1)

        show = (i < 4) or (i >= n_layers - 4) or (i % 5 == 0)
        if show:
            print(f"  {i:>5} {N:>5} {global_r:>10} {mean_cluster_r:>16.1f} "
                  f"{sil:>11.3f} {compression:>11.2f}x")
        results.append({
            "layer": i,
            "n_samples": int(N),
            "global_r90": int(global_r),
            "mean_cluster_r90": float(mean_cluster_r),
            "silhouette": float(sil),
            "atlas_compression_ratio": float(compression),
            "n_clusters_with_data": len(per_cluster_ranks),
        })

    # Summary
    print(f"\n=== summary ===")
    mean_sil = sum(r["silhouette"] for r in results) / len(results)
    mean_compression = sum(r["atlas_compression_ratio"] for r in results) / len(results)
    print(f"  mean silhouette across layers: {mean_sil:.3f}")
    print(f"  mean atlas compression ratio:  {mean_compression:.2f}x")
    if mean_compression > 1.5:
        print(f"  -> atlas helps materially (>1.5x rank reduction at 90% variance)")
    elif mean_compression > 1.1:
        print(f"  -> atlas helps modestly")
    else:
        print(f"  -> single global basis is enough; atlas adds complexity without gain")

    out_path = Path(args.out_dir) / f"stageE_atlas_observation_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "k_clusters": K,
            "mean_silhouette": mean_sil,
            "mean_atlas_compression_ratio": mean_compression,
            "per_layer": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
