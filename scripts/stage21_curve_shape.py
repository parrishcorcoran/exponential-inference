"""
Stage 21 — Rotation curve shape comparison + layer-1 rotation character.

Stage 20 showed all three tested models have their phase transition at
layer 0→1. This stage asks:

(A) Do the three models' adjacent-overlap CURVES follow the same shape
    when layer index is normalized to [0, 1]? If yes: scale-invariant
    rotation schedule. If no: shape is model/tokenizer specific.

(B) What's the CHARACTER of the universal layer-1 rotation? Compute
    all k principal cosines between P_embed (text-weighted) and
    P_act[1]. Is the rotation uniform (all k principal angles equal)
    or selective (some preserved, some fully rotated)?

No forward-pass cost beyond what stage 20 already used (models cached).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


CORPUS = [
    "The cell is the basic structural unit of life composed of cytoplasm enclosed by a membrane.",
    "Quantum mechanics describes matter and energy at atomic scales through wave-particle duality.",
    "Compilers translate source code into machine code through lexical analysis, parsing, and optimization.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Neural networks approximate functions through layered transformations trained by gradient descent.",
    "Plate tectonics describes the movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into three-dimensional structures determined by their amino acid sequences.",
    "The standard model unifies three fundamental forces with a set of gauge bosons and fermions.",
    "Evolution operates on heritable variation through differential reproduction across generations.",
    "Cryptography protects information using asymmetric mathematical operations.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges, with many practical applications.",
    "The Renaissance marked renewed interest in classical learning in Europe.",
    "Black holes are regions of spacetime where gravity prevents light from escaping.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotides.",
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle.",
    "The auditory system transduces air pressure oscillations into neural signals.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Microbiome research examines the communities of microorganisms living in hosts.",
]


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)


def subspace_overlap(P_a, P_b):
    k = P_a.shape[1]
    M = P_a.T @ P_b
    return float(torch.linalg.norm(M, ord="fro").item() / (k ** 0.5))


def principal_cosines(P_a, P_b):
    M = P_a.T @ P_b
    _, S, _ = torch.linalg.svd(M)
    return S.tolist()


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


def analyze(model_id, device, rank, corpus):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n=== {model_id} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    inputs = capture_layer_inputs(model, tokenizer, corpus, device)
    bases = [pca_basis(inputs[i], rank) for i in range(n_layers)]

    # (A) Adjacent-overlap curve
    adj = [subspace_overlap(bases[i], bases[i+1]) for i in range(n_layers-1)]
    normalized_depth = [i / max(n_layers - 2, 1) for i in range(n_layers - 1)]

    # (B) Layer-1 rotation character vs text-weighted embedding
    embed = model.model.embed_tokens.weight.detach().cpu().to(torch.float32)
    positions = torch.cat([tokenizer(t, return_tensors="pt",
                                      truncation=True, max_length=256).input_ids[0]
                           for t in corpus], dim=0)
    X_embed = embed[positions]
    P_embed = pca_basis(X_embed, rank)
    embed_to_1_cosines = principal_cosines(P_embed, bases[1])
    embed_to_1_overlap = subspace_overlap(P_embed, bases[1])

    # Free
    del model, tokenizer, inputs, bases, embed
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_size": d,
        "rank": rank,
        "adjacent_overlap_curve": adj,
        "normalized_depth": normalized_depth,
        "layer_0_to_1_overlap": adj[0],
        "embed_to_layer1_overlap": embed_to_1_overlap,
        "embed_to_layer1_principal_cosines": embed_to_1_cosines,
    }


def interpolate_curve(xs, ys, new_xs):
    """Linear interpolation on a curve (xs, ys) at new_xs positions."""
    out = []
    for x in new_xs:
        # Find bracketing indices
        if x <= xs[0]:
            out.append(ys[0])
        elif x >= xs[-1]:
            out.append(ys[-1])
        else:
            for i in range(len(xs) - 1):
                if xs[i] <= x <= xs[i+1]:
                    t = (x - xs[i]) / max(xs[i+1] - xs[i], 1e-10)
                    out.append(ys[i] + t * (ys[i+1] - ys[i]))
                    break
    return out


def curve_similarity(ys_a, ys_b):
    """Pearson correlation + mean absolute diff between two curves of equal length."""
    n = len(ys_a)
    ma = sum(ys_a) / n
    mb = sum(ys_b) / n
    va = sum((y - ma) ** 2 for y in ys_a) / n
    vb = sum((y - mb) ** 2 for y in ys_b) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(ys_a, ys_b)) / n
    r = cov / max((va ** 0.5) * (vb ** 0.5), 1e-12)
    mad = sum(abs(a - b) for a, b in zip(ys_a, ys_b)) / n
    return r, mad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,microsoft/phi-2")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage21_curve_shape.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}  rank={args.rank}")

    models = [m.strip() for m in args.models.split(",")]
    results = []
    for m in models:
        try:
            results.append(analyze(m, device, args.rank, CORPUS))
        except Exception as e:
            print(f"  ! {m} failed: {e}")

    # (A) Compare curve shapes by resampling to common depth grid
    n_grid = 20
    grid = [i / (n_grid - 1) for i in range(n_grid)]
    print(f"\n=== rotation curve shape comparison (20-point normalized grid) ===")
    resampled = []
    for r in results:
        ys = interpolate_curve(r["normalized_depth"], r["adjacent_overlap_curve"], grid)
        resampled.append(ys)
        print(f"  {r['model_id']:>24}  "
              f"start={ys[0]:.3f}  mid={ys[len(grid)//2]:.3f}  end={ys[-1]:.3f}")

    print(f"\n=== pairwise curve similarity ===")
    print(f"  {'pair':>40}  {'pearson_r':>10}  {'mean_abs_diff':>15}")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            r, mad = curve_similarity(resampled[i], resampled[j])
            label = f"{results[i]['model_id']} vs {results[j]['model_id']}"
            print(f"  {label:>40}  {r:>10.3f}  {mad:>15.4f}")

    # (B) Layer-1 rotation character
    print(f"\n=== layer-1 rotation character (embedding vs P_act[1]) ===")
    print(f"  {'model':>24}  {'overlap':>8}  {'cos spread':>12}  "
          f"{'top-5 cosines':>30}")
    for r in results:
        cos = r["embed_to_layer1_principal_cosines"]
        top5 = cos[:5]
        spread = max(cos) - min(cos)
        label = r["model_id"]
        print(f"  {label:>24}  {r['embed_to_layer1_overlap']:>8.3f}  {spread:>12.3f}  "
              f"{' '.join(f'{c:.2f}' for c in top5)}")

    # Interpretation helper
    print(f"\n  cos spread = max principal cosine - min principal cosine")
    print(f"  uniform rotation: low spread; selective rotation: high spread")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "models": results,
            "resampled_grid": grid,
            "resampled_curves": resampled,
            "pairwise_similarity": [
                {
                    "a": results[i]["model_id"],
                    "b": results[j]["model_id"],
                    "pearson_r": curve_similarity(resampled[i], resampled[j])[0],
                    "mean_abs_diff": curve_similarity(resampled[i], resampled[j])[1],
                }
                for i in range(len(results)) for j in range(i+1, len(results))
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
