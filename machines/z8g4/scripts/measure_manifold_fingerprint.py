"""
Z8G4: cross-model manifold fingerprint.

Extends the single-number TwoNN dim estimate into a richer fingerprint
that captures the two-mode rotation structure (stage 58/59 findings)
and lets us compare manifold geometry across scales and tokenizer
families.

Output: a single JSON per model with:
  - per_layer_twonn_bootstrap: mean ± std across 20 subsamples
  - per_layer_carry_fraction, per_layer_flip_fraction, per_layer_mid_fraction
  - per_transition_angle_histogram (24 bins, 0 to π)
  - per_transition_mode_concentration (fraction of directions at 0 or π)
  - adjacent_carry_overlap_per_transition (persistence)
  - rotation_curve: per-transition mean rotation angle
  - corpus_meta: tokens used, model config
  - timings

CPU-friendly: no SDPA, no GPU-specific ops. Streams the forward pass
and does eigendecomposition on CPU fp32. Handles models up to the
machine's memory capacity.

Usage:
    python machines/z8g4/scripts/measure_manifold_fingerprint.py \\
        --model Qwen/Qwen3-32B \\
        --out machines/z8g4/results/fingerprint_qwen3_32b.json

For models requiring spanned NUMA:
    numactl -N 0-1 -m 0-1 python machines/z8g4/scripts/measure_manifold_fingerprint.py ...
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


# Diverse corpus — runs forward on the model, doesn't need to be huge.
# 20 diverse wikitext-like sentences, ~400 tokens total.
CALIB_TEXTS = [
    "The cell is the basic structural unit of life, composed of cytoplasm enclosed within a membrane.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
    "Neural networks consist of parameterized layers trained by gradient descent to approximate functions.",
    "Plate tectonics describes the slow movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into complex three-dimensional structures determined by their amino acid sequences.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions.",
    "Evolution by natural selection operates on heritable variation in populations.",
    "Cryptography protects information using mathematical operations that are easy to compute.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges across many practical applications.",
    "Black holes are regions of spacetime from which nothing, not even light, can escape.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Bayesian inference updates a prior probability distribution using observed data.",
    "The immune system recognizes pathogens through pattern recognition receptors.",
    "The Riemann zeta function encodes deep information about the distribution of primes.",
]


def load_model(model_id, dtype_str="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_str, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
    return model, tokenizer


def collect_per_layer_hiddens(model, tokenizer, texts, max_len=256):
    L = len(model.model.layers)
    samples = [[] for _ in range(L + 1)]

    def make_hook(i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            samples[i].append(h_flat)
        return hook

    def embed_hook(mod, inputs, output):
        h = output
        h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
        samples[0].append(h_flat)
    h0 = model.model.embed_tokens.register_forward_hook(embed_hook)
    handles = [h0] + [model.model.layers[i].register_forward_hook(make_hook(i + 1))
                       for i in range(L)]
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()
    return {i: torch.cat(samples[i], dim=0) for i in range(L + 1)}


def pca_basis(H, k):
    mu = H.mean(dim=0, keepdim=True)
    Hc = H - mu
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1])


def twonn_dim(X):
    X = X.to(torch.float32)
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    sorted_d, _ = D.sort(dim=1)
    r1, r2 = sorted_d[:, 0], sorted_d[:, 1]
    mask = (r1 > 1e-8) & (r2 > r1 + 1e-10)
    if mask.sum() < 10:
        return float("nan")
    mu = r2[mask] / r1[mask]
    return float(mask.sum().item() / torch.log(mu).sum().item())


def twonn_bootstrap(X, n_boot=20, subsample_frac=0.8, seed=0):
    N = X.shape[0]
    n_sub = max(10, int(subsample_frac * N))
    estimates = []
    for b in range(n_boot):
        torch.manual_seed(seed + b)
        idx = torch.randperm(N)[:n_sub]
        estimates.append(twonn_dim(X[idx]))
    est = torch.tensor([e for e in estimates if math.isfinite(e)], dtype=torch.float32)
    if len(est) == 0:
        return float("nan"), float("nan")
    return float(est.mean()), float(est.std())


def rotation_angles(R):
    R = R.to(torch.float32)
    try:
        eigvals = torch.linalg.eigvals(R)
    except Exception:
        return None
    theta = torch.atan2(eigvals.imag, eigvals.real).abs()
    return theta.sort().values


def principal_angle_overlap(A, B):
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    try:
        Qa, _ = torch.linalg.qr(A.to(torch.float32))
        Qb, _ = torch.linalg.qr(B.to(torch.float32))
    except Exception:
        return float("nan")
    M = Qa.T @ Qb
    return float(torch.linalg.svdvals(M).mean().item())


def extract_subspaces(eigvals, eigvecs, carry_tol=0.3, flip_tol=0.3):
    angles = torch.atan2(eigvals.imag, eigvals.real).abs()
    carry_mask = angles < carry_tol
    flip_mask = angles > (math.pi - flip_tol)
    mid_mask = ~carry_mask & ~flip_mask
    return {
        "carry_vecs": eigvecs[:, carry_mask].real,
        "flip_vecs": eigvecs[:, flip_mask].real,
        "n_carry": int(carry_mask.sum().item()),
        "n_flip": int(flip_mask.sum().item()),
        "n_mid": int(mid_mask.sum().item()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rank", type=int, default=64, help="PCA basis rank per layer")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"model={args.model}  rank={args.rank}  dtype={args.dtype}")

    t0 = time.perf_counter()
    print(f"\n[1/5] loading model...")
    model, tokenizer = load_model(args.model, args.dtype)
    L = len(model.model.layers)
    H = model.config.hidden_size
    V = model.config.vocab_size
    print(f"  L={L}  H={H}  V={V}  load time: {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    print(f"\n[2/5] collecting per-layer hidden states...")
    hiddens = collect_per_layer_hiddens(model, tokenizer, CALIB_TEXTS)
    N_total = hiddens[0].shape[0]
    print(f"  {N_total} tokens, {L + 1} layers including embedding")
    print(f"  forward time: {time.perf_counter()-t1:.1f}s")

    t2 = time.perf_counter()
    print(f"\n[3/5] per-layer bootstrap TwoNN (20 subsamples)...")
    twonn_per_layer_mean = []
    twonn_per_layer_std = []
    for i in range(L + 1):
        mean, std = twonn_bootstrap(hiddens[i])
        twonn_per_layer_mean.append(mean)
        twonn_per_layer_std.append(std)
    print(f"  done in {time.perf_counter()-t2:.1f}s")

    t3 = time.perf_counter()
    print(f"\n[4/5] per-layer PCA bases + per-transition rotation spectra...")
    bases = {i: pca_basis(hiddens[i], args.rank) for i in range(L + 1)}

    # Histogram bins for rotation angles (24 bins from 0 to π)
    bins = torch.linspace(0, math.pi, 25)
    per_transition = []
    subspace_store = []
    for i in range(L):
        R = bases[i + 1].T @ bases[i]
        angles = rotation_angles(R)
        if angles is None:
            per_transition.append(None); subspace_store.append(None); continue
        hist = torch.histogram(angles, bins=bins).hist.tolist()
        # Classify directions
        try:
            eigvals, eigvecs = torch.linalg.eig(R)
        except Exception:
            per_transition.append(None); subspace_store.append(None); continue
        parts = extract_subspaces(eigvals, eigvecs)
        per_transition.append({
            "mean_angle": float(angles.mean()),
            "median_angle": float(angles.median()),
            "max_angle": float(angles.max()),
            "hist_counts": hist,
            "n_carry": parts["n_carry"],
            "n_flip": parts["n_flip"],
            "n_mid": parts["n_mid"],
            "mode_concentration": (parts["n_carry"] + parts["n_flip"]) / args.rank,
        })
        subspace_store.append(parts)
    print(f"  done in {time.perf_counter()-t3:.1f}s")

    t4 = time.perf_counter()
    print(f"\n[5/5] adjacent-transition carry overlap (persistence)...")
    carry_overlap_per_transition = []
    for i in range(L - 1):
        si = subspace_store[i]; si1 = subspace_store[i + 1]
        if si is None or si1 is None:
            carry_overlap_per_transition.append(None); continue
        A = si["carry_vecs"]; B = si1["carry_vecs"]
        if A.shape[1] == 0 or B.shape[1] == 0:
            carry_overlap_per_transition.append(None); continue
        A_global = bases[i] @ A
        B_global = bases[i + 1] @ B
        carry_overlap_per_transition.append(principal_angle_overlap(A_global, B_global))

    # First-to-last overlap
    first_last_overlap = None
    if subspace_store[0] is not None and subspace_store[-1] is not None:
        A = subspace_store[0]["carry_vecs"]; B = subspace_store[-1]["carry_vecs"]
        if A.shape[1] > 0 and B.shape[1] > 0:
            first_last_overlap = principal_angle_overlap(
                bases[0] @ A, bases[L - 1] @ B)
    print(f"  done in {time.perf_counter()-t4:.1f}s")

    # Build the fingerprint
    fingerprint = {
        "model": args.model,
        "L": L, "H": H, "V": V,
        "n_tokens": N_total,
        "rank": args.rank,
        "dtype": args.dtype,
        "twonn_mean_per_layer": twonn_per_layer_mean,
        "twonn_std_per_layer": twonn_per_layer_std,
        "twonn_mean_grand": sum(x for x in twonn_per_layer_mean if math.isfinite(x))
                             / sum(1 for x in twonn_per_layer_mean if math.isfinite(x)),
        "per_transition": per_transition,
        "adjacent_carry_overlap": carry_overlap_per_transition,
        "first_to_last_carry_overlap": first_last_overlap,
        "mean_rotation_angle_per_transition": [
            t["mean_angle"] if t is not None else None for t in per_transition],
        "mode_concentration_per_transition": [
            t["mode_concentration"] if t is not None else None for t in per_transition],
        "carry_fraction_per_transition": [
            t["n_carry"] / args.rank if t is not None else None for t in per_transition],
        "flip_fraction_per_transition": [
            t["n_flip"] / args.rank if t is not None else None for t in per_transition],
        "total_wall_seconds": time.perf_counter() - t0,
    }

    # Summary console output
    print(f"\n=== fingerprint summary for {args.model} ===")
    tw = fingerprint["twonn_mean_grand"]
    print(f"  TwoNN grand mean (across layers):    {tw:.3f}")
    valid_transitions = [t for t in per_transition if t is not None]
    if valid_transitions:
        mean_angle = sum(t["mean_angle"] for t in valid_transitions) / len(valid_transitions)
        mean_mode_conc = sum(t["mode_concentration"] for t in valid_transitions) / len(valid_transitions)
        mean_carry = sum(t["n_carry"] for t in valid_transitions) / (len(valid_transitions) * args.rank)
        mean_flip = sum(t["n_flip"] for t in valid_transitions) / (len(valid_transitions) * args.rank)
        print(f"  mean rotation angle per transition:  {mean_angle:.3f} rad")
        print(f"  mean mode concentration (carry+flip/total): {mean_mode_conc:.3f}")
        print(f"  mean carry fraction:                 {mean_carry:.3f}")
        print(f"  mean flip fraction:                  {mean_flip:.3f}")
    if carry_overlap_per_transition:
        valid = [o for o in carry_overlap_per_transition if o is not None]
        print(f"  mean adjacent carry overlap:         {sum(valid)/len(valid):.3f}")
    if first_last_overlap is not None:
        print(f"  first-to-last carry overlap:         {first_last_overlap:.3f}")
    print(f"  total wall seconds:                   {fingerprint['total_wall_seconds']:.0f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
