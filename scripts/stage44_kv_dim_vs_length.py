"""
Stage 44 — KV cache operational dim as a function of context length.

Tests the claim: each prior token in the KV cache adds a small amount of
operational dim to the forward-pass projection, with the contribution
decreasing per token and eventually saturating.

Protocol on Qwen3-0.6B (n_kv_heads=8, head_dim=128, d_kv=1024):
  1. Tokenize a long passage (target ~2000 tokens).
  2. Forward through the model, capturing past_key_values. For each layer,
     the K cache is [1, n_kv_heads, T, head_dim] and similarly for V.
  3. At context snapshots N ∈ {10, 30, 100, 300, 1000, 2000}, compute:
     - Effective rank of the K and V caches at each layer (PCA over the N
       token rows, ranking by variance captured).
     - Specifically: flatten to [N, n_kv_heads * head_dim], center, SVD,
       count singular values to hit {90%, 95%, 99%} variance.
  4. Plot/print effective rank vs N per layer and averaged.

Prediction (saturating claim): effective rank grows rapidly at small N,
then the marginal rank-per-token drops. Should asymptote at some cap
(probably < d_kv = 1024).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# A longer coherent passage — stitched from calibration texts to make a
# ~1500-2000 token corpus.
LONG_TEXT = (
    "The cell is the basic structural unit of life, composed of cytoplasm "
    "enclosed within a membrane. Quantum mechanics describes the behavior of "
    "matter and energy at atomic and subatomic scales. The history of computing "
    "began with mechanical calculators and evolved through vacuum tubes, then "
    "transistors, and now deep integrated circuits at the nanometer scale. "
    "Photosynthesis uses sunlight to convert carbon dioxide and water into "
    "glucose and oxygen, powering nearly all life on Earth through a chain of "
    "trophic dependencies. Neural networks consist of parameterized layers "
    "trained by gradient descent to approximate functions over large data "
    "distributions. Plate tectonics describes the slow movement of Earth's "
    "lithospheric plates over the mantle, giving rise to earthquakes, "
    "volcanism, and the rearrangement of continents over geological time. "
    "Proteins fold into complex three-dimensional structures determined by "
    "their amino acid sequences and the thermodynamic landscape of the solvent "
    "they are immersed in. The standard model of particle physics unifies "
    "electromagnetic, weak, and strong interactions in a single quantum field "
    "framework. Evolution by natural selection operates on heritable variation "
    "in populations, producing over time the apparent design that Darwin "
    "first noticed and William Paley had earlier taken as evidence of a "
    "designer. Cryptography protects information using mathematical operations "
    "that are easy to compute in one direction and hard to invert without the "
    "key. Thermodynamics relates heat, work, energy, and entropy in macroscopic "
    "systems, providing the bridge between microscopic statistics and observed "
    "behavior. Graph theory studies vertices connected by edges across many "
    "practical applications, from network routing to protein interaction maps. "
    "Black holes are regions of spacetime from which nothing, not even light, "
    "can escape, bounded by an event horizon that in some formulations may "
    "encode information about the interior on the boundary itself. DNA encodes "
    "genetic information in a double-helix structure of paired nucleotide "
    "bases whose sequence determines the catalogue of proteins an organism "
    "can synthesize. Volcanoes form at tectonic plate boundaries and hot "
    "spots in Earth's mantle, venting heat and material from the deep "
    "interior. Linear algebra provides the mathematical foundation for many "
    "machine learning algorithms, and more broadly for any computational "
    "treatment of high-dimensional data. Game theory analyzes strategic "
    "interactions between rational decision makers, providing a language for "
    "cooperation and conflict in both biological and economic contexts. "
    "Bayesian inference updates a prior probability distribution using "
    "observed data, producing a posterior that ideally reflects both prior "
    "beliefs and new evidence in proportion to their respective informational "
    "content. The immune system recognizes pathogens through pattern "
    "recognition receptors on innate cells, adaptive antibody repertoires, "
    "and a memory system that retains exposures. The Riemann zeta function "
    "encodes deep information about the distribution of primes through its "
    "analytic continuation and the location of its non-trivial zeros. The "
    "process of learning in large language models involves adjusting trillions "
    "of parameters so that predicted probability distributions match the "
    "observed distribution of text, which is a high-dimensional maximum "
    "likelihood problem with a very curved loss landscape. Calculus allows us "
    "to reason about continuous change by taking limits, assembling tangent "
    "lines and areas from infinitesimal pieces. Statistical mechanics derives "
    "thermodynamic laws from the behavior of large ensembles of microscopic "
    "particles, using probability distributions over microstates consistent "
    "with macroscopic constraints. The theory of computation distinguishes "
    "between tractable and intractable problems by studying the resources "
    "required to solve them. Electromagnetism, codified by Maxwell's equations, "
    "describes how electric and magnetic fields propagate and interact with "
    "charges, a unification that hinted at the deeper symmetries later "
    "discovered by Einstein's special relativity. The ribosome is a molecular "
    "machine that synthesizes proteins from messenger RNA, translating "
    "three-nucleotide codons into amino acids. Tensor calculus extends "
    "linear algebra to objects that transform in prescribed ways under "
    "coordinate changes, a prerequisite for general relativity and certain "
    "formulations of gauge theory. Renormalization in quantum field theory "
    "handles the divergences that arise when one tries to compute amplitudes "
    "to all orders of perturbation; the running of coupling constants with "
    "energy scale is one of its most important outputs. Signal processing "
    "decomposes functions into frequency components to analyze and manipulate "
    "them, with the Fourier transform serving as the canonical tool. Weather "
    "prediction couples numerical integration of fluid dynamics equations with "
    "data assimilation from satellite and ground measurements, producing "
    "forecasts whose skill decays with lead time. Control theory designs "
    "feedback loops to stabilize systems in the face of disturbances, a "
    "mathematical field with applications from aircraft autopilots to medical "
    "device regulation. The solar system formed from a collapsing cloud of gas "
    "and dust about 4.6 billion years ago, with the Sun at the center and "
    "planets condensing from the protoplanetary disk."
)


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def effective_rank_from_variance(X, thresholds=(0.90, 0.95, 0.99)):
    """X: [N, d]. Returns dict threshold -> rank needed."""
    N, d = X.shape
    if N <= 1:
        return {t: 1 for t in thresholds}
    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD; singular values squared ~ variances
    try:
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return {t: min(N, d) for t in thresholds}
    var = S ** 2
    total = var.sum()
    if total <= 0:
        return {t: 1 for t in thresholds}
    cum = torch.cumsum(var, dim=0) / total
    out = {}
    for th in thresholds:
        r = int((cum < th).sum().item()) + 1
        out[th] = max(1, min(r, min(N, d)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--snapshots", default="10,30,100,300,1000,2000")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage44_kv_dim_vs_length.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    L = len(model.model.layers)
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    d_kv = n_kv_heads * head_dim
    print(f"  L={L}  n_kv_heads={n_kv_heads}  head_dim={head_dim}  d_kv={d_kv}")

    ids = tokenizer(LONG_TEXT, return_tensors="pt").input_ids.to(device)
    T_total = ids.shape[1]
    print(f"\n=== tokenized prompt: {T_total} tokens ===")

    snapshots = [int(x) for x in args.snapshots.split(",") if int(x) <= T_total]
    print(f"  snapshots: {snapshots}")

    # One forward pass, capture past_key_values. Then truncate to each snapshot.
    print(f"\n=== forward pass (capturing full KV) ===")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    print(f"  {time.perf_counter()-t0:.1f}s")

    # past is a tuple of (K, V) per layer, each [batch, n_kv_heads, T_total, head_dim]
    # Gather K, V per layer.
    all_K = []; all_V = []
    for layer_idx in range(L):
        layer_past = past[layer_idx] if hasattr(past, '__getitem__') else past.layers[layer_idx]
        if isinstance(layer_past, tuple):
            K, V = layer_past
        else:
            K = layer_past.key_cache if hasattr(layer_past, 'key_cache') else layer_past[0]
            V = layer_past.value_cache if hasattr(layer_past, 'value_cache') else layer_past[1]
        # K, V: [1, n_kv_heads, T_total, head_dim]
        all_K.append(K.detach().to(torch.float32).cpu())
        all_V.append(V.detach().to(torch.float32).cpu())
    print(f"  captured K/V for {L} layers")

    thresholds = (0.90, 0.95, 0.99)
    results = []
    for N in snapshots:
        print(f"\n=== N = {N} ===")
        per_layer_k = {th: [] for th in thresholds}
        per_layer_v = {th: [] for th in thresholds}
        for l in range(L):
            # Reshape to [N, n_kv_heads * head_dim]
            K = all_K[l][0, :, :N, :].permute(1, 0, 2).reshape(N, d_kv)
            V = all_V[l][0, :, :N, :].permute(1, 0, 2).reshape(N, d_kv)
            rk = effective_rank_from_variance(K, thresholds)
            rv = effective_rank_from_variance(V, thresholds)
            for th in thresholds:
                per_layer_k[th].append(rk[th])
                per_layer_v[th].append(rv[th])

        # Report means
        print(f"  K effective rank (mean across {L} layers):")
        for th in thresholds:
            vals = torch.tensor(per_layer_k[th], dtype=torch.float32)
            print(f"    {th:.2f}: mean={float(vals.mean()):.1f}  "
                  f"min={int(vals.min())}  max={int(vals.max())}")
        print(f"  V effective rank (mean across {L} layers):")
        for th in thresholds:
            vals = torch.tensor(per_layer_v[th], dtype=torch.float32)
            print(f"    {th:.2f}: mean={float(vals.mean()):.1f}  "
                  f"min={int(vals.min())}  max={int(vals.max())}")

        results.append({
            "N": N,
            "K_mean_rank": {str(th): float(torch.tensor(per_layer_k[th], dtype=torch.float32).mean()) for th in thresholds},
            "V_mean_rank": {str(th): float(torch.tensor(per_layer_v[th], dtype=torch.float32).mean()) for th in thresholds},
            "K_per_layer": per_layer_k,
            "V_per_layer": per_layer_v,
        })

    print(f"\n=== summary (saturation check) ===")
    print(f"  {'N':>6}  {'K@0.95':>8}  {'V@0.95':>8}  {'dim/token_since_prev':>24}")
    prev_k = 0; prev_n = 0
    for r in results:
        k95 = r["K_mean_rank"]["0.95"]
        v95 = r["V_mean_rank"]["0.95"]
        delta_n = r["N"] - prev_n
        delta_k = k95 - prev_k
        rate = delta_k / max(delta_n, 1)
        print(f"  {r['N']:>6}  {k95:>8.1f}  {v95:>8.1f}  {rate:>24.4f}")
        prev_k = k95; prev_n = r["N"]

    print(f"\n  If rate decreases with N, user's saturation claim is confirmed.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "d_kv": d_kv,
            "total_tokens": T_total,
            "snapshots": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
