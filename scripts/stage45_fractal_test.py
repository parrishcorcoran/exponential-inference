"""
Stage 45 — Frame 1: Fractal self-similarity test.

Tests whether the operational-rank-vs-context-length scaling is the same at
per-layer scale and per-head scale. If so, the model has self-similar
(fractal) structure. If scaling exponents differ materially, it isn't
fractal in this sense.

Protocol (Qwen3-0.6B):
  1. Forward on long text, capture K cache per layer.
  2. At snapshots N ∈ {10, 30, 100, 300}, compute:
     - Per-layer K rank (cache as [N, n_kv_heads*head_dim])
     - Per-head K rank per layer (cache as [N, head_dim] for each of 8 heads)
  3. Fit rank ~ N^α at both scales.
  4. If α_layer ≈ α_head → fractal. If |α_layer - α_head| > 0.1 → not fractal.
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
    "discovered by Einstein's special relativity."
)


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def effective_rank(X, threshold=0.95):
    N, d = X.shape
    if N <= 1: return 1
    Xc = X - X.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return min(N, d)
    var = S ** 2
    total = var.sum()
    if total <= 0: return 1
    cum = torch.cumsum(var, dim=0) / total
    return max(1, int((cum < threshold).sum().item()) + 1)


def fit_power_law(ns, ranks):
    """Fit rank = C * N^alpha via log-log linear regression."""
    ns = torch.tensor(ns, dtype=torch.float64)
    rs = torch.tensor(ranks, dtype=torch.float64)
    ln = ns.log()
    lr = rs.log()
    # Linear regression: lr = log(C) + alpha * ln
    ln_mean = ln.mean(); lr_mean = lr.mean()
    num = ((ln - ln_mean) * (lr - lr_mean)).sum()
    den = ((ln - ln_mean) ** 2).sum()
    alpha = float(num / den) if den > 0 else 0.0
    log_C = float(lr_mean - alpha * ln_mean)
    # R²
    pred = log_C + alpha * ln
    ss_res = ((lr - pred) ** 2).sum()
    ss_tot = ((lr - lr_mean) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return alpha, math.exp(log_C), r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--snapshots", default="10,30,100,300")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage45_fractal_test.json")
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
    print(f"\n=== tokenized: {T_total} tokens ===")

    snapshots = [int(x) for x in args.snapshots.split(",") if int(x) <= T_total]

    print(f"\n=== forward pass ===")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    print(f"  {time.perf_counter()-t0:.1f}s")

    all_K = []
    for layer_idx in range(L):
        layer_past = past[layer_idx] if hasattr(past, '__getitem__') else past.layers[layer_idx]
        if isinstance(layer_past, tuple):
            K = layer_past[0]
        else:
            K = layer_past.key_cache if hasattr(layer_past, 'key_cache') else layer_past[0]
        all_K.append(K.detach().to(torch.float32).cpu())

    # Compute rank at two scales at each snapshot
    results = []
    layer_ranks_by_N = {N: [] for N in snapshots}
    head_ranks_by_N = {N: [] for N in snapshots}  # list of per-(layer, head) ranks

    for N in snapshots:
        for l in range(L):
            # Per-layer scale: cache as [N, d_kv]
            K_layer = all_K[l][0, :, :N, :].permute(1, 0, 2).reshape(N, d_kv)
            r_layer = effective_rank(K_layer, threshold=0.95)
            layer_ranks_by_N[N].append(r_layer)

            # Per-head scale: cache as [N, head_dim] for each head
            for h in range(n_kv_heads):
                K_head = all_K[l][0, h, :N, :]  # [N, head_dim]
                r_head = effective_rank(K_head, threshold=0.95)
                head_ranks_by_N[N].append(r_head)

    # Mean rank per scale per N
    print(f"\n=== rank vs N at two scales ===")
    print(f"  {'N':>6}  {'layer_mean':>10}  {'head_mean':>10}  {'head_max_possible':>18}")
    layer_means = []
    head_means = []
    for N in snapshots:
        lm = sum(layer_ranks_by_N[N]) / len(layer_ranks_by_N[N])
        hm = sum(head_ranks_by_N[N]) / len(head_ranks_by_N[N])
        layer_means.append(lm)
        head_means.append(hm)
        hmp = min(N, head_dim)
        print(f"  {N:>6}  {lm:>10.2f}  {hm:>10.2f}  {hmp:>18}")

    # Power law fits
    alpha_layer, C_layer, r2_layer = fit_power_law(snapshots, layer_means)
    alpha_head, C_head, r2_head = fit_power_law(snapshots, head_means)

    print(f"\n=== power law fits ===")
    print(f"  per-layer:  rank = {C_layer:.3f} * N^{alpha_layer:.3f}   R²={r2_layer:.4f}")
    print(f"  per-head:   rank = {C_head:.3f} * N^{alpha_head:.3f}    R²={r2_head:.4f}")
    print(f"  alpha_layer - alpha_head = {alpha_layer - alpha_head:+.3f}")

    if abs(alpha_layer - alpha_head) < 0.05:
        print(f"\n  VERDICT: |Δα| < 0.05  →  fractal self-similarity likely")
    elif abs(alpha_layer - alpha_head) < 0.10:
        print(f"\n  VERDICT: |Δα| ∈ [0.05, 0.10)  →  weak evidence for fractal")
    else:
        print(f"\n  VERDICT: |Δα| ≥ 0.10  →  NOT fractal in this sense")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L, "n_kv_heads": n_kv_heads, "head_dim": head_dim, "d_kv": d_kv,
            "snapshots": snapshots,
            "layer_mean_rank": layer_means,
            "head_mean_rank": head_means,
            "alpha_layer": alpha_layer, "C_layer": C_layer, "r2_layer": r2_layer,
            "alpha_head": alpha_head, "C_head": C_head, "r2_head": r2_head,
            "delta_alpha": alpha_layer - alpha_head,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
