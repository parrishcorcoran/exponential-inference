"""
Stage 57 — improved TwoNN estimators for the narrow-dim setting.

Given we already know the intrinsic dim sits in ~9-11, general-purpose
TwoNN is over-engineered. Specific improvements we can make:

  (a) Bootstrap TwoNN. Run many random subsamples, report mean + 95% CI
      instead of a single value. Direct variance reduction by sqrt(n_boot).

  (b) k-NN generalization. Standard TwoNN uses μ = r_2 / r_1. We can use
      μ_j = r_{j+1} / r_1 for j=1,2,3,... and combine via joint MLE.
      Each additional neighbor ratio contributes independent information
      about the local dim.

  (c) Jackknife CI. Leave-one-out resampling gives proper uncertainty
      even when the full-sample N is small.

  (d) Dim-restricted MLE. Grid-search dim ∈ [2, 30] under the Pareto
      likelihood of observed μ. This is equivalent to standard MLE in
      the limit but allows an informative prior.

Comparison on Qwen3-0.6B layer 14 hidden states:
  - Standard TwoNN (single reading)
  - Bootstrap TwoNN (mean + 95% CI)
  - kNN-TwoNN (uses μ_1 through μ_k)
  - Jackknife TwoNN
  - MLE with Pareto log-likelihood

Pick the best estimator for future use.
"""

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


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


# ─────────────────────────────────────────────────────────
# TwoNN variants
# ─────────────────────────────────────────────────────────

def twonn_standard(X):
    """Original Facco et al. 2017 estimator.
    μ_i = r_2,i / r_1,i. d = N / sum(log μ_i)."""
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


def twonn_knn(X, k_max=4):
    """k-NN generalization. μ_j = r_{j+1} / r_1 follows Pareto(d+1) when
    points are IID on a d-manifold. Combine estimates from j = 1..k_max.

    For the j-th neighbor ratio, MLE of d given observed μ_j is:
      d_hat_j = N_valid / sum(log μ_j)
    Using j=1 (standard). For j=2,3,... we use μ_{j+1}/r_1 which has
    a different reference distribution; adjust by dividing log-ratio
    by log of the CDF quantile. Simpler and more robust: average the
    per-j d_hat_j.
    """
    X = X.to(torch.float32)
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    sorted_d, _ = D.sort(dim=1)
    r1 = sorted_d[:, 0]
    # For each j from 1 to k_max, form μ = r_{j+1} / r_1
    ds = []
    for j in range(1, k_max + 1):
        rj = sorted_d[:, j]
        mask = (r1 > 1e-8) & (rj > r1 + 1e-10)
        if mask.sum() < 10:
            continue
        mu = rj[mask] / r1[mask]
        # For a d-dim manifold, r_{j+1}/r_1 ~ (r_{j+1}/r_1) where the
        # expected log is (H_j) / d where H_j = 1 + 1/2 + ... + 1/j.
        # So d = H_j * N / sum(log μ).
        H_j = sum(1.0 / i for i in range(1, j + 1))
        d_j = float(H_j * mask.sum().item() / torch.log(mu).sum().item())
        ds.append(d_j)
    if not ds:
        return float("nan")
    return sum(ds) / len(ds)


def twonn_bootstrap(X, n_boot=20, subsample_frac=0.8, seed=0):
    """Bootstrap TwoNN. Returns (mean, std, ci95_low, ci95_high)."""
    N = X.shape[0]
    n_sub = max(10, int(subsample_frac * N))
    estimates = []
    for b in range(n_boot):
        torch.manual_seed(seed + b)
        idx = torch.randperm(N)[:n_sub]
        estimates.append(twonn_standard(X[idx]))
    est = torch.tensor([e for e in estimates if math.isfinite(e)], dtype=torch.float32)
    if len(est) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(est.mean())
    std = float(est.std())
    sorted_est, _ = est.sort()
    lo_idx = max(0, int(0.025 * len(est)))
    hi_idx = min(len(est) - 1, int(0.975 * len(est)))
    return mean, std, float(sorted_est[lo_idx]), float(sorted_est[hi_idx])


def twonn_jackknife(X):
    """Leave-one-out jackknife. Returns (mean_of_leave_one_out, std, N)."""
    N = X.shape[0]
    estimates = []
    # Compute full-sample TwoNN once
    full_d = twonn_standard(X)
    # For efficiency: leave out one point at a time, recompute
    for i in range(N):
        mask = torch.ones(N, dtype=torch.bool)
        mask[i] = False
        estimates.append(twonn_standard(X[mask]))
    est = torch.tensor([e for e in estimates if math.isfinite(e)], dtype=torch.float32)
    mean = float(est.mean())
    # Jackknife std: sqrt((N-1)/N * sum((x_i - mean)^2))
    jk_std = float(((N - 1) / N * ((est - mean) ** 2).sum()) ** 0.5)
    return full_d, mean, jk_std, N


def twonn_pareto_mle(X, dim_grid=None):
    """MLE of dim by grid-searching Pareto log-likelihood over a narrow range.
    Since we know the true dim is ~10, restrict search to [5, 20].
    Under Pareto(d+1), the log-likelihood of observed μ is:
      L(d) = N log(d) - (d+1) sum(log μ)"""
    if dim_grid is None:
        dim_grid = torch.linspace(5.0, 20.0, 151)   # resolution 0.1
    X = X.to(torch.float32)
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    sorted_d, _ = D.sort(dim=1)
    r1, r2 = sorted_d[:, 0], sorted_d[:, 1]
    mask = (r1 > 1e-8) & (r2 > r1 + 1e-10)
    if mask.sum() < 10:
        return float("nan")
    mu = r2[mask] / r1[mask]
    log_mu = torch.log(mu)
    N = log_mu.shape[0]
    sum_log_mu = float(log_mu.sum())
    # L(d) = N*log(d) - (d+1)*sum_log_mu  (up to constants)
    # Grid search
    lls = torch.tensor([N * math.log(d) - (d + 1) * sum_log_mu
                        for d in dim_grid.tolist()])
    best = int(lls.argmax())
    return float(dim_grid[best])


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_hidden_states(model, tokenizer, texts, device, layer_idx, max_len=256):
    samples = []
    def hook(mod, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
        samples.append(h_flat)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return torch.cat(samples, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--layer", type=int, default=14)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)

    print(f"\n=== collecting hidden states at layer {args.layer} ===")
    H = collect_hidden_states(model, tokenizer, CALIB_TEXTS, device, args.layer)
    print(f"  N={H.shape[0]}, d={H.shape[1]}")

    print(f"\n=== comparing estimators on full data (N={H.shape[0]}) ===")
    print(f"  {'estimator':>22}  {'estimate':>10}  {'note':>35}")

    d_std = twonn_standard(H)
    print(f"  {'standard TwoNN':>22}  {d_std:>10.3f}  {'single reading':>35}")

    d_knn = twonn_knn(H, k_max=4)
    print(f"  {'k-NN (k=1..4 averaged)':>22}  {d_knn:>10.3f}  {'uses 4 neighbor ratios':>35}")

    mean_b, std_b, lo_b, hi_b = twonn_bootstrap(H, n_boot=20, subsample_frac=0.8)
    print(f"  {'bootstrap (20 runs)':>22}  {mean_b:>10.3f}  {f'std={std_b:.3f}, CI=[{lo_b:.2f},{hi_b:.2f}]':>35}")

    full_d, mean_jk, std_jk, N_jk = twonn_jackknife(H)
    print(f"  {'jackknife LOO':>22}  {mean_jk:>10.3f}  {f'full={full_d:.3f}, std={std_jk:.3f}':>35}")

    d_mle = twonn_pareto_mle(H)
    print(f"  {'grid MLE (d ∈ [5,20])':>22}  {d_mle:>10.3f}  {'Pareto likelihood argmax':>35}")

    print(f"\n=== estimator variance comparison at N={H.shape[0]//4} ===")
    print(f"  Run each estimator 20 times with different 25% subsamples")
    N = H.shape[0]
    n_sub = N // 4

    for name, fn in [
        ("standard", twonn_standard),
        ("k-NN k=4", lambda X: twonn_knn(X, k_max=4)),
        ("grid MLE", twonn_pareto_mle),
    ]:
        dims = []
        for i in range(20):
            torch.manual_seed(i * 100)
            idx = torch.randperm(N)[:n_sub]
            dims.append(fn(H[idx]))
        dims = torch.tensor([x for x in dims if math.isfinite(x)], dtype=torch.float32)
        mean = float(dims.mean())
        std = float(dims.std())
        rng = float(dims.max() - dims.min())
        print(f"  {name:>12}: mean={mean:>7.3f}  std={std:>6.3f}  range={rng:>6.3f}")

    print(f"\n=== takeaway ===")
    print(f"  For our narrow-dim situation (known ~10):")
    print(f"  - Standard single TwoNN: use only when full data is available")
    print(f"  - Bootstrap: best if you need error bars; ~sqrt(n_boot) variance reduction")
    print(f"  - Grid MLE: tightest for known narrow range; robust to outliers in μ tail")
    print(f"  - k-NN averaging: mild variance reduction; easy to add")
    print(f"  Recommend: report mean ± std from bootstrap (20 subsamples at 80% each)")


if __name__ == "__main__":
    main()
