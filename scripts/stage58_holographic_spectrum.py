"""
Stage 58 — Holographic interference test via rotation-operator spectra.

The claim: if the manifold is holographic, layer-to-layer transitions
encode interference patterns — frequency/phase structure, not just
linear projection. Measure this via the ROTATION OPERATOR between
consecutive layers' PCA bases.

Rotation from layer-i basis to layer-(i+1) basis:
  R_{i→i+1} = P_{i+1} P_i^T

Properties to examine:
  (a) Eigenvalues: a pure rotation has complex conjugate pairs e^{±iθ};
      the angles θ_k are the "rotation angles". Holographic structure
      predicts ANGLES CLUSTER AT SPECIFIC VALUES across the stack —
      universal frequencies.

  (b) Are these angles CONSISTENT across layer transitions? If layers
      rotate by the same discrete set of angles, we have frequency
      structure. If random, it's not holographic in this sense.

  (c) Does the distribution of angles change through the stack (phase
      transition at 0→1, Finding 03)?

Predictions:
  - Holographic: a few dominant angles appear across layers (discrete spectrum)
  - Non-holographic: continuous spectrum, angles vary freely layer-to-layer
  - Finding 02 (universal curve) would be *explained* by specific
    universal angles appearing in the spectrum.
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


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_per_layer_hiddens(model, tokenizer, texts, device, max_len=256):
    """Returns a dict layer_idx -> [N, d] hidden states."""
    L = len(model.model.layers)
    samples = [[] for _ in range(L + 1)]

    def make_hook(i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            samples[i].append(h_flat)
        return hook

    # Include embedding output (layer -1 / hidden[0]) via a hook on embed_tokens
    def embed_hook(mod, inputs, output):
        h = output
        h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
        samples[0].append(h_flat)
    h0 = model.model.embed_tokens.register_forward_hook(embed_hook)

    handles = [h0] + [
        model.model.layers[i].register_forward_hook(make_hook(i + 1))
        for i in range(L)
    ]
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()

    return {i: torch.cat(samples[i], dim=0) for i in range(L + 1)}


def pca_basis(H, k):
    """Top-k principal directions of H (mean-centered). H: [N, d]."""
    mu = H.mean(dim=0, keepdim=True)
    Hc = H - mu
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1])
    return P  # [d, k], columns are orthonormal


def rotation_angles(R):
    """Given a near-rotation matrix R [d, d], compute its rotation angles.
    A proper rotation has complex eigenvalues e^{±iθ_k}. We extract θ_k
    from the complex eigenvalues.

    R should be close to orthogonal; we use torch.linalg.eig."""
    R = R.to(torch.float32)
    try:
        eigvals = torch.linalg.eigvals(R)  # complex
    except Exception:
        return None
    # Each eigenvalue is a complex number. Extract angles.
    theta = torch.atan2(eigvals.imag, eigvals.real)  # [-pi, pi]
    # Only take one of each conjugate pair: abs(theta)
    # (a rotation has each nonzero angle twice, once +θ once -θ)
    return theta.abs().sort().values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=64,
                   help="PCA basis rank per layer (we'll compute rotations between these)")
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
    L = len(model.model.layers)
    print(f"  L = {L}")

    print(f"\n=== collecting per-layer hidden states ===")
    hiddens = collect_per_layer_hiddens(model, tokenizer, CALIB_TEXTS, device)
    for i in range(0, L + 1, max(1, L // 4)):
        print(f"  layer {i}: {hiddens[i].shape}")

    print(f"\n=== computing per-layer PCA basis (rank {args.rank}) ===")
    bases = {}
    for i in range(L + 1):
        bases[i] = pca_basis(hiddens[i], args.rank)  # [d, k]
    print(f"  {L+1} bases, each [d={bases[0].shape[0]}, k={bases[0].shape[1]}]")

    # Rotation operator between consecutive layers in rank-k subspace
    # R_{i,i+1} = P_{i+1}^T @ P_i, shape [k, k]
    # This is the rotation FROM layer-i subspace TO layer-(i+1) subspace
    print(f"\n=== computing rotation operators (layer i → i+1) ===")
    print(f"  per-layer rotation angles (top 5) — angles close to 0 = small rotation")
    print(f"  {'i→i+1':>6}  {'max_θ':>7}  {'mean_θ':>7}  {'top-5 angles (rad)':>40}")
    all_angles = []
    per_layer_angles = {}
    for i in range(L):
        R_ki = bases[i + 1].T @ bases[i]  # [k, k]
        # Symmetric form: use R_ki; this is a rotation/projection matrix in k-space
        angles = rotation_angles(R_ki)
        if angles is None: continue
        top5 = angles[-5:].tolist()
        per_layer_angles[i] = angles.tolist()
        all_angles.extend(angles.tolist())
        if i % max(1, L // 8) == 0 or i == L - 1:
            print(f"  {i:>3}→{i+1:<2}  {float(angles.max()):>7.3f}  "
                  f"{float(angles.mean()):>7.3f}  "
                  f"[{', '.join(f'{x:.2f}' for x in top5)}]")

    # Aggregate: is there a discrete set of angles across layers?
    print(f"\n=== distribution of rotation angles across ALL layer transitions ===")
    a = torch.tensor(all_angles, dtype=torch.float32)
    print(f"  total angle samples: {len(a)}")
    print(f"  mean: {float(a.mean()):.3f}  std: {float(a.std()):.3f}")
    print(f"  median: {float(a.median()):.3f}")
    print(f"  quartiles: [{float(a.quantile(0.25)):.3f}, "
          f"{float(a.quantile(0.5)):.3f}, "
          f"{float(a.quantile(0.75)):.3f}]")

    # Histogram: are angles concentrated at specific values (holographic)?
    print(f"\n=== histogram of rotation angles (24 bins, 0 to π) ===")
    bins = torch.linspace(0, math.pi, 25)
    counts = torch.histogram(a, bins=bins).hist
    max_count = float(counts.max())
    print(f"  {'bin (rad)':>18}  {'count':>6}  histogram")
    for i in range(24):
        lo = float(bins[i]); hi = float(bins[i + 1])
        c = int(counts[i].item())
        bar_len = int(40 * c / max_count) if max_count > 0 else 0
        print(f"  [{lo:>5.3f}, {hi:>5.3f})  {c:>6}  {'█' * bar_len}")

    # Characterize: is this distribution uniform, skewed, or multi-modal?
    print(f"\n=== interpretation ===")
    # If holographic/discrete: a few bins would dominate (multi-modal)
    # If near-identity rotations: concentration near 0 (small angles)
    # If random rotation: roughly uniform across [0, π]
    n_bins_above_half_max = int((counts >= 0.5 * max_count).sum().item())
    print(f"  bins ≥ 50% of peak count: {n_bins_above_half_max}/24")
    near_zero_frac = float((a < 0.1).float().mean())
    print(f"  fraction of angles near 0 (< 0.1 rad): {near_zero_frac:.3f}")
    near_pi_frac = float((a > math.pi - 0.1).float().mean())
    print(f"  fraction of angles near π (> π-0.1): {near_pi_frac:.3f}")

    if n_bins_above_half_max <= 3:
        print(f"  VERDICT: DISCRETE SPECTRUM → holographic interference plausible")
    elif near_zero_frac > 0.7:
        print(f"  VERDICT: MOSTLY IDENTITY → layers are near-identity, minimal rotation")
    elif n_bins_above_half_max >= 10:
        print(f"  VERDICT: CONTINUOUS SPECTRUM → not holographic in this sense")
    else:
        print(f"  VERDICT: INTERMEDIATE — some structure but not fully discrete")


if __name__ == "__main__":
    main()
