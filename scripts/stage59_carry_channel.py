"""
Stage 59 — Characterize the two-mode rotation structure: is there a
persistent "carry channel"?

Stage 58 found that rotation angles between consecutive layers cluster
at 0 (carry) and π (flip). The next question: are the 0-mode (carry)
directions the SAME subspace across layer transitions? Or does each
transition have its own 0-mode directions?

If the SAME subspace carries forward across many transitions, we've
identified a persistent "carry channel" — a stable axis set preserved
through the stack. That would give mechanistic meaning to Finding 10's
boundary-vs-bulk: the carry channel IS the boundary.

Protocol:
  For each layer transition R_{i→i+1}, partition the rank-64 subspace
  into three parts via eigen-decomposition of R_{i→i+1}:
    - carry directions: eigenvectors with angle ≈ 0 (real eigenvalue ≈ +1)
    - flip directions: eigenvectors with angle ≈ π (real eigenvalue ≈ -1)
    - mid directions: other rotations

  Then measure ALIGNMENT between carry-subspaces of adjacent transitions:
    overlap(C_i, C_{i+1}) via principal angles / Grassmann distance

  If overlap is high across many transitions, the carry channel is
  persistent. If low, each transition has its own carry set.
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
    """Top-k principal directions (orthonormal), as columns of [d, k]."""
    mu = H.mean(dim=0, keepdim=True)
    Hc = H - mu
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1])


def eigendecompose_rotation(R):
    """R: [k, k] (close to a rotation in the PCA subspace). Returns
    (eigvals_complex, eigvecs_complex) — eigenvectors are the mode
    directions, eigenvalues encode the rotation angles."""
    R = R.to(torch.float32)
    try:
        eigvals, eigvecs = torch.linalg.eig(R)
    except Exception:
        return None, None
    return eigvals, eigvecs


def extract_subspaces(eigvals, eigvecs, carry_tol=0.3, flip_tol=0.3):
    """Given complex eigenvalues/eigenvectors of a rotation matrix,
    partition into carry / flip / mid subspaces based on rotation angle.

    carry = angle < carry_tol (eigenvalue close to +1 real)
    flip  = angle > π - flip_tol (eigenvalue close to -1 real)
    mid   = the rest
    """
    angles = torch.atan2(eigvals.imag, eigvals.real).abs()   # [0, π]
    carry_mask = angles < carry_tol
    flip_mask = angles > (math.pi - flip_tol)
    mid_mask = ~carry_mask & ~flip_mask

    # For carry/flip (real-eigenvalue directions), just take the real part
    # For complex conjugate pairs, we need to handle carefully; for the
    # carry/flip cases, eigenvalues should be near real.
    carry_vecs = eigvecs[:, carry_mask].real  # [k, n_carry]
    flip_vecs = eigvecs[:, flip_mask].real
    mid_vecs = eigvecs[:, mid_mask]
    return {
        "carry": (carry_vecs, angles[carry_mask]),
        "flip": (flip_vecs, angles[flip_mask]),
        "mid": (mid_vecs, angles[mid_mask]),
        "n_carry": int(carry_mask.sum().item()),
        "n_flip": int(flip_mask.sum().item()),
        "n_mid": int(mid_mask.sum().item()),
    }


def principal_angle_overlap(A, B):
    """Given two real matrices A [k, n_a] and B [k, n_b] (columns span
    two subspaces of R^k), compute the mean cosine of principal angles
    between them. 1 = identical subspace, 0 = orthogonal."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float("nan")
    # QR for orthonormal bases
    try:
        Qa, _ = torch.linalg.qr(A.to(torch.float32))
        Qb, _ = torch.linalg.qr(B.to(torch.float32))
    except Exception:
        return float("nan")
    M = Qa.T @ Qb                                    # [n_a, n_b]
    svals = torch.linalg.svdvals(M)                   # singular values ∈ [0, 1]
    # Mean of the singular values up to min(n_a, n_b)
    return float(svals.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--carry-tol", type=float, default=0.3)
    p.add_argument("--flip-tol", type=float, default=0.3)
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

    print(f"\n=== per-layer PCA bases (rank {args.rank}) ===")
    bases = {i: pca_basis(hiddens[i], args.rank) for i in range(L + 1)}

    # For each layer transition i → i+1:
    #   - compute R = P_{i+1}^T @ P_i
    #   - eigendecompose; partition into carry/flip/mid
    # Then compare carry subspaces across adjacent transitions.
    print(f"\n=== per-transition carry/flip/mid partition ===")
    print(f"  (carry: |angle| < {args.carry_tol};  flip: |angle| > π - {args.flip_tol})")
    print(f"  {'i→i+1':>7}  {'n_carry':>7}  {'n_flip':>6}  {'n_mid':>5}")
    transition_subspaces = []
    for i in range(L):
        R = bases[i + 1].T @ bases[i]
        eigvals, eigvecs = eigendecompose_rotation(R)
        if eigvals is None:
            transition_subspaces.append(None)
            continue
        parts = extract_subspaces(eigvals, eigvecs, args.carry_tol, args.flip_tol)
        transition_subspaces.append(parts)
        if i % max(1, L // 8) == 0 or i == L - 1:
            print(f"  {i:>3}→{i+1:<3}  {parts['n_carry']:>7}  "
                  f"{parts['n_flip']:>6}  {parts['n_mid']:>5}")

    total_carry = sum(s["n_carry"] for s in transition_subspaces if s is not None)
    total_flip = sum(s["n_flip"] for s in transition_subspaces if s is not None)
    total_mid = sum(s["n_mid"] for s in transition_subspaces if s is not None)
    total = total_carry + total_flip + total_mid
    print(f"\n  overall partition ratio:")
    print(f"    carry: {total_carry}/{total} ({total_carry/total:.3f})")
    print(f"    flip:  {total_flip}/{total} ({total_flip/total:.3f})")
    print(f"    mid:   {total_mid}/{total} ({total_mid/total:.3f})")

    # Persistent carry channel: overlap between consecutive transitions' carry subspaces
    print(f"\n=== carry-channel persistence: overlap between adjacent transitions' carry subspaces ===")
    print(f"  higher = more persistent (same directions carry forward)")
    overlaps = []
    for i in range(L - 1):
        si = transition_subspaces[i]
        si1 = transition_subspaces[i + 1]
        if si is None or si1 is None: continue
        A = si["carry"][0]          # carry directions of transition i→i+1 (in layer i's basis)
        B = si1["carry"][0]         # carry directions of transition i+1→i+2 (in layer i+1's basis)
        # These live in different bases! We need to put them in the same frame.
        # A is expressed in layer-i's rank-64 PCA basis → global: bases[i] @ A [d, n_carry_i]
        # B is expressed in layer-(i+1)'s rank-64 PCA basis → global: bases[i+1] @ B
        A_global = bases[i] @ A        # [d, n_carry_i]
        B_global = bases[i + 1] @ B    # [d, n_carry_{i+1}]
        ov = principal_angle_overlap(A_global, B_global)
        overlaps.append(ov)

    if overlaps:
        ov_t = torch.tensor(overlaps, dtype=torch.float32)
        print(f"  n measured: {len(overlaps)}")
        print(f"  mean overlap: {float(ov_t.mean()):.3f}")
        print(f"  std:          {float(ov_t.std()):.3f}")
        print(f"  range:        [{float(ov_t.min()):.3f}, {float(ov_t.max()):.3f}]")
        # Also: compare overlap to random baseline (random n-dim subspaces in R^d)
        # Expected overlap of random n-dim subspaces in R^d ≈ sqrt(n/d) for small n
        # With n ~ 10-30, d = 1024, expected ≈ sqrt(0.02) ≈ 0.14
        print(f"  baseline (random subspaces in R^1024 at same n): ≈ 0.10-0.20")
        if float(ov_t.mean()) > 0.5:
            print(f"  VERDICT: PERSISTENT CARRY CHANNEL — carry directions stable across transitions")
        elif float(ov_t.mean()) > 0.25:
            print(f"  VERDICT: PARTIAL carry persistence — some stable + some drifting")
        else:
            print(f"  VERDICT: NO persistent carry channel — carry directions drift layer-to-layer")

    # Likewise for flip subspaces
    print(f"\n=== flip-channel persistence: overlap between adjacent transitions' flip subspaces ===")
    flip_overlaps = []
    for i in range(L - 1):
        si = transition_subspaces[i]
        si1 = transition_subspaces[i + 1]
        if si is None or si1 is None: continue
        A = si["flip"][0]
        B = si1["flip"][0]
        if A.shape[1] == 0 or B.shape[1] == 0: continue
        A_global = bases[i] @ A
        B_global = bases[i + 1] @ B
        flip_overlaps.append(principal_angle_overlap(A_global, B_global))
    if flip_overlaps:
        f_t = torch.tensor(flip_overlaps, dtype=torch.float32)
        print(f"  mean flip overlap: {float(f_t.mean()):.3f}  "
              f"(std {float(f_t.std()):.3f}, range [{float(f_t.min()):.3f}, {float(f_t.max()):.3f}])")

    print(f"\n=== persistence to the END: do first-transition carries survive to the last? ===")
    if transition_subspaces[0] is not None and transition_subspaces[-1] is not None:
        A = transition_subspaces[0]["carry"][0]
        B = transition_subspaces[-1]["carry"][0]
        if A.shape[1] > 0 and B.shape[1] > 0:
            A_global = bases[0] @ A
            B_global = bases[L - 1] @ B
            ov_end = principal_angle_overlap(A_global, B_global)
            print(f"  first-carry vs last-carry overlap: {ov_end:.3f}")
            print(f"  (if high: SAME carry directions from embedding to final layer)")


if __name__ == "__main__":
    main()
