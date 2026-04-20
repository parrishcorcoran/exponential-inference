"""
Stage 49 — Frame 6: Quantum measurement / density matrix evolution.

Treats the hidden state as a quantum density matrix. Each token's
normalized hidden state is a pure state |ψ_i>. The layer-wise average
density matrix is

  ρ_layer = (1/N) Σ_i |ψ_i><ψ_i|

Two quantum quantities per layer:
  - Purity γ = tr(ρ²). Range [1/d, 1]. 1 = pure state, 1/d = max mixed.
  - Von Neumann entropy S(ρ) = -Σ_k λ_k log λ_k.

Predictions:
  - Measurement / pointer-basis selection (Frame 6a): purity INCREASES
    toward final layer (state contracts to the measurement eigenbasis).
  - Standard decoherence (Frame 6b): purity DECREASES (environment
    entangles with state, mixing it).

If purity rises and entropy falls through the stack, the transformer is
performing a gradual measurement. This is a distinct claim from RG flow
(Frame 3), which is formally a classical statistical statement. Quantum
measurement would be a fundamentally different picture.

Implementation note:
  - The eigenvalues of ρ = (1/N) X^T X (when rows of X are unit-normalized
    hidden states) are (1/N) σ_i², where σ_i are singular values of X.
    So SVD of X gives us everything.
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
    "The cell is the basic structural unit of life. Quantum mechanics "
    "describes the behavior of matter at atomic scales. Photosynthesis uses "
    "sunlight to convert carbon dioxide and water into glucose. Neural "
    "networks consist of parameterized layers trained by gradient descent. "
    "Plate tectonics describes the slow movement of Earth's lithospheric "
    "plates. Proteins fold into complex three-dimensional structures. The "
    "standard model unifies electromagnetic, weak, and strong interactions. "
    "Evolution by natural selection operates on heritable variation. "
    "Cryptography protects information using mathematical operations. "
    "Thermodynamics relates heat, work, energy, and entropy. Graph theory "
    "studies vertices connected by edges. Black holes are regions of "
    "spacetime from which nothing can escape. DNA encodes genetic information. "
    "Volcanoes form at tectonic plate boundaries. Linear algebra provides the "
    "mathematical foundation for machine learning. Game theory analyzes "
    "strategic interactions. Bayesian inference updates a prior probability. "
    "The immune system recognizes pathogens. The Riemann zeta function "
    "encodes deep information about primes."
)


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def density_matrix_stats(X):
    """X: [N, d] of token hidden states. Normalize each row to unit vector,
    then compute purity and von Neumann entropy of the average density matrix.

    Equivalent eigenvalues = (1/N) σ_i² where σ_i are SVD singular values
    of the normalized row matrix."""
    X = X.to(torch.float32)
    # Unit-normalize each row (each token is a pure state)
    norms = X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    X_unit = X / norms
    N, d = X_unit.shape
    # SVD
    try:
        _, S, _ = torch.linalg.svd(X_unit, full_matrices=False)
    except Exception:
        return None

    eigvals = (S ** 2) / N        # eigenvalues of ρ
    eigvals = eigvals.clamp_min(0.0)
    tot = eigvals.sum()
    if tot <= 0:
        return None
    eigvals = eigvals / tot       # ensure trace = 1
    purity = float((eigvals ** 2).sum())
    # Von Neumann entropy in nats
    ev_pos = eigvals[eigvals > 1e-12]
    vn = float(-(ev_pos * ev_pos.log()).sum())
    # In bits
    vn_bits = vn / math.log(2)
    # Effective rank (participation ratio from eigenvalues)
    eff_rank = float(1.0 / (eigvals ** 2).sum()) if (eigvals ** 2).sum() > 0 else 0.0
    # Max mixed entropy for this dim
    max_vn = math.log(min(N, d))
    return {
        "purity": purity,
        "vn_entropy_nats": vn,
        "vn_entropy_bits": vn_bits,
        "vn_entropy_normalized": vn / max_vn if max_vn > 0 else 0.0,
        "effective_rank": eff_rank,
        "max_vn_nats": max_vn,
        "n_tokens": N,
        "dim": d,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage49_quantum_measurement_test.json")
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
    print(f"  L={L}")

    ids = tokenizer(LONG_TEXT, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]
    print(f"\n=== tokenized {T} tokens ===")

    with torch.inference_mode():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states

    print(f"\n=== per-layer density matrix stats ===")
    per_layer = []
    print(f"  {'layer':>5}  {'purity':>7}  {'VN(nats)':>9}  {'VN_norm':>8}  {'eff_rank':>9}  {'note':>14}")
    prev_purity = None
    monotonic_up = 0
    monotonic_down = 0
    for i in range(L + 1):
        X = hidden[i][0].to(torch.float32).cpu()
        s = density_matrix_stats(X)
        if s is None: continue
        s["layer"] = i
        per_layer.append(s)
        note = ""
        if i == 0: note = "(embed)"
        elif i == L: note = "(final)"
        elif prev_purity is not None:
            if s["purity"] > prev_purity: monotonic_up += 1
            elif s["purity"] < prev_purity: monotonic_down += 1
        print(f"  {i:>5}  {s['purity']:>7.4f}  "
              f"{s['vn_entropy_nats']:>9.3f}  "
              f"{s['vn_entropy_normalized']:>8.3f}  "
              f"{s['effective_rank']:>9.1f}  {note:>14}")
        prev_purity = s["purity"]

    # Overall trend
    first_purity = per_layer[0]["purity"]
    last_purity = per_layer[-1]["purity"]
    first_vn = per_layer[0]["vn_entropy_nats"]
    last_vn = per_layer[-1]["vn_entropy_nats"]

    print(f"\n=== trajectory ===")
    print(f"  purity  layer-0 = {first_purity:.4f} → layer-{L} = {last_purity:.4f}  "
          f"(Δ = {last_purity - first_purity:+.4f})")
    print(f"  VN(nats) layer-0 = {first_vn:.3f} → layer-{L} = {last_vn:.3f}  "
          f"(Δ = {last_vn - first_vn:+.3f})")
    print(f"  transitions: {monotonic_up} purity increases, {monotonic_down} decreases")

    # Verdict
    print(f"\n=== verdict ===")
    if last_purity > first_purity and last_vn < first_vn:
        if monotonic_up > 2 * monotonic_down:
            print(f"  Frame 6a (measurement / pointer selection): SUPPORTED")
            print(f"    state purifies toward the final layer (state-collapse analog)")
        else:
            print(f"  Frame 6a: partially supported — overall purifies but non-monotonic")
    elif last_purity < first_purity and last_vn > first_vn:
        print(f"  Frame 6b (standard decoherence): SUPPORTED")
        print(f"    state mixes through the stack")
    else:
        print(f"  Frame 6: mixed signal, neither measurement nor decoherence cleanly fit")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "per_layer": per_layer,
            "monotonic_up": monotonic_up,
            "monotonic_down": monotonic_down,
            "first_purity": first_purity,
            "last_purity": last_purity,
            "first_vn": first_vn,
            "last_vn": last_vn,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
