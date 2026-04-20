"""
Stage 48 — Frame 5: Parallel-transport isometry test.

Parallel transport preserves metric: inner products between tokens should
be invariant under the layer-to-layer rotation. This translates to:

  - Hidden state norms should be approximately constant across layers.
  - Pairwise token distances should be approximately constant across layers.

If either is strongly violated, parallel transport is NOT the right frame.

Frame 3 (RG flow) predicts contraction: norms + distances shrink.
Frame 5 (parallel transport) predicts isometry: norms + distances preserved.

Protocol: forward on long text, measure per-layer norm statistics and
per-layer pairwise-distance variance. Compare to find which frame wins.
"""

import argparse
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage48_parallel_transport_test.json")
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

    print(f"\n=== per-layer norm and distance stats ===")
    # Sample token pairs once; track the SAME pairs across layers for distance
    torch.manual_seed(0)
    ia = torch.randint(0, T, (args.n_pairs,))
    ib = torch.randint(0, T, (args.n_pairs,))
    mask = ia != ib
    ia, ib = ia[mask], ib[mask]

    per_layer = []
    print(f"  {'layer':>5}  {'mean |h|':>10}  {'std |h|':>9}  {'mean d(a,b)':>12}  {'std d':>7}")
    for i in range(L + 1):
        H = hidden[i][0].to(torch.float32).cpu()
        norms = H.norm(dim=-1)
        diff = (H[ia] - H[ib])
        dists = diff.norm(dim=-1)
        stats = {
            "layer": i,
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "mean_dist": float(dists.mean()),
            "std_dist": float(dists.std()),
        }
        per_layer.append(stats)
        marker = ""
        if i == 0: marker = "  (embed)"
        elif i == L: marker = "  (final)"
        print(f"  {i:>5}  {stats['mean_norm']:>10.3f}  {stats['std_norm']:>9.3f}  "
              f"{stats['mean_dist']:>12.3f}  {stats['std_dist']:>7.3f}{marker}")

    # Parallel-transport: mean_norm and mean_dist should be ~constant across layers.
    # RG flow: mean_dist should shrink monotonically.
    norms = [s["mean_norm"] for s in per_layer]
    dists = [s["mean_dist"] for s in per_layer]

    norm_ratio_max_min = max(norms) / min(norms)
    dist_ratio_max_min = max(dists) / min(dists)

    print(f"\n=== summary ===")
    print(f"  norm  max/min: {norm_ratio_max_min:.2f}x   (parallel-transport predicts ≈1)")
    print(f"  dist  max/min: {dist_ratio_max_min:.2f}x   (parallel-transport predicts ≈1)")

    # Check dist monotonic decrease (RG signature)
    dist_drops = sum(1 for i in range(1, len(dists)) if dists[i] < dists[i-1])
    print(f"  layers where pairwise dist shrunk vs prev: {dist_drops}/{len(dists)-1}")

    if dist_ratio_max_min < 1.3 and norm_ratio_max_min < 1.3:
        print(f"  VERDICT: near-isometric → parallel transport supported")
    elif dist_drops > 0.7 * (len(dists) - 1):
        print(f"  VERDICT: strong contraction → Frame 5 FALSIFIED, Frame 3 (RG) reconfirmed")
    else:
        print(f"  VERDICT: mixed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "L": L, "n_tokens": T,
            "n_pairs": int(mask.sum().item()),
            "per_layer": per_layer,
            "norm_max_min_ratio": norm_ratio_max_min,
            "dist_max_min_ratio": dist_ratio_max_min,
            "dist_monotonic_drops": dist_drops,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
