"""
Stage 47 — Frame 4: Parisi P(q) overlap-distribution test for RSB.

In spin-glass theory, the distribution of overlaps P(q) between replicas
reveals the phase:
  - Replica-symmetric (RS, high-T paramagnet):  narrow single peak
  - 1-step RSB:                                 bimodal, two peaks
  - Full continuous RSB (SK low-T glass):       smooth support on interval

Treating each token's hidden state as a "replica" (same weights, different
input), we compute pairwise overlaps q = <h_a · h_b> / (|h_a||h_b|) at
each layer and build P(q).

Protocol (Qwen3-0.6B):
  1. Forward on a long passage, capture per-layer hidden state at every
     token position.
  2. At each layer, sample many pairs of token positions, compute cosine
     overlap q.
  3. Build histogram per layer. Report modality and spread.

Predictions:
  - If RS: single narrow peak at each layer
  - If RSB: multimodal distribution or wide support
  - If RSB and rotation-layer correspondence holds: P(q) shape should
    change with layer depth (different RSB levels at different depths)
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
    "The cell is the basic structural unit of life, composed of cytoplasm "
    "enclosed within a membrane. Quantum mechanics describes the behavior of "
    "matter and energy at atomic and subatomic scales. The history of computing "
    "began with mechanical calculators and evolved through vacuum tubes. "
    "Photosynthesis uses sunlight to convert carbon dioxide and water into "
    "glucose and oxygen. Neural networks consist of parameterized layers "
    "trained by gradient descent to approximate functions. Plate tectonics "
    "describes the slow movement of Earth's lithospheric plates over the "
    "mantle. Proteins fold into complex three-dimensional structures "
    "determined by their amino acid sequences. The standard model of particle "
    "physics unifies electromagnetic, weak, and strong interactions. Evolution "
    "by natural selection operates on heritable variation in populations. "
    "Cryptography protects information using mathematical operations that are "
    "easy to compute. Thermodynamics relates heat, work, energy, and entropy "
    "in macroscopic systems. Graph theory studies vertices connected by edges "
    "across many practical applications. Black holes are regions of spacetime "
    "from which nothing, not even light, can escape. DNA encodes genetic "
    "information in a double-helix structure. Volcanoes form at tectonic "
    "plate boundaries. Linear algebra provides the mathematical foundation "
    "for many machine learning algorithms. Game theory analyzes strategic "
    "interactions. Bayesian inference updates a prior probability "
    "distribution using observed data."
)


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def overlap_histogram(H, n_pairs=2000, n_bins=40):
    """Compute cosine overlaps between random pairs of rows. Returns (bins, counts, stats)."""
    N = H.shape[0]
    if N < 2:
        return None, None, None
    H_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
    ia = torch.randint(0, N, (n_pairs,))
    ib = torch.randint(0, N, (n_pairs,))
    # Avoid self-pairs (q=1 trivially)
    mask = ia != ib
    ia, ib = ia[mask], ib[mask]
    q = (H_norm[ia] * H_norm[ib]).sum(dim=-1).to(torch.float32)
    bins = torch.linspace(-1.0, 1.0, n_bins + 1)
    counts = torch.histogram(q, bins=bins).hist
    stats = {
        "mean": float(q.mean()),
        "std": float(q.std()),
        "min": float(q.min()),
        "max": float(q.max()),
        "n_pairs": int(mask.sum().item()),
    }
    return bins.tolist(), counts.tolist(), stats


def count_peaks(counts, min_prominence=0.1):
    """Count peaks in the histogram with minimum prominence (fraction of max)."""
    counts = torch.tensor(counts, dtype=torch.float32)
    if counts.sum() == 0:
        return 0
    counts = counts / counts.max()
    peaks = 0
    n = len(counts)
    for i in range(1, n - 1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1] and counts[i] > min_prominence:
            peaks += 1
    # endpoints
    if len(counts) >= 2 and counts[0] > counts[1] and counts[0] > min_prominence:
        peaks += 1
    if len(counts) >= 2 and counts[-1] > counts[-2] and counts[-1] > min_prominence:
        peaks += 1
    return peaks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n-pairs", type=int, default=3000)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage47_parisi_pq_test.json")
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
    hidden = out.hidden_states  # tuple len L+1, each [1, T, hidden]

    print(f"\n=== per-layer P(q) overlap distributions ===")
    print(f"  sampling {args.n_pairs} random token pairs per layer")
    per_layer_stats = []
    per_layer_hist = []

    for i in range(L + 1):
        H = hidden[i][0].to(torch.float32).cpu()  # [T, hidden]
        bins, counts, stats = overlap_histogram(H, n_pairs=args.n_pairs)
        peaks = count_peaks(counts, min_prominence=0.15)
        stats["peaks"] = peaks
        per_layer_stats.append(stats)
        per_layer_hist.append({"bins": bins, "counts": counts})

    print(f"  {'layer':>5}  {'mean q':>8}  {'std':>6}  {'peaks':>5}  {'label':>10}")
    for i in range(L + 1):
        s = per_layer_stats[i]
        if s["peaks"] == 1:
            label = "RS"
        elif s["peaks"] == 2:
            label = "1-step RSB"
        elif s["peaks"] >= 3:
            label = "multi-RSB"
        else:
            label = "flat/wide"
        marker = ""
        if i == 0:
            marker = " (embed)"
        elif i == L:
            marker = " (final)"
        print(f"  {i:>5}  {s['mean']:>8.3f}  {s['std']:>6.3f}  {s['peaks']:>5}  {label:>10}{marker}")

    # Summary: RS vs RSB counts across layers
    rs_count = sum(1 for s in per_layer_stats if s["peaks"] == 1)
    rsb_count = sum(1 for s in per_layer_stats if s["peaks"] >= 2)

    print(f"\n=== summary ===")
    print(f"  RS (single peak): {rs_count} / {L+1} layers")
    print(f"  RSB (multi peak): {rsb_count} / {L+1} layers")

    if rsb_count > rs_count:
        print(f"  VERDICT: RSB behavior dominant → Frame 4 (Parisi) supported")
    elif rs_count > 2 * rsb_count:
        print(f"  VERDICT: RS dominant → Frame 4 weakly supported at best")
    else:
        print(f"  VERDICT: Mixed → inconclusive")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "n_tokens": T,
            "n_pairs": args.n_pairs,
            "per_layer_stats": per_layer_stats,
            "per_layer_hist": per_layer_hist,
            "rs_count": rs_count,
            "rsb_count": rsb_count,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
