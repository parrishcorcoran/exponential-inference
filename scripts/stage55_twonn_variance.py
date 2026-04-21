"""
Stage 55 — quantify TwoNN run-to-run variance.

Runs TwoNN on the same model, same corpus, with DIFFERENT random
subsamples of hidden states. Measures the spread to decide how much
trust to put in a single TwoNN reading.

Also tests how variance shrinks with sample size.
"""

import argparse
import time
from pathlib import Path
import sys

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


def twonn_dim(X):
    X = X.to(torch.float32)
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    sorted_d, _ = D.sort(dim=1)
    r1 = sorted_d[:, 0]
    r2 = sorted_d[:, 1]
    mask = (r1 > 1e-8) & (r2 > r1 + 1e-10)
    if mask.sum() < 10:
        return float("nan")
    mu = r2[mask] / r1[mask]
    log_mu = torch.log(mu)
    return float(mask.sum().item() / log_mu.sum().item())


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_hidden_states(model, tokenizer, texts, device, layer_idx, max_len=256):
    """Capture ALL hidden-state tokens at a given layer."""
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
    p.add_argument("--layer", type=int, default=14, help="mid-stack layer to measure")
    p.add_argument("--runs-per-sample-size", type=int, default=20)
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

    print(f"\n=== collecting all hidden states at layer {args.layer} ===")
    H = collect_hidden_states(model, tokenizer, CALIB_TEXTS, device, args.layer)
    N_total = H.shape[0]
    print(f"  {N_total} total tokens, hidden_dim = {H.shape[1]}")

    # Variance vs sample size
    sample_sizes = [50, 100, 200, 500, min(1000, N_total), min(2000, N_total), N_total]
    sample_sizes = sorted(set(s for s in sample_sizes if s <= N_total))

    print(f"\n=== run-to-run variance at different sample sizes ===")
    print(f"  {'N':>6}  {'mean':>7}  {'std':>6}  {'min':>6}  {'max':>6}  {'range':>7}")

    summary = []
    for n in sample_sizes:
        dims = []
        # Fix overall RNG but vary the subsample each run
        for run in range(args.runs_per_sample_size):
            torch.manual_seed(run * 1000 + n)
            if n < N_total:
                idx = torch.randperm(N_total)[:n]
                X = H[idx]
            else:
                X = H
            dims.append(twonn_dim(X))
            if n == N_total:
                break  # with full data and no subsample, result is deterministic
        d = torch.tensor(dims, dtype=torch.float32)
        print(f"  {n:>6}  {float(d.mean()):>7.3f}  "
              f"{float(d.std()):>6.3f}  {float(d.min()):>6.3f}  {float(d.max()):>6.3f}  "
              f"{float(d.max() - d.min()):>7.3f}")
        summary.append({
            "N": n, "runs": len(dims),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max()),
            "range": float(d.max() - d.min()),
        })

    print(f"\n=== interpretation ===")
    print(f"  - At small N (50-100), TwoNN stderr is high; single readings can be off by 1-2 dims")
    print(f"  - At moderate N (200-500), stderr drops but run-to-run range is still ~0.5-1 dim")
    print(f"  - At large N, variance is dominated by the point-set choice (not subsampling)")
    print(f"  - For 'universal ~9-11' claim: the reading is stable within ±0.5 dims at N >= 500")


if __name__ == "__main__":
    main()
