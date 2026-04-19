"""
Stage 38b — Per-layer KV rank schedule.

Stage 38 showed uniform rank 128 (8x compression) preserves coherence, below
that output collapses. Under the rotation framing (Findings 02/03), rotation
rate varies with depth — so different layers may tolerate different KV ranks.
A per-layer schedule aligned with each layer's effective rank should reach
a lower mean rank than the uniform floor.

Protocol:
  1. Collect per-layer K/V output covariances (same as stage 38).
  2. Compute effective rank per (layer, type) at variance thresholds
     {90%, 95%, 99%, 99.9%}.
  3. For each threshold, install per-layer rank bottleneck and generate.
  4. Report mean rank per schedule and match/sample quality.

If a threshold yields mean rank << 128 with preserved coherence, the 8x
floor was an artifact of uniform compression, not a real physics limit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

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


def find_kv_projs(model):
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in ("k_proj", "v_proj"):
            result.append((name, mod))
    return result


def collect_output_covariances(model, tokenizer, texts, modules, device, max_len=256):
    covs = {name: None for name, _ in modules}

    def make_hook(n):
        def hook(mod, inputs, output):
            y = output.detach()
            y_flat = y.reshape(-1, y.shape[-1]).to(torch.float32).cpu()
            if covs[n] is None:
                covs[n] = torch.zeros(y_flat.shape[1], y_flat.shape[1], dtype=torch.float32)
            covs[n] += y_flat.T @ y_flat
        return hook

    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in modules]
    try:
        model.eval()
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {n: c.to(torch.float64) for n, c in covs.items()}


def eigen_decomp(cov):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(dims=[0])
    eigvecs = eigvecs.flip(dims=[1])
    eigvals = torch.clamp(eigvals, min=0.0)
    return eigvals, eigvecs


def rank_for_variance(eigvals, threshold):
    """Smallest k such that sum of top-k eigenvalues >= threshold * total."""
    total = eigvals.sum()
    if total <= 0:
        return 1
    cumsum = torch.cumsum(eigvals, dim=0)
    ratio = cumsum / total
    k = int((ratio < threshold).sum().item()) + 1
    return max(1, k)


def top_k_basis_from_eigvecs(eigvecs, k):
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, :k].to(torch.float32)


def install_bottleneck_hooks(modules, bases, dtype, device):
    handles = []
    for name, mod in modules:
        P = bases[name].to(dtype).to(device)
        PPt = (P @ P.T).contiguous()

        def make_hook(projector):
            def hook(mod, inputs, output):
                return output @ projector
            return hook

        handles.append(mod.register_forward_hook(make_hook(PPt)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--thresholds", default="0.90,0.95,0.99,0.999,0.9999")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage38b_per_layer_kv.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    modules = find_kv_projs(model)
    d_kv = modules[0][1].out_features
    print(f"  {len(modules)} kv-projections, d_kv={d_kv}")

    print(f"\n=== baseline ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    print(f"\n=== calibrating ===")
    t0 = time.perf_counter()
    covs = collect_output_covariances(model, tokenizer, CALIB_TEXTS, modules, device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"\n=== per-layer effective rank spectrum ===")
    eigs = {}
    for name, _ in modules:
        vals, vecs = eigen_decomp(covs[name])
        eigs[name] = (vals, vecs)

    # Print a summary: for each threshold, min/median/max effective rank across (layer, K/V).
    thresholds = [float(x) for x in args.thresholds.split(",")]
    print(f"  {'threshold':>10}  {'mean_rank':>10}  {'min':>5}  {'median':>6}  {'max':>5}")
    thresh_ranks = {}
    for th in thresholds:
        ranks = [rank_for_variance(eigs[name][0], th) for name, _ in modules]
        thresh_ranks[th] = ranks
        r = torch.tensor(ranks, dtype=torch.float32)
        print(f"  {th:>10.4f}  {float(r.mean()):>10.1f}  "
              f"{int(r.min()):>5}  {int(r.median()):>6}  {int(r.max()):>5}")

    # Also show K vs V split at threshold 0.95
    print(f"\n  K vs V per-layer rank at threshold 0.95 (first 14 of 28 layers):")
    k_ranks = []; v_ranks = []
    for name, _ in modules:
        vals, _ = eigs[name]
        r = rank_for_variance(vals, 0.95)
        if name.endswith(".k_proj"): k_ranks.append(r)
        else: v_ranks.append(r)
    for i in range(min(14, len(k_ranks))):
        print(f"    layer {i:2d}  K={k_ranks[i]:>4}  V={v_ranks[i]:>4}")

    print(f"\n=== generation with per-layer schedules ===")
    results = []
    dtype = modules[0][1].weight.dtype
    for th in thresholds:
        ranks_list = thresh_ranks[th]
        mean_rank = sum(ranks_list) / len(ranks_list)
        print(f"\n-- threshold {th}  mean_rank={mean_rank:.1f}  "
              f"(compression {d_kv/mean_rank:.1f}x) --")
        t0 = time.perf_counter()
        bases = {}
        for (name, _), k in zip(modules, ranks_list):
            vals, vecs = eigs[name]
            bases[name] = top_k_basis_from_eigvecs(vecs, k)
        handles = install_bottleneck_hooks(modules, bases, dtype, device)
        try:
            tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        finally:
            remove_hooks(handles)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        n = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:n], tokens[:n]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens)) if a != b), n)
        print(f"  {time.perf_counter()-t0:.1f}s  match {match}/{n}  first_div @ {first_div}")
        print(f"  {text[:160]}")
        results.append({
            "threshold": th,
            "mean_rank": mean_rank,
            "compression_ratio": d_kv / mean_rank,
            "min_rank": min(ranks_list),
            "max_rank": max(ranks_list),
            "match": match, "total": n,
            "match_ratio": match / max(n, 1),
            "first_divergence": first_div,
            "sample": text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  d_kv={d_kv}")
    print(f"  {'thresh':>8}  {'mean_k':>7}  {'compress':>8}  {'match':>10}  {'first_div':>9}")
    for r in results:
        print(f"  {r['threshold']:>8.4f}  {r['mean_rank']:>7.1f}  "
              f"{r['compression_ratio']:>7.1f}x  "
              f"{r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "d_kv": d_kv,
            "baseline_sample": base_text[:400],
            "schedules": results,
            "per_layer_ranks": {str(t): thresh_ranks[t] for t in thresholds},
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
