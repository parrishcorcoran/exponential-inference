"""
Stage 38 — KV cache rank-k compression test.

Claim: the KV cache stores d_kv-dim vectors per token per layer (1024 dims
for Qwen3-0.6B), but under the manifold framing the information that's
actually needed is a rank-k geometric position (k ~ 10-50), not the full
dims. Most of the stored K/V is redundant with the geometric framing.

Protocol:
  1. Collect per-layer PCA basis for k_proj and v_proj outputs from
     calibration texts.
  2. Install forward-hooks on every layer's k_proj and v_proj that apply a
     rank-k bottleneck: y -> y @ P @ P.T.
  3. This simulates storing rank-k compressed K/V in cache while
     preserving attention's expected shapes.
  4. Generate tokens and measure match vs uncompressed baseline.
  5. Sweep ranks to find the compression threshold.

Falsification:
  - If rank-k ≈ manifold dim (8-16) preserves generation, the manifold
    framing of KV cache is right; cache is ~100x compressible.
  - If full d_kv is needed even at high k (256+), the non-geometric
    information in K/V is load-bearing.
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
    """Return list of (name, module) for every k_proj and v_proj in attention."""
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in ("k_proj", "v_proj"):
            result.append((name, mod))
    return result


def collect_output_covariances(model, tokenizer, texts, modules, device, max_len=256):
    """Hook each target module's OUTPUT, accumulate y^T y across tokens."""
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


def top_k_basis(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)
    return P


def install_bottleneck_hooks(modules, bases, dtype, device):
    """Install forward hooks that apply y -> y @ P @ P.T (rank-k bottleneck)."""
    handles = []
    for name, mod in modules:
        P = bases[name].to(dtype).to(device)           # [d, k]
        PPt = (P @ P.T).contiguous()                   # [d, d] rank-k projector

        def make_hook(projector):
            def hook(mod, inputs, output):
                y = output
                # y: [..., d]. Apply projector.
                return y @ projector
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
    p.add_argument("--ranks", default="4,8,16,32,64,128,256,512")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage38_kv_compression.json")
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
    print(f"  found {len(modules)} kv-projections, d_kv={d_kv}")

    print(f"\n=== baseline (no compression) ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  generated in {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    print(f"\n=== calibrating K/V output covariances ===")
    t0 = time.perf_counter()
    covs = collect_output_covariances(model, tokenizer, CALIB_TEXTS, modules, device)
    print(f"  {len(covs)} covs in {time.perf_counter()-t0:.1f}s")

    ranks = [int(x) for x in args.ranks.split(",")]
    results = []
    for k in ranks:
        print(f"\n=== rank {k} KV compression ===")
        t0 = time.perf_counter()
        bases = {n: top_k_basis(covs[n], k) for n, _ in modules}
        dtype = modules[0][1].weight.dtype
        handles = install_bottleneck_hooks(modules, bases, dtype, device)
        try:
            tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        finally:
            remove_hooks(handles)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        n = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:n], tokens[:n]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens)) if a != b), n)
        compression = d_kv / k
        print(f"  {time.perf_counter()-t0:.1f}s  compression {compression:.1f}x  "
              f"match {match}/{n}  first_div @ {first_div}")
        print(f"  {text[:150]}")
        results.append({
            "rank": k,
            "compression_ratio": compression,
            "match": match, "total": n,
            "match_ratio": match / max(n, 1),
            "first_divergence": first_div,
            "sample": text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  d_kv={d_kv}")
    print(f"  baseline: {base_text[:80]}")
    print(f"  {'rank':>5}  {'compress':>8}  {'match':>10}  {'first div':>9}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['compression_ratio']:>7.1f}x  "
              f"{r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "d_kv": d_kv,
            "baseline_sample": base_text[:400],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
