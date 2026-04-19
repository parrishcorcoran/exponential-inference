"""
Stage 38c — KV cache rank from measured per-layer manifold dim.

Uses TwoNN (Facco et al. 2017) to measure the per-layer intrinsic manifold
dim from hidden-state samples collected during calibration. The measured
d_layer is then used as the per-layer KV cache rank (optionally scaled by a
safety factor).

No variance thresholds. The rank schedule comes from the physics.

Protocol:
  1. Forward on calibration texts, collect per-layer hidden states AND
     per-layer K/V output covariances.
  2. Compute TwoNN intrinsic dim per layer from hidden states.
  3. For each safety factor s in {1, 2, 4, 8}: set per-layer KV rank =
     ceil(d_layer * s), install bottleneck hooks, generate, measure.
"""

import argparse
import json
import math
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


def twonn_dim(X):
    """Facco et al. 2017 MLE estimator of intrinsic dim.
    X: [N, D] float32 tensor of sample points (on CPU). Returns float d."""
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
    # MLE: d = N / sum(log mu)
    return float(mask.sum().item() / log_mu.sum().item())


def find_kv_projs(model):
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in ("k_proj", "v_proj"):
            result.append((name, mod))
    return result


def find_decoder_layers(model):
    return list(model.model.layers)


def calibrate(model, tokenizer, texts, device, max_len=256, max_samples_per_layer=500):
    """Single pass: collect per-layer hidden-state samples + per-module K/V covs."""
    kv_modules = find_kv_projs(model)
    layers = find_decoder_layers(model)
    L = len(layers)

    kv_covs = {name: None for name, _ in kv_modules}
    hidden_samples = [[] for _ in range(L)]

    def make_kv_hook(n):
        def hook(mod, inputs, output):
            y = output.detach()
            y_flat = y.reshape(-1, y.shape[-1]).to(torch.float32).cpu()
            if kv_covs[n] is None:
                kv_covs[n] = torch.zeros(y_flat.shape[1], y_flat.shape[1], dtype=torch.float32)
            kv_covs[n] += y_flat.T @ y_flat
        return hook

    def make_layer_hook(i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            hidden_samples[i].append(h_flat)
        return hook

    handles = []
    for name, mod in kv_modules:
        handles.append(mod.register_forward_hook(make_kv_hook(name)))
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_layer_hook(i)))

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

    # Concatenate and subsample hidden per layer
    hidden_per_layer = []
    for i in range(L):
        cat = torch.cat(hidden_samples[i], dim=0)
        if cat.shape[0] > max_samples_per_layer:
            idx = torch.randperm(cat.shape[0])[:max_samples_per_layer]
            cat = cat[idx]
        hidden_per_layer.append(cat)

    kv_covs = {n: c.to(torch.float64) for n, c in kv_covs.items()}
    return kv_modules, layers, hidden_per_layer, kv_covs


def top_k_basis_from_cov(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)


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
    p.add_argument("--factors", default="1,2,4,8")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage38c_manifold_kv.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)

    print(f"\n=== baseline ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    print(f"\n=== calibrating (hidden + K/V) ===")
    t0 = time.perf_counter()
    kv_modules, layers, hidden_per_layer, kv_covs = calibrate(
        model, tokenizer, CALIB_TEXTS, device)
    L = len(layers)
    d_kv = kv_modules[0][1].out_features
    print(f"  L={L} layers, {len(kv_modules)} kv-proj, d_kv={d_kv}, "
          f"calibration {time.perf_counter()-t0:.1f}s")

    print(f"\n=== per-layer TwoNN intrinsic dim ===")
    layer_dims = []
    for i, H in enumerate(hidden_per_layer):
        d = twonn_dim(H)
        layer_dims.append(d)
    print(f"  {'layer':>5}  {'n_samples':>9}  {'TwoNN_dim':>10}")
    for i in range(L):
        print(f"  {i:>5}  {hidden_per_layer[i].shape[0]:>9}  {layer_dims[i]:>10.2f}")
    mean_dim = sum(layer_dims) / L
    print(f"  mean TwoNN dim: {mean_dim:.2f}")

    results = []
    dtype = kv_modules[0][1].weight.dtype
    factors = [float(x) for x in args.factors.split(",")]

    # Map layer index -> rank, then to k_proj/v_proj name lookups.
    # k_proj/v_proj naming: model.layers.<i>.self_attn.k_proj / .v_proj
    for factor in factors:
        per_layer_rank = [max(1, int(math.ceil(d * factor))) for d in layer_dims]
        mean_rank = sum(per_layer_rank) / L
        print(f"\n-- factor={factor}  mean_rank={mean_rank:.1f}  "
              f"(compression {d_kv/mean_rank:.1f}x) --")

        bases = {}
        for name, _ in kv_modules:
            # parse layer index from name like "model.layers.7.self_attn.k_proj"
            toks = name.split(".")
            idx = int(toks[toks.index("layers") + 1])
            k = per_layer_rank[idx]
            bases[name] = top_k_basis_from_cov(kv_covs[name], k)

        t0 = time.perf_counter()
        handles = install_bottleneck_hooks(kv_modules, bases, dtype, device)
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
            "factor": factor,
            "mean_rank": mean_rank,
            "compression_ratio": d_kv / mean_rank,
            "per_layer_rank": per_layer_rank,
            "match": match, "total": n,
            "match_ratio": match / max(n, 1),
            "first_divergence": first_div,
            "sample": text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  d_kv={d_kv}   mean_TwoNN_dim={mean_dim:.2f}")
    print(f"  {'factor':>6}  {'mean_k':>6}  {'compress':>8}  {'match':>10}  {'first_div':>9}")
    for r in results:
        print(f"  {r['factor']:>6.1f}  {r['mean_rank']:>6.1f}  "
              f"{r['compression_ratio']:>7.1f}x  "
              f"{r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "d_kv": d_kv,
            "mean_twonn_dim": mean_dim,
            "per_layer_twonn_dim": layer_dims,
            "baseline_sample": base_text[:400],
            "factors": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
