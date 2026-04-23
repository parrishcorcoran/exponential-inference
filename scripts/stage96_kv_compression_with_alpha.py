"""
Stage 96 — Rerun stage 38 KV rank-k compression WITH per-head α compensation.

Stage 38 (2026-02-xx) tested rank-k bottleneck on k_proj/v_proj outputs and
observed that rank 128 (8× compression) diverged at token 1 and matched
only 2/80 tokens vs baseline — called "not lossless."

Today's finding: the failure may be structural preservation + magnitude
loss, uncompensated. The rank-128 PROJECTION preserved the direction of
each K/V vector in the top-128 subspace, but the RMS of the projected
signal is smaller than the original — attention softmax then concentrates
differently and argmax diverges.

Compensation: per-head, per-layer scalar α chosen to restore the RMS of
the projected (k or v) output to match the baseline. No training; a single
calibration pass derives α.

Protocol:
  1. Teacher pass on calibration texts — record per-head, per-position
     ||k|| and ||v|| before any modification.
  2. Teacher pass with rank-k projection hook — record per-head, per-position
     ||k_proj|| and ||v_proj||.
  3. α = ratio of RMS(before) / RMS(after) per head per layer.
  4. Install hook that applies rank-k projection AND scales by α.
  5. Generate tokens on stage-38's prompt; compare to baseline (same
     comparison as stage 38).

Sweep rank k ∈ {4, 8, 16, 32, 64, 128, 256, 512}.
Report match ratio and first-divergence per rank.

If rank 128 jumps from 2/80 (2.5%) to something much higher with α, the
conservation thesis is validated on a real measurement.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def collect_rms_per_head(model, tokenizer, texts, modules, device, head_dim, n_kv_heads, max_len=256):
    """Collect RMS per head for each module's output."""
    rms_by_name = {name: None for name, _ in modules}
    counts = {name: 0 for name, _ in modules}
    def make_hook(n):
        def hook(mod, inputs, output):
            y = output.detach().float().cpu()          # [B, T, n_kv_heads * head_dim]
            B, T, D = y.shape
            y_heads = y.view(B, T, n_kv_heads, head_dim)    # [B, T, n_kv, head_dim]
            sq = y_heads.pow(2).sum(dim=-1).sqrt()          # [B, T, n_kv]  ||.|| per head per pos
            # accumulate mean ||.|| per head across pos×batch
            if rms_by_name[n] is None:
                rms_by_name[n] = torch.zeros(n_kv_heads, dtype=torch.float64)
            rms_by_name[n] += sq.double().reshape(-1, n_kv_heads).mean(dim=0) * (B * T)
            counts[n] += B * T
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
    # Mean RMS per head
    return {n: (rms_by_name[n] / max(counts[n], 1)).to(torch.float32) for n in rms_by_name}


def install_bottleneck_alpha_hooks(modules, bases, alphas, dtype, device, n_kv_heads, head_dim):
    """Install forward hooks that apply:
         y -> (y @ P @ P.T) * alpha_per_head
       where alpha_per_head is broadcast across head_dim for each head."""
    handles = []
    for name, mod in modules:
        P = bases[name].to(dtype).to(device)               # [d, k]
        PPt = (P @ P.T).contiguous()                       # [d, d]
        alpha = alphas[name].to(dtype).to(device)          # [n_kv_heads]
        # Build broadcast-ready scale: [1, 1, n_kv_heads*head_dim]
        alpha_full = alpha.unsqueeze(-1).expand(n_kv_heads, head_dim).reshape(-1)

        def make_hook(projector, scale):
            def hook(mod, inputs, output):
                y = output @ projector                     # rank-k bottleneck
                return y * scale                           # per-head scalar compensation
            return hook

        handles.append(mod.register_forward_hook(make_hook(PPt, alpha_full)))
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
    p.add_argument("--ranks", default="8,16,32,64,128,256,512")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage96_kv_alpha.json")
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
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else d_kv // n_kv_heads
    print(f"  found {len(modules)} kv-projections, d_kv={d_kv}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    print(f"\n=== baseline (no compression) ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  generated in {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    print(f"\n=== calibrating K/V covariances + per-head RMS (teacher) ===")
    t0 = time.perf_counter()
    covs = collect_output_covariances(model, tokenizer, CALIB_TEXTS, modules, device)
    teacher_rms = collect_rms_per_head(model, tokenizer, CALIB_TEXTS, modules, device,
                                       head_dim, n_kv_heads)
    print(f"  {len(covs)} covs, {len(teacher_rms)} rms vectors in {time.perf_counter()-t0:.1f}s")

    ranks = [int(x) for x in args.ranks.split(",")]
    results = []
    for k in ranks:
        print(f"\n=== rank {k} compression (with α) ===")
        t0 = time.perf_counter()
        bases = {n: top_k_basis(covs[n], k) for n, _ in modules}
        # ---- measure projected RMS per head with these bases (no α yet) ----
        dtype = modules[0][1].weight.dtype

        # Install rank-k bottleneck only, no scaling, measure RMS
        def make_bottleneck_only_hook(n, P_on_dev):
            PPt = (P_on_dev @ P_on_dev.T).contiguous()
            def hook(mod, inputs, output):
                return output @ PPt
            return hook
        base_dev = {n: bases[n].to(dtype).to(device) for n, _ in modules}
        handles = []
        for name, mod in modules:
            handles.append(mod.register_forward_hook(make_bottleneck_only_hook(name, base_dev[name])))
        projected_rms = collect_rms_per_head(model, tokenizer, CALIB_TEXTS, modules, device,
                                             head_dim, n_kv_heads)
        remove_hooks(handles)
        # ---- derive per-head α ----
        alphas = {}
        for name, _ in modules:
            eps = 1e-6
            alpha = teacher_rms[name] / (projected_rms[name] + eps)
            alphas[name] = alpha

        # ---- install bottleneck + α, generate ----
        handles = install_bottleneck_alpha_hooks(
            modules, bases, alphas, dtype, device, n_kv_heads, head_dim)
        try:
            tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        finally:
            remove_hooks(handles)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        n = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:n], tokens[:n]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens)) if a != b), n)
        compression = d_kv / k
        avg_alpha = float(torch.stack([a.mean() for a in alphas.values()]).mean())
        print(f"  {time.perf_counter()-t0:.1f}s  compression {compression:.1f}x  "
              f"match {match}/{n}  first_div @ {first_div}  avg_alpha={avg_alpha:.2f}")
        print(f"  {text[:150]}")
        results.append({
            "rank": k,
            "compression_ratio": compression,
            "match": match, "total": n,
            "match_ratio": match / max(n, 1),
            "first_divergence": first_div,
            "avg_alpha": avg_alpha,
            "sample": text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  d_kv={d_kv}  n_kv_heads={n_kv_heads}  head_dim={head_dim}")
    print(f"  baseline: {base_text[:80]}")
    print(f"  {'rank':>5}  {'compress':>8}  {'match':>10}  {'first div':>9}  {'avg_alpha':>10}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['compression_ratio']:>7.1f}x  "
              f"{r['match']}/{r['total']:<4}  {r['first_divergence']:>9}  {r['avg_alpha']:>10.3f}")

    # Compare to stage 38 if available
    stage38_path = Path("results/stage38_kv_compression.json")
    if stage38_path.exists():
        print(f"\n=== delta vs stage 38 (without α compensation) ===")
        s38 = json.load(open(stage38_path))
        s38_by_rank = {r["rank"]: r for r in s38["ranks"]}
        print(f"  {'rank':>5}  {'stage38 match':>14}  {'stage96 match':>14}  {'delta':>8}")
        for r in results:
            k = r["rank"]
            if k in s38_by_rank:
                s = s38_by_rank[k]
                delta = r["match_ratio"] - s["match_ratio"]
                print(f"  {k:>5}  {s['match']}/{s['total']:<11}  "
                      f"{r['match']}/{r['total']:<11}  {delta:>+8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "d_kv": d_kv, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
            "baseline_sample": base_text[:400],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
