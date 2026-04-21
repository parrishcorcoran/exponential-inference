"""
Stage 51 — Test theory #1: 1 KV head per layer is sufficient.

Premise: if each token sits at an *exact* manifold coordinate (not an
approximation), one KV head per layer is enough — the ensemble
averaging across multiple KV heads is error-correction machinery that
the exact-point framing makes unnecessary.

Tests on Qwen3-0.6B (n_kv_heads=8 native):
  (a) baseline: full 8 KV heads
  (b) KV head 0 only: zero the other 7 kv heads' outputs
  (c) mean-of-KV-heads: replace each head with the mean across heads
  (d) rank-1 PCA of KV output: keep only top-1 direction per layer

If (b), (c), or (d) produces coherent output close to baseline, the
1-KV-sufficient claim is supported. If all degrade, the claim fails.

Implementation via forward hooks on k_proj and v_proj outputs.
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
    "The cell is the basic structural unit of life.",
    "Quantum mechanics describes matter and energy at atomic scales.",
    "Photosynthesis uses sunlight to convert carbon dioxide into glucose.",
    "Neural networks consist of parameterized layers trained by gradient descent.",
    "The immune system recognizes pathogens through pattern-recognition receptors.",
    "Plate tectonics describes the movement of lithospheric plates over the mantle.",
    "Proteins fold into three-dimensional structures determined by amino-acid sequences.",
    "Black holes are regions of spacetime from which nothing can escape.",
    "DNA encodes genetic information in a double-helix of paired nucleotide bases.",
    "Linear algebra provides the mathematical foundation for machine learning.",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def find_kv_projs(model):
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in ("k_proj", "v_proj"):
            result.append((name, mod))
    return result


def install_keep_head0(modules, n_kv_heads, head_dim):
    """Zero all KV heads except head 0 (per layer)."""
    handles = []
    for name, mod in modules:
        def make_hook(n_heads=n_kv_heads, hd=head_dim):
            def hook(m, inputs, output):
                # output: [B, T, n_kv_heads * head_dim]
                y = output.reshape(*output.shape[:-1], n_heads, hd).clone()
                y[..., 1:, :] = 0.0
                return y.reshape(*output.shape[:-1], n_heads * hd)
            return hook
        handles.append(mod.register_forward_hook(make_hook()))
    return handles


def install_mean_heads(modules, n_kv_heads, head_dim):
    """Replace each KV head's output with the mean across heads."""
    handles = []
    for name, mod in modules:
        def make_hook(n_heads=n_kv_heads, hd=head_dim):
            def hook(m, inputs, output):
                y = output.reshape(*output.shape[:-1], n_heads, hd)
                mean_head = y.mean(dim=-2, keepdim=True)  # [..., 1, hd]
                y_mean = mean_head.expand_as(y)
                return y_mean.reshape(*output.shape[:-1], n_heads * hd)
            return hook
        handles.append(mod.register_forward_hook(make_hook()))
    return handles


def collect_cov(model, tokenizer, texts, modules, device, max_len=256):
    """Collect output covariance for rank-1 basis computation."""
    covs = {name: None for name, _ in modules}

    def make_hook(n):
        def hook(m, inputs, output):
            y = output.detach().reshape(-1, output.shape[-1]).to(torch.float32).cpu()
            if covs[n] is None:
                covs[n] = torch.zeros(y.shape[1], y.shape[1], dtype=torch.float32)
            covs[n] += y.T @ y
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
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)


def install_rank1_bottleneck(modules, covs, dtype, device):
    """Project KV output through rank-1 PCA basis."""
    handles = []
    for name, mod in modules:
        P = top_k_basis(covs[name], 1).to(dtype).to(device)     # [d, 1]
        PPt = (P @ P.T).contiguous()                              # [d, d], rank 1
        def make_hook(proj):
            def hook(m, inputs, output):
                return output @ proj
            return hook
        handles.append(mod.register_forward_hook(make_hook(PPt)))
    return handles


def install_rank_n_bottleneck(modules, covs, rank, dtype, device):
    """Project KV output through rank-k bottleneck (for comparisons)."""
    handles = []
    for name, mod in modules:
        P = top_k_basis(covs[name], rank).to(dtype).to(device)
        PPt = (P @ P.T).contiguous()
        def make_hook(proj):
            def hook(m, inputs, output):
                return output @ proj
            return hook
        handles.append(mod.register_forward_hook(make_hook(PPt)))
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


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
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage51_one_kv_head.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    d_kv = n_kv_heads * head_dim
    dtype = model.model.layers[0].self_attn.k_proj.weight.dtype
    print(f"  n_kv_heads={n_kv_heads}  head_dim={head_dim}  d_kv={d_kv}")

    modules = find_kv_projs(model)
    print(f"  {len(modules)} KV projections")

    print(f"\n=== baseline (all 8 KV heads) ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    print(f"\n=== collecting covariances for rank-1 and rank-128 baseline ===")
    t0 = time.perf_counter()
    covs = collect_cov(model, tokenizer, CALIB_TEXTS, modules, device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    results = {}

    def run_variant(name, installer):
        handles = installer()
        try:
            t0 = time.perf_counter()
            tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
        finally:
            remove_hooks(handles)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        n = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:n], tokens[:n]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens))
                          if a != b), n)
        print(f"  [{name}]  {time.perf_counter()-t0:.1f}s  "
              f"match {match}/{n}  first_div @ {first_div}")
        print(f"   sample: {text[:160]}")
        results[name] = {
            "match": match, "total": n, "match_ratio": match / max(n, 1),
            "first_divergence": first_div, "sample": text[:300],
        }

    print(f"\n=== (a) keep head 0 only ===")
    run_variant("keep_head0", lambda: install_keep_head0(modules, n_kv_heads, head_dim))

    print(f"\n=== (b) mean of KV heads (each head replaced by mean) ===")
    run_variant("mean_heads", lambda: install_mean_heads(modules, n_kv_heads, head_dim))

    print(f"\n=== (c) rank-1 PCA bottleneck on KV output ===")
    run_variant("rank1_pca", lambda: install_rank1_bottleneck(modules, covs, dtype, device))

    print(f"\n=== (d) rank-128 PCA bottleneck (stage 38 floor reference) ===")
    run_variant("rank128_pca", lambda: install_rank_n_bottleneck(modules, covs, 128, dtype, device))

    print(f"\n=== summary ===")
    print(f"  baseline: {base_text[:80]}")
    print(f"  {'variant':>14}  {'match':>10}  {'first_div':>9}  sample preview")
    print(f"  {'baseline':>14}  {len(base_tokens)}/{len(base_tokens):<4}  "
          f"{len(base_tokens):>9}  (full)")
    for name, r in results.items():
        print(f"  {name:>14}  {r['match']}/{r['total']:<4}  "
              f"{r['first_divergence']:>9}  {r['sample'][:60]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_kv_heads": n_kv_heads, "head_dim": head_dim, "d_kv": d_kv,
            "baseline_sample": base_text[:400],
            "variants": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
