"""
Stage 43 — Compositionality test: head pruning + KV compression simultaneously.

Verifies whether head pruning (Finding 04) and KV compression (stage 38) are
independent compression axes. If independent, stacking both should preserve
output coherence comparable to either alone. If they overlap, stacking
should break output where neither alone does.

Four conditions on Qwen3-0.6B:
  (1) baseline            — no compression
  (2) heads only          — keep 20% of Q-heads per layer (by calibration norm)
  (3) KV only             — KV outputs projected through rank-128 bottleneck
  (4) heads + KV stacked  — both (2) and (3) active

Report token match and sample per condition. If (4) is as coherent as (3)
alone, the axes stack.
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


def find_o_projs(model):
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last == "o_proj":
            result.append((name, mod))
    return result


# ---------- KV compression (from stage 38) ----------

def collect_output_cov(model, tokenizer, texts, modules, device, max_len=256):
    covs = {name: None for name, _ in modules}

    def make_hook(n):
        def hook(mod, inputs, output):
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


def install_kv_bottleneck(modules, covs, rank, dtype, device):
    handles = []
    for name, mod in modules:
        P = top_k_basis(covs[name], rank).to(dtype).to(device)
        PPt = (P @ P.T).contiguous()
        def make_hook(projector):
            def hook(mod, inputs, output):
                return output @ projector
            return hook
        handles.append(mod.register_forward_hook(make_hook(PPt)))
    return handles


# ---------- Head pruning ----------

def collect_head_norms(model, tokenizer, texts, o_modules, device, n_heads, head_dim, max_len=256):
    """Per-layer per-head mean output norm from calibration."""
    sums = {name: torch.zeros(n_heads, dtype=torch.float32) for name, _ in o_modules}
    counts = {name: 0 for name, _ in o_modules}

    def make_prehook(n):
        def hook(mod, inputs):
            x = inputs[0].detach()  # [B, T, n_heads*head_dim]
            B, T, _ = x.shape
            x_heads = x.reshape(B, T, n_heads, head_dim).to(torch.float32).cpu()
            norms = x_heads.pow(2).sum(dim=-1).sqrt()  # [B, T, n_heads]
            sums[n] += norms.reshape(-1, n_heads).sum(dim=0)
            counts[n] += B * T
        return hook

    handles = [mod.register_forward_pre_hook(make_prehook(name)) for name, mod in o_modules]
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
    return {n: (sums[n] / max(counts[n], 1)) for n in sums}


def install_head_mask(o_modules, head_norms, n_heads, head_dim, keep_frac, dtype, device):
    """Keep the top-`keep_frac` heads by calibration norm; zero the rest."""
    handles = []
    for name, mod in o_modules:
        norms = head_norms[name]
        keep_k = max(1, int(round(keep_frac * n_heads)))
        keep_idx = norms.topk(keep_k).indices.tolist()
        mask = torch.zeros(n_heads, dtype=dtype, device=device)
        mask[keep_idx] = 1.0
        # mask shape [n_heads]; broadcast over [B, T, n_heads, head_dim]
        mask_hd = mask.unsqueeze(-1).expand(n_heads, head_dim).reshape(-1)  # [n_heads*head_dim]

        def make_prehook(m):
            def hook(mod, inputs):
                x = inputs[0]
                return (x * m,)
            return hook

        handles.append(mod.register_forward_pre_hook(make_prehook(mask_hd)))
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
    p.add_argument("--head-keep-frac", type=float, default=0.20)
    p.add_argument("--kv-rank", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage43_compose_heads_kv.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim",
                        model.config.hidden_size // model.config.num_attention_heads)
    print(f"  n_heads={n_heads}  head_dim={head_dim}  "
          f"n_kv_heads={model.config.num_key_value_heads}")

    kv_modules = find_kv_projs(model)
    o_modules = find_o_projs(model)
    print(f"  kv projs: {len(kv_modules)}   o projs: {len(o_modules)}")

    print(f"\n=== baseline ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:160]}")

    print(f"\n=== calibrating K/V covs + per-head norms ===")
    t0 = time.perf_counter()
    kv_covs = collect_output_cov(model, tokenizer, CALIB_TEXTS, kv_modules, device)
    head_norms = collect_head_norms(model, tokenizer, CALIB_TEXTS, o_modules,
                                    device, n_heads, head_dim)
    print(f"  {time.perf_counter()-t0:.1f}s")

    dtype = kv_modules[0][1].weight.dtype
    results = {}

    def run_condition(name, install_kv, install_heads):
        handles = []
        if install_kv:
            handles += install_kv_bottleneck(kv_modules, kv_covs, args.kv_rank, dtype, device)
        if install_heads:
            handles += install_head_mask(o_modules, head_norms, n_heads, head_dim,
                                         args.head_keep_frac, dtype, device)
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
            "first_divergence": first_div, "sample": text[:400],
        }

    print(f"\n=== heads-only ({int(args.head_keep_frac*100)}% kept) ===")
    run_condition("heads_only", install_kv=False, install_heads=True)

    print(f"\n=== KV-only (rank {args.kv_rank}) ===")
    run_condition("kv_only", install_kv=True, install_heads=False)

    print(f"\n=== heads + KV stacked ===")
    run_condition("heads_kv_stacked", install_kv=True, install_heads=True)

    print(f"\n=== summary ===")
    print(f"  baseline: {base_text[:80]}")
    print(f"  {'condition':>20}  {'match':>10}  {'first_div':>9}")
    print(f"  {'baseline':>20}  {len(base_tokens)}/{len(base_tokens):<4}  "
          f"{len(base_tokens):>9}")
    for n, r in results.items():
        print(f"  {n:>20}  {r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "head_keep_frac": args.head_keep_frac,
            "kv_rank": args.kv_rank,
            "baseline_sample": base_text[:400],
            "conditions": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
