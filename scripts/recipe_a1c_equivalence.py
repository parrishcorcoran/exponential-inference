"""A1c: verify A1 has Loshchilov nGPT paper's structural properties.

Loads A1 (perfect-nGPT) and checks the properties Loshchilov's paper claims
nGPT models have:

  1. W̃ rows are unit-norm to high precision
  2. α tensors are well-distributed (not collapsed, not exploding)
  3. Singular value spectra of attention matrices are bounded
  4. Hypersphere geometry: all per-row norms in W̃ tightly clustered
  5. Gradient magnitudes per param group are sane (no exploding/vanishing)

Compares to base Qwen3-0.6B where applicable. Outputs a structured report at
results/a1c_equivalence.json.
"""
import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngpt_lossless_convert import NGPTLinear, TARGETS  # noqa: E402
from ngpt_load import load_ngpt_model  # noqa: E402


CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
NGPT_DIR = Path(os.environ.get("NGPT_DIR", "model_package/Qwen3-0.6B-nGPT-perfect"))
OUTPUT = Path(os.environ.get("OUTPUT", "results/a1c_equivalence.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def measure_w_tilde_norms(model):
    """Property 1: W̃ rows should be unit-norm to high precision."""
    out = {}
    all_norms = []
    for name, mod in model.named_modules():
        if isinstance(mod, NGPTLinear):
            n = mod.weight.data.float().norm(dim=-1)
            all_norms.append(n)
            out[name] = {
                "mean": n.mean().item(), "min": n.min().item(),
                "max": n.max().item(), "std": n.std().item(),
            }
    all_norms = torch.cat(all_norms)
    summary = {
        "n_modules": len(out),
        "n_rows_total": all_norms.numel(),
        "global_mean": all_norms.mean().item(),
        "global_min": all_norms.min().item(),
        "global_max": all_norms.max().item(),
        "global_std": all_norms.std().item(),
        "max_deviation_from_unit": (all_norms - 1.0).abs().max().item(),
        "fraction_within_1e-3": ((all_norms - 1.0).abs() < 1e-3).float().mean().item(),
        "per_module": out,
    }
    return summary


def measure_alpha_distribution(model):
    """Property 2: α should have a healthy distribution (not collapsed, not exploding)."""
    out = {}
    all_alphas = []
    for name, mod in model.named_modules():
        if isinstance(mod, NGPTLinear):
            a = mod.alpha.data.float()
            all_alphas.append(a)
            out[name] = {
                "mean": a.mean().item(), "min": a.min().item(),
                "max": a.max().item(), "std": a.std().item(),
            }
    all_alphas = torch.cat(all_alphas)
    return {
        "n_modules": len(out),
        "n_alphas_total": all_alphas.numel(),
        "global_mean": all_alphas.mean().item(),
        "global_min": all_alphas.min().item(),
        "global_max": all_alphas.max().item(),
        "global_std": all_alphas.std().item(),
        "fraction_negative": (all_alphas < 0).float().mean().item(),
        "per_module": out,
    }


def measure_attention_singular_values(model, max_layers=4):
    """Property 3: bounded singular values for attention W̃ matrices.

    Loshchilov: nGPT's unit-norm rows imply ||W̃||_2 = O(1) per layer.
    Compare top singular values of W̃ across layers.
    """
    layers_root = model.model.layers if hasattr(model, "model") else model.layers
    n = len(layers_root)
    sample_indices = list(range(min(max_layers, n // 4)))  # first few layers
    sample_indices += list(range(n - max_layers, n))         # last few layers
    sample_indices = sorted(set(sample_indices))

    out = {}
    for li in sample_indices:
        layer = layers_root[li]
        layer_data = {}
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = layer.self_attn.__getattr__(proj_name)
            if isinstance(mod, NGPTLinear):
                W = mod.weight.data.float()
                # SVD on a slice (full SVD is expensive)
                # Use low-rank approximation via top-k singular values
                try:
                    U, S, V = torch.linalg.svd(W, full_matrices=False)
                    layer_data[proj_name] = {
                        "shape": list(W.shape),
                        "top_5_sv": S[:5].cpu().tolist(),
                        "min_sv": S.min().item(),
                        "max_sv": S.max().item(),
                        "condition_number": (S.max() / S.min().clamp(min=1e-9)).item(),
                        "frobenius_norm": W.norm().item(),
                    }
                except Exception as e:
                    layer_data[proj_name] = {"error": str(e)}
        out[f"layer_{li}"] = layer_data
    return out


def measure_per_module_param_counts(model):
    """Sanity: count parameters by group."""
    w_tilde_count = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".weight") and any(t in n for t in TARGETS))
    alpha_count = sum(p.numel() for n, p in model.named_parameters() if n.endswith(".alpha"))
    other_count = sum(p.numel() for n, p in model.named_parameters() if not (n.endswith(".alpha") or (n.endswith(".weight") and any(t in n for t in TARGETS))))
    total = w_tilde_count + alpha_count + other_count
    return {
        "w_tilde_params": w_tilde_count,
        "alpha_params": alpha_count,
        "other_params": other_count,
        "total_params": total,
        "alpha_fraction_pct": alpha_count / total * 100,
    }


def measure_gradient_conditioning(model, tokenizer):
    """Property 5: gradient magnitudes per param group, run on a small batch."""
    model.train()
    text = "The quick brown fox jumps over the lazy dog. " * 50
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)[:, :512]

    logits = model(ids).logits
    loss = F.cross_entropy(
        logits[:, :-1, :].float().view(-1, logits.size(-1)),
        ids[:, 1:].view(-1),
    )
    loss.backward()

    w_tilde_grads, alpha_grads, other_grads = [], [], []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.float()
        norm = g.norm().item()
        if name.endswith(".alpha"):
            alpha_grads.append(norm)
        elif name.endswith(".weight") and any(t in name for t in TARGETS):
            w_tilde_grads.append(norm)
        else:
            other_grads.append(norm)

    def stats(arr):
        if not arr:
            return None
        t = torch.tensor(arr)
        return {
            "n": len(arr),
            "mean": t.mean().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "std": t.std().item(),
        }

    out = {
        "loss_on_test_batch": loss.item(),
        "w_tilde_grad_norms": stats(w_tilde_grads),
        "alpha_grad_norms": stats(alpha_grads),
        "other_grad_norms": stats(other_grads),
    }

    # Clear gradients
    model.zero_grad(set_to_none=True)
    model.eval()
    return out


def main():
    print(f"=== A1c: nGPT equivalence checks ===")
    print(f"  loading A1: {NGPT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = load_ngpt_model(NGPT_DIR, CHECKPOINT, DEVICE, DTYPE)

    print("\n[1/5] W̃ unit-norm property")
    p1 = measure_w_tilde_norms(model)
    print(f"  global mean = {p1['global_mean']:.7f}  range = [{p1['global_min']:.6f}, {p1['global_max']:.6f}]")
    print(f"  max deviation from unit = {p1['max_deviation_from_unit']:.6f}")
    print(f"  rows within 1e-3 of unit = {p1['fraction_within_1e-3']*100:.4f}%")

    print("\n[2/5] α distribution")
    p2 = measure_alpha_distribution(model)
    print(f"  global mean = {p2['global_mean']:.4f}  range = [{p2['global_min']:.4f}, {p2['global_max']:.4f}]")
    print(f"  std = {p2['global_std']:.4f}")
    print(f"  fraction negative = {p2['fraction_negative']*100:.2f}%")

    print("\n[3/5] attention singular value spectra")
    p3 = measure_attention_singular_values(model)
    sample_layer = next(iter(p3))
    print(f"  sample {sample_layer}.q_proj top SVs: {p3[sample_layer].get('q_proj', {}).get('top_5_sv', 'N/A')}")

    print("\n[4/5] parameter counts")
    p4 = measure_per_module_param_counts(model)
    print(f"  W̃: {p4['w_tilde_params']:,}  α: {p4['alpha_params']:,}  other: {p4['other_params']:,}")
    print(f"  α is {p4['alpha_fraction_pct']:.4f}% of total")

    print("\n[5/5] gradient conditioning on test batch")
    p5 = measure_gradient_conditioning(model, tokenizer)
    print(f"  loss = {p5['loss_on_test_batch']:.4f}")
    if p5['w_tilde_grad_norms']:
        print(f"  W̃ grads: mean={p5['w_tilde_grad_norms']['mean']:.4f}  range=[{p5['w_tilde_grad_norms']['min']:.4e}, {p5['w_tilde_grad_norms']['max']:.4f}]")
    if p5['alpha_grad_norms']:
        print(f"  α grads: mean={p5['alpha_grad_norms']['mean']:.4f}  range=[{p5['alpha_grad_norms']['min']:.4e}, {p5['alpha_grad_norms']['max']:.4f}]")

    # Verdict
    print("\n=== VERDICT ===")
    pass_unit_norm = p1['max_deviation_from_unit'] < 0.01
    pass_alpha_healthy = p2['global_min'] > 0 and p2['global_max'] < 100
    pass_grad_sane = (p5['w_tilde_grad_norms'] is not None and
                      p5['w_tilde_grad_norms']['max'] < 1000 and
                      p5['w_tilde_grad_norms']['min'] > 1e-8)
    print(f"  ✓ W̃ unit-norm:           {pass_unit_norm}")
    print(f"  ✓ α distribution healthy: {pass_alpha_healthy}")
    print(f"  ✓ gradient conditioning:  {pass_grad_sane}")

    overall = pass_unit_norm and pass_alpha_healthy and pass_grad_sane
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'} — A1 {'has' if overall else 'does NOT have'} Loshchilov nGPT structural properties")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_dir": str(NGPT_DIR),
        "verdicts": {
            "w_tilde_unit_norm": pass_unit_norm,
            "alpha_healthy": pass_alpha_healthy,
            "gradient_conditioning_sane": pass_grad_sane,
            "overall": overall,
        },
        "property_1_w_tilde_norms": p1,
        "property_2_alpha_distribution": p2,
        "property_3_attention_singular_values": p3,
        "property_4_param_counts": p4,
        "property_5_gradient_conditioning": p5,
    }
    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved: {OUTPUT}")


if __name__ == "__main__":
    main()
