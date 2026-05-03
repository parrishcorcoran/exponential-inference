"""
Per-layer attention budget based on PCA dimensionality.

Wall layers (high-D): full attention
Throat layers (1D): skip attention or sliding window
Budget allocated by measured manifold structure.

Test strategies per layer:
  - "full": standard Q@K^T softmax (expensive)
  - "window": attend to only last W tokens (cheap)
  - "skip": pass hidden state through unchanged (free)

Assign strategy based on PCA dim:
  99% variance in 1D → skip or window
  99% variance in 10D+ → full

Measure PPL at various budget allocations.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(seq_len=256, max_val=100_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    val_t = tokens[:max_val]
    def chunk(t):
        n = len(t) // (seq_len + 1)
        return t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return chunk(val_t)


@torch.inference_mode()
def eval_ppl(model, chunks, seq_len=256, n=20):
    model.eval()
    total = 0; c = 0
    for i in range(min(n, len(chunks))):
        inp = chunks[i:i+1, :seq_len]
        tgt = chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        c += 1
    return math.exp(total / max(c, 1))


def measure_pca(model, val_chunks, seq_len=256):
    """Measure PCA dimensionality at every layer."""
    with torch.inference_mode():
        inp = val_chunks[0:1, :seq_len]
        out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
        hidden_states = [h.squeeze(0) for h in out.hidden_states]

    L = len(hidden_states) - 1  # exclude embedding layer from count
    pca_dims = []

    for li in range(L + 1):
        h = hidden_states[li].float()
        h_c = h - h.mean(dim=0)
        S = torch.linalg.svdvals(h_c)
        var = (S ** 2) / (S ** 2).sum()
        cum = var.cumsum(0)
        d90 = (cum < 0.90).sum().item() + 1
        d95 = (cum < 0.95).sum().item() + 1
        d99 = (cum < 0.99).sum().item() + 1
        pca_dims.append({
            "layer": li, "d90": d90, "d95": d95, "d99": d99,
            "top1_var": var[0].item(),
        })

    return pca_dims


def assign_strategies(pca_dims, throat_threshold=3, narrow_threshold=20):
    """Assign attention strategy per layer based on PCA dim.

    d99 <= throat_threshold → skip (1D, attention is useless)
    d99 <= narrow_threshold → window (low-D, local attention enough)
    d99 > narrow_threshold → full (high-D, need global attention)
    """
    strategies = []
    for p in pca_dims[1:]:  # skip embedding layer
        if p["d99"] <= throat_threshold:
            strategies.append("skip")
        elif p["d99"] <= narrow_threshold:
            strategies.append("window")
        else:
            strategies.append("full")
    return strategies


def install_weighted_attention(model, strategies, window_size=16):
    """Install per-layer attention hooks based on strategy."""
    L = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim',
                       model.config.hidden_size // n_heads)
    hooks = []

    for li in range(L):
        strategy = strategies[li]

        if strategy == "full":
            continue  # don't hook, use default

        attn_mod = model.model.layers[li].self_attn
        orig_fwd = attn_mod.forward

        if strategy == "skip":
            def make_skip_hook(orig):
                def hooked(hidden_states, *args, **kwargs):
                    # Skip attention entirely — return zeros
                    # The residual connection in the transformer block
                    # will just pass hidden_states through
                    return (torch.zeros_like(hidden_states), None)
                return hooked
            attn_mod.forward = make_skip_hook(orig_fwd)

        elif strategy == "window":
            def make_window_hook(orig, ws):
                def hooked(hidden_states, *args, **kwargs):
                    B, S, D = hidden_states.shape
                    hd = head_dim
                    am = model.model.layers[li].self_attn

                    q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                    k = am.k_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)
                    v = am.v_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)

                    kv_groups = n_heads // n_kv_heads
                    K_e = k.unsqueeze(2).expand(B, n_kv_heads, kv_groups, S, hd).reshape(B, n_heads, S, hd)
                    V_e = v.unsqueeze(2).expand(B, n_kv_heads, kv_groups, S, hd).reshape(B, n_heads, S, hd)

                    # Standard attention but with window mask
                    scores = q @ K_e.transpose(-2, -1) / math.sqrt(hd)

                    # Causal + window mask
                    mask = torch.ones(S, S, device=scores.device, dtype=torch.bool)
                    for i in range(S):
                        start = max(0, i - ws + 1)
                        mask[i, start:i+1] = False  # unmask window
                    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                    attn = torch.softmax(scores, dim=-1)
                    out = attn @ V_e
                    out = out.transpose(1, 2).reshape(B, S, n_heads * hd)
                    out = am.o_proj(out)
                    return (out, None)
                return hooked
            attn_mod.forward = make_window_hook(orig_fwd, window_size)

        hooks.append((attn_mod, orig_fwd, strategy))

    return hooks


def restore_hooks(hooks):
    for attn_mod, orig_fwd, _ in hooks:
        attn_mod.forward = orig_fwd


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"WEIGHTED LAYER ATTENTION: {cli.model}")
    print(f"  Allocate attention budget by PCA dimensionality")
    print(f"  WALL layers: full attention")
    print(f"  THROAT layers: skip or sliding window")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  L={L}, d={d_model}")

    val_chunks = load_data()

    teacher_ppl = eval_ppl(model, val_chunks)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # Measure PCA
    print(f"\nMeasuring PCA dimensionality...", flush=True)
    pca_dims = measure_pca(model, val_chunks)
    print(f"\n  Layer-by-layer PCA:")
    for p in pca_dims:
        shape = "THROAT" if p["d99"] <= 3 else "narrow" if p["d99"] <= 20 else "WALL"
        print(f"    L{p['layer']:>3}: d99={p['d99']:>4}  top1={p['top1_var']:.4f}  [{shape}]")

    # Test different budget configurations
    print(f"\n{'='*60}")
    print(f"BUDGET CONFIGURATIONS")
    print(f"{'='*60}")
    print(f"  {'Config':>30} | {'Skip':>4} {'Win':>4} {'Full':>4} | {'PPL':>8} | {'Ratio':>6} | {'Savings'}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
    print(f"  {'all full':>30} | {0:>4} {0:>4} {L:>4} | {teacher_ppl:8.2f} | {'1.00x':>6} | 0%", flush=True)

    configs = [
        ("throat=skip (d99<=1)", 1, 0),
        ("throat=skip (d99<=3)", 3, 0),
        ("throat=skip (d99<=5)", 5, 0),
        ("throat=window (d99<=3)", 3, 20),
        ("throat=window (d99<=5)", 5, 20),
        ("throat+narrow=skip (d99<=20)", 20, 0),
        ("narrow=window (d99<=20)", 20, 20),
        ("aggressive (d99<=50)", 50, 0),
        ("aggressive+window (d99<=50)", 50, 50),
    ]

    results = []
    for name, skip_thresh, window_thresh in configs:
        del model; gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            cli.model, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="eager").eval()

        # Assign strategies
        strategies = []
        for p in pca_dims[1:]:
            if window_thresh > 0 and p["d99"] <= window_thresh:
                if p["d99"] <= skip_thresh:
                    strategies.append("skip")
                else:
                    strategies.append("window")
            elif p["d99"] <= skip_thresh:
                strategies.append("skip")
            else:
                strategies.append("full")

        n_skip = strategies.count("skip")
        n_window = strategies.count("window")
        n_full = strategies.count("full")
        savings = (n_skip + 0.5 * n_window) / L * 100

        hooks = install_weighted_attention(model, strategies, window_size=16)
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl

        print(f"  {name:>30} | {n_skip:>4} {n_window:>4} {n_full:>4} | {ppl:8.2f} | {ratio:5.2f}x | {savings:.0f}%",
              flush=True)

        results.append({
            "config": name, "skip": n_skip, "window": n_window, "full": n_full,
            "ppl": round(ppl, 2), "ratio": round(ratio, 4), "savings_pct": round(savings, 1),
        })

        restore_hooks(hooks)

    # Timing comparison for best config
    print(f"\n{'='*60}")
    print(f"TIMING: best config vs full")
    print(f"{'='*60}")

    # Find best config within 5% quality
    good = [r for r in results if r["ratio"] <= 1.05]
    if good:
        best = max(good, key=lambda r: r["savings_pct"])
        print(f"  Best within 5%: {best['config']} ({best['savings_pct']}% savings, {best['ratio']:.2f}x)")
    else:
        best = min(results, key=lambda r: r["ratio"])
        print(f"  Best overall: {best['config']} ({best['ratio']:.2f}x)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher: {teacher_ppl:.2f}")
    for r in results:
        marker = " <<<" if r["ratio"] <= 1.05 else ""
        print(f"    {r['config']:>30}: PPL={r['ppl']:.2f} ({r['ratio']:.2f}x) "
              f"savings={r['savings_pct']}%{marker}")

    all_results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "pca": pca_dims, "configs": results,
    }
    with open(Path(save_dir) / "weighted_layer_attention.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {save_dir}/weighted_layer_attention.json")


if __name__ == "__main__":
    main()
