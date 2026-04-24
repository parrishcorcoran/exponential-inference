"""
Stage 134 — Shape-aware KV subspace projection (proof of concept).

Stage 132 measured per-layer K/V rank: K is rank 1-5, V is rank 12-46.
This stage TESTS whether projecting K/V into their measured rank
subspace at inference time preserves quality.

Procedure:
  1. Calibrate: run training corpus through model. For each layer l,
     stack K_l and V_l across all positions/sentences. SVD → top-k basis.
  2. Install forward hooks on k_proj and v_proj for each layer that
     project the output through the rank-k subspace: out → out @ U @ U^T.
  3. Measure PPL on WikiText-2 val.

Sweep over (rank_K, rank_V) configurations:
  - Uniform aggressive: rank_K=1, rank_V=10
  - Uniform measured: rank_K=5, rank_V=50
  - Uniform generous: rank_K=10, rank_V=100
  - Per-layer measured: use each layer's actual rank from stage 132
  - Baseline: full rank (no projection)

Outputs:
  - PPL per configuration
  - Compression ratio per configuration
  - Quality/compression frontier
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


@torch.no_grad()
def calibrate_subspaces(model, tokens, seq_len, n_chunks, device):
    """Run several chunks through model, collect K and V per layer.
       Returns U_K[l], U_V[l] = full SVD basis for each layer."""
    L = model.config.num_hidden_layers
    K_accum = [[] for _ in range(L)]
    V_accum = [[] for _ in range(L)]

    for c in range(n_chunks):
        start = c * seq_len
        if start + seq_len > len(tokens): break
        ids = torch.tensor([tokens[start:start+seq_len]],
                           dtype=torch.long, device=device)
        out = model(ids, use_cache=True, output_hidden_states=False)
        kv = out.past_key_values
        if hasattr(kv, "layers") and kv.layers:
            pairs = [(layer_cache.keys, layer_cache.values) for layer_cache in kv.layers]
        elif hasattr(kv, "to_legacy_cache"):
            pairs = kv.to_legacy_cache()
        else:
            pairs = list(kv)
        for l, (K, V) in enumerate(pairs):
            K_flat = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
            V_flat = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
            K_accum[l].append(K_flat)
            V_accum[l].append(V_flat)

    U_K = []
    U_V = []
    S_K = []
    S_V = []
    for l in range(L):
        K_stack = torch.cat(K_accum[l], dim=0)  # [N_total, d_kv]
        V_stack = torch.cat(V_accum[l], dim=0)
        # SVD
        _, sk, vkt = torch.linalg.svd(K_stack, full_matrices=False)
        _, sv, vvt = torch.linalg.svd(V_stack, full_matrices=False)
        U_K.append(vkt.T)  # [d_kv, rank]
        U_V.append(vvt.T)
        S_K.append(sk)
        S_V.append(sv)
    return U_K, U_V, S_K, S_V


def make_subspace_hook(U_k, device, dtype):
    """Forward hook for k_proj or v_proj that projects output to rank-k subspace.
       output: [batch, seq, num_kv_heads × head_dim]
       projection: output @ U_k @ U_k^T  in [d_kv, d_kv]"""
    P = (U_k @ U_k.T).to(device).to(dtype)

    def hook(module, inputs, output):
        # output is the attention's k_proj or v_proj output
        # In Qwen3 with GQA, output shape is [batch, seq, num_kv_heads * head_dim]
        # which is what was stored as flat in calibration
        return output @ P
    return hook


def install_kv_hooks(model, U_K_list, U_V_list, rank_K_per_layer, rank_V_per_layer, device):
    """Install hooks on each layer's k_proj and v_proj."""
    handles = []
    L = len(U_K_list)
    dtype = next(model.parameters()).dtype
    for l in range(L):
        rk = min(rank_K_per_layer[l], U_K_list[l].shape[1])
        rv = min(rank_V_per_layer[l], U_V_list[l].shape[1])
        U_K = U_K_list[l][:, :rk]
        U_V = U_V_list[l][:, :rv]
        layer = model.model.layers[l]
        h_k = layer.self_attn.k_proj.register_forward_hook(
            make_subspace_hook(U_K, device, dtype))
        h_v = layer.self_attn.v_proj.register_forward_hook(
            make_subspace_hook(U_V, device, dtype))
        handles.extend([h_k, h_v])
    return handles


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, n_batches=12):
    model.eval()
    total_loss = 0.0
    n = 0
    for i in range(n_batches):
        start = i * seq_len
        if start + seq_len + 1 > len(tokens):
            break
        ids = torch.tensor([tokens[start:start+seq_len+1]],
                            dtype=torch.long, device=device)
        out = model(ids[:, :-1], use_cache=False)
        logits = out.logits.reshape(-1, out.logits.shape[-1])
        targets = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.float(), targets)
        total_loss += loss.item()
        n += 1
    return total_loss / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage134_kv_subspace.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--calib-chunks", type=int, default=10)
    p.add_argument("--eval-batches", type=int, default=15)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    print(f"L={L}")

    # Tokens
    print("loading WikiText-2 tokens...")
    train_tokens = load_tokens(tok, args.seq_len * args.calib_chunks * 2, "train")
    val_tokens = load_tokens(tok, args.seq_len * args.eval_batches * 2, "validation")

    # Calibrate
    print(f"calibrating subspaces from {args.calib_chunks} chunks of {args.seq_len} tokens...")
    t0 = time.time()
    U_K, U_V, S_K, S_V = calibrate_subspaces(model, train_tokens,
                                              args.seq_len, args.calib_chunks, device)
    d_kv = U_K[0].shape[0]
    print(f"  done in {time.time()-t0:.0f}s  d_kv={d_kv}")

    # Print spectrum at a few layers for sanity
    for l in [0, L//4, L//2, 3*L//4, L-1]:
        s_k = S_K[l]
        s_v = S_V[l]
        evr_k_5 = (s_k[:5].pow(2).sum() / s_k.pow(2).sum()).item()
        evr_v_50 = (s_v[:50].pow(2).sum() / s_v.pow(2).sum()).item()
        print(f"  L{l}: K EVR@5={evr_k_5:.3f}  V EVR@50={evr_v_50:.3f}")

    # Baseline PPL
    print("\nbaseline PPL (no projection)...")
    base_loss = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    base_ppl = float(np.exp(base_loss))
    print(f"  baseline: loss={base_loss:.4f}  PPL={base_ppl:.2f}")

    results = {"model": args.model, "d_kv": d_kv,
                "baseline_loss": base_loss, "baseline_ppl": base_ppl,
                "configs": []}

    # Configurations to test
    configs = [
        ("aggressive", [1] * L, [10] * L),
        ("measured", [5] * L, [50] * L),
        ("generous", [10] * L, [100] * L),
        ("loose", [20] * L, [200] * L),
    ]

    for name, rank_K, rank_V in configs:
        # Compute compression ratio
        # Storage per layer: (rank_K + rank_V) × seq vs d_kv × 2 × seq
        avg_rank_K = sum(rank_K) / L
        avg_rank_V = sum(rank_V) / L
        comp_ratio = (2 * d_kv) / (avg_rank_K + avg_rank_V)
        # Install hooks, eval, remove
        handles = install_kv_hooks(model, U_K, U_V, rank_K, rank_V, device)
        loss = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
        ppl = float(np.exp(loss))
        for h in handles: h.remove()

        delta_loss = loss - base_loss
        marker = " ✓ " if delta_loss < 0.05 else \
                 " ~ " if delta_loss < 0.2 else \
                 " ! " if delta_loss < 1.0 else "XXX"
        print(f"\n  {name:>12s}  rank_K={int(avg_rank_K):>3d}  rank_V={int(avg_rank_V):>3d}  "
              f"compression={comp_ratio:>5.1f}×  PPL={ppl:>7.2f}  Δloss={delta_loss:+.4f}  {marker}")
        results["configs"].append({
            "name": name,
            "rank_K_avg": avg_rank_K,
            "rank_V_avg": avg_rank_V,
            "compression_ratio": comp_ratio,
            "loss": loss, "ppl": ppl, "delta_loss": delta_loss,
        })

    # Per-layer measured (use 95% EVR threshold per layer)
    rank_K_per_layer = []
    rank_V_per_layer = []
    for l in range(L):
        # Find min rank to capture 95% EVR
        evr_K = S_K[l].pow(2).cumsum(0) / S_K[l].pow(2).sum()
        evr_V = S_V[l].pow(2).cumsum(0) / S_V[l].pow(2).sum()
        rk = max(1, (evr_K < 0.95).sum().item() + 1)
        rv = max(1, (evr_V < 0.95).sum().item() + 1)
        rank_K_per_layer.append(rk)
        rank_V_per_layer.append(rv)
    avg_rk = sum(rank_K_per_layer) / L
    avg_rv = sum(rank_V_per_layer) / L

    print(f"\n  per-layer at 95% EVR: avg rank_K={avg_rk:.1f}  rank_V={avg_rv:.1f}")
    print(f"    rank_K per layer: {rank_K_per_layer}")
    print(f"    rank_V per layer: {rank_V_per_layer}")

    handles = install_kv_hooks(model, U_K, U_V, rank_K_per_layer, rank_V_per_layer, device)
    loss = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    ppl = float(np.exp(loss))
    for h in handles: h.remove()
    comp_ratio = (2 * d_kv) / (avg_rk + avg_rv)
    delta_loss = loss - base_loss
    marker = " ✓ " if delta_loss < 0.05 else " ~ " if delta_loss < 0.2 else " ! " if delta_loss < 1.0 else "XXX"
    print(f"  per-layer 95EVR  rank_K={avg_rk:.1f}  rank_V={avg_rv:.1f}  "
          f"compression={comp_ratio:.1f}×  PPL={ppl:.2f}  Δloss={delta_loss:+.4f}  {marker}")
    results["configs"].append({
        "name": "per_layer_95evr",
        "rank_K_per_layer": rank_K_per_layer,
        "rank_V_per_layer": rank_V_per_layer,
        "rank_K_avg": avg_rk,
        "rank_V_avg": avg_rv,
        "compression_ratio": comp_ratio,
        "loss": loss, "ppl": ppl, "delta_loss": delta_loss,
    })

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  baseline PPL: {base_ppl:.2f}")
    print(f"  {'config':>16s}  {'compression':>12s}  {'PPL':>8s}  {'Δloss':>8s}")
    for c in results["configs"]:
        print(f"  {c['name']:>16s}  {c['compression_ratio']:>11.1f}×  {c['ppl']:>8.2f}  "
              f"{c['delta_loss']:>+8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
