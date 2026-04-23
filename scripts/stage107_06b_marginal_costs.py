"""
Stage 107 — 0.6B marginal cost analysis (untested axes).

Completes the Strix 14B marginal-cost picture for 0.6B. Tests each
compression axis POST-HOC (no fine-tune) so results are comparable to
stage 38, stage 92, and Strix's Qwen Halo matrix.

Axes tested here:
  A. Weight Q16 → Q8 per-channel
  B. Weight Q16 → Q4 per-channel
  C. Embed Q16 → Q8
  D. Embed Q16 → Q4
  E. MLP d_ffn shrink (keep first k columns of gate/up, first k rows of down)
  F. LM head Q8

All measurements in val_ppl vs teacher on wikitext-2. No fine-tune.
Output directly comparable to Strix's marginal cost verdict.
"""

import argparse
import json
import math
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


def iter_batches(tokens, seq_len, batch_size, device):
    n = (len(tokens) - 1) // seq_len
    for i in range(n):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        t = torch.tensor([window], dtype=torch.long, device=device)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    return total / max(count, 1)


# -------- quantization ops (post-hoc, fake-quantize: apply then dequant) --------

def quantize_tensor_int(W, bits, per_channel_dim=None):
    """Symmetric int quantization. bits: target bit count (e.g., 8, 6, 4).
       per_channel_dim: if given, scale is per-row (output channel). Else per-tensor."""
    levels = 2 ** bits
    if per_channel_dim is not None:
        max_abs = W.abs().amax(dim=1 - per_channel_dim, keepdim=True)
    else:
        max_abs = W.abs().max()
    scale = (max_abs / (levels / 2 - 1)).clamp(min=1e-8)
    q = torch.round(W / scale).clamp(-(levels/2 - 1), levels/2 - 1)
    return q * scale


def quantize_all_body_weights(model, bits):
    """Quantize every nn.Linear weight in attention + MLP bodies."""
    for layer in model.model.layers:
        for parent, name in [(layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                             (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                             (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                             (layer.mlp, "down_proj")]:
            old = getattr(parent, name)
            W_orig = old.weight.data.clone()
            W_q = quantize_tensor_int(W_orig.float(), bits, per_channel_dim=0)
            old.weight.data.copy_(W_q.to(old.weight.dtype))


def quantize_embed(model, bits):
    W = model.get_input_embeddings().weight.data
    W_q = quantize_tensor_int(W.float(), bits, per_channel_dim=0)
    model.get_input_embeddings().weight.data.copy_(W_q.to(W.dtype))


def quantize_lm_head(model, bits):
    if hasattr(model, "lm_head") and not (model.lm_head.weight is model.get_input_embeddings().weight):
        W = model.lm_head.weight.data
        W_q = quantize_tensor_int(W.float(), bits, per_channel_dim=0)
        model.lm_head.weight.data.copy_(W_q.to(W.dtype))


def shrink_dffn(model, keep):
    """Shrink d_ffn to `keep` columns/rows. In-place modification of weight shapes."""
    for layer in model.model.layers:
        g = layer.mlp.gate_proj
        u = layer.mlp.up_proj
        d = layer.mlp.down_proj
        k = min(keep, g.weight.shape[0])
        # gate: [d_ffn, d_model] — keep first k rows
        g.weight.data = g.weight.data[:k].contiguous()
        u.weight.data = u.weight.data[:k].contiguous()
        # down: [d_model, d_ffn] — keep first k columns
        d.weight.data = d.weight.data[:, :k].contiguous()
        # Update out_features / in_features for correctness
        g.out_features = k
        u.out_features = k
        d.in_features = k


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    m.eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--out", default="results/stage107_06b_marginal.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 30, split="validation")

    # Teacher baseline
    print("teacher baseline...", flush=True)
    model = load_fresh(args.model, device)
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}  val_ce={teacher_ce:.4f}", flush=True)
    del model; import gc; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    tests = []

    # Weight bit sweep
    for bits in [8, 6, 4, 3, 2]:
        print(f"\n--- weight Q{bits} per-channel ---", flush=True)
        t0 = time.time()
        model = load_fresh(args.model, device)
        quantize_all_body_weights(model, bits)
        model = model.to(device)
        ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
        ppl = math.exp(ce)
        tests.append({"axis": "weight_bits", "value": bits, "val_ce": ce, "val_ppl": ppl,
                     "delta_ppl": ppl - teacher_ppl, "delta_ce": ce - teacher_ce,
                     "time": time.time()-t0})
        print(f"  val_ppl={ppl:.3f}  Δ={ppl-teacher_ppl:+.3f} from teacher  "
              f"(teacher={teacher_ppl:.3f})", flush=True)
        del model; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    # Embed bit sweep
    for bits in [8, 6, 4, 3, 2]:
        print(f"\n--- embed Q{bits} ---", flush=True)
        t0 = time.time()
        model = load_fresh(args.model, device)
        quantize_embed(model, bits)
        model = model.to(device)
        ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
        ppl = math.exp(ce)
        tests.append({"axis": "embed_bits", "value": bits, "val_ce": ce, "val_ppl": ppl,
                     "delta_ppl": ppl - teacher_ppl, "delta_ce": ce - teacher_ce,
                     "time": time.time()-t0})
        print(f"  val_ppl={ppl:.3f}  Δ={ppl-teacher_ppl:+.3f}", flush=True)
        del model; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    # LM head bit sweep (if not tied)
    model_check = load_fresh(args.model, device)
    tied = model_check.lm_head.weight is model_check.get_input_embeddings().weight
    del model_check
    if tied:
        print(f"\nLM head tied with embed; covered by embed sweep", flush=True)
    else:
        for bits in [8, 6, 4]:
            print(f"\n--- lm_head Q{bits} ---", flush=True)
            t0 = time.time()
            model = load_fresh(args.model, device)
            quantize_lm_head(model, bits)
            model = model.to(device)
            ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
            ppl = math.exp(ce)
            tests.append({"axis": "lm_head_bits", "value": bits, "val_ce": ce, "val_ppl": ppl,
                         "delta_ppl": ppl - teacher_ppl, "delta_ce": ce - teacher_ce,
                         "time": time.time()-t0})
            print(f"  val_ppl={ppl:.3f}  Δ={ppl-teacher_ppl:+.3f}", flush=True)
            del model; gc.collect()
            if device == "mps": torch.mps.empty_cache()

    # d_ffn shrink
    for keep in [3072, 2048, 1536, 1024, 768, 512, 384, 256]:
        print(f"\n--- d_ffn = {keep} ---", flush=True)
        t0 = time.time()
        model = load_fresh(args.model, device)
        if keep < 3072:
            shrink_dffn(model, keep)
        model = model.to(device)
        try:
            ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
            ppl = math.exp(ce)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            ppl = float('inf'); ce = float('inf')
        tests.append({"axis": "d_ffn", "value": keep, "val_ce": ce, "val_ppl": ppl,
                     "delta_ppl": ppl - teacher_ppl if math.isfinite(ppl) else float('inf'),
                     "delta_ce": ce - teacher_ce if math.isfinite(ce) else float('inf'),
                     "time": time.time()-t0})
        print(f"  val_ppl={ppl:.3f}  Δ={ppl - teacher_ppl:+.3f}", flush=True)
        del model; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    print(f"\n{'='*60}\n=== 0.6B marginal cost summary ===\n{'='*60}", flush=True)
    print(f"teacher val_ppl: {teacher_ppl:.3f}")
    print(f"\n{'axis':>16}  {'value':>8}  {'val_ppl':>10}  {'Δ_ppl':>+10}  cost")
    for r in tests:
        marker = "FREE WIN!" if r['delta_ppl'] < -0.1 else ("cheap" if r['delta_ppl'] < 1 else
                 ("moderate" if r['delta_ppl'] < 10 else ("expensive" if r['delta_ppl'] < 100 else "broken")))
        print(f"  {r['axis']:>16}  {r['value']:>8}  {r['val_ppl']:>10.3f}  "
              f"{r['delta_ppl']:>+10.3f}  {marker}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "teacher_val_ce": teacher_ce,
                   "teacher_val_ppl": teacher_ppl, "tests": tests}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
