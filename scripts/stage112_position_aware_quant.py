"""
Stage 112 — Position-aware quantization on 0.6B.

Tests the bathtub prediction (Finding 13): middle layers (L3-24) have
rank-1 activation trajectories and can tolerate aggressive weight
quantization. Edge layers (L0-2, L25-27) need higher precision because
their activations are high-rank.

Compare uniform vs hybrid schedules at post-hoc (no fine-tune).

Variants:
  1. Teacher (baseline)
  2. Uniform Q8 (sanity)
  3. Uniform Q6
  4. Uniform Q4 (known cliff from stage 107)
  5. HYBRID Q8 edges + Q4 middle
  6. HYBRID Q6 edges + Q4 middle
  7. HYBRID Q8 edges + Q3 middle
  8. HYBRID Q6 edges + ternary middle (Q2)
  9. HYBRID Q8 edges + ternary middle (most aggressive)

Edge layers: L0-2 and L25-27 (6 of 28 layers, 21%)
Middle layers: L3-24 (22 of 28 layers, 79%)

Hypothesis: hybrid configs with ternary middle should beat uniform Q4
because middle's rank-1 activation tolerates ternary, while edges get
the precision they need.
"""

import argparse
import json
import math
import time
import gc
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


def iter_batches(tokens, seq_len, device):
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
    for inp, tgt in iter_batches(tokens, seq_len, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    return total / max(count, 1)


def quantize_tensor_int(W, bits):
    """Per-channel symmetric int quantization, per output row (dim 0)."""
    levels = 2 ** bits
    max_abs = W.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / (levels / 2 - 1)).clamp(min=1e-8)
    q = torch.round(W / scale).clamp(-(levels/2 - 1), levels/2 - 1)
    return q * scale


def quantize_layer_weights(layer, bits):
    """Quantize all linear weights in a single transformer layer."""
    for parent, name in [(layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                         (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                         (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                         (layer.mlp, "down_proj")]:
        old = getattr(parent, name)
        W_orig = old.weight.data.clone().float()
        W_q = quantize_tensor_int(W_orig, bits)
        old.weight.data.copy_(W_q.to(old.weight.dtype))


def apply_per_layer_schedule(model, schedule):
    """schedule: list of (layer_idx, bits). Quantize each layer at its specified bits."""
    L = len(model.model.layers)
    for layer_idx, bits in schedule:
        if layer_idx >= L or bits >= 16:
            continue
        quantize_layer_weights(model.model.layers[layer_idx], bits)


def make_schedule(L, edge_layers_each_side, edge_bits, middle_bits):
    """Return list of (layer_idx, bits) based on edge/middle split."""
    schedule = []
    for l in range(L):
        if l < edge_layers_each_side or l >= L - edge_layers_each_side:
            schedule.append((l, edge_bits))
        else:
            schedule.append((l, middle_bits))
    return schedule


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--edge-width", type=int, default=3,
                   help="Number of edge layers on each side (default 3 → 6 total edges, 22 middle)")
    p.add_argument("--out", default="results/stage112_position_aware.json")
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
    L = model.config.num_hidden_layers
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}  L={L}", flush=True)
    del model; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # Variants to test
    edge_w = args.edge_width
    variants = [
        ("uniform_Q8",                    lambda: [(l, 8)  for l in range(L)]),
        ("uniform_Q6",                    lambda: [(l, 6)  for l in range(L)]),
        ("uniform_Q4",                    lambda: [(l, 4)  for l in range(L)]),
        ("uniform_Q3",                    lambda: [(l, 3)  for l in range(L)]),
        ("uniform_Q2_ternary",            lambda: [(l, 2)  for l in range(L)]),
        ("hybrid_Q8_edge__Q4_mid",        lambda: make_schedule(L, edge_w, 8, 4)),
        ("hybrid_Q6_edge__Q4_mid",        lambda: make_schedule(L, edge_w, 6, 4)),
        ("hybrid_Q8_edge__Q3_mid",        lambda: make_schedule(L, edge_w, 8, 3)),
        ("hybrid_Q6_edge__Q3_mid",        lambda: make_schedule(L, edge_w, 6, 3)),
        ("hybrid_Q8_edge__Q2_mid",        lambda: make_schedule(L, edge_w, 8, 2)),
        ("hybrid_Q6_edge__Q2_mid",        lambda: make_schedule(L, edge_w, 6, 2)),
        ("hybrid_Q4_edge__Q2_mid",        lambda: make_schedule(L, edge_w, 4, 2)),
    ]

    tests = []
    for label, schedule_fn in variants:
        print(f"\n--- {label} ---", flush=True)
        t0 = time.time()
        model = load_fresh(args.model, device)
        schedule = schedule_fn()
        apply_per_layer_schedule(model, schedule)
        model = model.to(device)
        try:
            ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
            ppl = math.exp(ce)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            ce = float('inf'); ppl = float('inf')
        delta = ppl - teacher_ppl if math.isfinite(ppl) else float('inf')
        # Compute effective bits/weight across the schedule
        total_bits = sum(b for _, b in schedule)
        avg_bits = total_bits / L
        elapsed = time.time() - t0
        tests.append({
            "label": label, "avg_bits": avg_bits, "schedule": schedule,
            "val_ce": ce, "val_ppl": ppl, "delta_ppl": delta,
            "elapsed_sec": elapsed,
        })
        bucket = ("free" if abs(delta) < 0.5 else
                  "cheap" if delta < 2 else
                  "moderate" if delta < 10 else
                  "expensive" if delta < 100 else
                  "broken")
        print(f"  avg={avg_bits:.1f} bits  val_ppl={ppl:.3f}  Δ={delta:+.3f}  [{bucket}]",
              flush=True)
        del model; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    # Summary
    print(f"\n=== SUMMARY ===\nteacher val_ppl: {teacher_ppl:.3f}\n", flush=True)
    print(f"{'config':>30}  {'avg_bits':>8}  {'val_ppl':>10}  {'Δ':>+10}  cost")
    for r in tests:
        bucket = ("free" if abs(r['delta_ppl']) < 0.5 else
                  "cheap" if r['delta_ppl'] < 2 else
                  "moderate" if r['delta_ppl'] < 10 else
                  "expensive" if r['delta_ppl'] < 100 else
                  "broken")
        print(f"  {r['label']:>30}  {r['avg_bits']:>8.1f}  {r['val_ppl']:>10.3f}  "
              f"{r['delta_ppl']:>+10.3f}  {bucket}")

    # Key comparison: uniform_Q4 vs hybrid_Q8_edge_Q4_mid vs hybrid_Q8_edge_Q3_mid
    print(f"\n=== bathtub prediction check ===", flush=True)
    uniform_q4 = next((t for t in tests if t["label"] == "uniform_Q4"), None)
    hybrid_84 = next((t for t in tests if t["label"] == "hybrid_Q8_edge__Q4_mid"), None)
    if uniform_q4 and hybrid_84:
        print(f"  uniform Q4  ({uniform_q4['avg_bits']:.1f} bits avg): {uniform_q4['val_ppl']:.2f} ppl")
        print(f"  hybrid 8+4  ({hybrid_84['avg_bits']:.1f} bits avg): {hybrid_84['val_ppl']:.2f} ppl")
        if hybrid_84['val_ppl'] < uniform_q4['val_ppl']:
            saved = uniform_q4['val_ppl'] - hybrid_84['val_ppl']
            print(f"  bathtub-aware saved {saved:.1f} ppl at ~same bits")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "L": L, "edge_width": edge_w,
                   "teacher_val_ce": teacher_ce, "teacher_val_ppl": teacher_ppl,
                   "tests": tests}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
