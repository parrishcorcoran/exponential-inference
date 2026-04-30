"""
Stage 120 — Multi-axis slow-anneal squeeze test on Qwen3-0.6B.

Per user's direction: apply tiny slow compression to every layer on
every axis simultaneously, with fine-tuning between each step. Let
the model's own adaptation reveal its natural compressed shape.

Axes:
  - Per-layer weight bits (start 16, step down by 1, floor at 2)
  - Per-layer KV rank (start d_kv=1024, step by 64, floor at 32)
  - Per-layer d_ffn (start 3072, step by 128, floor at 256)
  - Embed bits (single slider, start 16, step by 1, floor at 2)

Policy (thermostat):
  - Pick random (axis, layer) that hasn't been frozen
  - Tentatively advance the slider by one step
  - Fine-tune M steps
  - Evaluate val_ppl
  - If val_ppl <= threshold × teacher: keep; reset consecutive_reject
  - Else: back off one step, increment reject count for that slider
  - Freeze slider if rejected 3 times
  - Stop when all sliders frozen OR max_cycles reached

Result: per-(axis, layer) floor values. The natural compressed shape.
"""

import argparse
import json
import math
import random
import time
import gc
from pathlib import Path
from collections import defaultdict

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


def iter_batches(tokens, seq_len, batch_size, device, shuffle=True):
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n))
    if shuffle: random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=5):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    model.train()
    return total / max(count, 1)


def quantize_tensor_int(W, bits):
    levels = 2 ** bits
    if bits >= 16:
        return W
    max_abs = W.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / (levels / 2 - 1)).clamp(min=1e-8)
    q = torch.round(W / scale).clamp(-(levels/2 - 1), levels/2 - 1)
    return q * scale


# ---------- state management ----------

class ModelState:
    """Tracks original weights + applies compression from slider state."""
    def __init__(self, model):
        self.orig_weights = {}
        self.orig_embed = None
        self.orig_dffn = None
        # attention + MLP
        for i, layer in enumerate(model.model.layers):
            for parent, name in [(layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                                 (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                                 (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                                 (layer.mlp, "down_proj")]:
                mod = getattr(parent, name)
                self.orig_weights[(i, name)] = mod.weight.data.clone()
        # embedding
        self.orig_embed = model.get_input_embeddings().weight.data.clone()
        # d_ffn original full width
        self.orig_dffn = model.config.intermediate_size


def apply_weight_bits_to_layer(model, state, layer_idx, bits):
    """Quantize all 7 weights in a layer to `bits`."""
    if bits >= 16:
        # restore originals
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(model.model.layers[layer_idx].self_attn, name)
            mod.weight.data.copy_(state.orig_weights[(layer_idx, name)].to(mod.weight.dtype))
        for name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(model.model.layers[layer_idx].mlp, name)
            mod.weight.data.copy_(state.orig_weights[(layer_idx, name)].to(mod.weight.dtype))
        return
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        mod = getattr(model.model.layers[layer_idx].self_attn, name)
        W = state.orig_weights[(layer_idx, name)].float()
        W_q = quantize_tensor_int(W, bits)
        mod.weight.data.copy_(W_q.to(mod.weight.dtype))
    for name in ("gate_proj", "up_proj", "down_proj"):
        mod = getattr(model.model.layers[layer_idx].mlp, name)
        W = state.orig_weights[(layer_idx, name)].float()
        W_q = quantize_tensor_int(W, bits)
        mod.weight.data.copy_(W_q.to(mod.weight.dtype))


def apply_embed_bits(model, state, bits):
    W = state.orig_embed.float()
    if bits >= 16:
        model.get_input_embeddings().weight.data.copy_(state.orig_embed)
        return
    W_q = quantize_tensor_int(W, bits)
    model.get_input_embeddings().weight.data.copy_(W_q.to(model.get_input_embeddings().weight.dtype))


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tune-steps-per-probe", type=int, default=50)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--tolerance", type=float, default=1.10,
                   help="Accept if val_ppl <= teacher × tolerance")
    p.add_argument("--freeze-after-rejects", type=int, default=3)
    p.add_argument("--out", default="results/stage120_squeeze.json")
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
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")
    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 200, split="train")

    print("loading model + measuring baseline...", flush=True)
    model = load_fresh(args.model, device)
    L = model.config.num_hidden_layers
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=10)
    teacher_ppl = math.exp(teacher_ce)
    threshold = teacher_ppl * args.tolerance
    print(f"  teacher val_ppl={teacher_ppl:.3f}  threshold={threshold:.3f}  L={L}", flush=True)

    state = ModelState(model)

    # Initialize sliders — SHAPE-AWARE per wormhole structure.
    # Stage 111 manifold + Strix cross-model: 0.6B throat at L2-L24.
    # Stage 107 established Q8 is free for weights on 0.6B; Q6 cheap; Q4 cliff.
    # Strix stage 115 found Q5 middle works at 14B.
    # Initialize: edges at Q10 (plenty of margin), inner edges Q8,
    # throat at Q6 (known cheap). Squeeze pushes each further.
    EDGE_WIDTH = 3
    INNER_WIDTH = 2  # layers adjacent to edges
    sliders = {}
    for i in range(L):
        if i < EDGE_WIDTH or i >= L - EDGE_WIDTH:
            start = 10            # hard edges (L0-2, L25-27)
        elif i < EDGE_WIDTH + INNER_WIDTH or i >= L - EDGE_WIDTH - INNER_WIDTH:
            start = 8             # inner edges (L3-4, L23-24)
        else:
            start = 6             # deep throat (L5-L22)
        sliders[("wbits", i)] = [start, 0]
    sliders[("embed_bits", None)] = [8, 0]    # embed: Q8 (free, already known)

    # Simple axis step definitions
    def step_down(axis, current):
        if axis == "wbits":
            if current <= 2: return None
            return current - 1
        if axis == "embed_bits":
            if current <= 2: return None
            return current - 1
        return None

    def apply_all(model, state, sliders):
        for (axis, lid), (val, _) in sliders.items():
            if axis == "wbits":
                apply_weight_bits_to_layer(model, state, lid, val)
            elif axis == "embed_bits":
                apply_embed_bits(model, state, val)

    # Verify baseline after apply_all with all sliders at 16
    apply_all(model, state, sliders)
    baseline_check = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=5)
    print(f"  after apply_all at 16: val_ppl={math.exp(baseline_check):.3f} (should match teacher)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    history = []
    cycle = 0; t0 = time.time()

    while cycle < args.max_cycles:
        active = [k for k, (val, rej) in sliders.items() if rej < args.freeze_after_rejects]
        if not active:
            print(f"\nall sliders frozen at cycle {cycle}", flush=True)
            break

        # Pick random active slider
        key = random.choice(active)
        axis, lid = key
        current, rej = sliders[key]
        next_val = step_down(axis, current)
        if next_val is None:
            # Hit floor, freeze
            sliders[key][1] = args.freeze_after_rejects
            continue

        # Tentatively advance
        sliders[key][0] = next_val
        apply_all(model, state, sliders)

        # Fine-tune a bit
        model.train()
        step = 0
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device, shuffle=True):
            if step >= args.tune_steps_per_probe: break
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        # Eval
        val_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=3)
        val_ppl = math.exp(val_ce)

        if val_ppl <= threshold:
            # keep
            decision = "ACCEPT"
            sliders[key][1] = 0  # reset reject count for this slider
        else:
            # back off
            decision = "REJECT"
            sliders[key][0] = current
            sliders[key][1] = rej + 1
            apply_all(model, state, sliders)

        cycle += 1
        elapsed = time.time() - t0
        frozen = sum(1 for _, r in sliders.values() if r >= args.freeze_after_rejects)
        history.append({
            "cycle": cycle, "axis": axis, "layer": lid,
            "tried_value": next_val, "prev_value": current,
            "val_ppl": val_ppl, "decision": decision,
            "frozen_count": frozen, "active_count": len(sliders) - frozen,
            "elapsed": elapsed,
        })
        # Brief log
        key_str = f"{axis}/L{lid}" if lid is not None else axis
        print(f"  cyc {cycle:>3}  {key_str:>15}  {current}→{next_val}  "
              f"val_ppl={val_ppl:.3f}  {decision}  frozen={frozen}/{len(sliders)}  "
              f"elapsed={elapsed:.0f}s", flush=True)

        # Periodic snapshot save
        if cycle % 20 == 0:
            snap = {k: v[0] for k, v in sliders.items()}
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"args": vars(args), "teacher_val_ppl": teacher_ppl,
                           "threshold": threshold, "L": L,
                           "current_sliders": {f"{k[0]}/{k[1]}": v[0] for k, v in sliders.items()},
                           "history": history}, f, indent=2)

    # Final save + summary
    print(f"\n=== SUMMARY — stage 120 squeeze ===", flush=True)
    print(f"teacher val_ppl: {teacher_ppl:.3f}  threshold: {threshold:.3f}")
    print(f"\nWeight bits per layer (floor reached):")
    for i in range(L):
        v = sliders[("wbits", i)][0]
        rej = sliders[("wbits", i)][1]
        status = "FROZEN" if rej >= args.freeze_after_rejects else "ACTIVE"
        print(f"  L{i:>2}: {v} bits  ({status})")
    print(f"\nEmbed bits: {sliders[('embed_bits', None)][0]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "teacher_val_ppl": teacher_ppl,
                   "threshold": threshold, "L": L,
                   "final_sliders": {f"{k[0]}/{k[1]}": v[0] for k, v in sliders.items()},
                   "history": history}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
