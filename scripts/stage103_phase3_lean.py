"""Qwen Halo Phase 3 — Lean version. Memory-safe for Strix Halo unified RAM.

The problem: QAT wrappers copy every weight tensor, doubling memory.
On Strix Halo (GPU VRAM = system RAM = 89GB), 14B model already uses 28GB.
No room for copies.

The fix: quantize weights IN-PLACE. No wrapper classes. No copies.
Just round the weight values and store one scale per tensor.

Round-robin compression:
  KV rank: 512 → 384 → 256 → 192 → 128 → 96 → 64 → 48 → 32 → 24 → 16
  Weight bits: 8 → 6 → 4
  Embed bits: 8 → 6 → 4

Each step: compress → eval → fine-tune (tiny params) → eval → next step.
Resume from Phase 1+2 checkpoints.
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
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
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
def eval_ppl(model, val_tokens, seq_len, batch_size, device):
    model.eval()
    total_loss = 0; count = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, batch_size, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total_loss += loss.item(); count += 1
        if count >= 10: break
    model.train()
    return total_loss / max(count, 1)


def quantize_tensor_inplace(w, n_bits):
    """Quantize a weight tensor in-place to n_bits. No copies."""
    if n_bits >= 16:
        return 1.0  # no-op
    levels = 2 ** n_bits
    half = levels // 2
    scale = w.float().abs().max().item() / (half - 1) if half > 1 else 1.0
    if scale < 1e-10: scale = 1e-10
    w.data = (w.float() / scale).round().clamp(-half + 1, half - 1).mul(scale).to(w.dtype)
    return scale


def compress_kv_inplace(model, rank):
    """Compress k_proj and v_proj to low-rank via SVD, in-place."""
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            W = proj.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = min(rank, len(S))
            W_approx = (U[:, :k] * S[:k].unsqueeze(0)) @ Vt[:k]
            proj.weight.data = W_approx.to(proj.weight.dtype)
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--tune-steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="results/qwen_halo_phase3.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")

    # Baseline
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"baseline val_ce={teacher_ce:.4f}  val_ppl={math.exp(teacher_ce):.2f}", flush=True)

    # Round-robin schedule
    kv_schedule = [512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16]
    weight_schedule = [8, 6, 4]
    embed_schedule = [8, 6, 4]

    # Interleave
    schedule = []
    max_len = max(len(kv_schedule), len(weight_schedule), len(embed_schedule))
    for i in range(max_len):
        if i < len(kv_schedule): schedule.append(("kv", kv_schedule[i]))
        if i < len(weight_schedule): schedule.append(("weights", weight_schedule[i]))
        if i < len(embed_schedule): schedule.append(("embed", embed_schedule[i]))

    history = []

    for mi, (axis, value) in enumerate(schedule):
        print(f"\n  -- step {mi+1}/{len(schedule)}: axis={axis} value={value} --", flush=True)

        # Compress
        if axis == "kv":
            compress_kv_inplace(model, int(value))
        elif axis == "weights":
            for layer in model.model.layers:
                for name in ["q_proj", "o_proj"]:
                    w = getattr(layer.self_attn, name).weight
                    quantize_tensor_inplace(w, float(value))
                for name in ["gate_proj", "up_proj", "down_proj"]:
                    w = getattr(layer.mlp, name).weight
                    quantize_tensor_inplace(w, float(value))
            torch.cuda.empty_cache()
        elif axis == "embed":
            quantize_tensor_inplace(model.get_input_embeddings().weight, float(value))

        # Eval after compression
        post_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
        print(f"  after compress: val_ce={post_ce:.4f}  val_ppl={math.exp(post_ce):.2f}", flush=True)

        # Fine-tune: only norms (tiny, always safe)
        for p_m in model.parameters(): p_m.requires_grad = False
        trainable = []
        for name, p_m in model.named_parameters():
            if "norm" in name or "layernorm" in name:
                p_m.requires_grad = True
                trainable.append(p_m)
        n_train = sum(p_m.numel() for p_m in trainable)
        print(f"  fine-tuning {n_train/1e6:.1f}M params (norms only)", flush=True)

        if trainable:
            opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
            model.train()
            step = 0
            for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
                if step >= args.tune_steps: break
                opt.zero_grad()
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                step += 1
            del opt
            torch.cuda.empty_cache()

        # Eval after tune
        tuned_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
        delta = tuned_ce - teacher_ce
        print(f"  after tune: val_ce={tuned_ce:.4f}  val_ppl={math.exp(tuned_ce):.2f}  "
              f"Δ={delta:+.4f}", flush=True)

        history.append({
            "step": mi + 1, "axis": axis, "value": value,
            "post_compress_ce": post_ce, "post_tune_ce": tuned_ce,
            "post_compress_ppl": math.exp(post_ce), "post_tune_ppl": math.exp(tuned_ce),
            "delta_from_teacher": delta,
        })

        # Check if broken
        if tuned_ce > teacher_ce * 3:
            print(f"  ⚠ quality degraded >3x teacher — stopping", flush=True)
            break

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"teacher_ce": teacher_ce, "history": history}, f, indent=2)
    print(f"\nsaved {args.out}", flush=True)
    print(f"\nFINAL: teacher_ppl={math.exp(teacher_ce):.2f}  "
          f"compressed_ppl={math.exp(tuned_ce):.2f}  "
          f"ratio={math.exp(tuned_ce)/math.exp(teacher_ce):.2f}x", flush=True)


if __name__ == "__main__":
    main()
