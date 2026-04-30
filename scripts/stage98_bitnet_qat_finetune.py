"""
Stage 98 — BitNet-style QAT fine-tune of Qwen3-0.6B.

Question: can we replicate BitNet's 10× compression by fine-tuning an
existing fp16 model to ternary weights, rather than training from scratch?

Method (standard QAT + BitNet b1.58 recipe):
  - Every nn.Linear in the transformer body (attention Q/K/V/O, MLP
    gate/up/down) is replaced by a QATTernaryLinear.
  - QATTernaryLinear holds an fp16/fp32 master weight W_fp.
  - Forward: quantize W_fp to W_ternary in {-1, 0, +1} via threshold
    τ = 0.7 * mean(|W_fp|). Apply learned per-tensor scalar α.
    Effective W = α * W_ternary.
  - Backward: straight-through estimator — gradients flow from the
    quantized forward into W_fp directly (identity on the sign gate).

At inference we'd store only W_ternary (1.58 bits/weight) + α (per tensor).
At training we pay fp for W_fp. The MODEL at inference is ~10× smaller.

Protocol:
  1. Load teacher (unmodified) — measure val perplexity.
  2. Convert body linears to QATTernaryLinear (init W_fp = teacher weights,
     α = mean(|W_fp|)). Measure val ppl at step 0 (quantized, not yet
     retrained).
  3. Fine-tune on wikitext-2 for N steps. Track val ppl.
  4. Report recovery curve. Target: quantized ≈ teacher val ppl.
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


# ---------------- QAT Ternary Linear ----------------

class TernaryQuantSTE(torch.autograd.Function):
    """Quantize W_fp to {-α, 0, +α} with threshold τ = 0.7 * mean(|W_fp|).
       Forward: returns quantized. Backward: identity (STE)."""
    @staticmethod
    def forward(ctx, W_fp):
        absW = W_fp.abs()
        tau = 0.7 * absW.mean()
        # sign where |W|>tau, else 0
        W_sign = torch.where(W_fp > tau, torch.ones_like(W_fp),
                  torch.where(W_fp < -tau, -torch.ones_like(W_fp), torch.zeros_like(W_fp)))
        return W_sign
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out  # STE: gradient passes through unchanged


class QATTernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.W_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.alpha = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.W_fp, std=0.02)
    @classmethod
    def from_linear(cls, linear):
        """Copy weights from an existing nn.Linear."""
        m = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        with torch.no_grad():
            # Keep original fp weights as the master; init α to match mean magnitude
            m.W_fp.data = linear.weight.data.clone().float()
            m.alpha.data = torch.tensor([m.W_fp.data.abs().mean().item()], dtype=torch.float32)
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m
    def forward(self, x):
        # Quantize on the fly
        W_sign = TernaryQuantSTE.apply(self.W_fp)
        W_eff = W_sign * self.alpha   # broadcast
        y = F.linear(x.float(), W_eff, self.bias)
        return y.to(x.dtype)


def convert_body_to_ternary(model):
    """Replace every nn.Linear in the transformer body (attention + MLP) with
       QATTernaryLinear. Keeps lm_head and embedding unchanged."""
    n_converted = 0
    for layer in model.model.layers:
        # Attention: q_proj, k_proj, v_proj, o_proj
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            old = getattr(layer.self_attn, name)
            new = QATTernaryLinear.from_linear(old)
            setattr(layer.self_attn, name, new)
            n_converted += 1
        # MLP: gate_proj, up_proj, down_proj
        for name in ("gate_proj", "up_proj", "down_proj"):
            old = getattr(layer.mlp, name)
            new = QATTernaryLinear.from_linear(old)
            setattr(layer.mlp, name, new)
            n_converted += 1
    return n_converted


# ---------------- data + training ----------------

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
def eval_ppl(model, tokens, seq_len, batch_size, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, batch_size, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    model.train()
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage98_bitnet_qat.json")
    p.add_argument("--force-fp32", action="store_true",
                   help="Force whole model fp32 (slow on MPS, safer numerics)")
    p.add_argument("--ckpt-every", type=int, default=500,
                   help="Save rotating checkpoint every N steps")
    p.add_argument("--ckpt-dir", default="checkpoints/stage98")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 1. Teacher baseline val ppl
    print(f"\n=== teacher baseline (fp16, unmodified) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 30, split="validation")
    teacher_ppl_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  teacher val_ce={teacher_ppl_ce:.4f}  val_ppl={math.exp(teacher_ppl_ce):.2f}", flush=True)
    del model
    import gc; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # 2. Load fresh and convert body to ternary
    print(f"\n=== loading Qwen3 and converting body to QAT ternary ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    # Keep base model in bf16 for MPS speed. QAT layers internally store fp32 master
    # weights and upcast activations inside the forward; the rest of the network
    # (norms, embedding, lm_head, RoPE, attention softmax) stays bf16 for tensor-core
    # speed. This is ~15x faster than model.float() on MPS.
    if args.force_fp32:
        model = model.float()
    n_converted = convert_body_to_ternary(model)
    model = model.to(device)
    print(f"  converted {n_converted} linear layers to QATTernaryLinear", flush=True)

    # 3. Step-0 eval (ternary but not retrained)
    print(f"\n=== step 0: ternary, pre-finetune ===", flush=True)
    init_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  val_ce={init_ce:.4f}  val_ppl={math.exp(init_ce):.2f}  "
          f"Δ from teacher: {init_ce - teacher_ppl_ce:+.4f}", flush=True)
    history = [{"step": 0, "val_ce": init_ce, "val_ppl": math.exp(init_ce),
                "delta": init_ce - teacher_ppl_ce}]

    # 4. Fine-tune
    print(f"\n=== fine-tuning {args.steps} steps at lr={args.lr} ===", flush=True)
    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 400, split="train")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    step = 0; t0 = time.time(); running = []
    while step < args.steps:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.steps: break
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(loss.item()); step += 1
            if step % args.eval_every == 0:
                tr = float(np.mean(running[-args.eval_every:]))
                val_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
                history.append({"step": step, "train_ce": tr, "val_ce": val_ce,
                               "val_ppl": math.exp(val_ce),
                               "delta": val_ce - teacher_ppl_ce,
                               "elapsed": time.time()-t0})
                print(f"  step {step}/{args.steps}  train_ce={tr:.4f}  val_ce={val_ce:.4f}  "
                      f"val_ppl={math.exp(val_ce):.2f}  Δ={val_ce-teacher_ppl_ce:+.4f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\n=== summary ===", flush=True)
    print(f"  teacher val_ppl:        {math.exp(teacher_ppl_ce):.2f}")
    for h in history:
        print(f"  step {h['step']:>5}  val_ppl={h['val_ppl']:.2f}  Δ={h['delta']:+.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "teacher_val_ce": teacher_ppl_ce,
                   "teacher_val_ppl": math.exp(teacher_ppl_ce),
                   "history": history}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
