"""
Stage 100 — Progressive rank compression with fine-tune between each step.

Hypothesis: jumping from rank-1024 to rank-16 in one shot puts the model
through a huge optimization landscape change — likely falls into a bad
local minimum. Instead, walk down the rank ladder step by step:

  rank 128 → fine-tune M steps → rank 64 → fine-tune → rank 32 → fine-tune → rank 16

At each transition, preserve the current effective weight matrix via SVD
truncation (keep the top-K components of W_up @ W_down). The model only
has to adapt to one small rank drop per stage.

Curriculum compression — known in general ML as "progressive pruning /
quantization," but applied specifically to aware low-rank KV with per-
stage recovery fine-tune is less explored for autoregressive LM.

This stage test is standalone (no QAT yet). If it works well at rank 16
here, we can stack with QAT ternary in a later stage.

Protocol:
  1. Load Qwen3-0.6B. Measure teacher val_ppl.
  2. Convert k_proj/v_proj to LowRankLinear at starting rank (e.g., 128).
  3. Measure val_ppl at init (SVD-init, no training).
  4. Fine-tune for M steps. Measure val_ppl.
  5. Rank step down: SVD the current W_up @ W_down, truncate to next rank,
     replace modules. Measure val_ppl post-truncation.
  6. Fine-tune M steps. Measure val_ppl.
  7. Repeat until target rank.
  8. Report the recovery curve: val_ppl at each (rank, training step)
     checkpoint.
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


class LowRankLinear(nn.Module):
    """W ≈ W_up @ W_down, both trainable."""
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features; self.rank = rank
        self.W_down = nn.Parameter(torch.empty(rank, in_features))
        self.W_up   = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear_svd(cls, linear, rank):
        m = cls(linear.in_features, linear.out_features, rank,
                bias=(linear.bias is not None))
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, S.shape[0])
        sqrt_S = S[:r].sqrt()
        with torch.no_grad():
            m.W_up.data = U[:, :r] * sqrt_S[None, :]
            m.W_down.data = sqrt_S[:, None] * Vh[:r, :]
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m

    @classmethod
    def from_lowrank_svd(cls, old, rank):
        """Reduce an existing LowRankLinear to smaller rank via SVD of its effective W."""
        m = cls(old.in_features, old.out_features, rank,
                bias=(old.bias is not None))
        W_eff = (old.W_up @ old.W_down).data.float()
        U, S, Vh = torch.linalg.svd(W_eff, full_matrices=False)
        r = min(rank, S.shape[0])
        sqrt_S = S[:r].sqrt()
        with torch.no_grad():
            m.W_up.data = U[:, :r] * sqrt_S[None, :]
            m.W_down.data = sqrt_S[:, None] * Vh[:r, :]
            if old.bias is not None:
                m.bias.data = old.bias.data.clone()
        return m

    def forward(self, x):
        h = F.linear(x.float(), self.W_down)
        y = F.linear(h, self.W_up, self.bias)
        return y.to(x.dtype)


def convert_kv_to_low_rank(model, rank):
    n = 0
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            old = getattr(layer.self_attn, name)
            new = LowRankLinear.from_linear_svd(old, rank)
            setattr(layer.self_attn, name, new)
            n += 1
    return n


def reduce_kv_rank(model, new_rank):
    n = 0
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            old = getattr(layer.self_attn, name)
            new = LowRankLinear.from_lowrank_svd(old, new_rank)
            setattr(layer.self_attn, name, new)
            n += 1
    return n


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


def train_phase(model, train_tokens, val_tokens, args, device, teacher_ppl_ce, phase_label):
    """Fine-tune one phase. Returns history dicts."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    step = 0; t0 = time.time(); running = []
    history = []
    while step < args.steps_per_phase:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.steps_per_phase: break
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
                history.append({"phase": phase_label, "step": step, "train_ce": tr,
                               "val_ce": val_ce, "val_ppl": math.exp(val_ce),
                               "delta": val_ce - teacher_ppl_ce,
                               "elapsed": time.time()-t0})
                print(f"  [{phase_label}] step {step}  train_ce={tr:.4f}  "
                      f"val_ce={val_ce:.4f}  val_ppl={math.exp(val_ce):.2f}  "
                      f"Δ={val_ce-teacher_ppl_ce:+.4f}", flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="128,64,32,16",
                   help="Progressive rank schedule, high→low")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps-per-phase", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage100_progressive_rank.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Teacher baseline
    print("\n=== teacher baseline (fp16) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 30, split="validation")
    teacher_ppl_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  teacher val_ce={teacher_ppl_ce:.4f}  val_ppl={math.exp(teacher_ppl_ce):.2f}", flush=True)
    del model
    import gc; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # Load fresh model for progressive compression
    print("\n=== loading model for progressive compression ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    model = model.float()

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 300, split="train")

    rank_schedule = [int(x) for x in args.ranks.split(",")]
    all_history = []

    # Initial conversion to first rank
    r0 = rank_schedule[0]
    print(f"\n=== initial conversion: rank {r0} ===", flush=True)
    n = convert_kv_to_low_rank(model, r0)
    model = model.to(device)
    print(f"  converted {n} kv projections", flush=True)
    init_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    init_ppl = math.exp(init_ce)
    print(f"  rank {r0} post-SVD (no tune): val_ppl={init_ppl:.2f}  Δ={init_ce-teacher_ppl_ce:+.4f}", flush=True)
    all_history.append({"phase": f"rank{r0}_init", "step": 0, "val_ce": init_ce,
                        "val_ppl": init_ppl, "delta": init_ce - teacher_ppl_ce})

    # Train at rank[0]
    h = train_phase(model, train_tokens, val_tokens, args, device, teacher_ppl_ce,
                    f"rank{r0}_train")
    all_history.extend(h)

    # Progressive rank reduction + training
    for r in rank_schedule[1:]:
        print(f"\n=== reducing rank to {r} ===", flush=True)
        n = reduce_kv_rank(model, r)
        model = model.to(device)
        print(f"  reduced {n} kv projections to rank {r}", flush=True)
        red_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
        red_ppl = math.exp(red_ce)
        print(f"  rank {r} post-reduction (no tune): val_ppl={red_ppl:.2f}  Δ={red_ce-teacher_ppl_ce:+.4f}", flush=True)
        all_history.append({"phase": f"rank{r}_reduced", "step": 0, "val_ce": red_ce,
                            "val_ppl": red_ppl, "delta": red_ce - teacher_ppl_ce})
        # Train at new rank
        h = train_phase(model, train_tokens, val_tokens, args, device, teacher_ppl_ce,
                        f"rank{r}_train")
        all_history.extend(h)

    print(f"\n=== summary ===", flush=True)
    print(f"  teacher val_ppl: {math.exp(teacher_ppl_ce):.2f}")
    print(f"  {'phase':>20}  {'step':>5}  {'val_ppl':>10}  {'Δ':>8}")
    # Print one summary line per phase (last step)
    phases_seen = {}
    for h in all_history:
        phases_seen[h["phase"]] = h
    for phase, h in phases_seen.items():
        print(f"  {phase:>20}  {h['step']:>5}  {h['val_ppl']:>10.2f}  {h['delta']:>+8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "teacher_val_ce": teacher_ppl_ce,
                   "teacher_val_ppl": math.exp(teacher_ppl_ce),
                   "history": all_history}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
