"""
Stage 99 — AWARE low-rank KV projection, with rank sweep.

Stage 97 failed because the rank-128 SVD projection was FIXED — gradient
never saw the constraint. Stage 98 is testing that QAT works when
compression is gradient-aware. This stage applies the same principle to
KV compression: replace each k_proj and v_proj with a learnable
factorization (W_down @ W_up) of rank r, so the projection itself can
adapt.

DeepSeek-V2 MLA did this at ~8× KV compression from scratch. We're
testing whether aware training can push further via fine-tune from
pretrained, at rank r = {16, 32, 64, 128}.

For each rank:
  1. Load Qwen3-0.6B.
  2. For every attention layer, replace k_proj/v_proj with a rank-r
     factorized pair: Linear(d, r) → Linear(r, d_kv). Initialize from
     SVD of teacher's weight.
  3. Fine-tune full model on wikitext-2 for N steps.
  4. Measure val_ppl vs teacher baseline.

If rank-16 recovers to near teacher val_ppl, we've beaten DeepSeek's
aware compression ratio via fine-tune. If even rank-128 fails to
recover, aware factorization alone isn't sufficient and we need more
compensation.
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


# ---------------- aware low-rank linear ----------------

class LowRankLinear(nn.Module):
    """W ≈ W_up @ W_down where W_down: d_in → r, W_up: r → d_out. Both trainable."""
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features; self.rank = rank
        self.W_down = nn.Parameter(torch.empty(rank, in_features))
        self.W_up   = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.normal_(self.W_down, std=0.02)
        nn.init.normal_(self.W_up, std=0.02)

    @classmethod
    def from_linear_svd(cls, linear, rank):
        """Initialize via truncated SVD of the teacher's weight matrix."""
        m = cls(linear.in_features, linear.out_features, rank, bias=(linear.bias is not None))
        W = linear.weight.data.float()          # [out, in]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)   # U [out,k], S [k], Vh [k,in]
        r = min(rank, S.shape[0])
        # W_up @ W_down should approximate W → store U*sqrt(S), sqrt(S)*Vh
        sqrt_S = S[:r].sqrt()
        W_up_init = U[:, :r] * sqrt_S[None, :]           # [out, r]
        W_down_init = sqrt_S[:, None] * Vh[:r, :]        # [r, in]
        with torch.no_grad():
            m.W_up.data = W_up_init
            m.W_down.data = W_down_init
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m

    def forward(self, x):
        # x: [..., in]. down to rank, up to out.
        h = F.linear(x.float(), self.W_down)                # [..., r]
        y = F.linear(h, self.W_up, self.bias)               # [..., out]
        return y.to(x.dtype)


def convert_kv_to_low_rank(model, rank):
    """Replace every layer.self_attn.k_proj and .v_proj with LowRankLinear."""
    n = 0
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            old = getattr(layer.self_attn, name)
            new = LowRankLinear.from_linear_svd(old, rank)
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


def run_one_rank(args, tokenizer, train_tokens, val_tokens, teacher_ppl_ce, device, rank):
    print(f"\n{'='*60}\n=== rank r={rank} ===\n{'='*60}", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    model = model.float()
    n = convert_kv_to_low_rank(model, rank)
    model = model.to(device)
    print(f"  converted {n} kv projections to rank-{rank}", flush=True)

    init_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  step 0 (pre-tune, just SVD init): val_ce={init_ce:.4f}  val_ppl={math.exp(init_ce):.2f}  "
          f"Δ={init_ce-teacher_ppl_ce:+.4f}", flush=True)
    history = [{"step": 0, "val_ce": init_ce, "val_ppl": math.exp(init_ce),
                "delta": init_ce - teacher_ppl_ce}]

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

    del model; import gc; gc.collect()
    if device == "mps": torch.mps.empty_cache()
    return {"rank": rank, "history": history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="16,32,64,128")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage99_aware_kv.json")
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

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 300, split="train")

    ranks = [int(x) for x in args.ranks.split(",")]
    all_results = []
    for r in ranks:
        result = run_one_rank(args, tokenizer, train_tokens, val_tokens,
                              teacher_ppl_ce, device, r)
        all_results.append(result)

    print(f"\n=== summary ===", flush=True)
    print(f"  teacher val_ppl: {math.exp(teacher_ppl_ce):.2f}")
    print(f"  {'rank':>5}  {'init_ppl':>10}  {'final_ppl':>10}  {'final_Δ':>8}")
    for r in all_results:
        init_p = r["history"][0]["val_ppl"]
        final_p = r["history"][-1]["val_ppl"]
        final_d = r["history"][-1]["delta"]
        print(f"  {r['rank']:>5}  {init_p:>10.2f}  {final_p:>10.2f}  {final_d:>+8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "teacher_val_ce": teacher_ppl_ce,
                   "teacher_val_ppl": math.exp(teacher_ppl_ce),
                   "results": all_results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
