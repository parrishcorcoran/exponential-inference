"""
Stage 135 — Slow-anneal W_K and W_V with fine-tuning.

Stage 134 showed post-hoc KV subspace projection fails catastrophically.
Stage 117/119 (Strix) showed factoring k_proj/v_proj weights at rank 3
on 14B works WITH fine-tuning. This stage applies the slow-anneal +
finetune methodology to 0.6B's KV projections to find the per-layer
floor with proper training.

Method:
  1. Factorize W_K and W_V at high rank (full or near-full)
  2. Fine-tune for N steps
  3. Reduce rank multiplicatively (×0.85)
  4. Fine-tune for N steps
  5. Repeat until PPL exceeds threshold
  6. Back off, freeze

Targets ALL layers (no zone selection — let the per-layer floor emerge
during the anneal).

This is the trained-aware version of stage 134. Runs on MPS but slow
(~2 hours). Z8 (which is currently finetuning 0.6B) is a better target.

NOTE: this is the script — running on Mac is slow. Designed for handoff
to Z8 or Strix for full execution.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FactoredLinear(nn.Module):
    """W ≈ A @ B, with explicit matmul (avoids MPS F.linear bug)."""
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)  # [out, rank]
        self.B = nn.Parameter(B)  # [rank, in]
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out


def factorize_linear(linear, rank, device, dtype):
    """SVD-truncate to target rank. SVD on CPU."""
    W = linear.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    bias = linear.bias.data.to(dtype).to(device) if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def refactorize(fac_linear, rank, device, dtype):
    """Re-SVD an already-factored layer to lower rank."""
    with torch.no_grad():
        W_eff = (fac_linear.A.data.float().cpu() @ fac_linear.B.data.float().cpu())
    U, S, Vt = torch.linalg.svd(W_eff, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A_new = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B_new = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    fac_linear.A = nn.Parameter(A_new)
    fac_linear.B = nn.Parameter(B_new)


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
    if shuffle:
        import random
        random.shuffle(idx)
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
def eval_loss(model, tokens, seq_len, device, n_batches=10):
    model.eval()
    total = 0.0; n = 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item()
        n += 1
        if n >= n_batches: break
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage135_kv_anneal_ft.json")
    p.add_argument("--device", default=None)
    p.add_argument("--ranks", default="1024,768,512,384,256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1")
    p.add_argument("--ft-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-tokens", type=int, default=100000)
    p.add_argument("--val-tokens", type=int, default=4000)
    p.add_argument("--early-stop-loss-delta", type=float, default=1.0,
                   help="stop if loss > baseline + this")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    ranks = [int(x) for x in args.ranks.split(",")]
    print(f"device={device}  dtype={dtype}  ranks={ranks}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"L={L}  d={d}")

    print("loading WikiText-2...")
    train_tokens = load_tokens(tokenizer, args.train_tokens, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")
    print(f"  train={len(train_tokens)}  val={len(val_tokens)}")

    # Baseline
    loss_base = eval_loss(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # Factor ALL layers' k_proj and v_proj at first (highest) rank
    proj_names = ["k_proj", "v_proj"]
    factored_modules = {}
    first_rank = ranks[0]
    print(f"\nfactorizing all layers' k_proj, v_proj at rank {first_rank}...")
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in proj_names:
            proj = getattr(attn, name)
            max_rank = min(proj.weight.shape)
            r = min(first_rank, max_rank)
            factored = factorize_linear(proj, r, device, dtype)
            setattr(attn, name, factored)
            factored_modules[(l, name)] = factored

    # Freeze most params, train only A/B + final norm
    for p_ in model.parameters(): p_.requires_grad = False
    for m in factored_modules.values():
        m.A.requires_grad = True; m.B.requires_grad = True
    for p_ in model.model.norm.parameters(): p_.requires_grad = True

    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  trainable params: {n_trainable:,}")

    results = {"baseline_loss": loss_base, "baseline_ppl": ppl_base,
                "ranks": ranks, "ft_steps": args.ft_steps, "stages": []}

    # Anneal
    for stage_idx, rank in enumerate(ranks):
        print(f"\n{'=' * 60}\n=== stage {stage_idx}: rank → {rank} ===\n{'=' * 60}")
        t0 = time.time()

        # Refactorize
        for (l, name), fac in factored_modules.items():
            max_r = min(fac.A.shape[0], fac.B.shape[1])
            r = min(rank, max_r)
            refactorize(fac, r, device, dtype)
            fac.A.requires_grad = True; fac.B.requires_grad = True

        # Build trainable list and optimizer
        trainable = []
        for m in factored_modules.values():
            trainable += [m.A, m.B]
        for p_ in model.model.norm.parameters():
            trainable.append(p_)
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

        # Pre-FT loss
        loss_pre = eval_loss(model, val_tokens, args.seq_len, device)
        print(f"  pre-FT:  loss={loss_pre:.4f}  PPL={np.exp(loss_pre):.2f}")

        # Sanity check at first stage
        if stage_idx == 0 and loss_pre - loss_base > 0.5:
            print(f"  FACTORIZATION SANITY CHECK FAILED — first-stage at full rank "
                  f"should be near-identity but loss jumped {loss_pre - loss_base:+.3f}")
            break

        # Fine-tune
        model.train()
        step = 0
        train_losses = []
        while step < args.ft_steps:
            for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
                if step >= args.ft_steps: break
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                train_losses.append(loss.item())
                if step % 20 == 0:
                    print(f"    step {step:>3d}/{args.ft_steps}  loss={loss.item():.4f}")
                step += 1

        loss_post = eval_loss(model, val_tokens, args.seq_len, device)
        ppl_post = float(np.exp(loss_post))
        delta = loss_post - loss_base
        dur = time.time() - t0
        print(f"  post-FT: loss={loss_post:.4f}  PPL={ppl_post:.2f}  "
              f"Δ baseline={delta:+.3f}  ({dur:.0f}s)")

        results["stages"].append({
            "rank": rank,
            "loss_pre_ft": loss_pre, "ppl_pre_ft": float(np.exp(loss_pre)),
            "loss_post_ft": loss_post, "ppl_post_ft": ppl_post,
            "delta_loss_baseline": delta,
            "train_loss_start": train_losses[0] if train_losses else None,
            "train_loss_end": train_losses[-1] if train_losses else None,
            "duration_s": dur,
        })

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Early stop
        if delta > args.early_stop_loss_delta:
            print(f"\n  EARLY STOP: Δloss {delta:+.3f} > {args.early_stop_loss_delta}. "
                  f"KV anneal floor identified near rank {rank}.")
            break

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  baseline: PPL={ppl_base:.2f}")
    for s in results["stages"]:
        marker = " ✓ " if s["delta_loss_baseline"] < 0.05 else \
                 " ~ " if s["delta_loss_baseline"] < 0.2 else \
                 " ! " if s["delta_loss_baseline"] < 1.0 else "XXX"
        print(f"  rank {s['rank']:>4d}: pre={s['ppl_pre_ft']:>8.1f}  post-FT={s['ppl_post_ft']:>8.1f}  "
              f"Δ={s['delta_loss_baseline']:+.3f}  {marker}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
