"""
Stage 126 — Slow-anneal throat factorization on 0.6B to find rank floor.

Strix showed instant rank-32 throat factorization on 14B gives coherent
text (stage 119). My stage 124b showed post-hoc activation rank
reduction on 0.6B fails even at k=640. But that was ACTIVATION rank;
this stage tests WEIGHT rank, and with fine-tuning between steps.

Schedule for 0.6B throat layers (L10-L17, deep middle):
  K/V/Q/O projections SVD-factored, rank annealed:
  512 → 256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
  Fine-tune 150 steps between each step. Eval PPL on held-out.

Stop when PPL > 2× baseline.

Frozen: all non-throat layers, all non-attention parts of throat.
Trainable: factored A/B matrices + model.norm.
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
    """Two thin matmuls replacing one fat one. W ≈ A @ B."""
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)  # [out, rank]
        self.B = nn.Parameter(B)  # [rank, in]
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        # Explicit matmul: F.linear has precision issues on MPS for non-square shapes.
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out


def factorize_linear(linear, rank, device, dtype):
    """SVD-truncate to target rank. Returns a FactoredLinear on device/dtype.
       SVD is done on CPU (MPS's SVD is numerically unstable)."""
    W = linear.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    bias = linear.bias.data.to(dtype).to(device) if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def refactorize(fac_linear, rank, device, dtype):
    """Re-SVD an already-factored FactoredLinear to a lower rank.
       W_eff = A @ B. SVD(W_eff) on CPU."""
    with torch.no_grad():
        W_eff = (fac_linear.A.data.float().cpu() @ fac_linear.B.data.float().cpu())
    U, S, Vt = torch.linalg.svd(W_eff, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A_new = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B_new = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    fac_linear.A = nn.Parameter(A_new)
    fac_linear.B = nn.Parameter(B_new)


def load_tokens(tokenizer, max_tokens, split="train"):
    """Load WikiText-2 tokens."""
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
def eval_ppl(model, tokens, seq_len, device, n_batches=12):
    model.eval()
    total = 0.0
    n = 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1))
        total += loss.item()
        n += 1
        if n >= n_batches:
            break
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage126_slow_anneal_06b.json")
    p.add_argument("--device", default=None)
    p.add_argument("--throat-start", type=int, default=10)
    p.add_argument("--throat-end", type=int, default=17)
    p.add_argument("--ranks", default="1024,768,512,384,256,192,128,96,64,48,32,24,16,12,8,6,4,3,2,1")
    p.add_argument("--ft-steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--train-tokens", type=int, default=30000)
    p.add_argument("--val-tokens", type=int, default=5000)
    p.add_argument("--early-stop-ppl-ratio", type=float, default=3.0,
                   help="stop if PPL > this × baseline")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    ranks = [int(x) for x in args.ranks.split(",")]
    print(f"device={device}  ranks={ranks}  ft_steps/rank={args.ft_steps}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    throat_layers = list(range(args.throat_start, args.throat_end + 1))
    print(f"L={L}  d={d}  throat layers: {throat_layers} ({len(throat_layers)} layers)")

    # Dataset
    print("\nloading WikiText-2...")
    train_tokens = load_tokens(tokenizer, args.train_tokens, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")
    print(f"  train tokens: {len(train_tokens)}  val tokens: {len(val_tokens)}")

    # Baseline PPL
    loss_base = eval_ppl(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline (no factorization): loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # Identify target projections & dimensions
    # 0.6B: d=1024. Probably head_dim=64, num_heads=16, num_kv_heads=8.
    # q_proj: [1024, 1024], k_proj/v_proj: [512, 1024], o_proj: [1024, 1024]
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Factor all throat projections initially at highest rank
    factored_modules = {}  # (layer_idx, proj_name) -> FactoredLinear
    first_rank = ranks[0]
    print(f"\nfactorizing throat projections initially at rank {first_rank}...")
    for l in throat_layers:
        attn = model.model.layers[l].self_attn
        for name in proj_names:
            proj = getattr(attn, name)
            max_rank = min(proj.weight.shape)
            r = min(first_rank, max_rank)
            factored = factorize_linear(proj, r, device, dtype)
            setattr(attn, name, factored)
            factored_modules[(l, name)] = factored

    # Freeze everything except factored A/B and final norm
    for p_ in model.parameters():
        p_.requires_grad = False
    trainable = []
    for m in factored_modules.values():
        m.A.requires_grad = True
        m.B.requires_grad = True
        trainable += [m.A, m.B]
    for p_ in model.model.norm.parameters():
        p_.requires_grad = True
        trainable.append(p_)
    n_trainable = sum(p_.numel() for p_ in trainable if p_.requires_grad)
    print(f"  trainable params: {n_trainable:,} "
          f"({n_trainable / sum(p_.numel() for p_ in model.parameters()) * 100:.2f}% of model)")

    results = {"baseline_loss": loss_base, "baseline_ppl": ppl_base,
               "throat_layers": throat_layers, "ranks": ranks,
               "ft_steps": args.ft_steps, "stages": []}

    # Anneal loop
    for stage_idx, rank in enumerate(ranks):
        print(f"\n{'=' * 60}")
        print(f"=== stage {stage_idx}: rank → {rank} ===")
        print(f"{'=' * 60}")

        # Set all factored modules to this rank (re-SVD from current W_eff)
        t0 = time.time()
        for (l, name), fac in factored_modules.items():
            max_rank = min(fac.A.shape[0], fac.B.shape[1])
            r = min(rank, max_rank)
            refactorize(fac, r, device, dtype)
            # restore requires_grad after reassigning Parameter
            fac.A.requires_grad = True
            fac.B.requires_grad = True
        # Rebuild trainable list after reassignment
        trainable = []
        for m in factored_modules.values():
            trainable += [m.A, m.B]
        for p_ in model.model.norm.parameters():
            trainable.append(p_)
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

        # PPL right after truncation (before finetune)
        loss_pre = eval_ppl(model, val_tokens, args.seq_len, device)
        ppl_pre = float(np.exp(loss_pre))
        print(f"  post-truncate (no FT): loss={loss_pre:.4f}  PPL={ppl_pre:.2f}")
        # Sanity check: at the first (max) rank, factorization should be identity
        if stage_idx == 0 and loss_pre - loss_base > 0.5:
            print(f"\n  SANITY CHECK FAILED: first-stage factorization at rank {rank} "
                  f"should be near-identity but baseline loss jumped "
                  f"{loss_pre - loss_base:+.3f}. Bug in factorization.")
            break

        # Finetune
        model.train()
        step = 0
        train_losses = []
        while step < args.ft_steps:
            for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
                if step >= args.ft_steps: break
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                train_losses.append(loss.item())
                if step % 30 == 0:
                    print(f"    step {step:3d}/{args.ft_steps}  loss={loss.item():.4f}")
                step += 1

        # Post-finetune PPL
        loss_post = eval_ppl(model, val_tokens, args.seq_len, device)
        ppl_post = float(np.exp(loss_post))
        dur = time.time() - t0
        print(f"  post-FT:    loss={loss_post:.4f}  PPL={ppl_post:.2f}  "
              f"(Δ baseline {loss_post - loss_base:+.3f})  [{dur:.0f}s]")

        # Count effective params
        n_fac_params = sum(m.A.numel() + m.B.numel() for m in factored_modules.values())
        results["stages"].append({
            "rank": rank,
            "loss_post_truncate": loss_pre,
            "ppl_post_truncate": ppl_pre,
            "loss_post_ft": loss_post,
            "ppl_post_ft": ppl_post,
            "delta_loss": loss_post - loss_base,
            "train_loss_start": train_losses[0] if train_losses else None,
            "train_loss_end": train_losses[-1] if train_losses else None,
            "n_factored_params": n_fac_params,
            "duration_s": dur,
        })

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        # Early stop
        if ppl_post > args.early_stop_ppl_ratio * ppl_base:
            print(f"\n  EARLY STOP: PPL {ppl_post:.2f} > {args.early_stop_ppl_ratio}× "
                  f"baseline {ppl_base:.2f}. Rank floor identified near {rank}.")
            break

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  baseline: PPL={ppl_base:.2f}")
    for s in results["stages"]:
        marker = " ✓ " if s["delta_loss"] < 0.1 else \
                 " ~ " if s["delta_loss"] < 0.4 else \
                 " ! " if s["delta_loss"] < 1.0 else "XXX"
        print(f"  rank {s['rank']:4d}: post-trunc PPL={s['ppl_post_truncate']:>7.1f}  "
              f"post-FT PPL={s['ppl_post_ft']:>7.1f}  Δloss={s['delta_loss']:+.3f}  {marker}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
