"""
Thermostat rank annealing: the rank follows the learning.

Instead of fixed schedule (drop rank every N steps), the rank
decreases ONLY when the model has recovered sufficiently from
the current compression. Like a thermostat:

- Temperature = rank (higher = more capacity)
- Target = teacher PPL
- When the model cools (PPL recovers to within threshold of
  teacher), lower the temperature (reduce rank by 1)
- If PPL spikes (model overheats), hold rank until recovery

The rank drops by 1 at a time, not by large jumps. Continuous,
smooth annealing — the model is always near its comfort zone.

Starting at full rank and walking down to the manifold dimension.
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


class AwareLowRankKV(nn.Module):
    """K or V projection factored as W_down @ W_up, both trainable.

    Supports dynamic rank reduction: drop the least important
    direction when told to compress.
    """
    def __init__(self, original_linear, rank):
        super().__init__()
        W = original_linear.weight.data.float()  # [out, in]
        out_f, in_f = W.shape
        self.in_features = in_f
        self.out_features = out_f

        # SVD init at current rank
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.W_down = nn.Parameter(Vh[:rank].clone())          # [rank, in]
        self.W_up = nn.Parameter((U[:, :rank] * S[:rank]).clone())  # [out, rank]
        self.rank = rank
        self.max_rank = rank

    def forward(self, x):
        # x @ W_down.T @ W_up.T = x @ (W_up @ W_down).T
        return F.linear(F.linear(x, self.W_down.to(x.dtype)), self.W_up.to(x.dtype))

    def reduce_rank(self):
        """Drop rank by 1: remove the least important direction."""
        if self.rank <= 1:
            return
        # Reconstruct current W, re-SVD, truncate by 1
        W_current = self.W_up.data @ self.W_down.data  # [out, in]
        U, S, Vh = torch.linalg.svd(W_current, full_matrices=False)
        new_rank = self.rank - 1
        self.W_down = nn.Parameter(Vh[:new_rank].clone())
        self.W_up = nn.Parameter((U[:, :new_rank] * S[:new_rank]).clone())
        self.rank = new_rank


def convert_kv_to_lowrank(model, init_rank):
    """Replace all k_proj and v_proj with AwareLowRankKV."""
    count = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ['k_proj', 'v_proj']:
            orig = getattr(attn, name)
            lr = AwareLowRankKV(orig, init_rank)
            setattr(attn, name, lr)
            count += 1
    return count


def get_all_lowrank_modules(model):
    """Get all AwareLowRankKV modules."""
    modules = []
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ['k_proj', 'v_proj']:
            mod = getattr(attn, name)
            if isinstance(mod, AwareLowRankKV):
                modules.append(mod)
    return modules


def reduce_all_ranks(model):
    """Reduce rank by 1 across all KV projections."""
    modules = get_all_lowrank_modules(model)
    for m in modules:
        m.reduce_rank()
    return modules[0].rank if modules else 0


def load_data(tokenizer, seq_len, max_tokens=2000000):
    """Load wikitext-2 train + val."""
    from datasets import load_dataset

    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize_and_chunk(ds, max_tok):
        text = "\n".join(ds["text"])
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        if max_tok:
            tokens = tokens[:max_tok]
        n = len(tokens) // (seq_len + 1)
        return tokens[:n * (seq_len + 1)].view(n, seq_len + 1)

    train_chunks = tokenize_and_chunk(ds_train, max_tokens)
    val_chunks = tokenize_and_chunk(ds_val, 50000)
    return train_chunks, val_chunks


@torch.inference_mode()
def eval_ppl(model, val_chunks, batch_size, device):
    model.eval()
    total_loss = 0
    n = 0
    for i in range(0, min(len(val_chunks), 100 * batch_size), batch_size):
        batch = val_chunks[i:i+batch_size].to(device)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    ce = total_loss / max(n, 1)
    return ce, math.exp(min(ce, 20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--init-rank", type=int, default=64,
                   help="Starting rank for KV projections")
    p.add_argument("--target-rank", type=int, default=8,
                   help="Target rank to anneal down to")
    p.add_argument("--recovery-threshold", type=float, default=1.5,
                   help="PPL ratio (student/teacher) below which rank drops")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--patience", type=int, default=100,
                   help="Steps without improvement before rank drop is forced")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="machines/z8g4/results/thermostat_anneal.json")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Thermostat Rank Annealing", flush=True)
    print(f"  init_rank={args.init_rank} → target_rank={args.target_rank}")
    print(f"  recovery_threshold={args.recovery_threshold}× teacher PPL")
    print(f"  device={device}", flush=True)

    # Tokenizer + data
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("\nLoading data...", flush=True)
    train_chunks, val_chunks = load_data(tokenizer, args.seq_len)
    print(f"  train: {len(train_chunks)} chunks, val: {len(val_chunks)} chunks")

    # Teacher baseline
    print("\nLoading teacher...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

    teacher_ce, teacher_ppl = eval_ppl(model, val_chunks, args.batch_size, device)
    print(f"  teacher: ce={teacher_ce:.4f} ppl={teacher_ppl:.2f}", flush=True)

    # Convert KV to low-rank
    print(f"\nConverting KV to rank {args.init_rank}...", flush=True)
    n_converted = convert_kv_to_lowrank(model, args.init_rank)
    print(f"  {n_converted} projections converted")

    # Check init quality
    init_ce, init_ppl = eval_ppl(model, val_chunks, args.batch_size, device)
    ppl_ratio = init_ppl / teacher_ppl
    print(f"  init: ce={init_ce:.4f} ppl={init_ppl:.2f} ratio={ppl_ratio:.2f}×", flush=True)

    # Optimizer — only train the low-rank parameters
    lr_params = []
    for m in get_all_lowrank_modules(model):
        lr_params.extend([m.W_down, m.W_up])
    optimizer = torch.optim.AdamW(lr_params, lr=args.lr, weight_decay=0.01)

    # Thermostat loop
    current_rank = args.init_rank
    history = []
    step = 0
    best_ppl_at_rank = float('inf')
    steps_since_improvement = 0
    train_idx = 0
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"THERMOSTAT ANNEALING: rank {args.init_rank} → {args.target_rank}")
    print(f"{'='*70}", flush=True)

    model.train()
    while step < args.max_steps and current_rank > args.target_rank:
        # Get batch
        if train_idx + args.batch_size > len(train_chunks):
            train_idx = 0
        batch = train_chunks[train_idx:train_idx + args.batch_size].to(device)
        train_idx += args.batch_size

        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lr_params, 1.0)
        optimizer.step()

        step += 1

        # Evaluate
        if step % args.eval_every == 0:
            val_ce, val_ppl = eval_ppl(model, val_chunks, args.batch_size, device)
            ppl_ratio = val_ppl / teacher_ppl
            elapsed = time.time() - t0

            entry = {
                "step": step, "rank": current_rank,
                "val_ce": val_ce, "val_ppl": val_ppl,
                "ppl_ratio": ppl_ratio, "train_loss": loss.item(),
                "elapsed": elapsed,
            }
            history.append(entry)

            improved = val_ppl < best_ppl_at_rank
            if improved:
                best_ppl_at_rank = val_ppl
                steps_since_improvement = 0
            else:
                steps_since_improvement += args.eval_every

            # Thermostat decision
            should_drop = False
            reason = ""

            if ppl_ratio <= args.recovery_threshold:
                should_drop = True
                reason = f"recovered to {ppl_ratio:.2f}× teacher"
            elif steps_since_improvement >= args.patience:
                should_drop = True
                reason = f"patience exhausted ({args.patience} steps)"

            status = f"step {step:5d} | rank {current_rank:3d} | ppl {val_ppl:8.2f} | ratio {ppl_ratio:.3f}× | loss {loss.item():.4f}"

            if should_drop:
                current_rank = reduce_all_ranks(model)
                # Re-create optimizer for new parameters
                lr_params = []
                for m in get_all_lowrank_modules(model):
                    lr_params.extend([m.W_down, m.W_up])
                optimizer = torch.optim.AdamW(lr_params, lr=args.lr, weight_decay=0.01)
                best_ppl_at_rank = float('inf')
                steps_since_improvement = 0
                print(f"  {status} | ▼ RANK DROP → {current_rank} ({reason})", flush=True)
            else:
                marker = "↑" if improved else "→"
                print(f"  {status} | {marker}", flush=True)

    # Final eval
    final_ce, final_ppl = eval_ppl(model, val_chunks, args.batch_size, device)
    final_ratio = final_ppl / teacher_ppl

    print(f"\n{'='*70}")
    print(f"THERMOSTAT RESULT")
    print(f"{'='*70}")
    print(f"  Teacher PPL:    {teacher_ppl:.2f}")
    print(f"  Final PPL:      {final_ppl:.2f}")
    print(f"  Final ratio:    {final_ratio:.2f}×")
    print(f"  Final rank:     {current_rank}")
    print(f"  Init rank:      {args.init_rank}")
    print(f"  Compression:    {args.init_rank}→{current_rank} ({current_rank/args.init_rank*100:.0f}% of init)")
    print(f"  Total steps:    {step}")
    print(f"  Total time:     {time.time()-t0:.0f}s")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "teacher_ppl": teacher_ppl,
            "init_rank": args.init_rank,
            "final_rank": current_rank,
            "target_rank": args.target_rank,
            "final_ppl": final_ppl,
            "final_ratio": final_ratio,
            "recovery_threshold": args.recovery_threshold,
            "total_steps": step,
            "history": history,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
