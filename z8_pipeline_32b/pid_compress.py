"""
PID-Controlled Compression Framework.

Applies one compression axis at a time with PID control that rides
the quality line at exactly target% above teacher PPL.

PID controller:
  - Setpoint: teacher_ppl * (1 + target_pct/100)
  - Process variable: current model PPL after eval
  - Control output: compression step size (how aggressively to compress)
  - P: react to current error
  - I: accumulated error prevents drift
  - D: rate of change prevents oscillation

Each axis is tested independently on 4B:
  1. Measure teacher baseline
  2. Apply axis at increasing levels with PID control
  3. Fine-tune between compression steps
  4. Find: free zone (quality improves), sweet spot, wall (quality breaks)
  5. Record everything

Usage:
  python pid_compress.py --axis k_rank
  python pid_compress.py --axis v_rank
  python pid_compress.py --axis weight_bits
  python pid_compress.py --axis mlp_width
  python pid_compress.py --axis q_heads
  python pid_compress.py --axis magnitude
  python pid_compress.py --axis mlp_rank
  etc.
"""

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── PID Controller ───────────────────────────────────────────────────

class PIDController:
    """PID controller for compression rate."""
    def __init__(self, setpoint, kp=1.0, ki=0.1, kd=0.3,
                 min_output=0.001, max_output=0.10):
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output  # minimum compression step (0.1%)
        self.max_output = max_output  # maximum compression step (10%)
        self.integral = 0.0
        self.prev_error = 0.0
        self.history = []

    def update(self, current_ppl):
        """Returns compression rate (0 to max_output).
        Positive = room to compress. Zero = stop."""
        error = self.setpoint - current_ppl  # positive = quality is good, compress more

        self.integral += error
        # Anti-windup: clamp integral
        self.integral = max(-20, min(20, self.integral))

        derivative = error - self.prev_error
        self.prev_error = error

        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        # Normalize: setpoint-relative
        output = output / max(self.setpoint, 1.0)

        # Clamp
        output = max(0.0, min(self.max_output, output))

        self.history.append({
            "ppl": current_ppl, "error": error,
            "p": self.kp * error, "i": self.ki * self.integral,
            "d": self.kd * derivative, "output": output,
        })

        return output


# ── Data ─────────────────────────────────────────────────────────────

def load_data(tokenizer, seq_len=256, max_train=2_000_000, max_val=100_000):
    from datasets import load_dataset
    print("  Loading OpenWebText...", flush=True)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts = []
    count = 0
    for ex in ds:
        texts.append(ex["text"])
        count += len(ex["text"]) // 4
        if count >= (max_train + max_val) * 1.2:
            break
    all_text = "\n\n".join(texts)
    tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]

    val_tokens = tokens[:max_val]
    train_tokens = tokens[max_val:max_val + max_train]

    def chunk(toks):
        n = len(toks) // (seq_len + 1)
        return toks[:n * (seq_len + 1)].view(n, seq_len + 1)

    train = chunk(train_tokens)
    val = chunk(val_tokens)
    print(f"  Train: {len(train)} chunks  Val: {len(val)} chunks")
    return train, val


@torch.inference_mode()
def eval_ppl(model, val_chunks, seq_len=256, n_eval=30):
    model.eval()
    total = 0
    n = 0
    for i in range(min(n_eval, len(val_chunks))):
        inp = val_chunks[i:i+1, :seq_len]
        tgt = val_chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1))
        total += loss.item()
        n += 1
    ce = total / max(n, 1)
    return math.exp(min(ce, 20))


def train_steps(model, train_chunks, trainable_params, n_steps, lr, seq_len=256):
    """Quick fine-tune for n_steps."""
    model.train()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    indices = list(range(len(train_chunks)))
    random.shuffle(indices)
    idx_iter = iter(indices)
    total_loss = 0

    for step in range(n_steps):
        try:
            idx = next(idx_iter)
        except StopIteration:
            random.shuffle(indices)
            idx_iter = iter(indices)
            idx = next(idx_iter)

        batch = train_chunks[idx:idx+1]
        inp = batch[:, :seq_len]
        tgt = batch[:, 1:seq_len+1]

        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    return total_loss / max(n_steps, 1)


# ── Compression Axes ─────────────────────────────────────────────────

ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]


class FactoredLinear(nn.Module):
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(bias) if bias is not None else None

    @property
    def rank(self):
        return self.A.shape[1]

    def forward(self, x):
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def reduce_rank(self, new_rank):
        if new_rank >= self.rank:
            return
        W = self.A.data @ self.B.data
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(new_rank, len(S))
        sqrt_S = S[:k].sqrt()
        self.A = nn.Parameter((U[:, :k] * sqrt_S).contiguous())
        self.B = nn.Parameter((sqrt_S.unsqueeze(1) * Vt[:k]).contiguous())


def get_proj(model, l, name):
    layer = model.model.layers[l]
    if name in ATTN_PROJS:
        return getattr(layer.self_attn, name)
    return getattr(layer.mlp, name)


def set_proj(model, l, name, mod):
    layer = model.model.layers[l]
    if name in ATTN_PROJS:
        setattr(layer.self_attn, name, mod)
    else:
        setattr(layer.mlp, name, mod)


def factorize_proj(model, l, name, rank):
    proj = get_proj(model, l, name)
    if isinstance(proj, FactoredLinear):
        proj.reduce_rank(rank)
    else:
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, len(S))
        sqrt_S = S[:k].sqrt()
        A = (U[:, :k] * sqrt_S).contiguous()
        B = (sqrt_S.unsqueeze(1) * Vt[:k]).contiguous()
        bias = proj.bias.data if proj.bias is not None else None
        fac = FactoredLinear(A, B, bias)
        set_proj(model, l, name, fac)


def collect_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for layer in model.model.layers:
        for sub in [layer.self_attn, layer.mlp]:
            for name, mod in sub.named_children():
                if isinstance(mod, FactoredLinear):
                    mod.A.requires_grad_(True)
                    mod.B.requires_grad_(True)
                    trainable.extend([mod.A, mod.B])
                    if mod.bias is not None:
                        mod.bias.requires_grad_(True)
                        trainable.append(mod.bias)
    # Also train norms (important for magnitude axis)
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


# ── Axis implementations ─────────────────────────────────────────────

def apply_k_rank(model, level, L):
    """Compress K projections. level = fraction to remove (0.0 to 1.0)."""
    for l in range(L):
        proj = get_proj(model, l, "k_proj")
        if isinstance(proj, FactoredLinear):
            max_r = proj.A.shape[0]  # out features
        else:
            max_r = min(proj.weight.shape)
        new_rank = max(1, int(max_r * (1.0 - level)))
        factorize_proj(model, l, "k_proj", new_rank)
    return collect_trainable(model)


def apply_v_rank(model, level, L):
    """Compress V projections."""
    for l in range(L):
        proj = get_proj(model, l, "v_proj")
        if isinstance(proj, FactoredLinear):
            max_r = proj.A.shape[0]
        else:
            max_r = min(proj.weight.shape)
        new_rank = max(1, int(max_r * (1.0 - level)))
        factorize_proj(model, l, "v_proj", new_rank)
    return collect_trainable(model)


def apply_q_rank(model, level, L):
    """Compress Q projections."""
    for l in range(L):
        proj = get_proj(model, l, "q_proj")
        if isinstance(proj, FactoredLinear):
            max_r = proj.A.shape[0]
        else:
            max_r = min(proj.weight.shape)
        new_rank = max(1, int(max_r * (1.0 - level)))
        factorize_proj(model, l, "q_proj", new_rank)
    return collect_trainable(model)


def apply_o_rank(model, level, L):
    """Compress O projections."""
    for l in range(L):
        proj = get_proj(model, l, "o_proj")
        if isinstance(proj, FactoredLinear):
            max_r = proj.A.shape[0]
        else:
            max_r = min(proj.weight.shape)
        new_rank = max(1, int(max_r * (1.0 - level)))
        factorize_proj(model, l, "o_proj", new_rank)
    return collect_trainable(model)


def apply_mlp_rank(model, level, L):
    """Compress MLP projections (gate, up, down)."""
    for l in range(L):
        for name in MLP_PROJS:
            proj = get_proj(model, l, name)
            if isinstance(proj, FactoredLinear):
                max_r = proj.A.shape[0]
            else:
                max_r = min(proj.weight.shape)
            new_rank = max(1, int(max_r * (1.0 - level)))
            factorize_proj(model, l, name, new_rank)
    return collect_trainable(model)


def apply_magnitude(model, level, L):
    """Scale all weights down by level fraction."""
    scale = 1.0 - level
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "weight" in name and "norm" not in name.lower() and "embed" not in name.lower():
                p.mul_(scale)
    # Train norms to absorb
    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad_(False)
        if "norm" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def apply_mlp_width(model, level, L):
    """Prune MLP rows by zeroing least important."""
    n_pruned_total = 0
    with torch.no_grad():
        for l in range(L):
            gate = get_proj(model, l, "gate_proj")
            up = get_proj(model, l, "up_proj")
            if isinstance(gate, FactoredLinear):
                continue  # skip if already factored
            # Importance = L2 norm of gate rows
            importance = gate.weight.data.norm(dim=1)
            n_prune = int(len(importance) * level)
            if n_prune == 0:
                continue
            _, indices = importance.topk(n_prune, largest=False)
            gate.weight.data[indices] = 0
            up.weight.data[indices] = 0
            n_pruned_total += n_prune
    # Train remaining weights
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "norm" in name.lower() or "mlp" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def apply_q_heads(model, level, L):
    """Zero out Q attention heads."""
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = d_model // n_heads
    n_prune = int(n_heads * level)
    if n_prune == 0:
        return []
    with torch.no_grad():
        for l in range(L):
            q = get_proj(model, l, "q_proj")
            o = get_proj(model, l, "o_proj")
            if isinstance(q, FactoredLinear):
                continue
            # Importance: norm of each head's q weights
            q_heads = q.weight.data.view(n_heads, head_dim, -1)
            importance = q_heads.norm(dim=(1, 2))
            _, idx = importance.topk(n_prune, largest=False)
            for h in idx:
                q.weight.data[h*head_dim:(h+1)*head_dim] = 0
                o.weight.data[:, h*head_dim:(h+1)*head_dim] = 0
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def apply_norm_squash(model, level, L):
    """Push all RMSNorm scale weights toward 1.0 (identity).
    level=0: original norms. level=1: all norms = 1.0.
    Tests if learned per-channel scaling carries information."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name.lower() and "weight" in name:
                # Interpolate toward ones: w = (1-level)*w + level*1.0
                p.data.mul_(1.0 - level).add_(level)
    # Train norms back
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


def apply_embed_rank(model, level, L):
    """SVD on embedding matrix. level = fraction of rank to remove.
    Embedding is [vocab_size, d_model] — often 151K x 2560 for 4B.
    If low-rank, massive savings on vocab table."""
    embed = model.model.embed_tokens
    W = embed.weight.data.float()  # [vocab, d_model]
    max_r = min(W.shape)
    new_rank = max(1, int(max_r * (1.0 - level)))
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    # Reconstruct at lower rank
    W_approx = (U[:, :new_rank] * S[:new_rank]) @ Vt[:new_rank]
    embed.weight.data.copy_(W_approx)
    # Also compress lm_head if tied or separate
    if hasattr(model, 'lm_head') and model.lm_head.weight.data_ptr() != embed.weight.data_ptr():
        model.lm_head.weight.data.copy_(W_approx)
    # Train embedding
    trainable = []
    for p in model.parameters():
        p.requires_grad_(False)
    embed.weight.requires_grad_(True)
    trainable.append(embed.weight)
    if hasattr(model, 'lm_head') and model.lm_head.weight.data_ptr() != embed.weight.data_ptr():
        model.lm_head.weight.requires_grad_(True)
        trainable.append(model.lm_head.weight)
    # Also norms
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


AXIS_FUNCTIONS = {
    "k_rank": apply_k_rank,
    "v_rank": apply_v_rank,
    "q_rank": apply_q_rank,
    "o_rank": apply_o_rank,
    "mlp_rank": apply_mlp_rank,
    "magnitude": apply_magnitude,
    "mlp_width": apply_mlp_width,
    "q_heads": apply_q_heads,
    "norm_squash": apply_norm_squash,
    "embed_rank": apply_embed_rank,
}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=list(AXIS_FUNCTIONS.keys()))
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--target-pct", type=float, default=5.0,
                   help="Target PPL as pct above teacher (5 = within 5%%)")
    p.add_argument("--max-level", type=float, default=0.95,
                   help="Max compression level (0.95 = remove 95%%)")
    p.add_argument("--ft-steps", type=int, default=100,
                   help="Fine-tune steps between compression steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--save-dir", default="z8_pipeline_32b/pid_results")
    args = p.parse_args()

    torch.set_num_threads(32)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"PID COMPRESSION: {args.axis} on {args.model}")
    print(f"  Target: {args.target_pct}% above teacher")
    print(f"  Fine-tune: {args.ft_steps} steps between compressions")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\nLoading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()
    L = model.config.num_hidden_layers
    print(f"  L={L}, d={model.config.hidden_size}", flush=True)

    print("\nLoading data...", flush=True)
    train_chunks, val_chunks = load_data(tokenizer)

    print("\nTeacher baseline...", flush=True)
    teacher_ppl = eval_ppl(model, val_chunks)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # PID setup
    setpoint = teacher_ppl * (1.0 + args.target_pct / 100.0)
    pid = PIDController(setpoint, kp=0.8, ki=0.05, kd=0.2,
                        min_output=0.005, max_output=0.05)
    print(f"  PID setpoint: {setpoint:.2f} ({args.target_pct}% above teacher)")

    apply_fn = AXIS_FUNCTIONS[args.axis]
    level = 0.0  # current compression level (0 = none, 1 = max)
    best_ppl = teacher_ppl
    best_level = 0.0
    free_zone_end = None
    wall_level = None

    results = {
        "axis": args.axis, "model": args.model,
        "teacher_ppl": teacher_ppl, "setpoint": setpoint,
        "target_pct": args.target_pct,
        "history": [],
    }

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"PID LOOP: compressing {args.axis}")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Level':>6} | {'PPL':>8} | {'Ratio':>6} | {'PID out':>7} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*10}", flush=True)

    for round_num in range(1, args.max_rounds + 1):
        # Eval current state
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl

        # PID update
        compress_rate = pid.update(ppl)

        # Track free zone and wall
        if ppl <= teacher_ppl and free_zone_end is None:
            pass  # still in free zone
        elif ppl > teacher_ppl and free_zone_end is None:
            free_zone_end = level

        if ppl > setpoint * 1.5 and wall_level is None:
            wall_level = level

        if ppl < best_ppl:
            best_ppl = ppl
            best_level = level

        # Status
        if ppl <= teacher_ppl:
            status = "FREE (improves)"
        elif ratio <= 1.0 + args.target_pct / 100:
            status = "ON TARGET"
        elif ratio <= 1.0 + args.target_pct * 2 / 100:
            status = "RECOVERING"
        else:
            status = "OVER BUDGET"

        print(f"  {round_num:5d} | {level:5.1%} | {ppl:8.2f} | {ratio:5.2f}x | {compress_rate:6.3f} | {status}",
              flush=True)

        results["history"].append({
            "round": round_num, "level": round(level, 4),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "pid_output": round(compress_rate, 4), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        })

        # Save incrementally
        rpath = Path(args.save_dir) / f"pid_{args.axis}.json"
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2)

        # Check termination
        if level >= args.max_level:
            print(f"\n  Reached max level {args.max_level}", flush=True)
            break
        if wall_level is not None and level > wall_level + 0.1:
            print(f"\n  Past wall ({wall_level:.1%}), stopping", flush=True)
            break
        if compress_rate <= 0.001 and ppl > setpoint:
            print(f"\n  PID output near zero, quality unrecoverable", flush=True)
            break

        # Apply compression
        new_level = min(args.max_level, level + compress_rate)
        if new_level > level:
            # Reload and recompress at new level (clean application)
            del model
            gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32,
                low_cpu_mem_usage=True, trust_remote_code=True,
                attn_implementation="eager").eval()
            trainable = apply_fn(model, new_level, L)
            level = new_level

            # Fine-tune
            if trainable:
                avg_loss = train_steps(model, train_chunks, trainable,
                                       args.ft_steps, args.lr)
        else:
            # Just fine-tune more at current level
            trainable = collect_trainable(model)
            if trainable:
                avg_loss = train_steps(model, train_chunks, trainable,
                                       args.ft_steps, args.lr)

    # Summary
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"RESULT: {args.axis}")
    print(f"{'='*60}")
    print(f"  Teacher PPL:  {teacher_ppl:.2f}")
    print(f"  Best PPL:     {best_ppl:.2f} at level {best_level:.1%}")
    print(f"  Free zone:    0 - {free_zone_end:.1%}" if free_zone_end else "  Free zone:    entire range")
    print(f"  Wall:         {wall_level:.1%}" if wall_level else "  Wall:         not reached")
    print(f"  Final level:  {level:.1%}")
    print(f"  Rounds:       {round_num}")
    print(f"  Time:         {elapsed_h:.2f}h")

    results["best_ppl"] = best_ppl
    results["best_level"] = best_level
    results["free_zone_end"] = free_zone_end
    results["wall_level"] = wall_level
    results["final_level"] = level
    results["elapsed_h"] = elapsed_h

    rpath = Path(args.save_dir) / f"pid_{args.axis}.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {rpath}")


if __name__ == "__main__":
    main()
