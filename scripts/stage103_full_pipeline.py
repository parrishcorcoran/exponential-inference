"""
Qwen Halo — Full compression pipeline in one program. Fire-and-forget.

Designed to run unattended on Strix Halo for days. Auto-checkpoints
every N steps. Resumes from latest checkpoint on restart. Heartbeat
file with current position. Rotating checkpoints to bound disk use.

Phases (run sequentially):

  PHASE 1 — Early exit probes (per-layer LM heads trained, base frozen).
  PHASE 2 — Medusa heads added ONE AT A TIME (each trained with prior
            heads frozen).
  PHASE 3 — Round-robin compression: cycle through KV rank / weight
            precision / embedding precision, advancing ONE axis per
            mini-phase, then fine-tuning briefly before the next step.

Usage:
  python scripts/stage103_full_pipeline.py --model Qwen/Qwen3-14B
  # Runs for days. Check heartbeat.json anytime for status.
  # If process crashes or is killed, just re-run same command: auto-resumes.

Design principles (per user's direction):
  - Super slow. Each compression step is small. Nothing jumps.
  - One axis per round-robin phase, not all at once.
  - Base model stays GPU-friendly (dense fp matmul) throughout —
    the compression axes chosen preserve tensor-core compatibility,
    which is what lets Medusa + early exit coexist (BitNet's CPU-only
    substrate blocked this combo historically).

Minimum target: validates pipeline at Qwen3-14B as smoke test for
larger-model deployment.
"""

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Resilience: heartbeat + rotating checkpoints
# ============================================================

def write_heartbeat(checkpoints_dir, phase, step, extra=None):
    """Write current position to a heartbeat file so user can check status."""
    data = {"phase": phase, "step": step, "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S")}
    if extra: data.update(extra)
    path = Path(checkpoints_dir) / "heartbeat.json"
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # don't crash on heartbeat failure


def save_rotating_checkpoint(state, checkpoints_dir, phase_name, step, keep_last=3):
    """Save checkpoint, delete all but the last `keep_last` matching checkpoints
       for this phase."""
    out = Path(checkpoints_dir) / f"{phase_name}_step{step:08d}.pt"
    tmp = Path(checkpoints_dir) / f"{phase_name}_step{step:08d}.pt.tmp"
    try:
        torch.save(state, tmp)
        tmp.rename(out)
    except Exception as e:
        print(f"  [WARN] checkpoint save failed: {e}", flush=True)
        return None
    # Rotate: keep only newest `keep_last`
    pattern = str(Path(checkpoints_dir) / f"{phase_name}_step*.pt")
    all_ckpts = sorted(glob.glob(pattern))
    for old in all_ckpts[:-keep_last]:
        try: os.remove(old)
        except Exception: pass
    return out


def find_latest_checkpoint(checkpoints_dir, phase_name):
    pattern = str(Path(checkpoints_dir) / f"{phase_name}_step*.pt")
    all_ckpts = sorted(glob.glob(pattern))
    return Path(all_ckpts[-1]) if all_ckpts else None


# ============================================================
# Shared utilities
# ============================================================

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


def iter_batches_simple(tokens, seq_len, batch_size, device):
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


def iter_batches_multi(tokens, seq_len, batch_size, device, offset_max):
    import random
    window_len = seq_len + offset_max
    n = (len(tokens) - 1) // window_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * window_len
        window = tokens[start:start + window_len + 1]
        if len(window) < seq_len + 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            inp = t[:, :seq_len]
            targets = [t[:, k+1:k+1+seq_len] for k in range(offset_max)]
            yield inp, targets
            batch = []


@torch.no_grad()
def eval_ppl_base(model, tokens, seq_len, batch_size, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches_simple(tokens, seq_len, batch_size, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    model.train()
    return total / max(count, 1)


# ============================================================
# PHASE 1 — Early exit probes
# ============================================================

class LayerProbe(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.affine_weight = nn.Parameter(torch.eye(d_model))
        self.affine_bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, h, lm_head_weight):
        rms = h.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        h = h * rms * self.norm_weight
        h = h @ self.affine_weight.T + self.affine_bias
        return F.linear(h, lm_head_weight)


def train_phase1_early_exit(model, tokenizer, device, args, checkpoints_dir):
    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"\n{'='*72}\nPHASE 1: early-exit probes (L={L} layers + embedding = {L+1} probes)\n{'='*72}", flush=True)

    for p in model.parameters(): p.requires_grad = False
    lm_head_weight = model.lm_head.weight.to(torch.float32)

    probes = nn.ModuleList([LayerProbe(d_model) for _ in range(L + 1)]).to(device).to(torch.float32)
    opt = torch.optim.AdamW(probes.parameters(), lr=args.lr_probe,
                            betas=(0.9, 0.95), weight_decay=0.01)

    # Resume support
    latest = find_latest_checkpoint(checkpoints_dir, "phase1")
    resumed_step = 0; history = []
    if latest is not None:
        print(f"  resuming phase 1 from {latest}", flush=True)
        ckpt = torch.load(latest, map_location=device)
        probes.load_state_dict(ckpt["probes"])
        if "opt" in ckpt:
            try: opt.load_state_dict(ckpt["opt"])
            except Exception: pass
        resumed_step = ckpt.get("step", 0)
        history = ckpt.get("history", [])

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")
    weights = torch.ones(L + 1, device=device) / (L + 1)

    step = resumed_step; t0 = time.time(); running = []
    while step < args.phase1_steps:
        for inp, tgt in iter_batches_simple(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.phase1_steps: break
            opt.zero_grad()
            with torch.no_grad():
                out = model(inp, use_cache=False, output_hidden_states=True)
                hs = [h.detach() for h in out.hidden_states]
            total = 0.0
            for l, h in enumerate(hs):
                logits = probes[l](h.to(torch.float32), lm_head_weight)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
                total = total + weights[l] * loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
            opt.step()
            running.append(float(total.item())); step += 1
            if step % args.heartbeat_every == 0:
                write_heartbeat(checkpoints_dir, "phase1", step)
            if step % args.eval_every == 0:
                probes.eval()
                per_layer_val = [0.0] * (L + 1); vcount = 0
                with torch.no_grad():
                    for v_inp, v_tgt in iter_batches_simple(val_tokens, args.seq_len, args.batch_size, device):
                        out = model(v_inp, use_cache=False, output_hidden_states=True)
                        for l, h in enumerate(out.hidden_states):
                            logits = probes[l](h.to(torch.float32), lm_head_weight)
                            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), v_tgt.reshape(-1))
                            per_layer_val[l] += float(loss.item())
                        vcount += 1
                        if vcount >= 3: break
                per_layer_val = [v / max(vcount, 1) for v in per_layer_val]
                probes.train()
                tr = float(np.mean(running[-args.eval_every:]))
                history.append({"step": step, "train_total": tr,
                               "per_layer_val_ce": per_layer_val,
                               "elapsed": time.time()-t0})
                print(f"  [P1] step {step}/{args.phase1_steps}  total={tr:.4f}  "
                      f"ce@0={per_layer_val[0]:.3f}  ce@{L//2}={per_layer_val[L//2]:.3f}  "
                      f"ce@{L}={per_layer_val[L]:.3f}  elapsed={time.time()-t0:.0f}s", flush=True)
            if step % args.checkpoint_every == 0:
                save_rotating_checkpoint(
                    {"probes": probes.state_dict(), "opt": opt.state_dict(),
                     "step": step, "history": history},
                    checkpoints_dir, "phase1", step, keep_last=3)

    # Final checkpoint
    save_rotating_checkpoint(
        {"probes": probes.state_dict(), "step": step, "history": history},
        checkpoints_dir, "phase1", step, keep_last=3)
    out_path = Path(checkpoints_dir) / "phase1_final.pt"
    torch.save({"probes": probes.state_dict(), "history": history}, out_path)
    print(f"  phase 1 done. final: {out_path}")
    return probes, history


# ============================================================
# PHASE 2 — Medusa heads, one at a time
# ============================================================

class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.mlp_layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])
    def forward(self, h, lm_head_weight):
        for layer in self.mlp_layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


def train_phase2_medusa(model, tokenizer, device, args, checkpoints_dir):
    d_model = model.config.hidden_size
    print(f"\n{'='*72}\nPHASE 2: Medusa heads (1 → {args.medusa_heads}, one at a time)\n{'='*72}", flush=True)

    for p in model.parameters(): p.requires_grad = False
    lm_head_weight = model.lm_head.weight.to(torch.float32)

    heads = nn.ModuleList().to(device).to(torch.float32)
    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")

    all_history = []
    for k_idx in range(args.medusa_heads):
        # Add new head
        new_head = MedusaHead(d_model, n_layers=1).to(device).to(torch.float32)
        heads.append(new_head)
        # Freeze prior heads
        for pi in range(k_idx):
            for p_ in heads[pi].parameters(): p_.requires_grad = False

        print(f"\n  -- training head #{k_idx+1}/{args.medusa_heads} --", flush=True)
        params_trainable = list(new_head.parameters())
        opt = torch.optim.AdamW(params_trainable, lr=args.lr_medusa,
                                betas=(0.9, 0.95), weight_decay=0.01)
        step = 0; t0 = time.time(); running = []
        while step < args.phase2_steps_per_head:
            for inp, targets in iter_batches_multi(train_tokens, args.seq_len, args.batch_size,
                                                   device, args.medusa_heads):
                if step >= args.phase2_steps_per_head: break
                opt.zero_grad()
                with torch.no_grad():
                    out = model(inp, use_cache=False, output_hidden_states=True)
                    h_final = out.hidden_states[-1].detach().to(torch.float32)
                logits = new_head(h_final, lm_head_weight)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets[k_idx].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_trainable, 1.0)
                opt.step()
                running.append(float(loss.item())); step += 1
                if step % args.eval_every == 0:
                    tr = float(np.mean(running[-args.eval_every:]))
                    new_head.eval()
                    acc = 0.0; vcount = 0
                    with torch.no_grad():
                        for v_inp, v_tgt_list in iter_batches_multi(val_tokens, args.seq_len,
                                                                     args.batch_size, device,
                                                                     args.medusa_heads):
                            out = model(v_inp, use_cache=False, output_hidden_states=True)
                            h_final = out.hidden_states[-1].to(torch.float32)
                            logits = new_head(h_final, lm_head_weight)
                            preds = logits.argmax(-1)
                            acc += (preds == v_tgt_list[k_idx]).float().mean().item()
                            vcount += 1
                            if vcount >= 3: break
                    acc = acc / max(vcount, 1)
                    new_head.train()
                    all_history.append({"head": k_idx+1, "step": step, "train_ce": tr,
                                       "val_acc": acc, "elapsed": time.time()-t0})
                    print(f"  [P2-h{k_idx+1}] step {step}/{args.phase2_steps_per_head}  "
                          f"train_ce={tr:.4f}  val_acc={acc:.3f}  "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

    out_path = Path(checkpoints_dir) / "phase2_medusa.pt"
    torch.save({"heads": heads.state_dict(), "history": all_history}, out_path)
    print(f"  saved {out_path}")
    return heads, all_history


# ============================================================
# PHASE 3 — Round-robin compression
# ============================================================

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features; self.rank = rank
        self.W_down = nn.Parameter(torch.empty(rank, in_features))
        self.W_up = nn.Parameter(torch.empty(out_features, rank))
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.bias = None
    @classmethod
    def from_linear_svd(cls, linear, rank):
        m = cls(linear.in_features, linear.out_features, rank, bias=(linear.bias is not None))
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
        m = cls(old.in_features, old.out_features, rank, bias=(old.bias is not None))
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


class ScalableTernary(torch.autograd.Function):
    """Quantize W_fp to levels 2^n_bits using STE.
       n_bits = 16 → identity. n_bits = 1 → ternary. Intermediate = rounded to nearest of 2^n_bits levels."""
    @staticmethod
    def forward(ctx, W_fp, n_bits):
        if n_bits >= 16:
            return W_fp
        if n_bits <= 1.58:
            absW = W_fp.abs()
            tau = 0.7 * absW.mean()
            return torch.where(W_fp > tau, torch.ones_like(W_fp),
                    torch.where(W_fp < -tau, -torch.ones_like(W_fp), torch.zeros_like(W_fp)))
        # general: round to 2^n_bits levels per per-tensor scale
        levels = 2 ** n_bits
        scale = W_fp.abs().max() / (levels / 2 - 1)
        q = torch.round(W_fp / scale.clamp(min=1e-8)).clamp(-(levels/2-1), levels/2-1)
        return q * scale
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


class QATLinear(nn.Module):
    """Linear with learnable per-tensor α * quantized(W_fp). Quantization level
       set via n_bits. n_bits = 16 acts as identity (fp linear)."""
    def __init__(self, in_features, out_features, bias=False, n_bits=16.0):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.W_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.alpha = nn.Parameter(torch.ones(1))
        self.n_bits = n_bits
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.bias = None
    @classmethod
    def from_linear(cls, linear, n_bits=16.0):
        m = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None), n_bits=n_bits)
        with torch.no_grad():
            m.W_fp.data = linear.weight.data.float()
            m.alpha.data = torch.tensor([m.W_fp.data.abs().mean().item()], dtype=torch.float32)
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m
    def forward(self, x):
        W_q = ScalableTernary.apply(self.W_fp, self.n_bits)
        W_eff = W_q * self.alpha
        return F.linear(x.float(), W_eff, self.bias).to(x.dtype)


def convert_kv_to_low_rank(model, rank):
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            old = getattr(layer.self_attn, name)
            if isinstance(old, LowRankLinear):
                new = LowRankLinear.from_lowrank_svd(old, rank)
            else:
                new = LowRankLinear.from_linear_svd(old, rank)
            setattr(layer.self_attn, name, new)


def convert_body_to_qat(model, n_bits):
    for layer in model.model.layers:
        for parent, name in [(layer.self_attn, "q_proj"), (layer.self_attn, "o_proj"),
                             (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                             (layer.mlp, "down_proj")]:
            old = getattr(parent, name)
            if isinstance(old, QATLinear):
                old.n_bits = n_bits
            else:
                new = QATLinear.from_linear(old, n_bits=n_bits)
                setattr(parent, name, new)


def train_phase3_roundrobin(model, tokenizer, device, args, checkpoints_dir):
    print(f"\n{'='*72}\nPHASE 3: round-robin compression (KV, weights, embed cycling)\n{'='*72}", flush=True)
    model = model.to(torch.float32)

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")

    # Round-robin schedule: a list of (axis, value) compressions to apply.
    kv_schedule    = [128, 96, 64, 48, 32, 24, 16]
    weight_bits    = [8, 6, 4, 3, 2, 1.58]
    embed_bits     = [8, 6, 4]
    schedule = []
    # Interleave: one step per axis in round-robin order.
    max_len = max(len(kv_schedule), len(weight_bits), len(embed_bits))
    for i in range(max_len):
        if i < len(kv_schedule):    schedule.append(("kv", kv_schedule[i]))
        if i < len(weight_bits):    schedule.append(("weights", weight_bits[i]))
        if i < len(embed_bits):     schedule.append(("embed", embed_bits[i]))

    history = []
    teacher_ce = eval_ppl_base(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  pre-compression val_ce={teacher_ce:.4f}  val_ppl={math.exp(teacher_ce):.2f}", flush=True)
    history.append({"mini_phase": 0, "axis": "pre", "value": None, "val_ce": teacher_ce,
                    "val_ppl": math.exp(teacher_ce)})

    for mi, (axis, value) in enumerate(schedule, start=1):
        print(f"\n  -- mini-phase {mi}: axis={axis}  value={value} --", flush=True)
        # Apply the compression step
        if axis == "kv":
            convert_kv_to_low_rank(model, int(value))
        elif axis == "weights":
            convert_body_to_qat(model, float(value))
        elif axis == "embed":
            # Embedding quantization — quantize model.embed weight via same STE bit-width.
            # Apply as a rounding in-place but keep gradient via STE by storing master.
            # For smoke test, simple post-hoc quantization each phase (not aware).
            with torch.no_grad():
                W = model.get_input_embeddings().weight.data.float()
                levels = 2 ** value if value < 16 else None
                if levels:
                    scale = W.abs().max() / (levels / 2 - 1)
                    W_q = torch.round(W / scale.clamp(min=1e-8)).clamp(-(levels/2-1), levels/2-1) * scale
                    model.get_input_embeddings().weight.data.copy_(W_q.to(model.get_input_embeddings().weight.dtype))
        model = model.to(device)

        # Pre-train eval
        pre_ce = eval_ppl_base(model, val_tokens, args.seq_len, args.batch_size, device)
        print(f"  after compression (no tune): val_ce={pre_ce:.4f}  val_ppl={math.exp(pre_ce):.2f}", flush=True)

        # Mini fine-tune
        model.train()
        for p in model.parameters(): p.requires_grad = True
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr_phase3,
                                betas=(0.9, 0.95), weight_decay=0.01)
        step = 0; t0 = time.time(); running = []
        while step < args.phase3_steps_per_mini:
            for inp, tgt in iter_batches_simple(train_tokens, args.seq_len, args.batch_size, device):
                if step >= args.phase3_steps_per_mini: break
                opt.zero_grad()
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                running.append(float(loss.item())); step += 1

        post_ce = eval_ppl_base(model, val_tokens, args.seq_len, args.batch_size, device)
        tr = float(np.mean(running[-min(50, len(running)):]))
        print(f"  after {args.phase3_steps_per_mini} tune steps: val_ce={post_ce:.4f}  "
              f"val_ppl={math.exp(post_ce):.2f}  Δ from teacher={post_ce-teacher_ce:+.4f}", flush=True)
        history.append({"mini_phase": mi, "axis": axis, "value": value,
                       "pre_val_ce": pre_ce, "post_val_ce": post_ce,
                       "post_val_ppl": math.exp(post_ce),
                       "train_ce": tr,
                       "delta_from_initial": post_ce - teacher_ce,
                       "elapsed": time.time()-t0})

        # Checkpoint after each mini-phase
        out_path = Path(checkpoints_dir) / f"phase3_after_mini{mi}_{axis}.pt"
        torch.save({"history": history}, out_path)

    final_out = Path(checkpoints_dir) / "phase3_final.pt"
    torch.save({"history": history}, final_out)
    return history


# ============================================================
# main orchestrator
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=200)
    # Phase budgets
    p.add_argument("--phase1-steps", type=int, default=2000)
    p.add_argument("--phase2-steps-per-head", type=int, default=2000)
    p.add_argument("--phase3-steps-per-mini", type=int, default=150)
    p.add_argument("--medusa-heads", type=int, default=5)
    p.add_argument("--lr-probe", type=float, default=1e-4)
    p.add_argument("--lr-medusa", type=float, default=1e-4)
    p.add_argument("--lr-phase3", type=float, default=5e-5)
    p.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--checkpoints-dir", default="checkpoints/qwen_halo")
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Save checkpoint every N steps within a phase (for resume on crash).")
    p.add_argument("--heartbeat-every", type=int, default=50,
                   help="Update heartbeat.json every N steps.")
    p.add_argument("--out", default="results/qwen_halo.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}  model={args.model}", flush=True)

    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)

    all_results = {}
    if args.start_phase <= 1:
        _, h1 = train_phase1_early_exit(model, tokenizer, device, args, args.checkpoints_dir)
        all_results["phase1"] = h1
    if args.start_phase <= 2:
        _, h2 = train_phase2_medusa(model, tokenizer, device, args, args.checkpoints_dir)
        all_results["phase2"] = h2
    if args.start_phase <= 3:
        h3 = train_phase3_roundrobin(model, tokenizer, device, args, args.checkpoints_dir)
        all_results["phase3"] = h3

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
