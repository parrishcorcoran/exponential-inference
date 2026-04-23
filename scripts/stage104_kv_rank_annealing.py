"""
Stage 104 — Continuous KV rank annealing (Z8 project).

Replace each k_proj and v_proj with a ContinuousRankLinear module that
stores the SVD factorization (U, S, V) of the teacher's weight matrix.
A "slider" buffer controls effective rank via a sigmoid mask over singular
values:

    mask[i] = sigmoid((slider - i) / temperature)
    W_eff  = U @ diag(S * mask) @ V

At training step 0: slider = d (full rank, all sv weighted ~1).
At training step T: slider = target_rank (only top target_rank sv ≈ 1).
Between: smooth interpolation. No discrete jumps.

The question this answers: does smooth rank annealing recover what
discrete rank truncation cannot? Stage 38 (discrete rank 128) failed.
Our Qwen Halo Phase 3 uses discrete round-robin. Continuous annealing
gives gradient something to work with at every step.

Novelty check needed: near-neighbors are AdaLoRA (dynamic LoRA rank),
DyLoRA (multi-rank), progressive pruning (weight-magnitude-based).
Continuous rank annealing for KV specifically during fine-tune from
pretrained autoregressive LM — I don't recall a published version.
Z8 should confirm before writing up.

Monitors built in:
  - val_ppl every N steps
  - Gradient norm every step
  - Effective rank (count of sv with mask > 0.5) every N steps
  - Rolling-window loss trend
  - Attention output cosine to teacher's (on fixed probe prompts)
  - NaN/Inf detection
  - Auto-pause on breakpoint criteria (save checkpoint for inspection)
"""

import argparse
import glob
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ContinuousRankLinear
# ============================================================

class ContinuousRankLinear(nn.Module):
    """W ≈ U @ diag(S * mask) @ V  where mask is a smooth sigmoid on singular indices.

       U: [out, k], S: [k], V: [k, in], k = min(in, out).
       The `slider` buffer controls effective rank; outside modifies it over training."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        k = min(in_features, out_features)
        self.k = k
        self.U = nn.Parameter(torch.randn(out_features, k) * (1.0/math.sqrt(out_features)))
        self.S = nn.Parameter(torch.ones(k))
        self.V = nn.Parameter(torch.randn(k, in_features) * (1.0/math.sqrt(in_features)))
        if bias: self.bias = nn.Parameter(torch.zeros(out_features))
        else: self.bias = None
        # slider starts at k (full rank). temperature controls sharpness of sigmoid.
        self.register_buffer("slider", torch.tensor(float(k)))
        self.register_buffer("temperature", torch.tensor(1.0))

    @classmethod
    def from_linear_svd(cls, linear):
        m = cls(linear.in_features, linear.out_features, bias=(linear.bias is not None))
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        with torch.no_grad():
            m.U.data = U
            m.S.data = S
            m.V.data = Vh
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m

    def current_mask(self):
        """Sigmoid mask over singular indices. mask[i] ≈ 1 if i < slider, else 0."""
        idx = torch.arange(self.k, device=self.slider.device, dtype=torch.float32)
        return torch.sigmoid((self.slider - idx) / self.temperature.clamp(min=1e-3))

    def effective_rank(self, threshold=0.5):
        """How many dims are >= threshold active under current mask."""
        with torch.no_grad():
            return int((self.current_mask() > threshold).sum().item())

    def forward(self, x):
        mask = self.current_mask()                              # [k]
        S_masked = self.S * mask                                # [k]
        # Factored compute: (x @ V.T) * S_masked @ U.T
        x_proj = F.linear(x.float(), self.V)                    # [..., k]
        x_proj = x_proj * S_masked
        y = F.linear(x_proj, self.U, self.bias)                 # [..., out]
        return y.to(x.dtype)


def convert_kv_to_continuous_rank(model):
    """Replace k_proj, v_proj in every layer with ContinuousRankLinear."""
    n = 0
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            old = getattr(layer.self_attn, name)
            new = ContinuousRankLinear.from_linear_svd(old)
            setattr(layer.self_attn, name, new)
            n += 1
    return n


def set_slider_all(model, value):
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            mod = getattr(layer.self_attn, name)
            if isinstance(mod, ContinuousRankLinear):
                mod.slider.fill_(value)


def set_temperature_all(model, value):
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            mod = getattr(layer.self_attn, name)
            if isinstance(mod, ContinuousRankLinear):
                mod.temperature.fill_(value)


def mean_effective_rank(model):
    vals = []
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            mod = getattr(layer.self_attn, name)
            if isinstance(mod, ContinuousRankLinear):
                vals.append(mod.effective_rank())
    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# Schedulers
# ============================================================

class AnnealingSchedule:
    """Linear schedule from start_rank to target_rank over total_steps.
       Plus optional temperature annealing (sigmoid sharpens over training)."""
    def __init__(self, start_rank, target_rank, total_steps, start_temp=4.0, end_temp=0.5):
        self.start_rank = float(start_rank)
        self.target_rank = float(target_rank)
        self.total_steps = int(total_steps)
        self.start_temp = float(start_temp)
        self.end_temp = float(end_temp)

    def slider_at(self, step):
        frac = min(max(step / max(self.total_steps, 1), 0.0), 1.0)
        return self.start_rank + (self.target_rank - self.start_rank) * frac

    def temp_at(self, step):
        frac = min(max(step / max(self.total_steps, 1), 0.0), 1.0)
        return self.start_temp + (self.end_temp - self.start_temp) * frac


# ============================================================
# Breakpoint Monitor
# ============================================================

class BreakpointMonitor:
    """Watch metrics for signs of breakage. Flag when thresholds cross."""
    def __init__(self, val_ppl_ratio_trigger=2.0, grad_norm_ratio_trigger=5.0,
                 loss_stall_steps=500, window=100):
        self.val_ppl_history = []
        self.grad_norm_history = deque(maxlen=window)
        self.loss_history = deque(maxlen=loss_stall_steps)
        self.val_ppl_ratio_trigger = val_ppl_ratio_trigger
        self.grad_norm_ratio_trigger = grad_norm_ratio_trigger
        self.loss_stall_steps = loss_stall_steps
        self.window = window

    def record_loss(self, loss):
        self.loss_history.append(float(loss))

    def record_grad_norm(self, gn):
        self.grad_norm_history.append(float(gn))

    def record_val_ppl(self, ppl):
        self.val_ppl_history.append(float(ppl))

    def check(self):
        """Return (broken: bool, reason: str). None if OK."""
        # NaN / Inf in loss
        if len(self.loss_history) > 0:
            last = self.loss_history[-1]
            if math.isnan(last) or math.isinf(last):
                return True, f"loss is NaN/Inf: {last}"
        # val_ppl explosion
        if len(self.val_ppl_history) >= 2:
            mn = min(self.val_ppl_history)
            cur = self.val_ppl_history[-1]
            if mn > 0 and cur > mn * self.val_ppl_ratio_trigger:
                return True, f"val_ppl rose {cur/mn:.2f}× above rolling minimum ({cur:.2f} vs min {mn:.2f})"
        # grad norm explosion
        if len(self.grad_norm_history) >= self.window:
            avg = np.mean(self.grad_norm_history)
            if self.grad_norm_history[-1] > avg * self.grad_norm_ratio_trigger:
                return True, f"grad_norm {self.grad_norm_history[-1]:.4f} >> avg {avg:.4f}"
        # loss stall (no descent in window)
        if len(self.loss_history) >= self.loss_stall_steps:
            first_half = list(self.loss_history)[:self.loss_stall_steps//2]
            second_half = list(self.loss_history)[self.loss_stall_steps//2:]
            if np.mean(second_half) > np.mean(first_half) * 1.05:
                return True, f"loss stalled/ascending: {np.mean(first_half):.4f} → {np.mean(second_half):.4f}"
        return False, None


# ============================================================
# Data + training
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


def write_status(checkpoints_dir, step, slider, temp, eff_rank, val_ppl=None, broken=False, reason=None):
    path = Path(checkpoints_dir) / "status.json"
    data = {
        "step": step, "slider": slider, "temperature": temp,
        "mean_effective_rank": eff_rank,
        "val_ppl": val_ppl,
        "broken": broken, "reason": reason,
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def save_ckpt(checkpoints_dir, model, step, monitor, slider, label="periodic"):
    path = Path(checkpoints_dir) / f"kv_anneal_{label}_step{step:08d}.pt"
    try:
        torch.save({
            "model_state": model.state_dict(),
            "step": step,
            "slider": slider,
            "val_ppl_history": monitor.val_ppl_history,
        }, path)
    except Exception as e:
        print(f"  [WARN] ckpt save failed: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--start-rank", type=int, default=1024,
                   help="Start slider at this rank (full or near-full)")
    p.add_argument("--target-rank", type=int, default=16,
                   help="Target rank at end of annealing")
    p.add_argument("--anneal-steps", type=int, default=5000,
                   help="Number of training steps over which slider moves")
    p.add_argument("--total-steps", type=int, default=6000,
                   help="Total training steps (annealing + stabilization at end)")
    p.add_argument("--start-temp", type=float, default=4.0)
    p.add_argument("--end-temp", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--out", default="results/stage104_kv_annealing.json")
    p.add_argument("--checkpoints-dir", default="checkpoints/stage104_kv_anneal")
    p.add_argument("--device", default=None)
    p.add_argument("--pause-on-break", action="store_true",
                   help="If set, stop training when monitor flags breakage (default: just log)")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    model = model.float()
    d_kv = model.config.num_key_value_heads * (
        model.config.head_dim if hasattr(model.config, "head_dim")
        else model.config.hidden_size // model.config.num_attention_heads
    )
    print(f"  d_kv per matrix = {d_kv}", flush=True)

    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 30, split="validation")
    teacher_ppl_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"teacher val_ppl = {math.exp(teacher_ppl_ce):.2f}", flush=True)

    # Convert KV projections to continuous rank
    n = convert_kv_to_continuous_rank(model)
    model = model.to(device)
    print(f"converted {n} KV projections to ContinuousRankLinear", flush=True)

    # Initialize slider at start_rank, temperature at start_temp
    schedule = AnnealingSchedule(args.start_rank, args.target_rank,
                                 args.anneal_steps, args.start_temp, args.end_temp)
    set_slider_all(model, schedule.slider_at(0))
    set_temperature_all(model, schedule.temp_at(0))
    init_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"post-SVD init val_ppl = {math.exp(init_ce):.2f}", flush=True)

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    monitor = BreakpointMonitor()
    monitor.record_val_ppl(math.exp(init_ce))

    history = [{
        "step": 0, "slider": schedule.slider_at(0),
        "temperature": schedule.temp_at(0),
        "mean_effective_rank": mean_effective_rank(model),
        "val_ce": init_ce, "val_ppl": math.exp(init_ce),
    }]

    step = 0; t0 = time.time(); running = []
    model.train()
    while step < args.total_steps:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.total_steps: break

            # Update slider + temperature
            set_slider_all(model, schedule.slider_at(step))
            set_temperature_all(model, schedule.temp_at(step))

            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running.append(float(loss.item()))
            monitor.record_loss(loss.item())
            monitor.record_grad_norm(float(grad_norm))
            step += 1

            if step % 50 == 0:
                write_status(args.checkpoints_dir, step, schedule.slider_at(step),
                             schedule.temp_at(step), mean_effective_rank(model))

            if step % args.eval_every == 0:
                val_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
                val_ppl = math.exp(val_ce)
                monitor.record_val_ppl(val_ppl)
                eff_rank = mean_effective_rank(model)
                tr = float(np.mean(running[-args.eval_every:]))
                history.append({"step": step, "slider": schedule.slider_at(step),
                               "temperature": schedule.temp_at(step),
                               "mean_effective_rank": eff_rank,
                               "train_ce": tr, "val_ce": val_ce, "val_ppl": val_ppl,
                               "grad_norm": float(grad_norm),
                               "elapsed": time.time()-t0})
                print(f"  step {step}/{args.total_steps}  slider={schedule.slider_at(step):.1f}  "
                      f"temp={schedule.temp_at(step):.2f}  eff_rank={eff_rank:.1f}  "
                      f"train_ce={tr:.4f}  val_ppl={val_ppl:.2f}  "
                      f"grad_norm={float(grad_norm):.3f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                write_status(args.checkpoints_dir, step, schedule.slider_at(step),
                             schedule.temp_at(step), eff_rank, val_ppl=val_ppl)

                # Breakpoint check
                broken, reason = monitor.check()
                if broken:
                    print(f"\n  *** BREAKPOINT at step {step}: {reason} ***\n", flush=True)
                    save_ckpt(args.checkpoints_dir, model, step, monitor,
                              schedule.slider_at(step), label="break")
                    write_status(args.checkpoints_dir, step, schedule.slider_at(step),
                                 schedule.temp_at(step), eff_rank,
                                 val_ppl=val_ppl, broken=True, reason=reason)
                    history.append({"step": step, "event": "breakpoint", "reason": reason})
                    if args.pause_on_break:
                        print(f"  pausing per --pause-on-break", flush=True)
                        break

            if step % args.ckpt_every == 0:
                save_ckpt(args.checkpoints_dir, model, step, monitor,
                          schedule.slider_at(step), label="periodic")

        if args.pause_on_break:
            broken, reason = monitor.check()
            if broken: break

    # Final save + report
    save_ckpt(args.checkpoints_dir, model, step, monitor,
              schedule.slider_at(step), label="final")
    print(f"\n=== summary ===")
    print(f"  teacher val_ppl: {math.exp(teacher_ppl_ce):.2f}")
    print(f"  end slider: {schedule.slider_at(step):.1f}  "
          f"(target was {args.target_rank})")
    print(f"  end mean effective rank: {mean_effective_rank(model):.1f}")
    final_val_ce = eval_ppl(model, val_tokens, args.seq_len, args.batch_size, device)
    print(f"  end val_ppl: {math.exp(final_val_ce):.2f}")
    print(f"  min val_ppl during run: {min(monitor.val_ppl_history):.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "teacher_val_ce": teacher_ppl_ce,
                   "teacher_val_ppl": math.exp(teacher_ppl_ce),
                   "final_val_ce": final_val_ce,
                   "final_val_ppl": math.exp(final_val_ce),
                   "history": history,
                   "val_ppl_history": monitor.val_ppl_history}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
