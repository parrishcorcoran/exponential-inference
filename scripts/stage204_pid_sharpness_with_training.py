"""Stage 204: PID sharpness anneal WITH gradient body training. Real annealing.

Stage 203 walked statically — outlier cap descends, body can't adapt,
PID locks at T=6.5 with drift +0.5. The static plateau is the limit
of what outlier capping does ALONE.

Stage 204 adds the missing piece: body trains to absorb each cap step.
Each cycle:
  1. Cap RMSNorm gains: |gain| ≤ T (T descends slightly)
  2. Train body (o, down) + non-outlier norms for K steps to absorb
  3. Measure CE drift
  4. PID throttles T descent based on drift

If gradient training works, body finds per-channel compensation for the
clipped outliers. Drift stays bounded as T descends. Walk goes much
further than Stage 203's T=6.5 plateau.

Setpoint tighter (0.1 nat) since training should keep us near baseline.
"""
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR_BODY = 1e-5
LR_NORM = 5e-5
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage204_pid_sharpness_training.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
TRAINABLE_BODY = ("o_proj", "down_proj")

N_CYCLES = 100
TRAIN_STEPS_PER_CYCLE = 50
PID_SETPOINT = 0.1     # tighter — training should hold us close
RATE_MAX = 0.05
RATE_MIN = 0.001
QUALITY_LIMIT = 5.0
T_TARGET = 1.0


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt_cached():
    return torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                      weights_only=True).long()


def lm_ce(model, val_tokens):
    losses = []
    model.eval()
    for i in range(N_VAL_CHUNKS):
        s = i * SEQ_LEN
        window = val_tokens[s:s + SEQ_LEN + 1]
        if len(window) < SEQ_LEN + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            losses.append(F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def pid_rate(drift, setpoint):
    if drift <= 0:
        return RATE_MAX
    elif drift < setpoint:
        frac = drift / setpoint
        return RATE_MAX * (1 - frac) + RATE_MIN * frac
    elif drift < 2 * setpoint:
        return 0
    else:
        return -0.01


print(f"device={device} dtype={dtype}")
print(f"PID sharpness anneal WITH gradient training")
print(f"  setpoint={PID_SETPOINT}  T_target={T_TARGET}")
print(f"  train steps per cycle: {TRAIN_STEPS_PER_CYCLE}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("\nLoading val + train tokens...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 1024 : SEQ_LEN * 1024 + SEQ_LEN * 4096].tolist()


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

# Trainable body subset
body_params = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    if any(t in name for t in TRAINABLE_BODY):
        mod.weight.requires_grad = True
        body_params.append(mod.weight)

# Trainable norms (we cap these AND let them learn)
norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        norm_params.append(p)
original_norms = [p.data.clone() for p in norm_params]

n_body = sum(p.numel() for p in body_params)
n_norm = sum(p.numel() for p in norm_params)
print(f"  trainable: {n_body:,} body (o,down) + {n_norm:,} norm = {n_body + n_norm:,}")


opt = torch.optim.AdamW([
    {"params": body_params, "lr": LR_BODY},
    {"params": norm_params, "lr": LR_NORM},
], weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_steps(it, n_steps):
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(body_params + norm_params, GRAD_CLIP)
        opt.step()


T0 = lm_ce(model, val_tokens)
init_max = max(p.detach().float().abs().max().item() for p in norm_params)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial norm max: {init_max:.1f}")


print(f"\n{'='*70}")
print("PID sharpness + gradient training")
print('='*70)

trajectory = [{"cycle": 0, "T_cap": init_max, "ce": float(T0), "drift": 0.0,
               "norm_max": init_max, "n_clipped": 0}]
current_T = init_max
broke_at = None
it = iter_train()

for cycle in range(1, N_CYCLES + 1):
    prev_drift = trajectory[-1]["drift"]
    rate = pid_rate(prev_drift, PID_SETPOINT)
    new_T = current_T * (1 - rate)
    new_T = max(T_TARGET * 0.5, new_T)

    # Cap norms at new T (this is on TOP of any prior training)
    n_clipped = 0
    with torch.no_grad():
        for p in norm_params:
            sign = torch.sign(p.data)
            mag = p.data.abs()
            n_clipped += int((mag > new_T).sum().item())
            p.data = sign * torch.clamp(mag, max=new_T)

    # Train body + norms for K steps to absorb
    train_steps(it, TRAIN_STEPS_PER_CYCLE)

    # Measure
    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    cur_max = max(p.detach().float().abs().max().item() for p in norm_params)

    trajectory.append({
        "cycle": cycle, "T_cap": float(new_T),
        "ce": float(ce), "drift": float(drift),
        "norm_max": float(cur_max), "n_clipped": n_clipped,
        "rate": float(rate),
    })

    if cycle <= 3 or cycle % 5 == 0 or abs(drift - prev_drift) > 0.5:
        marker = "↓" if rate > 0 else ("↑" if rate < 0 else "·")
        print(f"  cycle {cycle:>3}/{N_CYCLES}  T={new_T:7.2f} {marker} rate={rate:.4f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  clipped={n_clipped} norm_max={cur_max:.1f}",
              flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING at +10")
        break
    if new_T <= T_TARGET:
        print(f"  Reached target T={T_TARGET}")
        break

    current_T = new_T


print("\n" + "=" * 70)
print("PID SHARPNESS + TRAINING COMPLETE")
print("=" * 70)
final = trajectory[-1]
best_idx = min(range(len(trajectory)), key=lambda i: trajectory[i]["drift"])
best = trajectory[best_idx]
print(f"  Final cycle: {final['cycle']}  T={final['T_cap']:.2f}  norm_max={final['norm_max']:.2f}  drift{final['drift']:+.4f}")
print(f"  Best drift: {best['drift']:+.4f} at cycle {best['cycle']}  T={best['T_cap']:.2f}")
print(f"\nCOMPARE to Stage 203 (no training):")
print(f"  Stage 203: locked at T=6.5, drift +0.5 (no training)")
print(f"  Stage 204: T={final['T_cap']:.2f}, drift{final['drift']:+.4f} (with training)")
print(f"  BitNet target: T=1.01")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "init_norm_max": float(init_max),
        "T_target": T_TARGET,
        "pid_setpoint": PID_SETPOINT,
        "n_cycles": N_CYCLES,
        "train_steps_per_cycle": TRAIN_STEPS_PER_CYCLE,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "best_drift": float(best["drift"]),
        "best_T_cap": float(best["T_cap"]),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
