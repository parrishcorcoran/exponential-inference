"""Stage 200: anneal RMSNorm UP slowly while body adapts via gradient.

Pattern from magnitude annealing (Stage 169-style): walk an axis slowly,
let the model gradient-train to absorb the change, find where it breaks.

Each cycle:
  1. Bump RMSNorm gain targets by small factor
  2. Train body (o,down) + norms for K steps to absorb
  3. Measure CE drift
  4. Continue until break

If body successfully absorbs each RMSNorm bump (drift stays near 0), we
walk far. If body runs out of compensation capacity, drift rises.

Uses cached 50M corpus (no streaming overhead). Tight trainable set
(no embed/lm_head — they overfit on small per-cycle data).
"""
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR_BODY = 1e-5
LR_NORM = 1e-4
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage200_rmsnorm_anneal_body_adapt.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
TRAINABLE_BODY = ("o_proj", "down_proj")

N_CYCLES = 60
TRAIN_STEPS_PER_CYCLE = 50
COMP_RATE = 0.005    # slow RMSNorm growth — annealing pace
QUALITY_LIMIT = 5.0


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


print(f"device={device} dtype={dtype}")
print(f"RMSNorm anneal-up with body adaptation")
print(f"  comp_rate={COMP_RATE} per cycle (slow)")
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

# Trainable body subset (o,down per Finding 27 bottleneck)
body_params = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    if any(t in name for t in TRAINABLE_BODY):
        mod.weight.requires_grad = True
        body_params.append(mod.weight)

# Trainable norms (all of them)
norm_params = []
original_norms = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        norm_params.append(p)
        original_norms.append(p.data.clone())

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
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial drift: +0.0000")


print(f"\n{'='*70}")
print(f"Anneal RMSNorm UP with body adaptation")
print('='*70)

it = iter_train()
trajectory = [{"cycle": 0, "comp_factor": 1.0, "ce": float(T0), "drift": 0.0}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    comp_factor = 1.0 + cycle * COMP_RATE

    # Step 1: bump RMSNorm gains to target
    with torch.no_grad():
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * comp_factor).to(p.dtype)

    # Step 2: train body + norms to absorb
    train_steps(it, TRAIN_STEPS_PER_CYCLE)

    # Step 3: measure
    ce = lm_ce(model, val_tokens)
    drift = ce - T0

    trajectory.append({
        "cycle": cycle, "comp_factor": float(comp_factor),
        "ce": float(ce), "drift": float(drift),
    })

    if cycle <= 3 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  comp×{comp_factor:.3f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  (after {TRAIN_STEPS_PER_CYCLE} train steps)",
              flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING: drift > 10")
        break


print("\n" + "=" * 70)
print("RMSNORM ANNEAL WITH BODY ADAPT COMPLETE")
print("=" * 70)
final = trajectory[-1]
print(f"  Final cycle: {final['cycle']}  comp×{final['comp_factor']:.3f}  drift {final['drift']:+.4f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} at cycle {broke_at}")
else:
    print(f"  Walked all {N_CYCLES} cycles without breaking")

print(f"\nCOMPARE:")
print(f"  Stage 199 (RMSNorm only, no train, fast rate): broke ~cycle 80 predicted")
print(f"  Stage 200 (RMSNorm anneal + body gradient): broke at cycle {broke_at}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "comp_rate": COMP_RATE,
        "n_cycles": N_CYCLES,
        "train_steps_per_cycle": TRAIN_STEPS_PER_CYCLE,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
