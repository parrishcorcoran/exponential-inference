"""Stage 194: linear walk — one up, one down. No training, no PID.

User's reframe: We have the baseline (FP). Just walk the path manually.
Each cycle: shape pressure goes UP a tiny bit, compensation goes UP a tiny
bit. Measure CE. Keep going until it breaks.

Two coupled manual knobs:
  - SHAPE: power transformation toward bimodal
      W → sign(W) × |W|^(1 - cycle × ε_shape)
    cycle=0: identity, weights unchanged
    cycle=large: weights collapse toward ±group_mean (bimodal)
  - COMPENSATION: multiplicative scaling on FP DOFs
      RMSNorm gains × (1 + cycle × ε_comp)
    cycle=0: original gains
    cycle=large: gains scaled up

No gradient. No training. No optimizer. Just pure forward measurement
along a deterministic path through (shape, compensation) space.

Walk the path until CE drift exceeds some quality bar (or finishes).
Snapshot the model at the end + apply K=1 binary, measure final drift.
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
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage194_linear_walk.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

# Linear walk parameters
N_CYCLES = 60
SHAPE_RATE = 0.01    # per-cycle reduction in power exponent
                     # cycle 60: power = 1 - 60*0.01 = 0.40 (strong bimodal)
COMP_RATE = 0.01     # per-cycle multiplicative growth on RMSNorm gains
QUALITY_LIMIT = 5.0  # drift cap; if drift exceeds, we've found break point


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
    return sum(losses) / max(len(losses), 1)


def shape_metrics(target_modules, group_size=128):
    """Within-row magnitude CV and K=1 residual error."""
    cv_all = []
    k1_errs = []
    for mod in target_modules:
        W = mod.weight.detach().float()
        out_features, in_features = W.shape
        if in_features % group_size != 0: continue
        n_groups = in_features // group_size
        grouped = W.reshape(out_features, n_groups, group_size)
        abs_w = grouped.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        cv = (abs_w.std(dim=-1) / mean_abs.squeeze(-1)).cpu().numpy().flatten()
        cv_all.extend(cv.tolist())

        scales = mean_abs
        W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
        rel_err = (W - W_q).norm() / W.norm().clamp(min=1e-8)
        k1_errs.append(rel_err.item())
    return {
        "shape_cv_mean": float(np.mean(cv_all)) if cv_all else 0.0,
        "k1_residual": float(np.mean(k1_errs)) if k1_errs else 0.0,
    }


print(f"device={device} dtype={dtype}")
print(f"Linear walk: {N_CYCLES} cycles, shape_rate={SHAPE_RATE}, comp_rate={COMP_RATE}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("\nLoading val tokens...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

# Find body modules
target_modules = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    target_modules.append(mod)

# Find norm params
norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        norm_params.append(p)

# Save originals for cumulative reshaping
original_body = [m.weight.data.clone() for m in target_modules]
original_norms = [p.data.clone() for p in norm_params]

# ─── Initial measurement ───
T0 = lm_ce(model, val_tokens)
init_metrics = shape_metrics(target_modules, GROUP_SIZE)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial shape CV: {init_metrics['shape_cv_mean']:.3f}")
print(f"Initial K=1 residual: {init_metrics['k1_residual']:.3f}")
print(f"Initial drift: +0.0000")


# ─── Linear walk ───
print(f"\n{'='*70}")
print("Linear walk: each cycle, shape ↑ and compensation ↑ by linear step")
print('='*70)

trajectory = [{
    "cycle": 0,
    "shape_pressure": 0.0,    # exponent reduction (1 - exponent)
    "comp_factor": 1.0,
    "ce": float(T0),
    "drift": 0.0,
    "shape_cv": init_metrics["shape_cv_mean"],
    "k1_residual": init_metrics["k1_residual"],
}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    # Linear schedule: shape pressure grows, comp factor grows
    shape_pressure = cycle * SHAPE_RATE
    exponent = max(0.0, 1.0 - shape_pressure)
    comp_factor = 1.0 + cycle * COMP_RATE

    # Apply shape: W = sign(W_orig) × |W_orig|^exponent
    with torch.no_grad():
        for m, w_orig in zip(target_modules, original_body):
            W = w_orig.float()
            sign_w = torch.sign(W)
            abs_w = W.abs()
            W_new = sign_w * abs_w.pow(exponent)
            m.weight.data = W_new.to(m.weight.dtype)

        # Apply compensation: RMSNorm gains scaled by comp_factor (from original)
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * comp_factor).to(p.dtype)

    # Measure
    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    metrics = shape_metrics(target_modules, GROUP_SIZE)

    trajectory.append({
        "cycle": cycle,
        "shape_pressure": float(shape_pressure),
        "exponent": float(exponent),
        "comp_factor": float(comp_factor),
        "ce": float(ce),
        "drift": float(drift),
        "shape_cv": float(metrics["shape_cv_mean"]),
        "k1_residual": float(metrics["k1_residual"]),
    })

    if cycle <= 5 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  exp={exponent:.2f} comp×{comp_factor:.3f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  CV={metrics['shape_cv_mean']:.3f}  "
              f"K1err={metrics['k1_residual']:.3f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} nat at cycle {cycle}")

    if drift > 10.0:
        print(f"  STOPPING: drift > 10 nat — model collapsed")
        break


# ─── Apply K=1 binary projection at end and measure ───
print(f"\nApplying K=1 binary projection to walked weights...")
saved = [m.weight.data.clone() for m in target_modules]
with torch.no_grad():
    for m in target_modules:
        W = m.weight.data.float()
        out_features, in_features = W.shape
        if in_features % GROUP_SIZE != 0:
            scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            W_q = torch.sign(W) * scale
        else:
            n_groups = in_features // GROUP_SIZE
            grouped = W.reshape(out_features, n_groups, GROUP_SIZE)
            scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
        m.weight.data = W_q.to(m.weight.dtype)

ce_post_k1 = lm_ce(model, val_tokens)
drift_post_k1 = ce_post_k1 - T0

# Restore (so saved JSON makes sense if anyone re-uses)
with torch.no_grad():
    for m, w in zip(target_modules, saved):
        m.weight.data = w


# ─── Summary ───
print("\n" + "=" * 70)
print("LINEAR WALK COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):                  {T0:.4f}")
print(f"  Initial: drift +0.000  CV={init_metrics['shape_cv_mean']:.3f}  K1err={init_metrics['k1_residual']:.3f}")
final = trajectory[-1]
print(f"  Final:   drift {final['drift']:+.4f}  CV={final['shape_cv']:.3f}  K1err={final['k1_residual']:.3f}")
print(f"  Final exponent:   {final.get('exponent', 1.0):.3f}  (1=no shape, 0=full bimodal)")
print(f"  Final comp factor: ×{final['comp_factor']:.3f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} nat at cycle {broke_at}")
print(f"\n  After applying K=1: drift {drift_post_k1:+.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "shape_rate": SHAPE_RATE,
        "comp_rate": COMP_RATE,
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "ce_post_k1": float(ce_post_k1),
        "drift_post_k1": float(drift_post_k1),
        "init_metrics": init_metrics,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
