"""Stage 196: shape-alone baseline. Pure shape op, NO compensation.

Walking with just one knob: shape pressure goes down. RMSNorm and
everything else stays at original FP values. No knobs UP.

Tells us how far pure shape pressure walks WITHOUT any compensation.
The walk-distance gap between this and Stage 194 (shape + RMSNorm) is
RMSNorm's actual compensation contribution.

If shape walks to cycle X alone, and shape + RMSNorm walks to cycle Y,
then RMSNorm contributes (Y - X) cycles of buffer. Each subsequent
candidate compensation will be measured against this baseline.
"""
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
RESULTS_PATH = Path("results/stage196_shape_alone.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 60
SHAPE_RATE = 0.01    # only knob — exponent decreases per cycle
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
    return sum(losses) / max(len(losses), 1)


def shape_metrics(target_modules, group_size=128):
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
print(f"Shape-alone baseline: {N_CYCLES} cycles, shape_rate={SHAPE_RATE}")
print("NO compensation knobs — pure shape pressure")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("\nLoading val tokens + model...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

target_modules = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    target_modules.append(mod)

original_body = [m.weight.data.clone() for m in target_modules]
original_row_norms = [w.float().norm(dim=-1, keepdim=True).clone() for w in original_body]


T0 = lm_ce(model, val_tokens)
init_metrics = shape_metrics(target_modules, GROUP_SIZE)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial shape CV: {init_metrics['shape_cv_mean']:.3f}")
print(f"Initial K=1 residual: {init_metrics['k1_residual']:.3f}")
print(f"Initial drift: +0.0000")


print(f"\n{'='*70}")
print("Shape-alone walk (no compensation)")
print('='*70)

trajectory = [{
    "cycle": 0, "exponent": 1.0,
    "ce": float(T0), "drift": 0.0,
    "shape_cv": init_metrics["shape_cv_mean"],
    "k1_residual": init_metrics["k1_residual"],
}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    exponent = max(0.0, 1.0 - cycle * SHAPE_RATE)

    with torch.no_grad():
        for m, w_orig, rn_orig in zip(target_modules, original_body, original_row_norms):
            W = w_orig.float()
            sign_w = torch.sign(W)
            abs_w = W.abs()
            W_new = sign_w * abs_w.pow(exponent)
            new_norms = W_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            W_new = W_new * (rn_orig / new_norms)
            m.weight.data = W_new.to(m.weight.dtype)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    metrics = shape_metrics(target_modules, GROUP_SIZE)

    trajectory.append({
        "cycle": cycle, "exponent": float(exponent),
        "ce": float(ce), "drift": float(drift),
        "shape_cv": float(metrics["shape_cv_mean"]),
        "k1_residual": float(metrics["k1_residual"]),
    })

    if cycle <= 5 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  exp={exponent:.2f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  CV={metrics['shape_cv_mean']:.3f}  "
              f"K1err={metrics['k1_residual']:.3f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} nat at cycle {cycle}")

    if drift > 10.0:
        print(f"  STOPPING: drift > 10 nat — model collapsed")
        break


print("\n" + "=" * 70)
print("SHAPE-ALONE BASELINE COMPLETE")
print("=" * 70)
final = trajectory[-1]
print(f"  Final cycle: {final['cycle']}")
print(f"  Final drift: {final['drift']:+.4f}")
print(f"  Final exponent: {final.get('exponent', 1.0):.3f}")
print(f"  Final CV: {final['shape_cv']:.3f}")
print(f"  Final K1err: {final['k1_residual']:.3f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} nat at cycle {broke_at}")
else:
    print(f"  Walked all {N_CYCLES} cycles without breaking")

print(f"\nCOMPARE:")
print(f"  Stage 194 (shape + RMSNorm gain): broke at cycle 22")
print(f"  Stage 195 (shape + RMSNorm + row mag): broke at cycle 16")
print(f"  Stage 196 (shape ALONE): broke at cycle {broke_at}")
if broke_at is not None:
    rmsnorm_contribution = 22 - broke_at
    print(f"  → RMSNorm gain compensation contribution: {rmsnorm_contribution} extra cycles")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "shape_rate": SHAPE_RATE,
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "init_metrics": init_metrics,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
