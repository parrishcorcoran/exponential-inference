"""Stage 195: three-knob linear walk — shape DOWN, body row magnitude UP,
RMSNorm gain UP.

User's reframe: Stage 194 broke at cycle 22 with only two knobs (shape
+ RMSNorm). Bonsai's data (Stage 185) shows BOTH body row-norms (×2.6)
AND RMSNorm gains changing direction. So two independent compensation
channels, not one. Adding body row magnitude as the third knob.

Three coupled manual knobs:
  Knob 1 (down):  shape pressure
                  W = sign(W_orig) × |W_orig|^(1 - cycle × SHAPE_RATE)
  Knob 2 (up):    body row magnitude
                  target row L2 = original_row_norm × (1 + cycle × ROW_RATE)
  Knob 3 (up):    RMSNorm gains × (1 + cycle × COMP_RATE)

Both up-knobs are pure compensation grown manually, no training.
Shape op pins each row's L2 to (knob 2) target — pure shape change
*at the right magnitude scale* for that cycle.

If two compensation channels track the damage curve better than one,
walk should reach further into bimodal territory before drift breaks.
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
RESULTS_PATH = Path("results/stage195_three_knob_walk.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

# Three-knob walk
N_CYCLES = 60
SHAPE_RATE = 0.01    # exponent decreases per cycle (knob 1, down)
ROW_RATE   = 0.015   # body row magnitude grows per cycle (knob 2, up)
COMP_RATE  = 0.01    # RMSNorm gain grows per cycle (knob 3, up)
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
print(f"Three-knob walk: {N_CYCLES} cycles")
print(f"  shape_rate={SHAPE_RATE} (down, exponent)")
print(f"  row_rate={ROW_RATE}   (up, body row magnitude)")
print(f"  comp_rate={COMP_RATE} (up, RMSNorm gains)")

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

target_modules = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    target_modules.append(mod)

norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        norm_params.append(p)

original_body = [m.weight.data.clone() for m in target_modules]
original_norms = [p.data.clone() for p in norm_params]
original_row_norms = [w.float().norm(dim=-1, keepdim=True).clone() for w in original_body]


T0 = lm_ce(model, val_tokens)
init_metrics = shape_metrics(target_modules, GROUP_SIZE)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial shape CV: {init_metrics['shape_cv_mean']:.3f}")
print(f"Initial K=1 residual: {init_metrics['k1_residual']:.3f}")
print(f"Initial drift: +0.0000")


print(f"\n{'='*70}")
print("Three-knob linear walk")
print('='*70)

trajectory = [{
    "cycle": 0,
    "exponent": 1.0,
    "row_factor": 1.0,
    "comp_factor": 1.0,
    "ce": float(T0),
    "drift": 0.0,
    "shape_cv": init_metrics["shape_cv_mean"],
    "k1_residual": init_metrics["k1_residual"],
}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    exponent = max(0.0, 1.0 - cycle * SHAPE_RATE)
    row_factor = 1.0 + cycle * ROW_RATE
    comp_factor = 1.0 + cycle * COMP_RATE

    with torch.no_grad():
        # Knob 1+2: shape op pinned to (original row norm × row_factor)
        for m, w_orig, rn_orig in zip(target_modules, original_body, original_row_norms):
            W = w_orig.float()
            sign_w = torch.sign(W)
            abs_w = W.abs()
            W_new = sign_w * abs_w.pow(exponent)
            new_norms = W_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            target_norms = rn_orig * row_factor
            W_new = W_new * (target_norms / new_norms)
            m.weight.data = W_new.to(m.weight.dtype)

        # Knob 3: RMSNorm gains × comp_factor (from original)
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * comp_factor).to(p.dtype)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    metrics = shape_metrics(target_modules, GROUP_SIZE)

    trajectory.append({
        "cycle": cycle,
        "exponent": float(exponent),
        "row_factor": float(row_factor),
        "comp_factor": float(comp_factor),
        "ce": float(ce),
        "drift": float(drift),
        "shape_cv": float(metrics["shape_cv_mean"]),
        "k1_residual": float(metrics["k1_residual"]),
    })

    if cycle <= 5 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  exp={exponent:.2f} row×{row_factor:.3f} comp×{comp_factor:.3f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  CV={metrics['shape_cv_mean']:.3f}  "
              f"K1err={metrics['k1_residual']:.3f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} nat at cycle {cycle}")

    if drift > 10.0:
        print(f"  STOPPING: drift > 10 nat — model collapsed")
        break


# K=1 binary projection at end
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
with torch.no_grad():
    for m, w in zip(target_modules, saved):
        m.weight.data = w


print("\n" + "=" * 70)
print("THREE-KNOB WALK COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):                  {T0:.4f}")
final = trajectory[-1]
print(f"  Final cycle: {final['cycle']}")
print(f"  Final state:")
print(f"    drift          {final['drift']:+.4f}")
print(f"    exponent       {final.get('exponent', 1.0):.3f}  (1=no shape, 0=full bimodal)")
print(f"    row factor     ×{final['row_factor']:.3f}")
print(f"    comp factor    ×{final['comp_factor']:.3f}")
print(f"    CV             {final['shape_cv']:.3f}")
print(f"    K1err          {final['k1_residual']:.3f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} nat at cycle {broke_at}")
else:
    print(f"  Walked all {N_CYCLES} cycles without breaking quality cap")
print(f"\n  After applying K=1: drift {drift_post_k1:+.4f}")

print(f"\nCOMPARE to Stage 194 v2 (two-knob):")
print(f"  Stage 194 broke at cycle 22 with drift ~5 nat")
print(f"  Stage 194 final K1err: 0.500")
print(f"  Stage 195 here: broke_at={broke_at}  final K1err={final['k1_residual']:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "shape_rate": SHAPE_RATE,
        "row_rate": ROW_RATE,
        "comp_rate": COMP_RATE,
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "ce_post_k1": float(ce_post_k1),
        "drift_post_k1": float(drift_post_k1),
        "init_metrics": init_metrics,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
