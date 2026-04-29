"""Stage 202: PID-controlled RMSNorm DOWN walk — see how far we can go.

Stage 201 found a U-shape with optimum at × 0.97 (drift -0.011), with
linear schedule breaking at cycle 25-30.

PID adaptive rate: aggressive when drift is improving (< 0), slow when
drift exceeds setpoint, stop when way past setpoint. Walks as far as
the model can absorb.

Setpoint: drift ≤ 0.5 nat (permissive — we're walking, not holding lossless).
Adaptive rate: 0.001 to 0.02 per cycle, based on current drift.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage202_pid_rmsnorm_down.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 200
PID_SETPOINT = 0.5
RATE_MAX = 0.02     # max shrink per cycle when in improvement zone
RATE_MIN = 0.0005   # min shrink per cycle when at setpoint
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


def measure_norm_stats(norm_params):
    all_g = []
    for p in norm_params:
        all_g.extend(p.detach().float().flatten().cpu().numpy().tolist())
    arr = np.array(all_g)
    return {"mean": float(arr.mean()), "max": float(arr.max())}


def pid_rate(drift, setpoint):
    """Return shrink rate based on current drift.
    drift < 0 (improving): full rate
    0 < drift < setpoint: linearly decreasing rate
    drift > setpoint: hold (rate=0) or back off (negative)
    """
    if drift <= 0:
        return RATE_MAX  # full speed when improving
    elif drift < setpoint:
        # Linearly interpolate from RATE_MAX to RATE_MIN as drift goes 0 → setpoint
        frac = drift / setpoint
        return RATE_MAX * (1 - frac) + RATE_MIN * frac
    elif drift < 2 * setpoint:
        return 0  # hold
    else:
        return -0.005  # back off


print(f"device={device} dtype={dtype}")
print(f"PID RMSNorm DOWN: setpoint={PID_SETPOINT}")
print(f"  rate range: {RATE_MIN} to {RATE_MAX}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("\nLoading val tokens + model...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        norm_params.append(p)
original_norms = [p.data.clone() for p in norm_params]


T0 = lm_ce(model, val_tokens)
init_stats = measure_norm_stats(norm_params)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial norm: mean={init_stats['mean']:.3f} max={init_stats['max']:.1f}")


print(f"\n{'='*70}")
print("PID-controlled RMSNorm shrink — see how far we can go")
print('='*70)

trajectory = [{"cycle": 0, "comp_factor": 1.0, "ce": float(T0), "drift": 0.0,
               "norm_max": init_stats["max"], "rate": 0.0}]

current_factor = 1.0
broke_at = None
for cycle in range(1, N_CYCLES + 1):
    # Compute previous drift (or 0 at cycle 1)
    prev_drift = trajectory[-1]["drift"]
    rate = pid_rate(prev_drift, PID_SETPOINT)
    new_factor = current_factor * (1 - rate)
    new_factor = max(0.001, new_factor)

    with torch.no_grad():
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * new_factor).to(p.dtype)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    stats = measure_norm_stats(norm_params)
    trajectory.append({
        "cycle": cycle, "comp_factor": float(new_factor),
        "ce": float(ce), "drift": float(drift),
        "norm_max": stats["max"], "norm_mean": stats["mean"],
        "rate": float(rate),
    })

    if cycle <= 5 or cycle % 10 == 0 or abs(drift - prev_drift) > 0.5:
        marker = "↓" if rate > 0 else ("↑" if rate < 0 else "·")
        print(f"  cycle {cycle:>3}/{N_CYCLES}  comp×{new_factor:.4f} {marker} rate={rate:.4f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  norm_max={stats['max']:.1f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING at +10")
        break

    current_factor = new_factor


print("\n" + "=" * 70)
print("PID RMSNORM-DOWN WALK COMPLETE")
print("=" * 70)
final = trajectory[-1]
best_idx = min(range(len(trajectory)), key=lambda i: trajectory[i]["drift"])
best = trajectory[best_idx]
print(f"  Final cycle: {final['cycle']}  comp×{final['comp_factor']:.4f}  drift {final['drift']:+.4f}")
print(f"  Final norm max: {final['norm_max']:.1f}  (BitNet target: 1.01)")
print(f"  Best drift seen: {best['drift']:+.4f} at cycle {best['cycle']}  comp×{best['comp_factor']:.4f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} at cycle {broke_at}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "pid_setpoint": PID_SETPOINT,
        "rate_range": [RATE_MIN, RATE_MAX],
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "best_drift": float(best["drift"]),
        "best_at_cycle": best["cycle"],
        "best_comp_factor": float(best["comp_factor"]),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
