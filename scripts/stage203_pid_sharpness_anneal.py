"""Stage 203: PID-controlled SHARPNESS anneal — outlier-cap RMSNorm gains.

User's reframe: we need RMSNorm SHARPNESS (compress outliers toward
bulk), not WIDTH (uniform scaling). Stages 201/202 were wrong — they
scaled the whole distribution without changing shape.

Mechanism: cap |gain| at threshold T. T descends each cycle under PID.
- Outliers (192, 100, 80...) get clipped to T
- Bulk gains (0.5-3) pass through unchanged
- Distribution variance shrinks; mean barely moves

Walk T down toward BitNet's 1.01 target. PID throttles when drift rises.

This isolates the OUTLIER question. If outliers are doing real work, we
break early (outliers are load-bearing relevant operators at the FP RG
attractor). If outliers are slack/redundant, we walk far.
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
RESULTS_PATH = Path("results/stage203_pid_sharpness_anneal.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 200
PID_SETPOINT = 0.5
RATE_MAX = 0.05      # multiplicative decay rate when in improvement zone
RATE_MIN = 0.001
QUALITY_LIMIT = 5.0
T_TARGET = 1.0       # BitNet-territory cap


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
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(arr.std() / max(abs(arr.mean()), 1e-12)),
        "max_abs": float(np.abs(arr).max()),
        "p99": float(np.percentile(np.abs(arr), 99)),
        "p99_9": float(np.percentile(np.abs(arr), 99.9)),
        "n_above_T": lambda T: int((np.abs(arr) > T).sum()),
    }


def measure_norm_stats_full(norm_params, T):
    all_g = []
    for p in norm_params:
        all_g.extend(p.detach().float().flatten().cpu().numpy().tolist())
    arr = np.array(all_g)
    return {
        "mean": float(arr.mean()),
        "max_abs": float(np.abs(arr).max()),
        "n_above_T": int((np.abs(arr) > T).sum()),
        "n_total": int(arr.size),
    }


def pid_rate(drift, setpoint):
    if drift <= 0:
        return RATE_MAX
    elif drift < setpoint:
        frac = drift / setpoint
        return RATE_MAX * (1 - frac) + RATE_MIN * frac
    elif drift < 2 * setpoint:
        return 0
    else:
        return -0.01  # back off (raise cap)


print(f"device={device} dtype={dtype}")
print(f"Sharpness anneal — outlier capping with PID")
print(f"  setpoint={PID_SETPOINT}, target T={T_TARGET}")
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
init_max = max(p.detach().float().abs().max().item() for p in norm_params)
init_stats = measure_norm_stats_full(norm_params, T=10)
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial norm: mean={init_stats['mean']:.3f} max_abs={init_max:.1f}")
print(f"  channels with |gain| > 10: {init_stats['n_above_T']}")


print(f"\n{'='*70}")
print("PID sharpness anneal (outlier capping)")
print('='*70)

trajectory = [{
    "cycle": 0, "T_cap": init_max,
    "ce": float(T0), "drift": 0.0,
    "norm_max": init_max, "n_clipped": 0,
}]

current_T = init_max
broke_at = None
for cycle in range(1, N_CYCLES + 1):
    prev_drift = trajectory[-1]["drift"]
    rate = pid_rate(prev_drift, PID_SETPOINT)
    new_T = current_T * (1 - rate)
    new_T = max(T_TARGET * 0.5, new_T)   # don't go below half target

    # Cap: clamp |gain| to new_T, preserving sign
    n_clipped_total = 0
    with torch.no_grad():
        for p, p_orig in zip(norm_params, original_norms):
            # Reset to original and re-apply cap (so we measure cumulative cap effect)
            p.data = p_orig.data.clone()
            sign = torch.sign(p.data)
            mag = p.data.abs()
            n_clipped_total += int((mag > new_T).sum().item())
            p.data = sign * torch.clamp(mag, max=new_T)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    cur_max = max(p.detach().float().abs().max().item() for p in norm_params)

    trajectory.append({
        "cycle": cycle, "T_cap": float(new_T),
        "ce": float(ce), "drift": float(drift),
        "norm_max": float(cur_max), "n_clipped": n_clipped_total,
        "rate": float(rate),
    })

    if cycle <= 5 or cycle % 5 == 0 or abs(drift - prev_drift) > 0.5:
        marker = "↓" if rate > 0 else ("↑" if rate < 0 else "·")
        print(f"  cycle {cycle:>3}/{N_CYCLES}  T={new_T:7.2f} {marker} rate={rate:.4f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  clipped={n_clipped_total} norm_max={cur_max:.1f}",
              flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING")
        break

    if new_T <= T_TARGET:
        print(f"  Reached target T={T_TARGET}")
        break

    current_T = new_T


print("\n" + "=" * 70)
print("PID SHARPNESS ANNEAL COMPLETE")
print("=" * 70)
final = trajectory[-1]
best_idx = min(range(len(trajectory)), key=lambda i: trajectory[i]["drift"])
best = trajectory[best_idx]
print(f"  Final cycle: {final['cycle']}  T={final['T_cap']:.2f}  norm_max={final['norm_max']:.1f}  drift{final['drift']:+.4f}")
print(f"  Channels clipped at end: {final['n_clipped']} / {init_stats['n_total']}")
print(f"  Best drift seen: {best['drift']:+.4f} at cycle {best['cycle']}  T={best['T_cap']:.2f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} at cycle {broke_at}")

print(f"\nCOMPARE:")
print(f"  Stage 202 (uniform scaling): plateau at norm_max=162 with drift +0.50")
print(f"  Stage 203 (sharpness):       reached norm_max={final['norm_max']:.1f} with drift {final['drift']:+.4f}")
print(f"  BitNet target:               norm_max ≈ 1.01")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "init_norm_max": float(init_max),
        "T_target": T_TARGET,
        "pid_setpoint": PID_SETPOINT,
        "rate_range": [RATE_MIN, RATE_MAX],
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "best_drift": float(best["drift"]),
        "best_T_cap": float(best["T_cap"]),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
