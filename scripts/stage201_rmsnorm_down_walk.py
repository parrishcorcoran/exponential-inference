"""Stage 201: RMSNorm shrinking alone, no shape change. Mirror of Stage 199.

Stage 199: RMSNorm × (1 + cycle×0.01) — grow up. Predicted break ~cycle 80.
Stage 201: RMSNorm × (1 - cycle×0.01) — shrink down. Break point?

Tests the symmetry / direction question. Going DOWN moves us toward
BitNet's RMSNorm shape (mean 0.47 vs Qwen 2.68; max 1.01 vs Qwen 192).
Outlier channels (Qwen's 192×) get squeezed toward unity.

If breaks happen at similar cycles (UP vs DOWN), RMSNorm scaling is
direction-agnostic damage. If DOWN walks further, it's slightly
beneficial direction (squeezing outliers toward target).
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
RESULTS_PATH = Path("results/stage201_rmsnorm_down_walk.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 100
SHRINK_RATE = 0.01    # RMSNorm shrinks per cycle
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
    all_gains = []
    for p in norm_params:
        all_gains.extend(p.detach().float().flatten().cpu().numpy().tolist())
    arr = np.array(all_gains)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "n_above_10x_mean": int((arr > 10 * arr.mean()).sum()),
    }


print(f"device={device} dtype={dtype}")
print(f"RMSNorm-DOWN walk: {N_CYCLES} cycles, shrink_rate={SHRINK_RATE}")

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
print(f"Initial RMSNorm gains: mean={init_stats['mean']:.3f} max={init_stats['max']:.1f}")
print(f"  channels above 10× mean: {init_stats['n_above_10x_mean']}")
print(f"Initial drift: +0.0000")


print(f"\n{'='*70}")
print("RMSNorm-DOWN walk")
print('='*70)

trajectory = [{
    "cycle": 0, "comp_factor": 1.0,
    "ce": float(T0), "drift": 0.0,
    "norm_max": init_stats["max"], "norm_mean": init_stats["mean"],
    "n_above_10x": init_stats["n_above_10x_mean"],
}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    comp_factor = max(0.001, 1.0 - cycle * SHRINK_RATE)

    with torch.no_grad():
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * comp_factor).to(p.dtype)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    stats = measure_norm_stats(norm_params)
    trajectory.append({
        "cycle": cycle, "comp_factor": float(comp_factor),
        "ce": float(ce), "drift": float(drift),
        "norm_max": stats["max"], "norm_mean": stats["mean"],
        "n_above_10x": stats["n_above_10x_mean"],
    })

    if cycle <= 5 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  comp×{comp_factor:.3f}  "
              f"CE={ce:.4f} drift={drift:+.4f}  norm_max={stats['max']:.1f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING: drift > 10")
        break


print("\n" + "=" * 70)
print("RMSNORM-DOWN WALK COMPLETE")
print("=" * 70)
final = trajectory[-1]
print(f"  Final cycle: {final['cycle']}  comp×{final['comp_factor']:.3f}  drift {final['drift']:+.4f}")
print(f"  Final norm: mean={final['norm_mean']:.3f} max={final['norm_max']:.1f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} at cycle {broke_at}")

print(f"\nSYMMETRY check:")
print(f"  Stage 199 RMSNorm UP (× 1.01/cycle): predicted break ~cycle 80")
print(f"  Stage 201 RMSNorm DOWN (× 0.99/cycle): broke at cycle {broke_at}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "shrink_rate": SHRINK_RATE,
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
