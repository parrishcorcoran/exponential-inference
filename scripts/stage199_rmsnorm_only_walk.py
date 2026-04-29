"""Stage 199: RMSNorm growing alone, no shape pressure.

Companion to Stage 196 (shape alone). Measures how much damage
RMSNorm × (1 + cycle×rate) does on its own with NO shape change.

Then we have two independent damage curves:
  - Stage 196: shape alone — broke at cycle 27 (~+5 nat)
  - Stage 199 (here): RMSNorm alone — breaks at cycle ?

If RMSNorm alone walks 27 cycles to break point, both axes contribute
roughly equal damage independently. Combined Stage 194 (broke at 22)
is just the sum of two damage sources.

If RMSNorm alone walks much further (e.g., 50+), RMSNorm growth itself
is benign. Stage 194's earlier break came from RMSNorm-shape
INTERACTION, not RMSNorm damage.

If RMSNorm alone walks shorter (e.g., 15), RMSNorm growth IS the bigger
damage source.
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
RESULTS_PATH = Path("results/stage199_rmsnorm_only_walk.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 80
COMP_RATE = 0.01     # only knob — RMSNorm grows per cycle
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


print(f"device={device} dtype={dtype}")
print(f"RMSNorm-only walk: {N_CYCLES} cycles, comp_rate={COMP_RATE}")
print("NO shape change — just RMSNorm grows each cycle")

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
print(f"\nT0 base FP CE: {T0:.4f}")
print(f"Initial drift: +0.0000")


print(f"\n{'='*70}")
print("RMSNorm-only walk (no shape change)")
print('='*70)

trajectory = [{"cycle": 0, "comp_factor": 1.0, "ce": float(T0), "drift": 0.0}]

broke_at = None
for cycle in range(1, N_CYCLES + 1):
    comp_factor = 1.0 + cycle * COMP_RATE

    with torch.no_grad():
        for p, p_orig in zip(norm_params, original_norms):
            p.data = (p_orig.float() * comp_factor).to(p.dtype)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    trajectory.append({"cycle": cycle, "comp_factor": float(comp_factor),
                       "ce": float(ce), "drift": float(drift)})

    if cycle <= 5 or cycle % 5 == 0:
        print(f"  cycle {cycle:>3}/{N_CYCLES}  comp×{comp_factor:.3f}  "
              f"CE={ce:.4f} drift={drift:+.4f}", flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING: drift > 10")
        break


print("\n" + "=" * 70)
print("RMSNORM-ONLY WALK COMPLETE")
print("=" * 70)
final = trajectory[-1]
print(f"  Final cycle: {final['cycle']}  comp×{final['comp_factor']:.3f}  drift{final['drift']:+.4f}")
if broke_at is not None:
    print(f"  Broke past +{QUALITY_LIMIT} nat at cycle {broke_at}")
else:
    print(f"  Walked all {N_CYCLES} cycles without breaking")

print(f"\nCOMPARE to Stage 196 (shape alone):")
print(f"  Stage 196 (shape only): broke at cycle 27, walked 36")
print(f"  Stage 199 (RMSNorm only): broke at cycle {broke_at}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "comp_rate": COMP_RATE,
        "n_cycles": N_CYCLES,
        "trajectory": trajectory,
        "broke_at_cycle": broke_at,
        "stage_196_baseline": 27,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
