"""Stage 198: RMSNorm bump sweep — fixed RMSNorm × factor, then walk shape.

User's hypothesis: maybe linear RMSNorm GROWING was the problem (each
cycle's compounding mismatch), but a fixed RMSNorm BUMP at the start
might extend the walk.

Sweep: try several bump values, walk shape alone, measure where each
breaks. The curve tells us whether RMSNorm scaling is helpful at any
fixed value, or hurts at every value.

Bump values: [1.00 (control), 0.95, 0.98, 1.02, 1.05, 1.10]

For each: scale RMSNorm gains × bump (one-time), then walk shape down
linearly until break or 60 cycles.
"""
import json
import math
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage198_rmsnorm_bump_sweep.json")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

N_CYCLES = 60
SHAPE_RATE = 0.01
QUALITY_LIMIT = 5.0
BUMP_VALUES = [1.00, 0.95, 0.98, 1.02, 1.05, 1.10]   # control + sweep


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


def k1_residual(target_modules, group_size=128):
    errs = []
    for mod in target_modules:
        W = mod.weight.detach().float()
        of, inf = W.shape
        if inf % group_size != 0: continue
        n_groups = inf // group_size
        grouped = W.reshape(of, n_groups, group_size)
        scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        W_q = (torch.sign(grouped) * scales).reshape(of, inf)
        errs.append(((W - W_q).norm() / W.norm().clamp(min=1e-8)).item())
    return float(np.mean(errs)) if errs else 0.0


print(f"device={device} dtype={dtype}")
print(f"RMSNorm bump sweep — bumps={BUMP_VALUES}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("\nLoading val tokens...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()


def run_walk(bump):
    """Load fresh model, apply RMSNorm bump, walk shape, return break cycle."""
    print(f"\n{'─' * 70}")
    print(f"Run with RMSNorm × {bump:.3f}")
    print('─' * 70)

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
    original_row_norms = [w.float().norm(dim=-1, keepdim=True).clone() for w in original_body]

    # Apply RMSNorm bump (one-time, then hold)
    with torch.no_grad():
        for p in norm_params:
            p.data = (p.data.float() * bump).to(p.dtype)

    T0 = lm_ce(model, val_tokens)
    init_k1err = k1_residual(target_modules, GROUP_SIZE)
    print(f"  T0 (post-bump): {T0:.4f}  K1err={init_k1err:.3f}")

    walk_traj = []
    broke_at = None
    for cycle in range(1, N_CYCLES + 1):
        exponent = max(0.0, 1.0 - cycle * SHAPE_RATE)
        with torch.no_grad():
            for m, w_orig, rn_orig in zip(target_modules, original_body, original_row_norms):
                W = w_orig.float()
                W_new = torch.sign(W) * W.abs().pow(exponent)
                new_norms = W_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                W_new = W_new * (rn_orig / new_norms)
                m.weight.data = W_new.to(m.weight.dtype)
        ce = lm_ce(model, val_tokens)
        drift = ce - T0
        k1err = k1_residual(target_modules, GROUP_SIZE)
        walk_traj.append({"cycle": cycle, "exp": exponent, "drift": drift, "k1err": k1err})

        if cycle <= 3 or cycle % 10 == 0:
            print(f"    cycle {cycle:>3}/{N_CYCLES}  exp={exponent:.2f}  drift={drift:+.4f}  K1err={k1err:.3f}", flush=True)

        if drift > QUALITY_LIMIT and broke_at is None:
            broke_at = cycle
            print(f"    ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
        if drift > 10.0:
            print(f"    STOPPING at +10 nat drift")
            break

    print(f"  Done bump={bump:.3f}: broke at cycle {broke_at}")
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return {"bump": bump, "T0_post_bump": float(T0), "init_k1err": float(init_k1err),
            "broke_at_cycle": broke_at, "trajectory": walk_traj}


results = []
for b in BUMP_VALUES:
    results.append(run_walk(b))


print("\n" + "=" * 70)
print("RMSNORM BUMP SWEEP COMPLETE")
print("=" * 70)
print(f"  {'bump':>8}  {'broke at cycle':>16}")
print(f"  {'-'*8}  {'-'*16}")
for r in results:
    bc = r["broke_at_cycle"] if r["broke_at_cycle"] is not None else "60+"
    print(f"  {r['bump']:>8.3f}  {bc:>16}")

print(f"\nBaseline (Stage 196, no bump): broke at cycle 27")
best = max(results, key=lambda r: r["broke_at_cycle"] or 0)
print(f"Best in this sweep: bump={best['bump']:.3f} broke at cycle {best['broke_at_cycle']}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "bump_values": BUMP_VALUES,
        "n_cycles": N_CYCLES,
        "shape_rate": SHAPE_RATE,
        "results": results,
        "stage_196_baseline": 27,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
