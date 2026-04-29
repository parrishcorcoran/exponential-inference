"""Stage 178: validate α theory by predicting from base weights.

Stage 169 trained α on top of unit-norm Qwen3-0.6B (synthetic τ=1.0)
and got Δ=-0.121 nats improvement. The α was initialized to base
row_norms and then trained.

Theory predicts: optimal α[i] should preserve output magnitude per
channel. Mathematically: α[i] = ||W_base[i, :]|| (the original row
norm), because then the projected forward equals the original forward
exactly (this is what T2 in stage 169 confirmed: Δ=0 with α=row_norm).

Stage 169 T3 (trained α) achieved Δ=-0.121 (better than identity).
Question: where did α drift? Was the change small (close to row_norm)?
Or did it find a structurally different optimum?

We can answer this theoretically by re-running Stage 169 briefly and
saving final α values. Compare to:
  - Initialization: α = row_norm
  - Theoretical "optimal": α that minimizes layer-output reconstruction
    loss (closed-form: it's just row_norm — but the LM loss optimum
    is different)

Output: histogram of (α_trained / α_init) ratios. If most are 1.0,
training barely moved α. If they vary, training found new structure.
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
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR = 2e-5
GRAD_CLIP = 1.0
TRAIN_STEPS = 500
RESULTS_PATH = Path("results/stage178_alpha_theory.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt(tokenizer, max_tokens, skip=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# Find target linears, capture row norms
target_mods = []
row_norms_init = {}  # full per-row L2 norm
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    rn = mod.weight.data.norm(dim=-1, keepdim=True).clone().float()
    row_norms_init[name] = rn

# Project to unit norm
for name, mod in target_mods:
    rn = row_norms_init[name].to(mod.weight.dtype)
    mod.weight.data = mod.weight.data / rn.clamp(min=1e-8)


class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight
        self.bias = original_module.bias
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone().to(self.weight.device).to(torch.float32))

    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


parent_lookup = {}
for name, mod in model.named_modules():
    for child_name, child_mod in mod.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (mod, child_name)
alphas = {}
for full_name, mod in target_mods:
    parent, child_attr = parent_lookup[full_name]
    new_layer = AlphaLinear(mod, row_norms_init[full_name])
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

# Save initial α (= row norms)
alpha_init_snapshot = {name: a.detach().clone().cpu().numpy() for name, a in alphas.items()}

# Train α for 500 steps (replicate stage 169 T3)
opt = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


it = iter_train()
print(f"\nTraining α for {TRAIN_STEPS} steps...")
model.train()
for step in range(TRAIN_STEPS):
    opt.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = model(ids[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_([a for a in alphas.values()], GRAD_CLIP)
    opt.step()
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}  loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

# Snapshot α after training
alpha_trained_snapshot = {name: a.detach().clone().cpu().numpy() for name, a in alphas.items()}


# ─── Compare ───
print(f"\n{'='*70}\nα drift analysis: trained vs initial (= row_norm)\n{'='*70}")

all_init = []
all_trained = []
all_ratios = []
per_type_ratios = {t: [] for t in TARGET_NAMES}
for name, a_init in alpha_init_snapshot.items():
    a_trained = alpha_trained_snapshot[name]
    ratio = a_trained / np.maximum(a_init, 1e-8)
    all_init.extend(a_init.tolist())
    all_trained.extend(a_trained.tolist())
    all_ratios.extend(ratio.tolist())
    for t in TARGET_NAMES:
        if t in name:
            per_type_ratios[t].extend(ratio.tolist())
            break

all_init = np.array(all_init)
all_trained = np.array(all_trained)
all_ratios = np.array(all_ratios)

print(f"\n  Initial α (= row_norm):")
print(f"    mean: {all_init.mean():.4f}, std: {all_init.std():.4f}, range: [{all_init.min():.3f}, {all_init.max():.3f}]")
print(f"  Trained α:")
print(f"    mean: {all_trained.mean():.4f}, std: {all_trained.std():.4f}, range: [{all_trained.min():.3f}, {all_trained.max():.3f}]")
print(f"\n  α_trained / α_init (drift ratio):")
print(f"    mean: {all_ratios.mean():.4f}    (1.0 = no drift)")
print(f"    std: {all_ratios.std():.4f}")
print(f"    p1: {np.percentile(all_ratios, 1):.4f}")
print(f"    p99: {np.percentile(all_ratios, 99):.4f}")
print(f"    fraction within [0.95, 1.05]: {((all_ratios > 0.95) & (all_ratios < 1.05)).mean():.4f}")

print(f"\n  By projection type — mean drift ratio:")
type_summary = {}
for t in TARGET_NAMES:
    rs = np.array(per_type_ratios[t])
    if len(rs) == 0: continue
    type_summary[t] = {
        "mean_ratio": float(rs.mean()),
        "std_ratio": float(rs.std()),
        "n": len(rs),
    }
    print(f"    {t:<14} n={len(rs):<8} mean_ratio={rs.mean():.4f}  std={rs.std():.4f}")


# Theoretical prediction check
print(f"\n{'='*70}\nTHEORY VERDICT\n{'='*70}")
within_5pct = ((all_ratios > 0.95) & (all_ratios < 1.05)).mean()
mean_drift = all_ratios.mean()
if abs(mean_drift - 1.0) < 0.05 and within_5pct > 0.5:
    print(f"  Most α stayed near init (drift mean={mean_drift:.3f}, {within_5pct*100:.0f}% within ±5%)")
    print(f"  Theory prediction (α ≈ row_norm) validated. Training found small refinement on top.")
else:
    print(f"  α drifted significantly (mean drift {mean_drift:.3f}, only {within_5pct*100:.0f}% within ±5%)")
    print(f"  Training found structurally different optimum than row_norm.")
    print(f"  → Init α to row_norm gives the right ballpark but not the exact target.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "train_steps": TRAIN_STEPS,
        "alpha_init_stats": {
            "mean": float(all_init.mean()),
            "std": float(all_init.std()),
            "min": float(all_init.min()),
            "max": float(all_init.max()),
        },
        "alpha_trained_stats": {
            "mean": float(all_trained.mean()),
            "std": float(all_trained.std()),
            "min": float(all_trained.min()),
            "max": float(all_trained.max()),
        },
        "drift_ratio_stats": {
            "mean": float(all_ratios.mean()),
            "std": float(all_ratios.std()),
            "p1": float(np.percentile(all_ratios, 1)),
            "p99": float(np.percentile(all_ratios, 99)),
            "fraction_within_5pct": float(within_5pct),
        },
        "by_type": type_summary,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
