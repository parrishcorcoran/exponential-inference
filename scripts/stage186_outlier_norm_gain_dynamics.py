"""Stage 186: do RMSNorm gain outliers move during compensation-only
training under binary forward?

Stage 185 found Qwen3-0.6B has RMSNorm gain max = 192 (CV 1.54), while
trained low-bit models flatten this dramatically (BitNet max=1.01,
Bonsai max=34). This stage asks: does our compensation-only training
(Stage 184 setup — binary forward, norms+α trainable, weights frozen)
naturally drift the gain distribution, or do outliers stick?

Procedure:
  1. Load Qwen3-0.6B FP, measure norm-gain distribution
  2. Project rows to unit norm, apply Bonsai-style binary
  3. Set norms + α trainable (Stage 184 setup)
  4. Train 600 steps; record norm-gain distribution every 50 steps

Outcomes:
  - Outliers shrink during training: compensation training does some
    flattening. May be enough; may need explicit regularization.
  - Outliers stay put: gradient descent has no incentive to flatten
    them under binary. Stage 187 (regularization) needed.
  - Outliers grow: training amplifies outliers (worst case for binary).

Cheap: same trainable budget as Stage 184 (~410K params), 600 steps.
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
LR = 5e-4
GRAD_CLIP = 1.0
TRAIN_STEPS = 600
MEASURE_EVERY = 50
GROUP_SIZE = 128
RESULTS_PATH = Path("results/stage186_outlier_norm_dynamics.json")
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
    return sum(losses) / len(losses)


def bonsai_style_quantize(W, group_size=128):
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    W_grouped = W.reshape(out_features, n_groups, group_size)
    scales = W_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return (torch.sign(W_grouped) * scales).reshape(out_features, in_features)


class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight
        self.bias = original_module.bias
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone()
                                  .to(self.weight.device).to(torch.float32))
    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


def measure_norm_gains(model):
    """Aggregate RMSNorm gain distribution across the body."""
    all_gains = []
    per_layer_max = []
    for n, p in model.named_parameters():
        if "norm" in n.lower() and "weight" in n:
            v = p.detach().float().flatten().cpu().numpy()
            all_gains.extend(v.tolist())
            per_layer_max.append({"name": n, "max": float(np.max(v)),
                                  "mean": float(v.mean()), "n": int(v.size)})
    arr = np.array(all_gains)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(arr.std() / max(arr.mean(), 1e-12)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "n_above_10x_mean": int((arr > 10 * arr.mean()).sum()),
        "n_above_50x_mean": int((arr > 50 * arr.mean()).sum()),
        "per_layer_max": per_layer_max[:5],  # top-N just for logging
    }


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)


# ─── Stage A: load FP, measure baseline ───
print("\n" + "=" * 70)
print("Stage A: FP baseline norm gain distribution")
print("=" * 70)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
fp_gains = measure_norm_gains(model)
print(f"  mean={fp_gains['mean']:.3f}  CV={fp_gains['cv']:.3f}  max={fp_gains['max']:.1f}  p99.9={fp_gains['p99_9']:.1f}")
print(f"  n above 10× mean: {fp_gains['n_above_10x_mean']}")
print(f"  n above 50× mean: {fp_gains['n_above_50x_mean']}")


# ─── Stage B: project rows to unit norm, apply Bonsai-binary, install α ───
print("\n" + "=" * 70)
print("Stage B: unit-norm + binary + α-bridge installed (no train yet)")
print("=" * 70)
target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))

# Save original row norms; project to unit; apply binary
original_row_norms = {}
for name, mod in target_mods:
    rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    original_row_norms[name] = rn.clone()
    mod.weight.data = (mod.weight.data.float() / rn).to(mod.weight.dtype)

for name, mod in target_mods:
    W_q = bonsai_style_quantize(mod.weight.data.float(), GROUP_SIZE)
    mod.weight.data = W_q.to(mod.weight.dtype)

# Install AlphaLinear with α = original_row_norm
parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

alphas = {}
for full_name, mod in target_mods:
    binary_rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mod.weight.data = (mod.weight.data.float() / binary_rn).to(mod.weight.dtype)
    new_layer = AlphaLinear(mod, original_row_norms[full_name])
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha

post_binary_gains = measure_norm_gains(model)
print(f"  mean={post_binary_gains['mean']:.3f}  CV={post_binary_gains['cv']:.3f}  max={post_binary_gains['max']:.1f}")
print(f"  (should be IDENTICAL to FP — we never touched norms)")


# ─── Stage C: train norms+α, track gain distribution ───
print("\n" + "=" * 70)
print("Stage C: train norms + α 600 steps, tracking norm gain dynamics")
print("=" * 70)
for p in model.parameters():
    p.requires_grad = False
trainable = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        trainable.append(p)
for a in alphas.values():
    a.requires_grad = True
    trainable.append(a)

opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)

print("\nLoading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


it = iter_train()
init_ce = lm_ce(model, val_tokens)
print(f"\ninit CE: {init_ce:.4f}")

trajectory = [{"step": 0, "ce": float(init_ce), "gains": post_binary_gains}]
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
    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    opt.step()

    if (step + 1) % MEASURE_EVERY == 0:
        ce = lm_ce(model, val_tokens)
        gains = measure_norm_gains(model)
        trajectory.append({"step": step + 1, "ce": float(ce), "gains": gains})
        print(f"  step {step+1:>4}: CE={ce:.4f}  gain_mean={gains['mean']:.3f}  CV={gains['cv']:.3f}  max={gains['max']:.1f}  n>10×: {gains['n_above_10x_mean']}",
              flush=True)


# ─── Summary ───
print("\n" + "=" * 70)
print("SUMMARY: did outliers move?")
print("=" * 70)
init_max = trajectory[0]["gains"]["max"]
final_max = trajectory[-1]["gains"]["max"]
init_cv = trajectory[0]["gains"]["cv"]
final_cv = trajectory[-1]["gains"]["cv"]
init_above10 = trajectory[0]["gains"]["n_above_10x_mean"]
final_above10 = trajectory[-1]["gains"]["n_above_10x_mean"]
print(f"  norm gain max:     {init_max:.1f} → {final_max:.1f}  (Δ = {final_max-init_max:+.1f})")
print(f"  norm gain CV:      {init_cv:.3f} → {final_cv:.3f}  (Δ = {final_cv-init_cv:+.3f})")
print(f"  channels >10× mean: {init_above10} → {final_above10}  (Δ = {final_above10-init_above10:+d})")

if final_max < init_max * 0.5:
    print(f"\n  ✓ OUTLIERS COLLAPSED: max gain halved or more during compensation training.")
elif final_max < init_max * 0.9:
    print(f"\n  ~ Outliers PARTIALLY collapsed (>10% reduction in max).")
elif final_max > init_max * 1.1:
    print(f"\n  ✗ OUTLIERS GREW: gradient amplifies outliers under binary.")
else:
    print(f"\n  - Outliers ESSENTIALLY UNCHANGED. Compensation training has no incentive to flatten them.")
    print(f"    Stage 187 (explicit regularization) is required to drive flattening.")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "fp_baseline_gains": fp_gains,
        "post_binary_gains_no_train": post_binary_gains,
        "trajectory": trajectory,
        "train_steps": TRAIN_STEPS,
        "lr": LR,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
