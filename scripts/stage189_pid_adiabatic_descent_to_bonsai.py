"""Stage 189: PID-controlled adiabatic descent from FP toward Bonsai.

User's reframe (2026-04-29): "split up the rows like bonsai but tons
more, full model with all the numbers, then PID controller and remove
each one until we reach binary."

The geometry: Bonsai's representation is sign × per-group-scale + bias.
At group_size=in_features (one scale per row), we have 1 bit/weight.
At group_size=1 (per-element scale), we have lossless FP.

Anywhere in between is a continuous knob:
  bits/weight = 1 + 16/group_size  (for FP16 scales)
  group=2     → 9   bits/weight  (almost lossless)
  group=8     → 3   bits/weight
  group=32    → 1.5 bits/weight
  group=128   → 1.125 bits/weight   (Bonsai's native)
  group=4096  → 1.004 bits/weight   (~pure binary)

Adiabatic descent: instead of one-shot binarization (Stage 188's
displacement off the attractor), grow group_size step by step. At each
step, train master + compensation to absorb the increase. PID throttle:
if CE drift exceeds setpoint, hold and train more before advancing.

This subsumes Stages 169 (= group=4096), 180 (= group=128), 184/187/188
(= group=128 + frozen master). Difference: master is TRAINABLE here.

Why this should work where Stage 188 didn't:
  Stage 188 displaced the system from its FP attractor in one step
  (clamped 8333 outliers, scaled embed by 2.5×, then asked compensation
  to recover) — got +3.76 nat plateau.
  Stage 189 keeps the system at its attractor at every level — the
  attractor *deforms continuously* under the constraint, and master
  weights track it.
"""
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 64
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR_MASTER = 1e-5
LR_NORMS_ALPHA = 5e-4
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage189_pid_adiabatic_descent.json")

# All target linears get GroupBinaryLinear wrapper. But only the
# BOTTLENECK ones (o_proj, down_proj per Finding 27) have trainable
# master — others stay at fixed group_size=128 (Bonsai) with frozen
# master. This tests whether master training on the bottleneck
# projections + PID descent closes the gap, while keeping memory
# manageable on Mac M4.
ALL_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
TRAINABLE_MASTER_NAMES = ("o_proj", "down_proj")  # subset that gets trained
FROZEN_GROUP_SIZE = 128  # group_size used for non-trainable linears

# Adiabatic schedule — only 3 levels for proof-of-concept
GROUP_SIZE_SCHEDULE = [16, 64, 128]

# PID parameters
PID_SETPOINT_DRIFT = 0.10   # target CE drift above base (nats)
PID_KP = 1.0                # proportional gain
PID_MAX_TRAIN_AT_LEVEL = 100  # cap on extra training steps per level
TRAIN_STEPS_BASELINE = 50   # default per-level training


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


class GroupBinaryLinear(nn.Module):
    """Linear with sign × per-group scale projection on forward, STE
    backward. group_size is mutable — change it to adjust quantization
    granularity during training."""
    def __init__(self, original_module, initial_group_size=2):
        super().__init__()
        self.weight = nn.Parameter(original_module.weight.data.clone())
        self.bias = original_module.bias
        self.group_size = initial_group_size
        # Per-row α to absorb residual scale
        rn = self.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.alpha = nn.Parameter(rn.squeeze(-1).clone().to(torch.float32))

    def project(self, W, group_size):
        out_features, in_features = W.shape
        if group_size == 1:
            return W
        if in_features % group_size != 0:
            scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            return torch.sign(W) * scale
        n_groups = in_features // group_size
        grouped = W.reshape(out_features, n_groups, group_size)
        scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return (torch.sign(grouped) * scales).reshape(out_features, in_features)

    def forward(self, x):
        w = self.weight
        w_q = self.project(w.float(), self.group_size).to(x.dtype)
        # STE: forward uses w_q, backward acts as identity through projection
        w_eff = w + (w_q - w).detach()
        # Renormalize to unit rows; α captures scale
        rn = w_eff.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_unit = (w_eff.float() / rn).to(x.dtype)
        out = F.linear(x, w_unit, self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


class GroupSizePID:
    """Simple PID-style controller for group_size advancement."""
    def __init__(self, setpoint=PID_SETPOINT_DRIFT, Kp=PID_KP):
        self.setpoint = setpoint
        self.Kp = Kp
        self.last_error = 0.0
        self.cumulative_error = 0.0

    def decide(self, observed_drift):
        """Return (advance: bool, extra_train_steps: int)."""
        error = observed_drift - self.setpoint
        self.cumulative_error += error
        derivative = error - self.last_error
        self.last_error = error

        if observed_drift > self.setpoint:
            # Drift exceeds setpoint — hold and train more
            extra = min(int(PID_MAX_TRAIN_AT_LEVEL * (self.Kp * error)),
                        PID_MAX_TRAIN_AT_LEVEL)
            return False, max(extra, 50)
        else:
            # Within tolerance — advance
            return True, 0


print(f"device={device} dtype={dtype}")
print(f"Schedule: {GROUP_SIZE_SCHEDULE}")
print(f"PID setpoint: drift ≤ {PID_SETPOINT_DRIFT} nats")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)


# ─── Reference: base FP CE ───
print("\nMeasuring base FP CE (reference)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

print("Loading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)

T0 = lm_ce(ref_model, val_tokens)
print(f"T0 base FP: CE={T0:.4f}  ppl={math.exp(T0):.2f}")
del ref_model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


# ─── Set up model with GroupBinaryLinear wrappers ───
print("\nLoading model and installing GroupBinaryLinear wrappers...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in ALL_TARGET_NAMES): continue
    target_mods.append((name, mod))

parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

# Wrap every target linear; track which ones get the descent treatment.
descending_layers = []   # o_proj + down_proj — group_size moves through schedule
fixed_layers = []        # q/k/v/gate/up — frozen at group=128, master frozen
for full_name, mod in target_mods:
    is_descending = any(t in full_name for t in TRAINABLE_MASTER_NAMES)
    init_group = GROUP_SIZE_SCHEDULE[0] if is_descending else FROZEN_GROUP_SIZE
    new_layer = GroupBinaryLinear(mod, initial_group_size=init_group)
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    if is_descending:
        descending_layers.append(new_layer)
    else:
        fixed_layers.append(new_layer)

# Trainable: master weights of o,down + α (all) + RMSNorm gains
master_params = [g.weight for g in descending_layers]   # only o,down masters trainable
alpha_params = [g.alpha for g in descending_layers + fixed_layers]
for g in fixed_layers:
    g.weight.requires_grad = False  # freeze non-bottleneck masters
norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n and "embed" not in n.lower():
        p.requires_grad = True
        norm_params.append(p)
for p in master_params:
    p.requires_grad = True
for p in alpha_params:
    p.requires_grad = True

n_master = sum(p.numel() for p in master_params)
n_alpha = sum(p.numel() for p in alpha_params)
n_norm = sum(p.numel() for p in norm_params)
print(f"  descending linears (o_proj+down_proj): {len(descending_layers)} (master trainable)")
print(f"  fixed linears (q/k/v/gate/up): {len(fixed_layers)} (frozen master, group=128)")
print(f"  trainable: {n_master:,} master + {n_alpha:,} α + {n_norm:,} norm = {n_master + n_alpha + n_norm:,}")


# Two LR groups: master gets lower LR (it's bigger), norms+α get higher
opt = torch.optim.AdamW([
    {"params": master_params, "lr": LR_MASTER},
    {"params": alpha_params, "lr": LR_NORMS_ALPHA},
    {"params": norm_params, "lr": LR_NORMS_ALPHA},
], weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_steps(it, n_steps, label=""):
    """Train n_steps with current group_size."""
    model.train()
    for step in range(n_steps):
        opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(master_params + alpha_params + norm_params, GRAD_CLIP)
        opt.step()


# ─── Adiabatic descent ───
it = iter_train()
trajectory = []
pid = GroupSizePID()

print("\n" + "=" * 70)
print("Starting adiabatic descent")
print("=" * 70)

for level_idx, group_size in enumerate(GROUP_SIZE_SCHEDULE):
    # Set new group_size on descending linears only
    for g in descending_layers:
        g.group_size = group_size

    # Initial measurement at this level
    init_ce = lm_ce(model, val_tokens)
    init_drift = init_ce - T0
    print(f"\nLevel {level_idx+1}/{len(GROUP_SIZE_SCHEDULE)}  group_size={group_size}  "
          f"bits/w≈{1 + 16/group_size:.2f}")
    print(f"  init CE={init_ce:.4f}  drift={init_drift:+.4f}")

    # Train baseline steps
    train_steps(it, TRAIN_STEPS_BASELINE)
    post_train_ce = lm_ce(model, val_tokens)
    post_drift = post_train_ce - T0

    # PID throttle: if drift still high, train more
    advance, extra_steps = pid.decide(post_drift)
    extra_done = 0
    while not advance and extra_done < PID_MAX_TRAIN_AT_LEVEL:
        train_steps(it, extra_steps)
        extra_done += extra_steps
        post_train_ce = lm_ce(model, val_tokens)
        post_drift = post_train_ce - T0
        advance, extra_steps = pid.decide(post_drift)
        print(f"    PID hold: extra {extra_done} steps total, drift={post_drift:+.4f}")

    final_ce = post_train_ce
    print(f"  level done  CE={final_ce:.4f}  drift={final_ce-T0:+.4f}  "
          f"(+{TRAIN_STEPS_BASELINE+extra_done} steps)")

    trajectory.append({
        "level": level_idx + 1,
        "group_size": group_size,
        "bits_per_weight": 1 + 16 / group_size,
        "init_ce": float(init_ce),
        "init_drift": float(init_drift),
        "final_ce": float(final_ce),
        "final_drift": float(final_ce - T0),
        "extra_train_steps": int(extra_done),
    })


# ─── Final summary ───
print("\n" + "=" * 70)
print("ADIABATIC DESCENT COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):  {T0:.4f}")
print(f"\n  {'level':>5}  {'group':>6}  {'bits/w':>7}  {'init Δ':>9}  {'final Δ':>9}  {'+steps':>7}")
for t in trajectory:
    print(f"  {t['level']:>5}  {t['group_size']:>6}  {t['bits_per_weight']:>7.2f}  "
          f"{t['init_drift']:>+9.4f}  {t['final_drift']:>+9.4f}  {t['extra_train_steps']:>7}")

print(f"\n  Final at group_size={GROUP_SIZE_SCHEDULE[-1]} (≈Bonsai): Δ={trajectory[-1]['final_drift']:+.4f}")
print(f"  Stage 188 (one-shot precondition):  Δ=+3.7595")

improvement = 3.7595 - trajectory[-1]['final_drift']
if improvement > 1.0:
    print(f"\n  ✓✓ MAJOR UNLOCK: adiabatic descent {improvement:.2f} nats below one-shot.")
elif improvement > 0.3:
    print(f"\n  ✓ Improvement: adiabatic {improvement:.2f} nats below one-shot.")
elif improvement > -0.1:
    print(f"\n  - Comparable: adiabatic descent ≈ one-shot ({improvement:+.2f}).")
else:
    print(f"\n  ✗ Adiabatic worse than one-shot by {-improvement:.2f}.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "schedule": GROUP_SIZE_SCHEDULE,
        "T0_base_ce": float(T0),
        "trajectory": trajectory,
        "final_drift": trajectory[-1]['final_drift'],
        "stage_188_baseline_delta": 3.7595,
        "improvement_vs_188": float(improvement),
        "config": {
            "lr_master": LR_MASTER,
            "lr_norms_alpha": LR_NORMS_ALPHA,
            "train_steps_baseline": TRAIN_STEPS_BASELINE,
            "pid_setpoint": PID_SETPOINT_DRIFT,
            "pid_kp": PID_KP,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
