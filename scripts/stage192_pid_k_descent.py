"""Stage 192: PID-controlled K-binary descent from K=2 toward K=1.

User's reframe (2026-04-29): "could we do PID K" / "K=1.9".

Continuous K between 1 and 2 via soft weight on second binary basis:

  W ≈ α₁·B₁ + w₂·α₂·B₂

where B_i is binary {-1, +1}, α_i is per-row FP scalar, and w₂ ∈ [0,1].

  w₂ = 1.0 → full K=2  (probably lossless, ≈2 bits/weight)
  w₂ = 0.0 → full K=1  (pure binary, 1 bit/weight)
  w₂ = 0.9 → "K=1.9"   (intermediate, ≈1.9 bits/weight equivalent)

Stage 191's κ-PID had the issue that κ=0 wasn't truly lossless (master and
α drifted apart). This formulation fixes that: at w₂=1.0 we're at full
K=2 representation which IS lossless within K=2's capacity (2.0 bits per
weight), and PID descends from there.

Mechanism:
  1. Start every layer at w₂=1.0 (full K=2)
  2. PID descends w₂ uniformly across layers based on CE drift setpoint
  3. Train master + aux throughout
  4. Stopping point = minimum sufficient K for lossless quality

If lossless reached at w₂=0.0 → pure binary works.
If lossless reached at w₂=0.5 → K=1.5 average is the floor.
If lossless lost partway → we've found the wall.
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
LR_AUX = 5e-4
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage192_pid_k_descent.json")

ALL_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
TRAINABLE_MASTER_NAMES = ("o_proj", "down_proj")
GROUP_SIZE = 128

# PID descent on w₂ (K=2 weight)
N_CYCLES = 50
TRAIN_STEPS_PER_CYCLE = 30
W2_INITIAL = 1.0    # start at K=2 lossless
W2_TARGET = 0.0     # try to reach K=1 (pure binary)
PID_SETPOINT_DRIFT = 0.05
PID_KP = 1.0
MAX_W2_STEP = (W2_INITIAL - W2_TARGET) / N_CYCLES   # nominal step


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


class K2BinaryLinear(nn.Module):
    """Linear with K=2 binary basis decomposition.

    Forward: W_eff = α₁·B₁ + w₂·α₂·B₂
    where B_i = sign(R_i), α_i = mean(|R_i|) per group, R₂ = R₁ - α₁·B₁
    and w₂ ∈ [0, 1] is the soft weight on the second basis (PID-controlled).

    STE backward: gradient flows to master weight as identity through
    the projection.
    """
    def __init__(self, original_module, group_size=128):
        super().__init__()
        self.weight = nn.Parameter(original_module.weight.data.clone())
        self.bias = original_module.bias
        self.group_size = group_size
        self.w2 = W2_INITIAL  # mutable, set externally by PID

    def k_binary_project(self, W, w2):
        """K=2 binary projection with continuous w2 weight on second basis."""
        out_features, in_features = W.shape
        if in_features % self.group_size != 0:
            # Fallback: per-row decomposition
            alpha1 = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            B1 = torch.sign(W)
            R1 = W - alpha1 * B1
            alpha2 = R1.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            B2 = torch.sign(R1)
            return alpha1 * B1 + w2 * alpha2 * B2

        n_groups = in_features // self.group_size
        W_grouped = W.reshape(out_features, n_groups, self.group_size)

        # First binary basis (per-group α₁)
        alpha1 = W_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        B1 = torch.sign(W_grouped)
        R1 = W_grouped - alpha1 * B1

        # Second binary basis on residual
        alpha2 = R1.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        B2 = torch.sign(R1)

        # Reconstruct
        W_q = alpha1 * B1 + w2 * alpha2 * B2
        return W_q.reshape(out_features, in_features)

    def forward(self, x):
        w = self.weight
        w_q = self.k_binary_project(w.float(), self.w2).to(x.dtype)
        # STE: forward uses w_q, backward acts as identity through projection
        w_eff = w + (w_q - w).detach()
        return F.linear(x, w_eff, self.bias.to(x.dtype) if self.bias is not None else None)


class W2PID:
    """PID controller for w₂ descent from W2_INITIAL toward W2_TARGET."""
    def __init__(self, setpoint=PID_SETPOINT_DRIFT, Kp=PID_KP, max_step=MAX_W2_STEP):
        self.setpoint = setpoint
        self.Kp = Kp
        self.max_step = max_step
        self.last_error = 0.0
        self.cumulative_error = 0.0

    def next_w2(self, current_drift, current_w2):
        """Decide next w₂ value (descending) based on observed drift."""
        error = current_drift - self.setpoint
        self.cumulative_error += error
        self.last_error = error
        # Descend faster when drift is below setpoint, slower (or back up) when above
        rate = 1.0 - self.Kp * (current_drift / max(self.setpoint, 1e-8))
        rate = max(-0.5, min(1.0, rate))
        # We're DESCENDING w2 toward target
        new_w2 = current_w2 - rate * self.max_step
        return max(W2_TARGET, min(W2_INITIAL, new_w2))


print(f"device={device} dtype={dtype}")
print(f"Schedule: w2 from {W2_INITIAL} → {W2_TARGET} (K=2 → K=1)")
print(f"N_CYCLES={N_CYCLES}  TRAIN_STEPS_PER_CYCLE={TRAIN_STEPS_PER_CYCLE}")
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


# ─── Set up model with K2BinaryLinear ───
print("\nLoading model and installing K2BinaryLinear wrappers...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in ALL_TARGET_NAMES): continue
    target_mods.append((name, mod))

parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

descending_layers = []   # o_proj + down_proj — master trainable
fixed_layers = []        # q/k/v/gate/up — master frozen

for full_name, mod in target_mods:
    is_descending = any(t in full_name for t in TRAINABLE_MASTER_NAMES)
    new_layer = K2BinaryLinear(mod, group_size=GROUP_SIZE)
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    if is_descending:
        descending_layers.append(new_layer)
    else:
        fixed_layers.append(new_layer)

# Trainable: master of descending + RMSNorm gains + embed + lm_head
master_params = [g.weight for g in descending_layers]
for g in fixed_layers:
    g.weight.requires_grad = False

aux_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
    if "embed_tokens" in n and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
    if "lm_head" in n and "weight" in n:
        p.requires_grad = True
        if p not in aux_params:
            aux_params.append(p)
for p in master_params:
    p.requires_grad = True

n_master = sum(p.numel() for p in master_params)
n_aux = sum(p.numel() for p in aux_params)
print(f"  descending: {len(descending_layers)} (o_proj+down_proj, master trainable)")
print(f"  fixed:      {len(fixed_layers)} (master frozen)")
print(f"  trainable:  {n_master:,} master + {n_aux:,} aux = {n_master + n_aux:,}")


opt = torch.optim.AdamW([
    {"params": master_params, "lr": LR_MASTER},
    {"params": aux_params, "lr": LR_AUX},
], weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_steps(it, n_steps):
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(master_params + aux_params, GRAD_CLIP)
        opt.step()


def set_w2(w):
    for g in descending_layers + fixed_layers:
        g.w2 = w


# ─── Initial measurement at full K=2 ───
set_w2(1.0)
init_ce_k2 = lm_ce(model, val_tokens)
print(f"\nInitial CE at full K=2 (w₂=1.0): {init_ce_k2:.4f}  drift={init_ce_k2-T0:+.4f}")


# ─── PID descent ───
it = iter_train()
trajectory = []
pid = W2PID()

print("\n" + "=" * 70)
print(f"Starting PID K descent (w₂: {W2_INITIAL} → {W2_TARGET})")
print("=" * 70)

current_w2 = W2_INITIAL
for cycle in range(N_CYCLES):
    set_w2(current_w2)
    train_steps(it, TRAIN_STEPS_PER_CYCLE)
    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    next_w2 = pid.next_w2(drift, current_w2)
    delta = next_w2 - current_w2
    K_eff = 1.0 + current_w2  # effective K (1.0 to 2.0)

    trajectory.append({
        "cycle": cycle + 1,
        "w2": float(current_w2),
        "K_eff": float(K_eff),
        "ce": float(ce),
        "drift": float(drift),
        "next_w2": float(next_w2),
    })

    if (cycle + 1) % 5 == 0 or cycle < 5 or cycle == N_CYCLES - 1:
        marker = "↓" if delta < 0 else ("↑" if delta > 0 else "·")
        print(f"  cycle {cycle+1:>3}/{N_CYCLES}  w2={current_w2:.4f} K_eff={K_eff:.3f} {marker} "
              f"CE={ce:.4f}  drift={drift:+.4f}", flush=True)

    current_w2 = next_w2


# Final settle at the last w2
print(f"\nFinal settle: training {TRAIN_STEPS_PER_CYCLE * 4} more steps at w₂={current_w2:.4f}...")
train_steps(it, TRAIN_STEPS_PER_CYCLE * 4)
final_ce = lm_ce(model, val_tokens)
final_drift = final_ce - T0
final_K = 1.0 + current_w2


# ─── Summary ───
print("\n" + "=" * 70)
print("PID K-DESCENT COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):                {T0:.4f}")
print(f"  Initial CE at full K=2:      {init_ce_k2:.4f}  drift {init_ce_k2-T0:+.4f}")
print(f"  Final w₂:                    {current_w2:.4f}  (K_eff={final_K:.3f})")
print(f"  Final CE:                    {final_ce:.4f}  drift {final_drift:+.4f}")
print(f"  Stage 189 baseline (K=1):    drift +3.159 nats")

if final_drift < 0.1 and final_K < 1.5:
    print(f"\n  ✓✓ MAJOR UNLOCK: lossless at K_eff={final_K:.2f} bits/weight effective.")
    print(f"     Validates sub-1.5 bit lossless hypothesis at LLM scale.")
elif final_drift < 0.5:
    print(f"\n  ✓ Near-lossless at K_eff={final_K:.2f}. Mixed precision could close residual.")
elif final_drift < 1.5:
    print(f"\n  ~ Moderate descent: K_eff={final_K:.2f}, drift {final_drift:.2f}.")
elif final_drift < final_K * 1.5:
    print(f"\n  - PID hit a wall around K_eff={final_K:.2f}. Map of where capacity binds.")
else:
    print(f"\n  ✗ Worse than expected. Recipe may need distillation or other lever.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_cycles": N_CYCLES,
        "T0_base_ce": float(T0),
        "init_ce_k2_lossless": float(init_ce_k2),
        "init_drift_k2": float(init_ce_k2 - T0),
        "trajectory": trajectory,
        "final_w2": float(current_w2),
        "final_K_eff": float(final_K),
        "final_ce": float(final_ce),
        "final_drift": float(final_drift),
        "stage_189_k1_baseline_delta": 3.159,
        "config": {
            "lr_master": LR_MASTER,
            "lr_aux": LR_AUX,
            "pid_setpoint": PID_SETPOINT_DRIFT,
            "pid_kp": PID_KP,
            "group_size": GROUP_SIZE,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
