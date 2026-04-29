"""Stage 191: continuous-κ laser PID descent — Qwen3-0.6B → binary body
weights, all other DOFs free to compensate.

Refines user's reframe (2026-04-29): "if our only goal is size through
quantization, the other factors don't apply." Body weights are 95%+ of
inference cost. Compress only those; leave embed, lm_head, RMSNorm
gains FP and trainable so they can develop whatever compensation
pattern the model needs.

This drops Stage 188's BitNet-shape preconditioning (cap, embed boost,
lm_head temp) — those were forcing the model into BitNet's
compensation pattern, which BitNet developed because its from-scratch
training was constrained low-bit everywhere. Qwen has no such
constraint on the non-body DOFs; let them run free.

Mechanism: single global κ ∈ [0, 1] controlling soft-mix between FP
master weights and Bonsai-style per-128-group binary projection:

  W_eff(κ) = (1 − κ) × W_master + κ × bonsai_project(W_master)

PID on CE drift advances κ continuously. Setpoint tight (0.02 nats).

Compares to:
  Stage 189 (3 discrete levels, descent on group_size):  +3.16 nats
  Stage 191 (continuous κ, soft mix, free FP DOFs):       ?

If continuous + free DOFs unlocks meaningful improvement over Stage
189's discrete schedule, we have the production recipe.
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
LR_AUX = 5e-4    # for embed/lm_head/norms/α
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage191_laser_pid_continuous.json")

ALL_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
TRAINABLE_MASTER_NAMES = ("o_proj", "down_proj")  # Mac memory subset
GROUP_SIZE = 128

# Continuous κ schedule
N_CYCLES = 50
TRAIN_STEPS_PER_CYCLE = 30
MAX_KAPPA_STEP = 1.0 / N_CYCLES   # nominal step if always advancing

# PID
PID_SETPOINT_DRIFT = 0.05         # tighter than Stage 189's 0.10
PID_KP = 1.0


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


def bonsai_project(W, group_size=128):
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    grouped = W.reshape(out_features, n_groups, group_size)
    scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return (torch.sign(grouped) * scales).reshape(out_features, in_features)


class KappaLinear(nn.Module):
    """Linear with soft-mix between FP master and Bonsai-projected weights.
    κ=0 is identity (full FP); κ=1 is full binary projection.
    Backward is STE (gradient flows to master as identity)."""
    def __init__(self, original_module):
        super().__init__()
        self.weight = nn.Parameter(original_module.weight.data.clone())
        self.bias = original_module.bias
        self.kappa = 0.0
        rn = self.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.alpha = nn.Parameter(rn.squeeze(-1).clone().to(torch.float32))

    def forward(self, x):
        w = self.weight
        if self.kappa <= 0.0:
            w_eff = w
        else:
            w_q = bonsai_project(w.float(), GROUP_SIZE).to(x.dtype)
            w_mix = (1 - self.kappa) * w + self.kappa * w_q
            # STE: forward uses w_mix, backward acts as identity through projection
            w_eff = w + (w_mix - w).detach()
        # Renormalize per row, α captures scale
        rn = w_eff.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_unit = (w_eff.float() / rn).to(x.dtype)
        out = F.linear(x, w_unit, self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


class LaserPID:
    def __init__(self, setpoint=PID_SETPOINT_DRIFT, Kp=PID_KP, max_step=MAX_KAPPA_STEP):
        self.setpoint = setpoint
        self.Kp = Kp
        self.max_step = max_step
        self.last_error = 0.0
        self.cumulative_error = 0.0

    def next_kappa(self, current_drift, current_kappa):
        """Decide next κ based on observed drift."""
        error = current_drift - self.setpoint
        self.cumulative_error += error
        # P-control: if drift below setpoint, advance fast; if above, hold or retreat.
        # Normalized rate: 1 when drift=0, 0 when drift=setpoint, negative above.
        rate = 1.0 - self.Kp * (current_drift / max(self.setpoint, 1e-8))
        rate = max(-0.5, min(1.0, rate))   # clamp
        new_kappa = current_kappa + rate * self.max_step
        return max(0.0, min(1.0, new_kappa))


print(f"device={device} dtype={dtype}")
print(f"checkpoint: {CHECKPOINT}")
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


# ─── Set up model ───
print("\nLoading model and installing KappaLinear wrappers...")
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

descending_layers = []  # o_proj + down_proj — master trainable
fixed_layers = []       # rest — master frozen, but α still trainable

for full_name, mod in target_mods:
    is_descending = any(t in full_name for t in TRAINABLE_MASTER_NAMES)
    new_layer = KappaLinear(mod)
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    if is_descending:
        descending_layers.append(new_layer)
    else:
        fixed_layers.append(new_layer)

# Trainable: master of descending + ALL αs + RMSNorm gains + embed + lm_head
master_params = [g.weight for g in descending_layers]
alpha_params = [g.alpha for g in descending_layers + fixed_layers]
for g in fixed_layers:
    g.weight.requires_grad = False

aux_params = list(alpha_params)  # α is auxiliary
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
    if "embed_tokens" in n and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
    if "lm_head" in n and "weight" in n:
        # may be tied to embed; if it is, this is no-op
        p.requires_grad = True
        if p not in aux_params:
            aux_params.append(p)

for p in master_params:
    p.requires_grad = True

n_master = sum(p.numel() for p in master_params)
n_aux = sum(p.numel() for p in aux_params)
print(f"  descending: {len(descending_layers)} (o_proj+down_proj, master trainable)")
print(f"  fixed:      {len(fixed_layers)} (master frozen, α only trainable)")
print(f"  trainable:  {n_master:,} master + {n_aux:,} aux = {n_master + n_aux:,}")
print(f"  aux includes: αs, RMSNorm gains, embeddings, lm_head — all FREE to compensate")


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


def set_kappa(k):
    for g in descending_layers + fixed_layers:
        g.kappa = k


# ─── Continuous laser descent ───
it = iter_train()
trajectory = []
pid = LaserPID()

print("\n" + "=" * 70)
print(f"Starting laser PID descent (κ: 0 → 1, target {N_CYCLES} cycles)")
print("=" * 70)

current_kappa = 0.0
for cycle in range(N_CYCLES):
    set_kappa(current_kappa)
    # Train at current κ
    train_steps(it, TRAIN_STEPS_PER_CYCLE)
    # Measure
    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    # PID update
    next_kappa = pid.next_kappa(drift, current_kappa)
    delta = next_kappa - current_kappa

    trajectory.append({
        "cycle": cycle + 1,
        "kappa": float(current_kappa),
        "ce": float(ce),
        "drift": float(drift),
        "next_kappa": float(next_kappa),
        "delta_kappa": float(delta),
    })
    if (cycle + 1) % 5 == 0 or cycle < 5:
        marker = "→" if delta > 0 else ("←" if delta < 0 else "·")
        print(f"  cycle {cycle+1:>3}/{N_CYCLES}  κ={current_kappa:.4f} {marker}κ_next={next_kappa:.4f}  "
              f"CE={ce:.4f}  drift={drift:+.4f}", flush=True)

    current_kappa = next_kappa
    if current_kappa >= 1.0 and cycle > N_CYCLES // 2:
        # Already at full binary, keep training to settle plateau
        pass


# Final settle: train more at κ=1.0 to converge
set_kappa(1.0)
print(f"\nFinal settle: training {TRAIN_STEPS_PER_CYCLE * 4} more steps at κ=1.0...")
train_steps(it, TRAIN_STEPS_PER_CYCLE * 4)
final_ce = lm_ce(model, val_tokens)
final_drift = final_ce - T0


# ─── Summary ───
print("\n" + "=" * 70)
print("LASER PID CONTINUOUS DESCENT COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):                   {T0:.4f}")
print(f"  Stage 189 plateau (3-level):    +3.159 nats")
print(f"  Stage 191 plateau (continuous): {final_drift:+.4f} nats")

improvement = 3.159 - final_drift
if improvement > 1.0:
    print(f"\n  ✓✓ MAJOR UNLOCK: laser continuous {improvement:.2f} nats below 3-level Stage 189.")
elif improvement > 0.3:
    print(f"\n  ✓ Improvement: laser {improvement:.2f} nats below Stage 189.")
elif improvement > -0.1:
    print(f"\n  - Comparable to Stage 189 ({improvement:+.2f}).")
else:
    print(f"\n  ✗ Worse than Stage 189 by {-improvement:.2f}.")

# κ distribution analysis
final_kappa = trajectory[-1]["next_kappa"] if trajectory else 0.0
print(f"\n  κ trajectory: started 0.0, ended training cycles at {final_kappa:.4f}")
print(f"  PID final cumulative_error: {pid.cumulative_error:+.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_cycles": N_CYCLES,
        "train_steps_per_cycle": TRAIN_STEPS_PER_CYCLE,
        "T0_base_ce": float(T0),
        "trajectory": trajectory,
        "final_ce": float(final_ce),
        "final_drift": float(final_drift),
        "stage_189_baseline_delta": 3.159,
        "improvement_vs_189": float(improvement),
        "config": {
            "lr_master": LR_MASTER,
            "lr_aux": LR_AUX,
            "pid_setpoint": PID_SETPOINT_DRIFT,
            "pid_kp": PID_KP,
            "max_kappa_step": MAX_KAPPA_STEP,
            "group_size": GROUP_SIZE,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
