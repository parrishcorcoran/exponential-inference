"""Stage 193: Slow shape annealing via PID-controlled regularization.

User's reframe (2026-04-29): if we slowly push Qwen weights into the
right (bimodal per-group) shape, the model's function migrates to that
shape, and K=1 binary projection has zero residual error. Equivalent
math identity to magnitude reduction, but for within-group distribution.

This experiment tests the shape-vs-capacity hypothesis directly.

**Mechanism (no quantization during training):**

Add a regularizer that penalizes within-group weight-magnitude variance:

  L_shape = Σ_groups var(|w_in_group| / mean(|w_in_group|))

Total loss:
  L_total = L_CE + λ × L_shape

PID throttles λ from 0 → large based on CE drift:
  - λ grows when drift ≤ setpoint (laser zone)
  - λ holds when drift > setpoint (preserve quality)
  - At λ=large, weights are forced bimodal-per-group

Trainable: master weights (must move to absorb shape pressure) + all FP DOFs.

**End state if successful:**
  - Body weights have low within-group variance → naturally bimodal
  - K=1 binary projection on shaped weights has near-zero residual error
  - Final CE matches FP base (drift ≈ 0)
  - This validates shape-not-capacity as the fundamental constraint

**End state if it fails:**
  - λ stalls at small value → PID says drift exceeds setpoint
  - Weights stay Gaussian-shaped → K=1 still catastrophic
  - Tells us: shape annealing alone isn't sufficient; need more (mixed precision, K>1, etc.)

User's calibration: "if we don't even get to ternary, we have more work to do."
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
LR_BODY = 1e-5
LR_AUX = 5e-4
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage193_shape_anneal.json")

ALL_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
TRAINABLE_BODY_NAMES = ("o_proj", "down_proj")  # subset for memory
GROUP_SIZE = 128

# PID parameters for λ ramp
# v2: loosened setpoint so PID actually engages despite training noise.
# Goal of this run is to MAP THE LANDSCAPE — find where the recipe breaks
# under shape pressure — not to hit lossless directly.
N_CYCLES = 80
TRAIN_STEPS_PER_CYCLE = 30
LAMBDA_INITIAL = 0.01  # start with non-zero so the regularizer has signal
LAMBDA_TARGET = 100.0
PID_SETPOINT_DRIFT = 0.5    # 0.5 nat ≈ Bonsai's 11% gap; permissive zone
PID_KP = 1.0
LAMBDA_GROWTH_RATE = 0.3

# Active compensation coaxing — PID-up on FP DOF magnitudes
# As shape pressure grows, gradient descent doesn't naturally grow compensation;
# we must actively push it up. Rate moderated by drift (laser).
AUX_GROWTH_RATE = 0.005  # max multiplicative growth per cycle (e.g., 0.5%)
COAX_EMBED = True       # actively scale embedding row norms upward
COAX_LMHEAD = True      # actively scale lm_head upward
COAX_NORMS = True       # actively scale RMSNorm gains


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


_CACHED_CORPUS = None

def load_owt_cached():
    """Load the pretokenized OWT corpus (data/owt_tokens_50M.pt). 58M tokens
    pre-tokenized with Qwen3 tokenizer. Disk read is instant vs streaming."""
    global _CACHED_CORPUS
    if _CACHED_CORPUS is None:
        _CACHED_CORPUS = torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                                     weights_only=True).long()
        print(f"  loaded cached corpus: {_CACHED_CORPUS.numel():,} tokens")
    return _CACHED_CORPUS


def load_owt(tokenizer, max_tokens, skip=0):
    """Slice from cached corpus instead of streaming."""
    corpus = load_owt_cached()
    end = min(skip + max_tokens, corpus.numel())
    return corpus[skip:end].tolist()


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


def shape_loss_fn(weight_list, group_size=128):
    """L_shape = mean over layers of mean within-group magnitude variance.

    Each layer's per-group |w| values are normalized by their group mean,
    and we penalize the variance. Lower = more bimodal-per-group.
    Zero variance = exactly bimodal (every |w| equals group mean).
    """
    total = 0.0
    n_layers = 0
    for W in weight_list:
        out_features, in_features = W.shape
        if in_features % group_size != 0:
            continue
        n_groups = in_features // group_size
        grouped = W.reshape(out_features, n_groups, group_size)
        abs_w = grouped.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = abs_w / mean_abs   # values around 1.0 each group
        var_in_group = ((normalized - 1.0) ** 2).mean(dim=-1)  # variance from 1.0
        total = total + var_in_group.mean()
        n_layers += 1
    return total / max(n_layers, 1)


def measure_shape_metrics(weight_list, group_size=128):
    """Diagnostic: how close are weights to bimodal-per-group?"""
    cv_per_group_all = []
    for W in weight_list:
        out_features, in_features = W.shape
        if in_features % group_size != 0: continue
        n_groups = in_features // group_size
        grouped = W.detach().float().reshape(out_features, n_groups, group_size)
        abs_w = grouped.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        std_abs = abs_w.std(dim=-1, keepdim=True)
        cv = (std_abs / mean_abs).cpu().numpy().flatten()
        cv_per_group_all.extend(cv.tolist())
    import numpy as np
    arr = np.array(cv_per_group_all)
    return {
        "mean_cv": float(arr.mean()),
        "median_cv": float(np.median(arr)),
        "max_cv": float(arr.max()),
        "p95_cv": float(np.percentile(arr, 95)),
    }


def k1_residual_error(weight_list, group_size=128):
    """Compute average per-layer relative residual error after K=1 binary
    projection. Lower = model is closer to perfect binary fit."""
    errors = []
    for W in weight_list:
        out_features, in_features = W.shape
        if in_features % group_size != 0: continue
        n_groups = in_features // group_size
        Wf = W.detach().float()
        grouped = Wf.reshape(out_features, n_groups, group_size)
        scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
        rel_err = (Wf - W_q).norm() / Wf.norm().clamp(min=1e-8)
        errors.append(rel_err.item())
    return sum(errors) / max(len(errors), 1)


print(f"device={device} dtype={dtype}")
print(f"PID setpoint: drift ≤ {PID_SETPOINT_DRIFT} (LASER zone)")
print(f"λ schedule: {LAMBDA_INITIAL} → {LAMBDA_TARGET} under PID throttle")

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


# ─── Set up trainable model ───
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

# Identify body linears; subset of those gets trainable master
target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in ALL_TARGET_NAMES): continue
    target_mods.append((name, mod))

trainable_body_modules = []
all_target_modules = []
for name, mod in target_mods:
    is_trainable_body = any(t in name for t in TRAINABLE_BODY_NAMES)
    if is_trainable_body:
        mod.weight.requires_grad = True
        trainable_body_modules.append(mod)
    all_target_modules.append(mod)

# Aux trainable: norms, embed, lm_head
aux_params = []
embed_param = None
lmhead_param = None
norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
        norm_params.append(p)
    if "embed_tokens" in n and "weight" in n:
        p.requires_grad = True
        aux_params.append(p)
        embed_param = p
    if "lm_head" in n and "weight" in n:
        p.requires_grad = True
        if p not in aux_params:
            aux_params.append(p)
        lmhead_param = p

body_params = [m.weight for m in trainable_body_modules]

n_body = sum(p.numel() for p in body_params)
n_aux = sum(p.numel() for p in aux_params)
print(f"  body trainable: {len(trainable_body_modules)} linears ({TRAINABLE_BODY_NAMES})")
print(f"  trainable params: {n_body:,} body + {n_aux:,} aux = {n_body + n_aux:,}")


opt = torch.optim.AdamW([
    {"params": body_params, "lr": LR_BODY},
    {"params": aux_params, "lr": LR_AUX},
], weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_steps_with_shape_reg(it, n_steps, lambda_val):
    """Train n_steps with L = L_CE + lambda * L_shape on body weights."""
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss_ce = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            if lambda_val > 0:
                loss_shape = (lambda_val / GRAD_ACCUM) * shape_loss_fn(body_params, GROUP_SIZE)
                (loss_ce + loss_shape).backward()
            else:
                loss_ce.backward()
        torch.nn.utils.clip_grad_norm_(body_params + aux_params, GRAD_CLIP)
        opt.step()


# ─── Initial measurements ───
init_ce = lm_ce(model, val_tokens)
init_metrics = measure_shape_metrics(body_params, GROUP_SIZE)
init_residual = k1_residual_error(body_params, GROUP_SIZE)
print(f"\nInitial state:")
print(f"  CE: {init_ce:.4f}  drift: {init_ce-T0:+.4f}")
print(f"  shape mean CV: {init_metrics['mean_cv']:.3f}  (lower = more bimodal)")
print(f"  shape p95 CV:  {init_metrics['p95_cv']:.3f}")
print(f"  K=1 residual error (rel): {init_residual:.3f}  (lower = closer to lossless K=1)")


# ─── PID-controlled λ ramp ───
print(f"\n{'='*70}\nStarting shape annealing under PID throttle\n{'='*70}")
trajectory = []
current_lambda = LAMBDA_INITIAL
it = iter_train()

for cycle in range(N_CYCLES):
    train_steps_with_shape_reg(it, TRAIN_STEPS_PER_CYCLE, current_lambda)
    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    metrics = measure_shape_metrics(body_params, GROUP_SIZE)
    residual = k1_residual_error(body_params, GROUP_SIZE)

    # PID decision on λ AND active compensation coaxing
    if drift <= PID_SETPOINT_DRIFT:
        # Laser zone — grow λ AND coax compensation upward
        rate = 1.0 - (drift / max(PID_SETPOINT_DRIFT, 1e-8))   # 1 when drift=0
        rate = max(0.0, min(1.0, rate))
        new_lambda = current_lambda * (1 + LAMBDA_GROWTH_RATE * rate) if current_lambda > 0 else 0.01

        # Active compensation coaxing: multiplicative growth on FP DOFs
        coax_factor = 1.0 + AUX_GROWTH_RATE * rate
        with torch.no_grad():
            if COAX_EMBED and embed_param is not None:
                embed_param.data = embed_param.data * coax_factor
            if COAX_LMHEAD and lmhead_param is not None and (lmhead_param is not embed_param):
                lmhead_param.data = lmhead_param.data * coax_factor
            if COAX_NORMS:
                for nm in norm_params:
                    nm.data = nm.data * coax_factor
    else:
        # Drift exceeds setpoint — hold or back off (no coax growth)
        excess = drift / PID_SETPOINT_DRIFT
        if excess > 2.0:
            new_lambda = current_lambda * 0.7  # back off hard
        else:
            new_lambda = current_lambda  # hold

    new_lambda = min(new_lambda, LAMBDA_TARGET)

    trajectory.append({
        "cycle": cycle + 1,
        "lambda": float(current_lambda),
        "next_lambda": float(new_lambda),
        "ce": float(ce),
        "drift": float(drift),
        "shape_mean_cv": metrics["mean_cv"],
        "shape_p95_cv": metrics["p95_cv"],
        "k1_residual": float(residual),
    })

    if (cycle + 1) % 5 == 0 or cycle < 5 or cycle == N_CYCLES - 1:
        marker = "↑" if new_lambda > current_lambda else ("↓" if new_lambda < current_lambda else "·")
        print(f"  cycle {cycle+1:>3}/{N_CYCLES}  λ={current_lambda:.3f}{marker}{new_lambda:.3f} "
              f"CE={ce:.4f} drift={drift:+.4f}  CV={metrics['mean_cv']:.3f} K1err={residual:.3f}",
              flush=True)

    current_lambda = new_lambda


# ─── Final measurement: apply K=1 binary and measure CE ───
print("\n" + "=" * 70)
print("Applying K=1 binary projection to shaped weights...")
print("=" * 70)

# Save current weights, apply K=1, measure, restore
saved_weights = [m.weight.data.clone() for m in trainable_body_modules]
for m in trainable_body_modules:
    W = m.weight.data.float()
    out_features, in_features = W.shape
    if in_features % GROUP_SIZE != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        W_q = torch.sign(W) * scale
    else:
        n_groups = in_features // GROUP_SIZE
        grouped = W.reshape(out_features, n_groups, GROUP_SIZE)
        scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
    m.weight.data = W_q.to(m.weight.dtype)

ce_post_k1 = lm_ce(model, val_tokens)
drift_post_k1 = ce_post_k1 - T0
metrics_final = measure_shape_metrics([w for w in saved_weights], GROUP_SIZE)

# Restore
for m, w in zip(trainable_body_modules, saved_weights):
    m.weight.data = w

print(f"\nFinal CE after K=1 projection: {ce_post_k1:.4f}  drift {drift_post_k1:+.4f}")


# ─── Summary ───
final_lambda = trajectory[-1]["lambda"] if trajectory else 0.0
final_drift_no_quant = trajectory[-1]["drift"] if trajectory else 0.0
final_cv = trajectory[-1]["shape_mean_cv"] if trajectory else init_metrics["mean_cv"]
final_residual = trajectory[-1]["k1_residual"] if trajectory else init_residual

print("\n" + "=" * 70)
print("SHAPE ANNEALING RESULT")
print("=" * 70)
print(f"  T0 (base FP):                {T0:.4f}")
print(f"  Initial state:")
print(f"    drift no-quant:            {init_ce-T0:+.4f}")
print(f"    shape mean CV:             {init_metrics['mean_cv']:.3f}")
print(f"    K=1 residual error:        {init_residual:.3f}")
print(f"  Final state (annealed FP):")
print(f"    drift no-quant:            {final_drift_no_quant:+.4f}")
print(f"    shape mean CV:             {final_cv:.3f}")
print(f"    K=1 residual error:        {final_residual:.3f}")
print(f"    λ reached:                 {final_lambda:.3f}")
print(f"  Final state with K=1 applied:")
print(f"    drift K=1:                 {drift_post_k1:+.4f}")

print(f"\nINTERPRETATION:")
if drift_post_k1 < 0.1:
    print(f"  ✓✓✓ K=1 LOSSLESS ACHIEVED via shape annealing.")
    print(f"      Validates user hypothesis: shape > capacity for binary lossless.")
elif drift_post_k1 < 0.5:
    print(f"  ✓✓ Near-K=1-lossless. Shape annealing nearly closed the binary gap.")
    print(f"     Compensation channels could finish via additional fine-tune.")
elif drift_post_k1 < 1.5:
    print(f"  ✓ Significant shape progress. K=1 still has gap — closer to ternary territory.")
elif drift_post_k1 < 3.0:
    print(f"  ~ Reached ternary-equivalent quality at K=1. More work needed for binary.")
else:
    print(f"  ✗ Did not reach ternary quality. User's diagnosis: more work to do.")

print(f"\n  Shape CV change:        {init_metrics['mean_cv']:.3f} → {final_cv:.3f}")
print(f"  K=1 residual change:    {init_residual:.3f} → {final_residual:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "init_ce": float(init_ce),
        "init_drift": float(init_ce - T0),
        "init_shape_metrics": init_metrics,
        "init_k1_residual": float(init_residual),
        "trajectory": trajectory,
        "final_lambda": float(final_lambda),
        "final_drift_no_quant": float(final_drift_no_quant),
        "final_shape_cv": float(final_cv),
        "final_k1_residual": float(final_residual),
        "ce_post_k1_projection": float(ce_post_k1),
        "drift_post_k1": float(drift_post_k1),
        "config": {
            "lr_body": LR_BODY,
            "lr_aux": LR_AUX,
            "pid_setpoint": PID_SETPOINT_DRIFT,
            "lambda_growth_rate": LAMBDA_GROWTH_RATE,
            "group_size": GROUP_SIZE,
            "n_cycles": N_CYCLES,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
