"""Stage 221 — Scaling test on Qwen3-1.7B with the 218 frozen-first protocol.

Per-stage learning: Stage 220 confirmed that "everything trainable simultaneously"
is unstable. Stage 218's frozen-first protocol with phase separation is what
actually delivers stable K=1 reduction (-3.85 nats on 0.6B).

This stage replicates 218's protocol on Qwen3-1.7B to test the scaling hypothesis:
does the recipe gain from more parametric slack at 1.7B vs 0.6B?

Protocol (218 frozen-first):
  Phase 1 (find laser zone) — 2000 steps:
    Body W_fp + β_g  FROZEN
    Compensation levers TRAINING
    γ slow-PID ramps 0 → 0.95 with drift band [0.05, 0.20]
    → levers find K=1 compensation environment

  Phase 2 (body settles + levers continue) — 2000 steps:
    Body W_fp + β_g  UNFROZEN
    Compensation levers STILL TRAINING (continuous compensation)
    γ HELD at phase-1-end value
    λ_bimodal = 1e-2 ACTIVE (FIXED, not PID-ramped — fix from 218)
    λ_variance = 1e-2 ACTIVE
    → body trains under regularizer pressure with levers continuously absorbing

Mac memory accommodations for 1.7B:
  - Body W_fp trainable only on DOWN_PROJ (Mac fp32 budget)
  - 2000+2000 = 4000 steps total (vs 218's 2000+2000 on 0.6B)
  - SEQ_LEN=64, BATCH=1
"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


CHECKPOINT = "Qwen/Qwen3-1.7B"
SEQ_LEN = 64
N_VAL_CHUNKS = 8
N_CALIB_TOKENS = 64
BATCH_SIZE = 1
PHASE_1_STEPS = 2000
PHASE_2_STEPS = 2000
EVAL_EVERY = 50
K1_DIAG_EVERY = 500

BODY_LR = 2e-5
LEVER_LR = 5e-4
BETA_LR = 5e-4

LAMBDA_BIMODAL = 1e-2
LAMBDA_VARIANCE = 1e-2

GAMMA_TARGET = 0.95
PID_STEP_FRAC = 0.05
DRIFT_TARGET = 0.05
DRIFT_HIGH = 0.20

RESULTS_PATH = Path("results/stage221_scaling_1p7b.json")
TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")
BODY_TRAIN_NAMES = ("down_proj",)   # Mac memory budget at 1.7B fp32
GROUP_SIZE = 128


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt_cached():
    return torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                      weights_only=True).long()


def lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS):
    losses = []
    model.eval()
    for i in range(n_chunks):
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


class AdiabaticQuantizedLinear(nn.Module):
    def __init__(self, original_linear, group_size=GROUP_SIZE):
        super().__init__()
        W_fp = original_linear.weight.data.clone()
        self.weight_fp = nn.Parameter(W_fp, requires_grad=False)
        out, in_ = W_fp.shape
        self.has_groups = (in_ % group_size == 0)
        if self.has_groups:
            n_groups = in_ // group_size
            Wg = W_fp.float().reshape(out, n_groups, group_size)
            alpha = Wg.abs().mean(dim=-1, keepdim=True)
            self.register_buffer("alpha", alpha.to(W_fp.dtype))
            self.beta_g = nn.Parameter(torch.zeros(
                out, n_groups, 1, device=W_fp.device, dtype=W_fp.dtype))
        else:
            self.register_buffer("alpha",
                                 W_fp.abs().mean(dim=-1, keepdim=True).to(W_fp.dtype))
            self.beta_g = None
        self.group_size = group_size
        self.out_features, self.in_features = out, in_
        self.register_buffer("gamma", torch.tensor(0.0, dtype=W_fp.dtype))
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(
                out, device=W_fp.device, dtype=W_fp.dtype))

    def forward(self, x):
        γ = self.gamma
        if self.has_groups:
            Wg_fp = self.weight_fp.reshape(
                self.out_features, self.in_features // self.group_size, self.group_size)
            mag_eff = γ * self.alpha + (1 - γ) * Wg_fp.abs()
            W_eff = torch.sign(Wg_fp) * mag_eff
            if self.beta_g is not None:
                W_eff = W_eff + γ * self.beta_g
            W_eff = W_eff.reshape(self.out_features, self.in_features)
        else:
            W_eff = torch.sign(self.weight_fp) * (
                γ * self.alpha + (1 - γ) * self.weight_fp.abs())
        return F.linear(x, W_eff, self.bias.to(x.dtype))


class SubLNLinear(nn.Module):
    def __init__(self, wrapped_linear, eps=1e-6):
        super().__init__()
        self.wrapped = wrapped_linear
        in_features = (wrapped_linear.weight_fp.shape[1] if hasattr(wrapped_linear, "weight_fp")
                       else wrapped_linear.weight.shape[1])
        device_ = (wrapped_linear.weight_fp if hasattr(wrapped_linear, "weight_fp")
                   else wrapped_linear.weight).device
        dtype_ = (wrapped_linear.weight_fp if hasattr(wrapped_linear, "weight_fp")
                  else wrapped_linear.weight).dtype
        self.subln_gain = nn.Parameter(torch.ones(in_features, device=device_, dtype=dtype_))
        self.subln_gate = nn.Parameter(torch.zeros((), device=device_, dtype=dtype_))
        self.eps = eps
        self.h_scale = None

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        normed = self.subln_gain * x / rms
        x = (1.0 - self.subln_gate) * x + self.subln_gate * normed
        return self.wrapped(x)


class TemperedLMHead(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        device_ = wrapped.weight.device
        dtype_ = wrapped.weight.dtype
        self.logit_tau = nn.Parameter(torch.ones((), device=device_, dtype=dtype_))

    def forward(self, x):
        return self.wrapped(x) / self.logit_tau


def calibrate_input_rms(model, calib_ids, target_suffixes):
    rms_sums, counts, hooks = {}, {}, []
    def make_hook(name):
        def hook(mod, inp):
            x = inp[0].detach().float()
            mean_sq = x.pow(2).mean(dim=tuple(range(x.dim() - 1)))
            rms = mean_sq.sqrt()
            if name not in rms_sums:
                rms_sums[name] = rms.clone(); counts[name] = 1
            else:
                rms_sums[name] += rms; counts[name] += 1
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in target_suffixes):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks: h.remove()
    return {name: (rms_sums[name] / counts[name]).cpu() for name in rms_sums}


def install_residual_gains_and_offsets(model):
    n_layers = 0
    for layer in model.model.layers:
        hidden_size = layer.input_layernorm.weight.shape[0]
        d, t = layer.input_layernorm.weight.device, layer.input_layernorm.weight.dtype
        layer.attn_gain = nn.Parameter(torch.ones(hidden_size, device=d, dtype=t))
        layer.mlp_gain = nn.Parameter(torch.ones(hidden_size, device=d, dtype=t))
        layer.attn_offset = nn.Parameter(torch.zeros(hidden_size, device=d, dtype=t))
        layer.mlp_offset = nn.Parameter(torch.zeros(hidden_size, device=d, dtype=t))

        def new_forward(self, hidden_states, **kwargs):
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(hidden_states=x, **kwargs)
            x = residual + self.attn_gain * attn_out + self.attn_offset
            residual = x
            x = self.post_attention_layernorm(x)
            mlp_out = self.mlp(x)
            x = residual + self.mlp_gain * mlp_out + self.mlp_offset
            return x

        layer.forward = types.MethodType(new_forward, layer)
        n_layers += 1
    return n_layers


def build_full_architecture(calib_ids):
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    n_layers = install_residual_gains_and_offsets(m)
    rms_table = calibrate_input_rms(m, calib_ids, ("down_proj",))

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n_quantized = 0; n_beta_g = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinear(mod)
        if any(t in name for t in BODY_TRAIN_NAMES):
            new_layer.weight_fp.requires_grad_(True)
        if new_layer.beta_g is not None:
            n_beta_g += new_layer.beta_g.numel()
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n_quantized += 1

    parent_lookup2 = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup2[full] = (mod, child_name)

    n_subln = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, AdiabaticQuantizedLinear): continue
        if not name.endswith("down_proj"): continue
        if name not in rms_table: continue
        gain = rms_table[name].to(device=mod.weight_fp.device, dtype=mod.weight_fp.dtype)
        new_layer = SubLNLinear(mod)
        with torch.no_grad():
            new_layer.subln_gain.data.copy_(gain)
        parent, child_attr = parent_lookup2[name]
        setattr(parent, child_attr, new_layer)
        n_subln += 1

    m.lm_head = TemperedLMHead(m.lm_head)
    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers,
                   n_subln=n_subln, n_beta_g_total=n_beta_g)


def is_body_master(name):
    return "weight_fp" in name and any(t in name for t in BODY_TRAIN_NAMES)


def is_beta_g(name):
    return "beta_g" in name


def is_compensation_lever(name):
    if any(t in name for t in (
        "subln_gate", "subln_gain", "h_scale", "attn_gain", "mlp_gain",
        "attn_offset", "mlp_offset", "logit_tau"
    )):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


def set_trainable(model, predicate, value):
    for n, p in model.named_parameters():
        if predicate(n):
            p.requires_grad_(value)


def freeze_all_else(model):
    for n, p in model.named_parameters():
        if not is_body_master(n) and not is_beta_g(n) and not is_compensation_lever(n):
            p.requires_grad_(False)


def set_gamma(model, gamma_value):
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear):
            mod.gamma.fill_(gamma_value)


def get_gamma(model):
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear):
            return float(mod.gamma.item())
    return 0.0


def k1_drift_now(model, val_tokens, T0):
    γ_save = get_gamma(model)
    set_gamma(model, 1.0)
    ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
    set_gamma(model, γ_save)
    return ce - T0


def bimodal_squeeze_loss(model):
    total = 0.0; n = 0
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear) and mod.has_groups:
            W = mod.weight_fp
            Wg = W.reshape(mod.out_features, mod.in_features // mod.group_size, mod.group_size)
            target = torch.sign(Wg) * mod.alpha
            total = total + ((Wg - target).float() ** 2).mean()
            n += 1
    return total / max(n, 1)


def variance_penalty_loss(model):
    total = 0.0; n = 0
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear) and mod.has_groups:
            W = mod.weight_fp
            Wg = W.reshape(mod.out_features, mod.in_features // mod.group_size, mod.group_size)
            total = total + Wg.abs().float().var(dim=-1).mean()
            n += 1
    return total / max(n, 1)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def pid_step_toward(current, target, drift, step_size):
    direction = 1 if target > current else -1
    if drift > DRIFT_HIGH:
        new = current - direction * step_size
    elif drift < DRIFT_TARGET:
        if direction > 0:
            new = min(current + step_size, target)
        else:
            new = max(current - step_size, target)
    else:
        new = current
    new = max(0.0, min(1.0, new))
    return new


print(f"device={device} dtype={dtype}  CHECKPOINT={CHECKPOINT}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 32].tolist()
train_tokens = corpus[SEQ_LEN * 32:SEQ_LEN * 32 + 1_000_000].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)

print("\nMeasuring T0 (base FP)...", flush=True)
m0 = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(m0, val_tokens)
cfg = m0.config
hidden_size = cfg.hidden_size
intermediate_size = cfg.intermediate_size
print(f"  T0 = {T0:.4f}  hidden={hidden_size}  intermediate={intermediate_size}",
      flush=True)
del m0; import gc; gc.collect()

print("\nBuilding architecture (1.7B, MLP-only, body=down_proj only)...", flush=True)
model, install_stats = build_full_architecture(calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
k1_initial = k1_drift_now(model, val_tokens, T0)
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}  K=1 initial: {k1_initial:+.4f}",
      flush=True)

freeze_all_else(model)
n_body = sum(p.numel() for n, p in model.named_parameters() if is_body_master(n))
n_beta = sum(p.numel() for n, p in model.named_parameters() if is_beta_g(n))
n_lever = sum(p.numel() for n, p in model.named_parameters() if is_compensation_lever(n))
print(f"\nBody (down_proj only): {n_body:,}")
print(f"β_g (Bonsai):           {n_beta:,}")
print(f"Compensation levers:    {n_lever:,}")
print(f"Total trainable:        {n_body + n_beta + n_lever:,}", flush=True)

body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
beta_params = [p for n, p in model.named_parameters() if is_beta_g(n)]
lever_params = [p for n, p in model.named_parameters() if is_compensation_lever(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": beta_params,  "lr": BETA_LR},
    {"params": lever_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


def train_step(batch, λ_bimodal=0.0, λ_var=0.0):
    out = model(batch[:, :-1], use_cache=False)
    ce_loss = F.cross_entropy(
        out.logits.float().reshape(-1, out.logits.size(-1)),
        batch[:, 1:].reshape(-1))
    total = ce_loss
    if λ_bimodal > 0:
        total = total + λ_bimodal * bimodal_squeeze_loss(model)
    if λ_var > 0:
        total = total + λ_var * variance_penalty_loss(model)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return float(ce_loss.item())


t_start = time.time()
history = [{"event": "init", "ce": ce_g0, "drift": drift_g0, "k1_drift": k1_initial}]


# ─── PHASE 1: find laser zone with body+β_g FROZEN ───
print(f"\n{'─'*60}")
print(f"PHASE 1 — find laser zone (body+β_g FROZEN, levers train, γ ramp)")
print('─'*60, flush=True)
set_trainable(model, is_body_master, False)
set_trainable(model, is_beta_g, False)
set_trainable(model, is_compensation_lever, True)

current_gamma = 0.0
gamma_step = GAMMA_TARGET * PID_STEP_FRAC
set_gamma(model, current_gamma)

best_k1 = k1_initial
k1_trajectory = [{"step": 0, "phase": "init", "k1_drift": k1_initial, "gamma": 0.0}]
model.train()
for step in range(1, PHASE_1_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    ce_loss = train_step(batch, λ_bimodal=0.0, λ_var=0.0)

    if step % EVAL_EVERY == 0 or step == PHASE_1_STEPS:
        val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        current_gamma = pid_step_toward(current_gamma, GAMMA_TARGET, drift, gamma_step)
        set_gamma(model, current_gamma)
        elapsed = time.time() - t_start
        print(f"  P1 step {step:>4} γ={current_gamma:.3f} ce={val_ce:.4f} "
              f"Δ={drift:+.4f} loss={ce_loss:.3f}  {elapsed:.0f}s", flush=True)
        history.append({"phase": "P1", "step": step, "gamma": float(current_gamma),
                        "ce": float(val_ce), "drift": float(drift),
                        "ce_loss": float(ce_loss)})
        model.train()

    if step % K1_DIAG_EVERY == 0:
        k1_now = k1_drift_now(model, val_tokens, T0)
        marker = " ⭐" if k1_now < best_k1 else ""
        print(f"  ── P1 step {step:>4} K=1 DIAG: drift={k1_now:+.4f}  "
              f"(initial: {k1_initial:+.4f}, best: {min(best_k1, k1_now):+.4f}){marker}",
              flush=True)
        k1_trajectory.append({"step": step, "phase": "P1",
                              "k1_drift": float(k1_now), "gamma": float(current_gamma)})
        if k1_now < best_k1:
            best_k1 = k1_now
        model.train()

phase1_end_gamma = get_gamma(model)
phase1_end_k1 = k1_drift_now(model, val_tokens, T0)
print(f"\n  P1 END: γ={phase1_end_gamma:.3f}  K=1_drift={phase1_end_k1:+.4f}  "
      f"(best K=1 in P1: {min(best_k1, phase1_end_k1):+.4f})", flush=True)
if phase1_end_k1 < best_k1:
    best_k1 = phase1_end_k1


# ─── PHASE 2: body+β_g UNFROZEN, levers continue, γ HELD, regularizers ACTIVE ───
print(f"\n{'─'*60}")
print(f"PHASE 2 — body+β_g UNFROZEN, levers continue, γ held at {phase1_end_gamma:.3f}")
print(f"          λ_bimodal={LAMBDA_BIMODAL} λ_variance={LAMBDA_VARIANCE} (FIXED, ALWAYS ACTIVE)")
print('─'*60, flush=True)
set_trainable(model, is_body_master, True)
set_trainable(model, is_beta_g, True)
set_trainable(model, is_compensation_lever, True)
set_gamma(model, phase1_end_gamma)

model.train()
for step in range(1, PHASE_2_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    ce_loss = train_step(batch, λ_bimodal=LAMBDA_BIMODAL, λ_var=LAMBDA_VARIANCE)

    if step % EVAL_EVERY == 0 or step == PHASE_2_STEPS:
        val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        print(f"  P2 step {step:>4} γ={get_gamma(model):.3f} ce={val_ce:.4f} "
              f"Δ={drift:+.4f} loss={ce_loss:.3f}  {elapsed:.0f}s", flush=True)
        history.append({"phase": "P2", "step": step, "gamma": float(get_gamma(model)),
                        "ce": float(val_ce), "drift": float(drift),
                        "ce_loss": float(ce_loss)})
        model.train()

    if step % K1_DIAG_EVERY == 0:
        k1_now = k1_drift_now(model, val_tokens, T0)
        marker = " ⭐" if k1_now < best_k1 else ""
        print(f"  ── P2 step {step:>4} K=1 DIAG: drift={k1_now:+.4f}  "
              f"(best: {min(best_k1, k1_now):+.4f}){marker}", flush=True)
        k1_trajectory.append({"step": step, "phase": "P2",
                              "k1_drift": float(k1_now), "gamma": float(get_gamma(model))})
        if k1_now < best_k1:
            best_k1 = k1_now
        model.train()


# Final
final_k1 = k1_drift_now(model, val_tokens, T0)
print(f"\n{'─'*60}")
print("STAGE 221 RESULT (Qwen3-1.7B, frozen-first protocol):")
print('─'*60)
print(f"  Model:             Qwen3-1.7B")
print(f"  T0:                {T0:.4f}")
print(f"  K=1 initial:       {k1_initial:+.4f}")
print(f"  After P1:          {phase1_end_k1:+.4f}  (γ={phase1_end_gamma:.3f})")
print(f"  After P2 (final):  {final_k1:+.4f}")
print(f"  Best K=1 seen:     {best_k1:+.4f}")
print(f"  Reduction (final): {k1_initial - final_k1:+.4f} nats "
      f"({100*(1 - final_k1/max(k1_initial, 1e-6)):.1f}%)")
print(f"\n  vs 0.6B reference (Stage 218):  -3.85 nats (33.3%)")
print(f"  Scaling hypothesis:")
print(f"    1.7B reduction > 0.6B → VALIDATED, recipe scales")
print(f"    1.7B ≈ 0.6B → recipe doesn't gain from scale")
print(f"    1.7B < 0.6B → different bottleneck at scale")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0": float(T0),
        "k1_initial_drift": float(k1_initial),
        "k1_after_phase1": float(phase1_end_k1),
        "phase1_end_gamma": float(phase1_end_gamma),
        "k1_final_drift": float(final_k1),
        "k1_best_drift": float(best_k1),
        "reduction": float(k1_initial - final_k1),
        "n_body_params": int(n_body),
        "n_beta_g_params": int(n_beta),
        "n_lever_params": int(n_lever),
        "phase_1_steps": PHASE_1_STEPS,
        "phase_2_steps": PHASE_2_STEPS,
        "lambda_bimodal": LAMBDA_BIMODAL,
        "lambda_variance": LAMBDA_VARIANCE,
        "k1_trajectory": k1_trajectory,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
