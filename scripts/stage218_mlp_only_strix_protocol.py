"""Stage 218 — MLP-only K=1 + Strix frozen-first protocol + combined regularizers.

Strategic frame (per user 2026-04-30):
  Attention quantization (q/k/v/o): use BitDistill ternary recipe later — it's
  proven, just engineering. Not part of this experiment.
  MLP quantization (gate/up/down): K=1 binary via OUR recipe — this is where
  the research value lives. Stage 218 isolates this.

Protocol (per Strix prior + user clarifications):

Phase 1 — find laser zone:
  Body W_fp + β_g:  FROZEN
  Other levers:     TRAINABLE
  γ ramps 0 → 0.95 (or wherever PID can stably reach)
  Levers find K=1 compensation environment while body sits at FP.

Phase 2 — body settles + levers absorb:
  Body W_fp + β_g:  UNFROZEN
  Other levers:     STILL TRAINABLE (continuous compensation)
  γ HELD at 0.95
  Loss = CE + λ_bimodal·||W − sign(W)·α||² + λ_var·Var(|W_g|)
  Both λ PID-ramped from 0 → max during phase 2.
  Body's gradient now has 3 components, all pointing at K=1 grid.
  Levers continuously absorb activation-level pressure as body moves.

End: measure K=1 drift at γ=1.0.

Per-lever gradient and displacement logging — to detect when compensation
is at its expressive limit (lever gradient norm sustained high while drift
also rises = capacity exceeded).

MLP-only quantization:
  TARGET_NAMES = (gate_proj, up_proj, down_proj)
  q/k/v/o stay as standard nn.Linear at FP. Untouched.

Effective bits/weight (with this stage successful + future BitDistill-attn):
  attention 40%: untouched here, ternary later → 1.58 bits
  MLP 60%:        K=1 + α + β_g → 1 + 32/128 = 1.25 bits
  Combined:       0.40·1.58 + 0.60·1.25 = 1.38 bits/weight
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


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 64
N_VAL_CHUNKS = 8
N_CALIB_TOKENS = 64
BATCH_SIZE = 1
PHASE_1_STEPS = 2000   # find laser zone (γ ramp with body frozen)
PHASE_2_STEPS = 2000   # body settles with regularizers (γ held)
EVAL_EVERY = 50
BODY_LR = 2e-5
LEVER_LR = 5e-4
BETA_LR = 5e-4

# Tight PID
DRIFT_TARGET = 0.02
DRIFT_HIGH = 0.05
GAMMA_PID_STEP_FRAC = 0.05
LAMBDA_PID_STEP_FRAC = 0.05

# Phase 2 regularizer targets (max λ values)
LAMBDA_BIMODAL_TARGET = 1e-2
LAMBDA_VARIANCE_TARGET = 1e-2

GAMMA_TARGET = 0.95   # phase 1 reaches here, phase 2 holds here

RESULTS_PATH = Path("results/stage218_mlp_only_strix.json")
# MLP-ONLY targets
TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")
BODY_TRAIN_NAMES = TARGET_NAMES  # train all 3 MLP body weights
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


# ─── AdiabaticQuantizedLinear (MLP-only, Bonsai-style with β_g) ───
class AdiabaticQuantizedLinear(nn.Module):
    """W_eff[i,j] = sign(W_fp[i,j])·(γ·α[i,g] + (1−γ)·|W_fp[i,j]|) + γ·β_g[i,g]"""
    def __init__(self, original_linear, group_size=GROUP_SIZE):
        super().__init__()
        W_fp = original_linear.weight.data.clone()
        self.weight_fp = nn.Parameter(W_fp, requires_grad=True)
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
    def __init__(self, wrapped_linear, num_heads=None, head_dim=None, eps=1e-6):
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
        if num_heads is not None and head_dim is not None:
            assert in_features == num_heads * head_dim
            self.h_scale = nn.Parameter(torch.ones(num_heads, device=device_, dtype=dtype_))
            self.num_heads = num_heads
            self.head_dim = head_dim
        else:
            self.h_scale = None

    def forward(self, x):
        if self.h_scale is not None:
            shape = x.shape
            x = x.reshape(*shape[:-1], self.num_heads, self.head_dim)
            x = x * self.h_scale.view(*([1] * (len(shape) - 1)), self.num_heads, 1)
            x = x.reshape(*shape)
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


def build_full_architecture(num_heads, head_dim, calib_ids):
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    n_layers = install_residual_gains_and_offsets(m)
    rms_table = calibrate_input_rms(m, calib_ids, ("down_proj",))   # only MLP wrap for SubLN

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n_quantized = 0
    n_beta_g = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinear(mod)
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
        if not name.endswith("down_proj"): continue   # SubLN only on down_proj for MLP-only
        if name not in rms_table: continue
        gain = rms_table[name].to(device=mod.weight_fp.device, dtype=mod.weight_fp.dtype)
        new_layer = SubLNLinear(mod, num_heads=None, head_dim=None)
        with torch.no_grad():
            new_layer.subln_gain.data.copy_(gain)
        parent, child_attr = parent_lookup2[name]
        setattr(parent, child_attr, new_layer)
        n_subln += 1

    m.lm_head = TemperedLMHead(m.lm_head)
    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers,
                   n_subln=n_subln, n_beta_g_total=n_beta_g)


# ─── Param helpers ───
def is_body_master(name):
    return "weight_fp" in name and any(t in name for t in BODY_TRAIN_NAMES)


def is_beta_g(name):
    return "beta_g" in name


def is_other_lever(name):
    if any(t in name for t in (
        "subln_gate", "subln_gain", "h_scale", "attn_gain", "mlp_gain",
        "attn_offset", "mlp_offset", "logit_tau"
    )):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


LEVER_GROUP_FILTERS = {
    "bias":         lambda n: "bias" in n and "norm" not in n,
    "subln_gate":   lambda n: "subln_gate" in n,
    "subln_gain":   lambda n: "subln_gain" in n,
    "attn_gain":    lambda n: "attn_gain" in n,
    "mlp_gain":     lambda n: "mlp_gain" in n,
    "attn_offset":  lambda n: "attn_offset" in n,
    "mlp_offset":   lambda n: "mlp_offset" in n,
    "logit_tau":    lambda n: "logit_tau" in n,
}


def set_trainable(model, predicate, value):
    for n, p in model.named_parameters():
        if predicate(n):
            p.requires_grad_(value)


def freeze_everything_else(model):
    for n, p in model.named_parameters():
        if not is_body_master(n) and not is_beta_g(n) and not is_other_lever(n):
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


def k1_drift(model, val_tokens, T0):
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


def snapshot_state(model):
    return {n: p.detach().clone()
            for n, p in model.named_parameters()
            if is_body_master(n) or is_beta_g(n) or is_other_lever(n)}


def per_lever_displacement_grad(model, snapshot):
    """Per-lever-group L2 displacement vs snapshot, and current gradient norm."""
    disp = {}
    grad_norm = {}
    for group_name, filter_fn in LEVER_GROUP_FILTERS.items():
        d, g = [], []
        for n, p in model.named_parameters():
            if not filter_fn(n): continue
            if n in snapshot:
                d.append((p.detach() - snapshot[n]).float().norm().item())
            if p.grad is not None:
                g.append(p.grad.detach().float().norm().item())
        disp[group_name] = float(np.mean(d)) if d else 0.0
        grad_norm[group_name] = float(np.mean(g)) if g else 0.0
    # β_g and body separately
    body_disp, body_grad = [], []
    for n, p in model.named_parameters():
        if is_body_master(n):
            if n in snapshot:
                body_disp.append((p.detach() - snapshot[n]).float().norm().item())
            if p.grad is not None:
                body_grad.append(p.grad.detach().float().norm().item())
    disp["body_W_fp"] = float(np.mean(body_disp)) if body_disp else 0.0
    grad_norm["body_W_fp"] = float(np.mean(body_grad)) if body_grad else 0.0

    beta_disp, beta_grad = [], []
    for n, p in model.named_parameters():
        if is_beta_g(n):
            if n in snapshot:
                beta_disp.append((p.detach() - snapshot[n]).float().norm().item())
            if p.grad is not None:
                beta_grad.append(p.grad.detach().float().norm().item())
    disp["beta_g"] = float(np.mean(beta_disp)) if beta_disp else 0.0
    grad_norm["beta_g"] = float(np.mean(beta_grad)) if beta_grad else 0.0
    return disp, grad_norm


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
    new = max(0.0, min(max(target, 1.0), new))
    return new


print(f"device={device} dtype={dtype}")
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
num_heads = cfg.num_attention_heads
head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
print(f"  T0 = {T0:.4f}", flush=True)
del m0; import gc; gc.collect()

print("\nBuilding architecture (MLP-only + Bonsai β_g)...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}  "
      f"({'lossless ✓' if abs(drift_g0)<1e-3 else 'distortion!'})", flush=True)
k1_initial = k1_drift(model, val_tokens, T0)
print(f"  K=1 initial drift: Δ={k1_initial:+.4f}", flush=True)

freeze_everything_else(model)
n_body = sum(p.numel() for n, p in model.named_parameters() if is_body_master(n))
n_beta = sum(p.numel() for n, p in model.named_parameters() if is_beta_g(n))
n_other = sum(p.numel() for n, p in model.named_parameters() if is_other_lever(n))
print(f"\nBody MLP master:    {n_body:,}")
print(f"β_g (Bonsai):       {n_beta:,}")
print(f"Other levers:       {n_other:,}", flush=True)

body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
beta_params = [p for n, p in model.named_parameters() if is_beta_g(n)]
other_params = [p for n, p in model.named_parameters() if is_other_lever(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": beta_params,  "lr": BETA_LR},
    {"params": other_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


def train_step(batch, lambda_bimodal=0.0, lambda_variance=0.0):
    out = model(batch[:, :-1], use_cache=False)
    ce_loss = F.cross_entropy(
        out.logits.float().reshape(-1, out.logits.size(-1)),
        batch[:, 1:].reshape(-1))
    total = ce_loss
    if lambda_bimodal > 0:
        total = total + lambda_bimodal * bimodal_squeeze_loss(model)
    if lambda_variance > 0:
        total = total + lambda_variance * variance_penalty_loss(model)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return float(ce_loss.item())


t_start = time.time()
history = [{"event": "init", "ce": ce_g0, "drift": drift_g0, "k1_drift": k1_initial}]

# ─── Phase 1: find laser zone ───
print(f"\n{'─'*60}")
print(f"PHASE 1 — find laser zone (body+β_g frozen, levers train, γ ramp 0→{GAMMA_TARGET})")
print('─'*60, flush=True)
set_trainable(model, is_body_master, False)
set_trainable(model, is_beta_g, False)
set_trainable(model, is_other_lever, True)

current_gamma = 0.0
gamma_step = GAMMA_TARGET * GAMMA_PID_STEP_FRAC
snap_phase1_start = snapshot_state(model)
model.train()
for step in range(1, PHASE_1_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    ce_loss = train_step(batch)
    if step % EVAL_EVERY == 0 or step == PHASE_1_STEPS:
        val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        current_gamma = pid_step_toward(current_gamma, GAMMA_TARGET, drift, gamma_step)
        set_gamma(model, current_gamma)
        disp, grad_norm = per_lever_displacement_grad(model, snap_phase1_start)
        elapsed = time.time() - t_start
        print(f"  P1 step {step:>4} γ={current_gamma:.3f} ce={val_ce:.4f} Δ={drift:+.4f} "
              f"loss={ce_loss:.3f}  body_disp={disp.get('body_W_fp', 0):.3f}  "
              f"top_lever_movers={', '.join(f'{k}={disp[k]:.2f}' for k in sorted(disp.keys(), key=lambda k: -disp[k])[:3])}  "
              f"{elapsed:.0f}s", flush=True)
        history.append({"phase": "P1", "step": step, "gamma": float(current_gamma),
                        "ce": float(val_ce), "drift": float(drift),
                        "ce_loss": float(ce_loss),
                        "lever_displacement": disp, "lever_grad_norm": grad_norm})
        model.train()

phase1_end_gamma = get_gamma(model)
phase1_end_drift = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS) - T0
phase1_end_k1 = k1_drift(model, val_tokens, T0)
print(f"\n  P1 END: γ={phase1_end_gamma:.3f}  drift={phase1_end_drift:+.4f}  "
      f"K=1_drift={phase1_end_k1:+.4f}  (initial K=1: {k1_initial:+.4f})", flush=True)


# ─── Phase 2: body settles + regularizers ramp ───
print(f"\n{'─'*60}")
print(f"PHASE 2 — body settles (γ held at {phase1_end_gamma:.3f}, body+β_g+levers all train)")
print(f"         + bimodal squeeze (λ→{LAMBDA_BIMODAL_TARGET}) + variance penalty (λ→{LAMBDA_VARIANCE_TARGET})")
print('─'*60, flush=True)
set_trainable(model, is_body_master, True)
set_trainable(model, is_beta_g, True)
set_trainable(model, is_other_lever, True)
set_gamma(model, phase1_end_gamma)

current_lambda_bimodal = 0.0
current_lambda_variance = 0.0
lambda_b_step = LAMBDA_BIMODAL_TARGET * LAMBDA_PID_STEP_FRAC
lambda_v_step = LAMBDA_VARIANCE_TARGET * LAMBDA_PID_STEP_FRAC
snap_phase2_start = snapshot_state(model)
model.train()
for step in range(1, PHASE_2_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    ce_loss = train_step(batch, lambda_bimodal=current_lambda_bimodal,
                          lambda_variance=current_lambda_variance)
    if step % EVAL_EVERY == 0 or step == PHASE_2_STEPS:
        val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        # Both λ ramped by same drift-PID
        current_lambda_bimodal = pid_step_toward(
            current_lambda_bimodal, LAMBDA_BIMODAL_TARGET, drift, lambda_b_step)
        current_lambda_variance = pid_step_toward(
            current_lambda_variance, LAMBDA_VARIANCE_TARGET, drift, lambda_v_step)
        disp, grad_norm = per_lever_displacement_grad(model, snap_phase2_start)
        elapsed = time.time() - t_start
        print(f"  P2 step {step:>4} λ_b={current_lambda_bimodal:.4f} λ_v={current_lambda_variance:.4f} "
              f"ce={val_ce:.4f} Δ={drift:+.4f} loss={ce_loss:.3f}  "
              f"body_disp={disp.get('body_W_fp', 0):.3f}  "
              f"β_g_disp={disp.get('beta_g', 0):.3f}  "
              f"top_lever_movers={', '.join(f'{k}={disp[k]:.2f}' for k in sorted(disp.keys(), key=lambda k: -disp[k])[:3])}  "
              f"{elapsed:.0f}s", flush=True)
        history.append({"phase": "P2", "step": step,
                        "lambda_bimodal": float(current_lambda_bimodal),
                        "lambda_variance": float(current_lambda_variance),
                        "ce": float(val_ce), "drift": float(drift),
                        "ce_loss": float(ce_loss),
                        "lever_displacement": disp, "lever_grad_norm": grad_norm})
        model.train()


# ─── Final K=1 ───
final_drift = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS) - T0
final_k1 = k1_drift(model, val_tokens, T0)

print(f"\n{'─'*60}")
print("STAGE 218 RESULT (MLP-only, Strix protocol):")
print('─'*60)
print(f"  T0:                          {T0:.4f}")
print(f"  K=1 initial drift:           {k1_initial:+.4f}")
print(f"  After P1 (laser zone):       {phase1_end_k1:+.4f}  (γ={phase1_end_gamma:.3f})")
print(f"  After P2 (body settled):     {final_k1:+.4f}  (γ held + regularizers)")
print(f"  Final lossless drift (γ=0):  {final_drift:+.4f}")
print(f"\n  Total K=1 drift reduction: {k1_initial - final_k1:+.4f} nats "
      f"({100*(1 - final_k1/max(k1_initial, 1e-6)):.1f}% of initial drift)")
print(f"  λ_bimodal final: {current_lambda_bimodal:.4f}  λ_variance final: {current_lambda_variance:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "k1_initial_drift": float(k1_initial),
        "k1_after_phase1": float(phase1_end_k1),
        "k1_final": float(final_k1),
        "final_lossless_drift": float(final_drift),
        "phase1_end_gamma": float(phase1_end_gamma),
        "lambda_bimodal_final": float(current_lambda_bimodal),
        "lambda_variance_final": float(current_lambda_variance),
        "n_body_params": int(n_body),
        "n_beta_g_params": int(n_beta),
        "n_other_levers": int(n_other),
        "phase_1_steps": PHASE_1_STEPS,
        "phase_2_steps": PHASE_2_STEPS,
        "body_lr": BODY_LR,
        "lever_lr": LEVER_LR,
        "beta_lr": BETA_LR,
        "drift_target": DRIFT_TARGET,
        "drift_high": DRIFT_HIGH,
        "lambda_bimodal_target": LAMBDA_BIMODAL_TARGET,
        "lambda_variance_target": LAMBDA_VARIANCE_TARGET,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
