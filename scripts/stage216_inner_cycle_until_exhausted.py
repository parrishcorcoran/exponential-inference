"""Stage 216 — Inner-cycle PID until each lever exhausted.

Refined protocol per user 2026-04-30:

Outer loop: 3 levers — [magnitude γ, bimodal squeeze, per-group variance penalty]
  All three apply mathematical pressure pointing at the K=1 binary target
  (W ∈ {±α_g} per group). Magnitude releases body to hypersphere, then the
  two regularizers directly pull weights to ±α_g grid points.

Inner loop per lever: phase A → phase B → diagnose K=1 drift.
  Repeat until K=1 drift stops dropping meaningfully (lever exhausted).
  Each inner iteration: PID can go DEEPER than the previous because phase B
  built more compensation pathways → more headroom for body movement.

Within each phase: SLOW PID (small step size, tight drift band).
  Lever moves only when drift is well within band. Backs off on drift breach.

Phase A: body trainable, ONLY this lever pulled (others frozen at accumulated state).
Phase B: body frozen, lever returns to identity, ALL OTHER levers trainable
         (they absorb the return pressure → reveal where compensation built).

Levers:
  1. magnitude γ:           architectural buffer in AdiabaticQuantizedLinear
                            identity=0, perturb=0.95 (capped — body has 0 grad at γ=1)
  2. bimodal squeeze λ:     loss regularizer λ·sum_i ||W_i - sign(W_i)·α_i||²
                            identity=0, perturb=1e-2 (calibrated by hand)
  3. per-group variance λ:  loss regularizer λ·sum_i Var_g(|W_i|)
                            identity=0, perturb=1e-2

Body trainable: o_proj + down_proj W_fp (Stage 189 / Finding 27 bottleneck).
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
PHASE_A_STEPS = 400
PHASE_B_STEPS = 400
EVAL_EVERY = 50
BODY_LR = 2e-5
LEVER_LR = 5e-4

# Slow PID
DRIFT_TARGET = 0.05
DRIFT_HIGH = 0.20
PID_STEP_FRAC = 0.05   # PID moves 5% of (perturb_target − identity) per eval

# Inner-cycle convergence
MAX_INNER_ITER = 4
INNER_TOLERANCE_NATS = 0.10   # K=1 drift must drop by at least this for inner to continue

RESULTS_PATH = Path("results/stage216_inner_cycle.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAIN_NAMES = ("o_proj", "down_proj")
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


# ─── AdiabaticQuantizedLinear (W_fp trainable, γ buffer-controlled) ───
class AdiabaticQuantizedLinear(nn.Module):
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
        else:
            self.register_buffer("alpha",
                                 W_fp.abs().mean(dim=-1, keepdim=True).to(W_fp.dtype))
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
            W_eff = (torch.sign(Wg_fp) * mag_eff).reshape(
                self.out_features, self.in_features)
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
    rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n_quantized = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinear(mod)
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
        is_o = name.endswith("o_proj"); is_d = name.endswith("down_proj")
        if not (is_o or is_d): continue
        if name not in rms_table: continue
        gain = rms_table[name].to(device=mod.weight_fp.device, dtype=mod.weight_fp.dtype)
        nh, hd = (num_heads, head_dim) if is_o else (None, None)
        new_layer = SubLNLinear(mod, num_heads=nh, head_dim=hd)
        with torch.no_grad():
            new_layer.subln_gain.data.copy_(gain)
        parent, child_attr = parent_lookup2[name]
        setattr(parent, child_attr, new_layer)
        n_subln += 1

    m.lm_head = TemperedLMHead(m.lm_head)
    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers, n_subln=n_subln)


# ─── Param helpers ───
def is_body_master(name):
    return "weight_fp" in name and any(t in name for t in BODY_TRAIN_NAMES)


def is_any_lever(name):
    if any(t in name for t in (
        "subln_gate", "subln_gain", "h_scale", "attn_gain", "mlp_gain",
        "attn_offset", "mlp_offset", "logit_tau"
    )):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


def set_body_trainable(model, trainable):
    for n, p in model.named_parameters():
        if is_body_master(n):
            p.requires_grad_(trainable)


def set_levers_trainable(model, trainable):
    for n, p in model.named_parameters():
        if is_any_lever(n):
            p.requires_grad_(trainable)


def freeze_everything_else(model):
    for n, p in model.named_parameters():
        if not is_body_master(n) and not is_any_lever(n):
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


# ─── Loss-side regularizers ───
def bimodal_squeeze_loss(model):
    """sum over body W_fp: ||W − sign(W)·α||² mean per parameter."""
    total = 0.0
    n = 0
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear) and mod.has_groups:
            W = mod.weight_fp
            Wg = W.reshape(mod.out_features, mod.in_features // mod.group_size, mod.group_size)
            target = torch.sign(Wg) * mod.alpha
            total = total + ((Wg - target).float() ** 2).mean()
            n += 1
    return total / max(n, 1)


def variance_penalty_loss(model):
    """sum over body W_fp: per-group Var(|W|), mean."""
    total = 0.0
    n = 0
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear) and mod.has_groups:
            W = mod.weight_fp
            Wg = W.reshape(mod.out_features, mod.in_features // mod.group_size, mod.group_size)
            total = total + Wg.abs().float().var(dim=-1).mean()
            n += 1
    return total / max(n, 1)


# ─── Lever definitions ───
LEVERS = [
    {"name": "magnitude",        "kind": "arch", "identity": 0.0,  "perturb": 0.95,
     "loss_fn": None},
    {"name": "bimodal_squeeze",  "kind": "loss", "identity": 0.0,  "perturb": 1e-2,
     "loss_fn": bimodal_squeeze_loss},
    {"name": "variance_penalty", "kind": "loss", "identity": 0.0,  "perturb": 1e-2,
     "loss_fn": variance_penalty_loss},
]


def apply_lever(model, lever, value):
    """Push the lever's current value to `value`. For arch levers, sets γ.
    For loss levers, returns the value (caller adds it to loss)."""
    if lever["kind"] == "arch" and lever["name"] == "magnitude":
        set_gamma(model, value)
    # Loss-lever value is returned via lever_current dict in caller


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def snapshot_lever_state(model):
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if is_any_lever(n)}


def lever_displacement(model, snapshot_before):
    groups = {"bias": [], "subln_gate": [], "subln_gain": [], "h_scale": [],
              "attn_gain": [], "mlp_gain": [], "attn_offset": [], "mlp_offset": [],
              "logit_tau": []}
    for n, p in model.named_parameters():
        if not is_any_lever(n): continue
        if n not in snapshot_before: continue
        diff = (p.detach() - snapshot_before[n]).float().norm().item()
        for key in groups:
            if key in n or (key == "bias" and "bias" in n and "norm" not in n):
                groups[key].append(diff)
                break
    return {k: float(np.mean(v)) if v else 0.0 for k, v in groups.items()}


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
    if direction > 0:
        new = max(0.0, min(1.0 if target > 0.5 else target, new))
    else:
        new = max(0.0, new)
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

print("\nBuilding architecture...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}", flush=True)
k1_initial = k1_drift(model, val_tokens, T0)
print(f"  K=1 initial drift: Δ={k1_initial:+.4f}  (target: shrink toward 0)", flush=True)

freeze_everything_else(model)
n_body = sum(p.numel() for n, p in model.named_parameters() if is_body_master(n))
n_levers = sum(p.numel() for n, p in model.named_parameters() if is_any_lever(n))
print(f"\nBody master params (o/down):  {n_body:,}")
print(f"Lever params (all):           {n_levers:,}", flush=True)

body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
lever_params = [p for n, p in model.named_parameters() if is_any_lever(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": lever_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


# ─── Training step with optional regularizer ───
def train_step(batch, lever_loss_lambda=0.0, lever_loss_fn=None):
    out = model(batch[:, :-1], use_cache=False)
    ce_loss = F.cross_entropy(
        out.logits.float().reshape(-1, out.logits.size(-1)),
        batch[:, 1:].reshape(-1))
    if lever_loss_lambda > 0 and lever_loss_fn is not None:
        reg = lever_loss_fn(model)
        total = ce_loss + lever_loss_lambda * reg
    else:
        total = ce_loss
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return float(ce_loss.item())


def run_phase(phase_name, lever, current_value, target_value,
              train_body, train_levers, n_steps, t_start, history):
    set_body_trainable(model, train_body)
    set_levers_trainable(model, train_levers)
    step_size = abs(lever["perturb"] - lever["identity"]) * PID_STEP_FRAC

    print(f"\n  ── {phase_name} (body={train_body} lev={train_levers}, "
          f"{lever['name']}: {current_value:.4f} → {target_value:.4f}, "
          f"step={step_size:.4f}, n={n_steps}) ──", flush=True)
    model.train()
    if lever["kind"] == "arch":
        apply_lever(model, lever, current_value)
        lever_lambda = 0.0
        loss_fn = None
    else:
        lever_lambda = current_value
        loss_fn = lever["loss_fn"]
    for step in range(1, n_steps + 1):
        batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
        ce_loss = train_step(batch, lever_loss_lambda=lever_lambda, lever_loss_fn=loss_fn)

        if step % EVAL_EVERY == 0 or step == n_steps:
            val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
            drift = val_ce - T0
            current_value = pid_step_toward(current_value, target_value, drift, step_size)
            if lever["kind"] == "arch":
                apply_lever(model, lever, current_value)
                lever_lambda = 0.0
            else:
                lever_lambda = current_value
            elapsed = time.time() - t_start
            print(f"    step {step:>4} {lever['name']}={current_value:.4f} "
                  f"ce={val_ce:.4f} Δ={drift:+.4f} loss={ce_loss:.3f}  {elapsed:.0f}s",
                  flush=True)
            history.append({"phase": phase_name, "step": step,
                            "lever": lever["name"],
                            "value": float(current_value),
                            "ce": float(val_ce), "drift": float(drift),
                            "ce_loss": float(ce_loss)})
            model.train()
    return current_value


t_start = time.time()
history = [{"event": "init", "ce": ce_g0, "drift": drift_g0, "k1_drift": k1_initial}]
all_diagnostics = []
prev_k1 = k1_initial
print(f"\n{'─'*60}")
print(f"Stage 216 — inner-cycle PID until exhausted")
print(f"  PID step frac: {PID_STEP_FRAC}  drift band [{DRIFT_TARGET}, {DRIFT_HIGH}]")
print(f"  Inner tolerance (K=1 drop required): {INNER_TOLERANCE_NATS} nats")
print(f"  Max inner iter per lever: {MAX_INNER_ITER}")
print('─'*60, flush=True)

for lever_idx, lever in enumerate(LEVERS, start=1):
    print(f"\n========== LEVER {lever_idx}/{len(LEVERS)}: {lever['name']} "
          f"(identity={lever['identity']}, perturb={lever['perturb']}) ==========", flush=True)

    inner_diagnostics = []
    last_k1 = prev_k1

    for inner_iter in range(1, MAX_INNER_ITER + 1):
        print(f"\n  --- inner iter {inner_iter}/{MAX_INNER_ITER} for {lever['name']} ---",
              flush=True)
        levers_before_inner = snapshot_lever_state(model)

        # Phase A: body trains, lever PID-driven AWAY from identity
        if lever["kind"] == "arch":
            apply_lever(model, lever, lever["identity"])
        current_value = run_phase(
            f"L{lever_idx}_inner{inner_iter}_phaseA", lever,
            current_value=lever["identity"], target_value=lever["perturb"],
            train_body=True, train_levers=False,
            n_steps=PHASE_A_STEPS, t_start=t_start, history=history)
        max_value_reached = current_value
        print(f"    phase A done. {lever['name']} reached {current_value:.4f} "
              f"(target was {lever['perturb']})", flush=True)
        levers_after_phaseA = snapshot_lever_state(model)

        # Phase B: body frozen, lever returns to identity, OTHER levers train
        current_value = run_phase(
            f"L{lever_idx}_inner{inner_iter}_phaseB", lever,
            current_value=current_value, target_value=lever["identity"],
            train_body=False, train_levers=True,
            n_steps=PHASE_B_STEPS, t_start=t_start, history=history)

        # Compensation displacement during phase B
        phaseB_disp = lever_displacement(model, levers_after_phaseA)
        print(f"    phase B compensation displacement:", flush=True)
        for k, v in sorted(phaseB_disp.items(), key=lambda kv: -kv[1]):
            if v > 0:
                print(f"      {k:18s} L2={v:.4f}", flush=True)

        # End-of-inner K=1 diagnostic
        if lever["kind"] == "arch":
            apply_lever(model, lever, lever["identity"])  # ensure γ=0 for diagnostic
        new_k1 = k1_drift(model, val_tokens, T0)
        improvement = last_k1 - new_k1
        print(f"\n    inner {inner_iter}: max {lever['name']} reached={max_value_reached:.4f}  "
              f"K=1 drift={new_k1:+.4f}  (improvement vs last inner: {improvement:+.4f})",
              flush=True)
        inner_diagnostics.append({
            "inner_iter": inner_iter,
            "max_value_reached": float(max_value_reached),
            "k1_drift": float(new_k1),
            "improvement_vs_last_inner": float(improvement),
            "phaseB_displacement": phaseB_disp,
        })

        if improvement < INNER_TOLERANCE_NATS:
            print(f"    ↳ improvement < {INNER_TOLERANCE_NATS} nats — lever {lever['name']} EXHAUSTED",
                  flush=True)
            last_k1 = new_k1
            break
        last_k1 = new_k1

    all_diagnostics.append({"lever": lever["name"], "inner_diagnostics": inner_diagnostics,
                             "k1_at_lever_end": float(last_k1)})
    prev_k1 = last_k1
    print(f"\n  >>> lever {lever['name']} done. K=1 drift now: {last_k1:+.4f}", flush=True)


# Final
print(f"\n{'─'*60}")
print("STAGE 216 RESULT (inner-cycle PID, mathematical levers):")
print('─'*60)
print(f"  T0:                {T0:.4f}")
print(f"  K=1 initial drift: {k1_initial:+.4f}")
print(f"\n  K=1 drift trajectory by lever:")
running = k1_initial
print(f"    initial:                 K=1 Δ={running:+.4f}")
for d in all_diagnostics:
    delta = d["k1_at_lever_end"] - running
    print(f"    after {d['lever']:18s}: K=1 Δ={d['k1_at_lever_end']:+.4f}  "
          f"(this lever: {delta:+.4f})")
    running = d["k1_at_lever_end"]
final_drift = running
print(f"\n  Total K=1 drift reduction: {k1_initial - final_drift:+.4f} nats "
      f"({100*(1 - final_drift/max(k1_initial,1e-6)):.1f}% of initial drift)")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "k1_initial_drift": float(k1_initial),
        "k1_final_drift": float(final_drift),
        "total_reduction": float(k1_initial - final_drift),
        "n_body_params": int(n_body),
        "n_lever_params": int(n_levers),
        "phase_a_steps": PHASE_A_STEPS,
        "phase_b_steps": PHASE_B_STEPS,
        "body_lr": BODY_LR,
        "lever_lr": LEVER_LR,
        "pid_step_frac": PID_STEP_FRAC,
        "drift_target": DRIFT_TARGET,
        "drift_high": DRIFT_HIGH,
        "inner_tolerance_nats": INNER_TOLERANCE_NATS,
        "max_inner_iter": MAX_INNER_ITER,
        "lever_diagnostics": all_diagnostics,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
