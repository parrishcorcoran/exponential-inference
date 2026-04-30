"""Stage 215 — Lever-cycle hysteresis with body flow (the actual recipe).

Validated protocol per Finding 31 + user clarification:

  Phase A (lever moves AWAY from identity):
    body weights:     UNFROZEN, train under PID
    wobbled lever:    PID-driven away from identity
    OTHER levers:     FROZEN at current accumulated values
    → body flows to absorb the perturbation (RG flow to deformed attractor)

  Phase B (lever RETURNS to identity):
    body weights:     FROZEN at flowed position
    wobbled lever:    PID-driven back to identity (now lossless)
    OTHER levers:     UNFROZEN, train to compensate the return pressure
    → body keeps its flowed geometry; other levers shift to compensate

  Each cycle: lever-of-the-cycle ends back at identity (architecture lossless w.r.t. it),
  body has moved (permanent), other levers have shifted (compensation accumulating).

This stage runs ONE lever for proof of concept: magnitude (γ in the
AdiabaticQuantizedLinear). γ=0 → W_eff = W_fp (FP, lossless start).
γ→1 → W_eff = sign(W_fp)·α_g (K=1). Phase A drives γ up; phase B drives γ down.

Ratcheting target schedule: each cycle's phase-A target γ grows
(0.2, 0.4, 0.6, 0.8, 1.0). Body sees progressively tighter constraints
across cycles. Other levers accumulate compensation across cycles.

End-of-cycle diagnostic: snapshot levers, set γ=1.0, measure K=1 CE
drift, restore. Want this DECREASING across cycles → body becoming
binary-capable.

Body trainable: W_fp on o_proj + down_proj only (Stage 189 setup,
~110M params, Finding 27 says these are the bottleneck).

PID setpoint: drift ≤ 0.10 nats per Stage 189 convention.
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
PHASE_A_STEPS = 300
PHASE_B_STEPS = 300
EVAL_EVERY = 50
BODY_LR = 2e-5      # small for adiabatic flow; Stage 189 used 5e-5
LEVER_LR = 5e-4

DRIFT_TARGET = 0.05
DRIFT_HIGH = 0.20
GAMMA_PID_STEP = 0.02   # how much γ moves per PID tick

GAMMA_TARGETS = [0.20, 0.40, 0.60, 0.80, 0.95]   # ratcheting per-cycle phase-A target
# Final γ=1.0 is measurement only (body has zero gradient through sign() at γ=1).

RESULTS_PATH = Path("results/stage215_lever_cycle.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAIN_NAMES = ("o_proj", "down_proj")   # Stage 189 / Finding 27 bottleneck
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


# ─── AdiabaticQuantizedLinear (no LoRA — body W_fp is now trainable) ───
class AdiabaticQuantizedLinear(nn.Module):
    """W_eff = sign(W_fp) · (γ · α_g + (1−γ) · |W_fp|).
    W_fp is now a TRAINABLE parameter (master weights).
    γ is a buffer (PID-controlled, not learned).
    α_g is a buffer (per-group magnitude, frozen at init).
    """
    def __init__(self, original_linear, group_size=GROUP_SIZE):
        super().__init__()
        W_fp = original_linear.weight.data.clone()
        self.weight_fp = nn.Parameter(W_fp, requires_grad=True)   # ← trainable
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


def is_body_master(name):
    """W_fp on o_proj or down_proj — Stage 189 / Finding 27 bottleneck."""
    return "weight_fp" in name and any(t in name for t in BODY_TRAIN_NAMES)


def is_lever_param(name):
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
        if is_lever_param(n):
            p.requires_grad_(trainable)


def freeze_everything_else(model):
    """Anything that's neither body master nor lever stays frozen."""
    for n, p in model.named_parameters():
        if not is_body_master(n) and not is_lever_param(n):
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


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    batch = torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)
    return batch


def k1_diagnostic_drift(model, val_tokens, T0):
    """Snapshot γ, set γ=1, measure CE drift, restore γ. Diagnostic only."""
    γ_save = get_gamma(model)
    set_gamma(model, 1.0)
    ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
    set_gamma(model, γ_save)
    return ce - T0


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
del m0
import gc; gc.collect()

print("\nBuilding architecture...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}", flush=True)
k1_initial = k1_diagnostic_drift(model, val_tokens, T0)
print(f"  K=1 initial drift: Δ={k1_initial:+.4f}  (this is what we want to shrink)", flush=True)

freeze_everything_else(model)
n_body = sum(p.numel() for n, p in model.named_parameters() if is_body_master(n))
n_levers = sum(p.numel() for n, p in model.named_parameters() if is_lever_param(n))
print(f"\nBody master params (o/down):  {n_body:,}")
print(f"Lever params:                  {n_levers:,}", flush=True)

# Single optimizer over all trainable; toggle requires_grad per phase.
all_trainable = [p for n, p in model.named_parameters()
                 if is_body_master(n) or is_lever_param(n)]
# Per-group LRs via param_groups
body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
lever_params = [p for n, p in model.named_parameters() if is_lever_param(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": lever_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)

set_gamma(model, 0.0)
history = [{"phase": "init", "gamma": 0.0, "ce": ce_g0, "drift": drift_g0,
            "k1_drift": k1_initial}]
t_start = time.time()


def pid_step_gamma_toward(model, target, drift, current_gamma):
    """Crude PID: move γ toward target only if drift in band; back off if drift too high.
    Phase B target is treated as a soft floor — PID stops short if drift won't allow
    full return to identity. The 'laser zone' falls out of this naturally."""
    direction = 1 if target > current_gamma else -1
    if drift > DRIFT_HIGH:
        new = current_gamma - direction * GAMMA_PID_STEP
    elif drift < DRIFT_TARGET:
        if direction > 0:
            new = min(current_gamma + GAMMA_PID_STEP, target)
        else:
            new = max(current_gamma - GAMMA_PID_STEP, target)
    else:
        new = current_gamma
    new = max(0.0, min(1.0, new))
    set_gamma(model, new)
    return new


def snapshot_lever_state(model):
    """Snapshot current values of all lever-type params. Returns dict[name → tensor]."""
    return {n: p.detach().clone()
            for n, p in model.named_parameters() if is_lever_param(n)}


def lever_displacement(model, snapshot_before):
    """For each lever-group, compute mean L2 displacement from snapshot.
    Reveals which levers absorbed pressure during a phase."""
    groups = {"bias": [], "subln_gate": [], "subln_gain": [], "h_scale": [],
              "attn_gain": [], "mlp_gain": [], "attn_offset": [], "mlp_offset": [],
              "logit_tau": []}
    for n, p in model.named_parameters():
        if not is_lever_param(n): continue
        if n not in snapshot_before: continue
        diff = (p.detach() - snapshot_before[n]).float().norm().item()
        for key in groups:
            if key in n or (key == "bias" and "bias" in n and "norm" not in n):
                groups[key].append(diff)
                break
    return {k: float(np.mean(v)) if v else 0.0 for k, v in groups.items()}


def run_phase(model, phase_name, train_body, train_levers,
              gamma_target, n_steps, train_tokens, val_tokens, T0, history, t_start):
    set_body_trainable(model, train_body)
    set_levers_trainable(model, train_levers)

    current_gamma = get_gamma(model)
    print(f"\n  ── {phase_name} (body_train={train_body}, lever_train={train_levers}, "
          f"γ {current_gamma:.2f} → target {gamma_target:.2f}, {n_steps} steps) ──", flush=True)
    model.train()
    for step in range(1, n_steps + 1):
        batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
        out = model(batch[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == n_steps:
            val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
            drift = val_ce - T0
            current_gamma = pid_step_gamma_toward(model, gamma_target, drift, current_gamma)
            elapsed = time.time() - t_start
            print(f"    step {step:>4} γ={current_gamma:.2f} ce={val_ce:.4f} Δ={drift:+.4f} "
                  f"loss={loss.item():.3f}  {elapsed:.0f}s", flush=True)
            history.append({"phase": phase_name, "step": step, "gamma": float(current_gamma),
                            "ce": float(val_ce), "drift": float(drift),
                            "loss": float(loss.item())})
            model.train()
    return get_gamma(model)


print(f"\n{'─'*60}")
print(f"Lever-cycle hysteresis: {len(GAMMA_TARGETS)} cycles, "
      f"phaseA={PHASE_A_STEPS} phaseB={PHASE_B_STEPS} steps each")
print(f"BODY_LR={BODY_LR}  LEVER_LR={LEVER_LR}  PID step={GAMMA_PID_STEP}")
print('─'*60, flush=True)

cycle_diagnostics = [{"cycle": 0, "k1_drift": k1_initial}]

for cycle_idx, gamma_target in enumerate(GAMMA_TARGETS, start=1):
    print(f"\n===== CYCLE {cycle_idx}/{len(GAMMA_TARGETS)}  (γ_target = {gamma_target:.2f}) =====",
          flush=True)

    # Snapshot lever values before cycle to measure compensation displacement
    levers_before_cycle = snapshot_lever_state(model)

    # Phase A: body trains, this lever (γ) drives UP toward target, others frozen
    set_gamma(model, 0.0)   # start each cycle at γ=0
    run_phase(model, f"cycle{cycle_idx}_phaseA",
              train_body=True, train_levers=False,
              gamma_target=gamma_target,
              n_steps=PHASE_A_STEPS,
              train_tokens=train_tokens, val_tokens=val_tokens, T0=T0,
              history=history, t_start=t_start)

    # Snapshot levers before phase B (should equal levers_before_cycle since phase A didn't touch them)
    levers_after_phaseA = snapshot_lever_state(model)

    # Phase B: body frozen, this lever drives DOWN to identity, others train
    run_phase(model, f"cycle{cycle_idx}_phaseB",
              train_body=False, train_levers=True,
              gamma_target=0.0,   # SOFT target — PID may stop at "laser zone" if drift won't allow
              n_steps=PHASE_B_STEPS,
              train_tokens=train_tokens, val_tokens=val_tokens, T0=T0,
              history=history, t_start=t_start)

    # Compensation diagnostic: how much did each lever group move during phase B?
    phaseB_displacement = lever_displacement(model, levers_after_phaseA)
    print(f"\n  cycle {cycle_idx} compensation displacement (phase B):", flush=True)
    for k, v in sorted(phaseB_displacement.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"    {k:18s} L2={v:.4f}", flush=True)

    # End-of-cycle diagnostic
    cur_g = get_gamma(model)
    end_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
    end_drift = end_ce - T0
    k1_drift = k1_diagnostic_drift(model, val_tokens, T0)
    print(f"\n  cycle {cycle_idx} END: γ={cur_g:.2f} (laser zone if not 0)  "
          f"ce={end_ce:.4f} Δ={end_drift:+.4f}  "
          f"K=1_drift={k1_drift:+.4f}  (initial K=1 was {k1_initial:+.4f})", flush=True)
    cycle_diagnostics.append({"cycle": cycle_idx, "gamma_target": gamma_target,
                              "end_gamma": float(cur_g), "end_ce": float(end_ce),
                              "end_drift": float(end_drift),
                              "k1_drift": float(k1_drift),
                              "phaseB_compensation_displacement": phaseB_displacement})

# ─── Final K=1 ───
print(f"\n{'─'*60}")
print("FINAL: applying K=1 to flowed body...")
print('─'*60)
set_gamma(model, 1.0)
final_ce = lm_ce(model, val_tokens)
final_drift = final_ce - T0
print(f"  T0:                {T0:.4f}")
print(f"  K=1 initial drift: {k1_initial:+.4f}  (raw FP body, no flow)")
print(f"  K=1 final drift:   {final_drift:+.4f}  (after {len(GAMMA_TARGETS)} lever cycles)")
print(f"  Improvement:       {k1_initial - final_drift:+.4f} nats")

print(f"\nK=1 drift trajectory across cycles:")
for d in cycle_diagnostics:
    cycle = d.get("cycle", 0)
    k1 = d["k1_drift"]
    print(f"  cycle {cycle}: K=1 Δ={k1:+.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "ce_gamma_0_init": float(ce_g0),
        "k1_initial_drift": float(k1_initial),
        "final_ce": float(final_ce),
        "final_drift": float(final_drift),
        "improvement": float(k1_initial - final_drift),
        "n_body_params": int(n_body),
        "n_lever_params": int(n_levers),
        "gamma_targets": GAMMA_TARGETS,
        "phase_a_steps": PHASE_A_STEPS,
        "phase_b_steps": PHASE_B_STEPS,
        "body_lr": BODY_LR,
        "lever_lr": LEVER_LR,
        "gamma_pid_step": GAMMA_PID_STEP,
        "drift_target": DRIFT_TARGET,
        "drift_high": DRIFT_HIGH,
        "cycle_diagnostics": cycle_diagnostics,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
