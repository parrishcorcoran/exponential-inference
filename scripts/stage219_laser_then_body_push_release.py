"""Stage 219 — Laser-zone discovery + body-push with release-valve cycles.

Per user 2026-04-30:

Pass 1 — find laser zone (no training, just PID sweep):
  Everything FROZEN. γ slow-PID ramps from 0 until it stalls.
  Output: γ_laser (deepest γ untouched arch tolerates without breaking drift band)

Pass 2 — body-push at γ_laser with release-valve cycles (max 5):
  γ HELD at γ_laser throughout pass 2.

  Each cycle:
    Body-push subphase:
      UNFROZEN: W_fp + β_g (MLP-only — these define binary geometry)
      FROZEN:   all compensation levers (residual gain/offset, SubLN,
                h_scale, logit τ, per-output bias)
      Loss = CE + λ_bimodal·||W − sign(W)·α||²    (regularizer ON — the WHOLE point)
      Train until body gradient plateaus (body's flow has saturated under
      current lever state).

    Release-valve subphase:
      FROZEN:   body + β_g (lock new flowed position)
      UNFROZEN: all compensation levers
      Loss = CE only (regularizer OFF)
      Train until lever movement slows (compensations have absorbed pressure).
      Drift should drop near zero.

    K=1 diagnostic at end of cycle. If K=1 drift didn't drop meaningfully
    (< 0.10 nat) vs cycle start, terminate (no more progress).

  Max 5 cycles, hard cap.

Final checkpoint: body has flowed as far as periodic release allows.
This is the "best body geometry achievable with these levers at this γ."
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

# Pass 1
PASS1_MAX_ITERS = 100             # PID sweep iterations (no training)
PASS1_GAMMA_STEP = 0.025
PASS1_STALL_TOLERANCE_ITERS = 5    # γ unchanged for this many iters → laser found

# Pass 2
MAX_CYCLES = 5
BODY_PUSH_MAX_STEPS = 1000
RELEASE_MAX_STEPS = 500
EVAL_EVERY = 50
PLATEAU_GRAD_THRESHOLD = 1e-3      # gradient norm below this for plateau
PLATEAU_PATIENCE = 3                # consecutive evals below threshold = plateau
CYCLE_PROGRESS_THRESHOLD = 0.10     # K=1 drift must drop by at least this
LAMBDA_BIMODAL = 1e-2

# PID
DRIFT_TARGET = 0.02
DRIFT_HIGH = 0.05

# LRs
BODY_LR = 2e-5
LEVER_LR = 5e-4
BETA_LR = 5e-4

RESULTS_PATH = Path("results/stage219_laser_body_push.json")
TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")
BODY_TRAIN_NAMES = TARGET_NAMES
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


# Reuse 218's architecture
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


def build_full_architecture(num_heads, head_dim, calib_ids):
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


def freeze_all(model):
    for n, p in model.named_parameters():
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


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def grad_norm_for(model, predicate):
    norms = []
    for n, p in model.named_parameters():
        if predicate(n) and p.grad is not None:
            norms.append(p.grad.detach().float().norm().item())
    return float(np.mean(norms)) if norms else 0.0


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

print("\nBuilding architecture (MLP-only + β_g)...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
k1_initial = k1_drift_now(model, val_tokens, T0)
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}  K=1 initial: {k1_initial:+.4f}",
      flush=True)

t_start = time.time()
history = []


# ─── PASS 1 — find laser zone (no training) ───
print(f"\n{'─'*60}")
print("PASS 1 — find laser zone (everything frozen, γ slow-PID sweep, no training)")
print('─'*60, flush=True)
freeze_all(model)
set_gamma(model, 0.0)

current_gamma = 0.0
gamma_history = [0.0]
last_advancing_iter = 0
γ_laser = 0.0

for iter_idx in range(1, PASS1_MAX_ITERS + 1):
    drift = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS) - T0
    history.append({"phase": "P1", "iter": iter_idx, "gamma": float(current_gamma),
                    "drift": float(drift)})
    elapsed = time.time() - t_start
    advancing = False
    if drift > DRIFT_HIGH:
        new_gamma = max(current_gamma - PASS1_GAMMA_STEP, 0.0)
    elif drift < DRIFT_TARGET:
        new_gamma = min(current_gamma + PASS1_GAMMA_STEP, 1.0)
        if new_gamma > current_gamma:
            advancing = True
    else:
        new_gamma = current_gamma
    print(f"  P1 iter {iter_idx:>3} γ={current_gamma:.3f} drift={drift:+.4f}  "
          f"{'→advance' if advancing else 'hold/back-off'}  next γ={new_gamma:.3f}  "
          f"{elapsed:.0f}s", flush=True)
    if advancing:
        last_advancing_iter = iter_idx
    current_gamma = new_gamma
    set_gamma(model, current_gamma)
    gamma_history.append(current_gamma)

    # Stall: hasn't advanced in PASS1_STALL_TOLERANCE_ITERS iters
    if iter_idx - last_advancing_iter >= PASS1_STALL_TOLERANCE_ITERS:
        γ_laser = current_gamma
        print(f"\n  P1 STALL — γ hasn't advanced in {PASS1_STALL_TOLERANCE_ITERS} iters. "
              f"γ_laser = {γ_laser:.3f}", flush=True)
        break
else:
    γ_laser = current_gamma
    print(f"\n  P1 hit max iters. γ_laser = {γ_laser:.3f}", flush=True)


# ─── PASS 2 — body-push at γ_laser with release-valve cycles ───
print(f"\n{'─'*60}")
print(f"PASS 2 — body-push at γ_laser={γ_laser:.3f} with release valve (max {MAX_CYCLES} cycles)")
print('─'*60, flush=True)
set_gamma(model, γ_laser)

# Adam on body+β_g and on compensation levers (we'll toggle requires_grad per subphase)
body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
beta_params = [p for n, p in model.named_parameters() if is_beta_g(n)]
lever_params = [p for n, p in model.named_parameters() if is_compensation_lever(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": beta_params,  "lr": BETA_LR},
    {"params": lever_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


def train_step(batch, regularizer_lambda=0.0):
    out = model(batch[:, :-1], use_cache=False)
    ce_loss = F.cross_entropy(
        out.logits.float().reshape(-1, out.logits.size(-1)),
        batch[:, 1:].reshape(-1))
    total = ce_loss
    if regularizer_lambda > 0:
        total = total + regularizer_lambda * bimodal_squeeze_loss(model)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return float(ce_loss.item())


cycle_diagnostics = []
prev_k1 = k1_initial
for cycle_idx in range(1, MAX_CYCLES + 1):
    print(f"\n===== CYCLE {cycle_idx}/{MAX_CYCLES} =====", flush=True)

    # ── Body-push subphase ──
    print(f"  Body-push subphase (body+β_g unfrozen, levers FROZEN, λ_bimodal={LAMBDA_BIMODAL})",
          flush=True)
    set_trainable(model, is_body_master, True)
    set_trainable(model, is_beta_g, True)
    set_trainable(model, is_compensation_lever, False)

    plateau_count = 0
    for step in range(1, BODY_PUSH_MAX_STEPS + 1):
        batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
        ce_loss = train_step(batch, regularizer_lambda=LAMBDA_BIMODAL)
        if step % EVAL_EVERY == 0:
            val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
            drift = val_ce - T0
            body_grad = grad_norm_for(model, is_body_master)
            beta_grad = grad_norm_for(model, is_beta_g)
            elapsed = time.time() - t_start
            print(f"    BP step {step:>4} ce={val_ce:.4f} Δ={drift:+.4f} "
                  f"loss={ce_loss:.3f} body_grad={body_grad:.5f} beta_grad={beta_grad:.5f}  "
                  f"{elapsed:.0f}s", flush=True)
            history.append({"phase": f"C{cycle_idx}_BP", "step": step,
                            "ce": float(val_ce), "drift": float(drift),
                            "body_grad": body_grad, "beta_grad": beta_grad})
            if body_grad < PLATEAU_GRAD_THRESHOLD and beta_grad < PLATEAU_GRAD_THRESHOLD:
                plateau_count += 1
                if plateau_count >= PLATEAU_PATIENCE:
                    print(f"    → body PLATEAU after {step} steps (grad < threshold for "
                          f"{PLATEAU_PATIENCE} consecutive evals)", flush=True)
                    break
            else:
                plateau_count = 0

    # ── Release-valve subphase ──
    print(f"  Release-valve subphase (body+β_g FROZEN, levers UNFROZEN, no regularizer)",
          flush=True)
    set_trainable(model, is_body_master, False)
    set_trainable(model, is_beta_g, False)
    set_trainable(model, is_compensation_lever, True)

    plateau_count = 0
    for step in range(1, RELEASE_MAX_STEPS + 1):
        batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
        ce_loss = train_step(batch, regularizer_lambda=0.0)
        if step % EVAL_EVERY == 0:
            val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
            drift = val_ce - T0
            lever_grad = grad_norm_for(model, is_compensation_lever)
            elapsed = time.time() - t_start
            print(f"    RV step {step:>4} ce={val_ce:.4f} Δ={drift:+.4f} "
                  f"loss={ce_loss:.3f} lever_grad={lever_grad:.5f}  "
                  f"{elapsed:.0f}s", flush=True)
            history.append({"phase": f"C{cycle_idx}_RV", "step": step,
                            "ce": float(val_ce), "drift": float(drift),
                            "lever_grad": lever_grad})
            if lever_grad < PLATEAU_GRAD_THRESHOLD:
                plateau_count += 1
                if plateau_count >= PLATEAU_PATIENCE:
                    print(f"    → lever PLATEAU after {step} steps", flush=True)
                    break
            else:
                plateau_count = 0

    # K=1 diagnostic
    k1_now = k1_drift_now(model, val_tokens, T0)
    improvement = prev_k1 - k1_now
    print(f"\n  CYCLE {cycle_idx} END: K=1 drift = {k1_now:+.4f}  "
          f"(Δ vs prev cycle: {-improvement:+.4f})", flush=True)
    cycle_diagnostics.append({"cycle": cycle_idx, "k1_drift": float(k1_now),
                               "improvement_vs_prev": float(improvement)})

    if improvement < CYCLE_PROGRESS_THRESHOLD:
        print(f"  → improvement < {CYCLE_PROGRESS_THRESHOLD} nats — TERMINATE (no more progress)",
              flush=True)
        prev_k1 = k1_now
        break
    prev_k1 = k1_now


# Final
final_k1 = prev_k1
print(f"\n{'─'*60}")
print("STAGE 219 RESULT:")
print('─'*60)
print(f"  T0:                {T0:.4f}")
print(f"  K=1 initial drift: {k1_initial:+.4f}")
print(f"  γ_laser found:     {γ_laser:.3f}")
print(f"  K=1 final drift:   {final_k1:+.4f}  (after {len(cycle_diagnostics)} cycles)")
print(f"  Total reduction:   {k1_initial - final_k1:+.4f} nats")
print(f"  Cycle trajectory:")
running = k1_initial
print(f"    initial:           {running:+.4f}")
for d in cycle_diagnostics:
    print(f"    after cycle {d['cycle']}:   {d['k1_drift']:+.4f}  "
          f"(this cycle: {-d['improvement_vs_prev']:+.4f})")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "k1_initial_drift": float(k1_initial),
        "gamma_laser": float(γ_laser),
        "k1_final_drift": float(final_k1),
        "total_reduction": float(k1_initial - final_k1),
        "max_cycles": MAX_CYCLES,
        "n_cycles_run": len(cycle_diagnostics),
        "cycle_diagnostics": cycle_diagnostics,
        "lambda_bimodal": LAMBDA_BIMODAL,
        "drift_target": DRIFT_TARGET,
        "drift_high": DRIFT_HIGH,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
