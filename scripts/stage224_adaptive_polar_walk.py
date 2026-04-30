"""Stage 224 — Adaptive walk of body weights toward polar (±α grid).

User's geometric framing: body weights "want to diffuse" (high entropy, Gaussian).
We're pushing against the system. Resistance grows exponentially near polar.
Levers must compensate equally — eventually pulling as hard as we're pushing.

Protocol:
  Save |W|_init at start (immutable reference).

  f = 0.0   (fraction of way from FP to polar)
  step_size = 0.01    (initial increment)
  K_inner = 50        (initial laminarization steps per increment)

  while f < 1.0:
    f += step_size, capped at 1.0

    With no_grad: set body magnitudes
      |W|_new = (1-f)·|W|_init + f·α_g
      W_fp ← sign(W_init) · |W|_new
    (Body has now moved a fraction f toward polar; signs preserved.)

    Train levers (frozen body) for K_inner steps under CE.
    → Flow re-laminarizes at the new body position.

    Measure: val CE drift, geometric distance, lever movement, K=1 drift periodically.

    Adapt:
      if drift > target_drift:  step_size halves, K_inner grows 1.5×
      elif drift < target/2:    step_size grows 1.5× (cap at max)
      else:                     hold

  When step_size hits floor and drift can't drop → that's the wall.
  Log everything around the wall so Stage 225 can design the right "pull" mechanism.

γ stays at 0 throughout (so body movement is meaningful in forward).
β_g stays at 0 throughout (γ·β_g = 0 at γ=0, doesn't contribute).
At the end: K=1 measurement at γ=1 (β_g still 0) ≈ val CE at γ=0.
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
EVAL_EVERY = 25      # within laminarization
K1_DIAG_EVERY_INCS = 5    # measure K=1 drift every N increments

LEVER_LR = 5e-4
BETA_LR = 5e-4

# Adaptive walk parameters
INIT_STEP_SIZE = 0.01
MIN_STEP_SIZE = 0.0005
MAX_STEP_SIZE = 0.05
INIT_K_INNER = 50
MAX_K_INNER = 400
TARGET_DRIFT = 0.05
DRIFT_HIGH = 0.20
MAX_INCREMENTS = 200       # safety bound

RESULTS_PATH = Path("results/stage224_adaptive_polar.json")
TARGET_NAMES = ("gate_proj", "up_proj", "down_proj")
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
    """Body W_fp + α_g + β_g + γ. Also stores |W|_init and sign(W)_init as buffers."""
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
        # Save original |W| and sign(W) for reconstruction at any f
        self.register_buffer("init_abs", W_fp.abs().clone())
        self.register_buffer("init_sign", torch.sign(W_fp).clone())
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


def is_beta_g(name): return "beta_g" in name


def is_compensation_lever(name):
    if any(t in name for t in (
        "subln_gate", "subln_gain", "h_scale", "attn_gain", "mlp_gain",
        "attn_offset", "mlp_offset", "logit_tau"
    )):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


def freeze_all(model):
    for n, p in model.named_parameters():
        p.requires_grad_(False)


def set_levers_trainable(model, value):
    for n, p in model.named_parameters():
        if is_beta_g(n) or is_compensation_lever(n):
            p.requires_grad_(value)


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


def update_body_to_fraction(model, f):
    """Set body W_fp to interpolated polar position based on init buffers + f."""
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, AdiabaticQuantizedLinear):
                # alpha shape: [out, n_groups, 1]; init_abs shape: [out, in]
                if mod.has_groups:
                    n_groups = mod.in_features // mod.group_size
                    init_abs_g = mod.init_abs.reshape(mod.out_features, n_groups, mod.group_size)
                    new_abs_g = (1 - f) * init_abs_g + f * mod.alpha   # broadcast
                    new_abs = new_abs_g.reshape(mod.out_features, mod.in_features)
                else:
                    new_abs = (1 - f) * mod.init_abs + f * mod.alpha
                mod.weight_fp.data.copy_(mod.init_sign * new_abs)


def geometric_distance(model):
    """Average ||W − sign(W)·α||² / ||W||² across body groups. 0 = polar; ~0.6 = Gaussian."""
    total_num = 0.0; total_den = 0.0
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear) and mod.has_groups:
            W = mod.weight_fp
            Wg = W.reshape(mod.out_features, mod.in_features // mod.group_size, mod.group_size)
            sign_Wg = torch.sign(Wg)
            target = sign_Wg * mod.alpha
            num = ((Wg - target) ** 2).float().sum().item()
            den = (Wg ** 2).float().sum().item()
            total_num += num
            total_den += den
    return total_num / max(total_den, 1e-8)


def lever_grad_norms(model):
    norms = {}
    for group_name in ["bias", "subln_gate", "subln_gain", "attn_gain", "mlp_gain",
                       "attn_offset", "mlp_offset", "logit_tau"]:
        gs = []
        for n, p in model.named_parameters():
            if p.grad is None: continue
            if group_name in n or (group_name == "bias" and "bias" in n and "norm" not in n):
                gs.append(p.grad.detach().float().norm().item())
        norms[group_name] = float(np.mean(gs)) if gs else 0.0
    return norms


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 32].tolist()
train_tokens = corpus[SEQ_LEN * 32:SEQ_LEN * 32 + 1_000_000].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)

print("\nMeasuring T0...", flush=True)
m0 = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(m0, val_tokens)
print(f"  T0 = {T0:.4f}", flush=True)
del m0; import gc; gc.collect()

print("\nBuilding architecture...", flush=True)
model, install_stats = build_full_architecture(calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
k1_initial = k1_drift_now(model, val_tokens, T0)
geom_initial = geometric_distance(model)
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}", flush=True)
print(f"  K=1 initial: {k1_initial:+.4f}", flush=True)
print(f"  Geometric distance initial: {geom_initial:.4f}  (Gaussian floor ≈ 0.60)", flush=True)

freeze_all(model)
set_levers_trainable(model, True)

beta_params = [p for n, p in model.named_parameters() if is_beta_g(n)]
lever_params = [p for n, p in model.named_parameters() if is_compensation_lever(n)]
optimizer = torch.optim.Adam([
    {"params": beta_params,  "lr": BETA_LR},
    {"params": lever_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


def laminarize(K, t_start, history):
    """Train levers under CE for K steps."""
    model.train()
    final_drift = None
    final_grad_norms = None
    for k in range(K):
        batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
        out = model(batch[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        # capture grad norms BEFORE step (last iter)
        if k == K - 1:
            final_grad_norms = lever_grad_norms(model)
        optimizer.step()
    val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
    final_drift = val_ce - T0
    return final_drift, final_grad_norms


t_start = time.time()
history = []
f = 0.0
step_size = INIT_STEP_SIZE
K_inner = INIT_K_INNER
best_k1 = k1_initial
best_f = 0.0

print(f"\n{'─'*60}")
print(f"Adaptive walk: target_drift={TARGET_DRIFT}, drift_high={DRIFT_HIGH}")
print(f"  init step={INIT_STEP_SIZE}, min={MIN_STEP_SIZE}, max={MAX_STEP_SIZE}")
print(f"  init K_inner={INIT_K_INNER}, max={MAX_K_INNER}")
print('─'*60, flush=True)

inc = 0
while f < 1.0 and inc < MAX_INCREMENTS:
    inc += 1
    f_old = f
    f = min(f + step_size, 1.0)

    # Move body
    update_body_to_fraction(model, f)

    # Laminarize
    drift, grad_norms = laminarize(K_inner, t_start, history)

    # Diagnostics
    geom = geometric_distance(model)
    elapsed = time.time() - t_start

    # K=1 only every K1_DIAG_EVERY_INCS increments
    if inc % K1_DIAG_EVERY_INCS == 0 or inc == 1:
        k1 = k1_drift_now(model, val_tokens, T0)
        if k1 < best_k1:
            best_k1 = k1
            best_f = f
        k1_str = f"K=1={k1:+.3f}"
    else:
        k1_str = ""

    print(f"  inc {inc:>3}  f={f:.3f}  step={step_size:.4f}  K={K_inner:>4}  "
          f"drift={drift:+.4f}  geom={geom:.4f}  {k1_str}  "
          f"top_grad: " +
          ", ".join(f"{k}={v:.3f}" for k, v in
                     sorted(grad_norms.items(), key=lambda kv: -kv[1])[:3]) +
          f"  {elapsed:.0f}s", flush=True)

    history.append({
        "increment": inc, "f": float(f), "step_size": float(step_size),
        "K_inner": int(K_inner), "drift": float(drift), "geom": float(geom),
        "k1_drift": float(k1) if "k1" in dir() and inc % K1_DIAG_EVERY_INCS == 0 else None,
        "lever_grad_norms": grad_norms,
    })

    # Adapt
    if drift > DRIFT_HIGH:
        # Severely turbulent — pull back, more lamination
        f = f_old   # rollback this increment
        step_size = max(step_size * 0.5, MIN_STEP_SIZE)
        K_inner = min(int(K_inner * 1.5), MAX_K_INNER)
        update_body_to_fraction(model, f)   # rollback body too
        print(f"    ↳ DRIFT_HIGH breach. Rolling back to f={f_old:.3f}, "
              f"new step={step_size:.4f}, K_inner={K_inner}", flush=True)
    elif drift > TARGET_DRIFT:
        # Borderline — slow down
        step_size = max(step_size * 0.7, MIN_STEP_SIZE)
        K_inner = min(int(K_inner * 1.2), MAX_K_INNER)
    elif drift < TARGET_DRIFT / 2:
        # Lots of headroom — speed up
        step_size = min(step_size * 1.5, MAX_STEP_SIZE)

    # Wall detection
    if step_size <= MIN_STEP_SIZE * 1.05 and drift > TARGET_DRIFT:
        print(f"\n  >>> WALL DETECTED at f={f:.3f}, geom={geom:.4f}, drift={drift:+.4f}",
              flush=True)
        # Don't break — keep trying with min step + max K_inner
        # Stage 225 will analyze the wall data


# Final
final_k1 = k1_drift_now(model, val_tokens, T0)
final_geom = geometric_distance(model)
final_drift = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS) - T0

print(f"\n{'─'*60}")
print(f"STAGE 224 RESULT (adaptive polar walk):")
print('─'*60)
print(f"  T0:              {T0:.4f}")
print(f"  Increments run:  {inc}")
print(f"  Final f:         {f:.4f}  (1.0 = polar)")
print(f"  Final geom dist: {final_geom:.4f}  (initial: {geom_initial:.4f}, polar=0)")
print(f"  Final val drift: {final_drift:+.4f}")
print(f"  Final K=1:       {final_k1:+.4f}  (initial: {k1_initial:+.4f})")
print(f"  Best K=1 seen:   {best_k1:+.4f} at f={best_f:.3f}")

with open(RESULTS_PATH, "w") as f_out:
    json.dump({
        "T0": float(T0),
        "k1_initial": float(k1_initial),
        "geom_initial": float(geom_initial),
        "n_increments": int(inc),
        "final_f": float(f),
        "final_geom": float(final_geom),
        "final_drift": float(final_drift),
        "final_k1": float(final_k1),
        "best_k1": float(best_k1),
        "best_f": float(best_f),
        "history": history,
    }, f_out, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
