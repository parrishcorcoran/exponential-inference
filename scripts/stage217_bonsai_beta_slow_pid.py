"""Stage 217 — Bonsai-class compensation + tight PID, single pass per lever.

Stage 216 inner iter 2 confirmed user's prediction: compensations built in
inner 1 CONSTRAIN body in inner 2 (γ ceiling dropped 0.285 → 0.2375).
Frozen-compensation mismatch with body's new flow direction. So inner-cycle
plan is wrong — better to ride the laser ONCE, deeply, on first pass.

Two changes from Stage 216:

1. Add Bonsai per-128-group bias β_g into AdiabaticQuantizedLinear.
   ~16× more compensation per layer than per-output bias alone.
   Forward: W_eff[i,j] = sign(W_fp[i,j])·(γ·α[i,g] + (1−γ)·|W_fp[i,j]|) + γ·β_g[i,g]
   At γ=0: lossless (β_g is multiplied out).
   At γ=1: full Bonsai sign·α + β representation.

2. Tighter PID, single pass per lever:
   - drift band [0.02, 0.05] (was [0.05, 0.20])
   - step 2% of perturb_target per eval (was 5%)
   - 800-step phases (was 400)
   - one phase A + one phase B per lever, no inner re-cycling
   - "ride the laser, max 5% loss" — never breach drift band

Three levers (same math as 216, all pointing at K=1 grid):
  1. magnitude γ        (architectural; identity 0 → 0.95 → 0)
  2. bimodal squeeze λ  (loss; identity 0 → 1e-2 → 0)
  3. variance penalty λ (loss; identity 0 → 1e-2 → 0)

Body trainable: o_proj + down_proj W_fp + β_g.
Other compensation levers train in phase B.
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
PHASE_A_STEPS = 1000
PHASE_B_STEPS = 1000
EVAL_EVERY = 50
BODY_LR = 2e-5
LEVER_LR = 5e-4
BETA_LR = 5e-4

# Tight PID — "ride the laser"
DRIFT_TARGET = 0.02
DRIFT_HIGH = 0.05
PID_STEP_FRAC = 0.05

RESULTS_PATH = Path("results/stage217_bonsai_beta.json")
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


# ─── AdiabaticQuantizedLinear with Bonsai per-group β_g ───
class AdiabaticQuantizedLinear(nn.Module):
    """W_eff[i, j_in_group_g] = sign(W_fp[i,j])·(γ·α[i,g] + (1−γ)·|W_fp[i,j]|) + γ·β_g[i,g]
    At γ=0: W_eff = W_fp (lossless).
    At γ=1: W_eff = sign(W_fp)·α + β_g (full Bonsai).
    """
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
            # NEW: Bonsai per-group bias β_g
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
            # Add γ·β_g (broadcast across group_size)
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
    rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))

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
    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers,
                   n_subln=n_subln, n_beta_g_total=n_beta_g)


# ─── Param helpers ───
def is_body_master(name):
    return "weight_fp" in name and any(t in name for t in BODY_TRAIN_NAMES)


def is_beta_g(name):
    return "beta_g" in name


def is_other_lever(name):
    """Levers that are neither body nor β_g — the other compensation pathway."""
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


LEVERS = [
    {"name": "magnitude",        "kind": "arch", "identity": 0.0,  "perturb": 0.95,
     "loss_fn": None},
    {"name": "bimodal_squeeze",  "kind": "loss", "identity": 0.0,  "perturb": 1e-2,
     "loss_fn": bimodal_squeeze_loss},
    {"name": "variance_penalty", "kind": "loss", "identity": 0.0,  "perturb": 1e-2,
     "loss_fn": variance_penalty_loss},
]


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def snapshot_state(model, predicate):
    return {n: p.detach().clone() for n, p in model.named_parameters() if predicate(n)}


def displacement_groups(model, snapshot, predicate):
    groups = {"bias": [], "subln_gate": [], "subln_gain": [], "h_scale": [],
              "attn_gain": [], "mlp_gain": [], "attn_offset": [], "mlp_offset": [],
              "logit_tau": [], "beta_g": []}
    for n, p in model.named_parameters():
        if not predicate(n): continue
        if n not in snapshot: continue
        diff = (p.detach() - snapshot[n]).float().norm().item()
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

print("\nBuilding architecture (with Bonsai β_g)...", flush=True)
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
print(f"\nBody master (o/down):     {n_body:,}")
print(f"β_g (Bonsai per-group):   {n_beta:,}")
print(f"Other levers:              {n_other:,}", flush=True)

body_params = [p for n, p in model.named_parameters() if is_body_master(n)]
beta_params = [p for n, p in model.named_parameters() if is_beta_g(n)]
other_params = [p for n, p in model.named_parameters() if is_other_lever(n)]
optimizer = torch.optim.Adam([
    {"params": body_params,  "lr": BODY_LR},
    {"params": beta_params,  "lr": BETA_LR},
    {"params": other_params, "lr": LEVER_LR},
])
rng = np.random.default_rng(42)


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
              train_body, train_beta, train_other, n_steps, t_start, history):
    set_trainable(model, is_body_master, train_body)
    set_trainable(model, is_beta_g, train_beta)
    set_trainable(model, is_other_lever, train_other)
    step_size = abs(lever["perturb"] - lever["identity"]) * PID_STEP_FRAC

    print(f"\n  ── {phase_name} (body={train_body} β_g={train_beta} other={train_other}, "
          f"{lever['name']}: {current_value:.4f} → {target_value:.4f}, "
          f"step={step_size:.4f}, n={n_steps}) ──", flush=True)
    model.train()
    if lever["kind"] == "arch":
        set_gamma(model, current_value)
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
                set_gamma(model, current_value)
                lever_lambda = 0.0
            else:
                lever_lambda = current_value
            elapsed = time.time() - t_start
            print(f"    step {step:>4} {lever['name']}={current_value:.4f} "
                  f"ce={val_ce:.4f} Δ={drift:+.4f} ce_loss={ce_loss:.3f}  {elapsed:.0f}s",
                  flush=True)
            history.append({"phase": phase_name, "step": step,
                            "lever": lever["name"], "value": float(current_value),
                            "ce": float(val_ce), "drift": float(drift),
                            "ce_loss": float(ce_loss)})
            model.train()
    return current_value


t_start = time.time()
history = [{"event": "init", "ce": ce_g0, "drift": drift_g0, "k1_drift": k1_initial}]
diagnostics = []
running_k1 = k1_initial
print(f"\n{'─'*60}")
print(f"Stage 217 — slow PID single-pass per lever")
print(f"  drift band [{DRIFT_TARGET}, {DRIFT_HIGH}]  step={PID_STEP_FRAC}  phases={PHASE_A_STEPS}/{PHASE_B_STEPS}")
print('─'*60, flush=True)

# Snapshot of all "compensation params" (β_g + other levers) for displacement tracking
all_comp_filter = lambda n: is_beta_g(n) or is_other_lever(n)

for lever_idx, lever in enumerate(LEVERS, start=1):
    print(f"\n========== LEVER {lever_idx}/{len(LEVERS)}: {lever['name']} "
          f"(identity={lever['identity']}, perturb={lever['perturb']}) ==========", flush=True)

    comp_before = snapshot_state(model, all_comp_filter)

    # Phase A: body + β_g trainable, other levers frozen, lever PID-driven AWAY
    if lever["kind"] == "arch":
        set_gamma(model, lever["identity"])
    current = run_phase(
        f"L{lever_idx}_phaseA", lever,
        current_value=lever["identity"], target_value=lever["perturb"],
        train_body=True, train_beta=True, train_other=False,
        n_steps=PHASE_A_STEPS, t_start=t_start, history=history)
    max_value = current
    print(f"\n    phase A done. {lever['name']} reached {current:.4f} "
          f"(target was {lever['perturb']})", flush=True)
    comp_after_phaseA = snapshot_state(model, all_comp_filter)

    # Phase B: body + β_g frozen, OTHER levers trainable, lever PID-driven BACK
    current = run_phase(
        f"L{lever_idx}_phaseB", lever,
        current_value=current, target_value=lever["identity"],
        train_body=False, train_beta=False, train_other=True,
        n_steps=PHASE_B_STEPS, t_start=t_start, history=history)

    # Compensation displacement during phase B
    phaseB_disp = displacement_groups(model, comp_after_phaseA, all_comp_filter)
    print(f"\n    phase B compensation displacement (other levers absorbing β_g+body):",
          flush=True)
    for k, v in sorted(phaseB_disp.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"      {k:18s} L2={v:.4f}", flush=True)

    # K=1 diagnostic
    if lever["kind"] == "arch":
        set_gamma(model, lever["identity"])
    new_k1 = k1_drift(model, val_tokens, T0)
    delta = running_k1 - new_k1
    print(f"\n    Lever {lever['name']} END: max reached={max_value:.4f}  "
          f"K=1 drift={new_k1:+.4f}  (Δ vs prev {-delta:+.4f})", flush=True)
    diagnostics.append({
        "lever": lever["name"],
        "max_value_reached": float(max_value),
        "k1_drift": float(new_k1),
        "delta_vs_prev": float(-delta),
        "phaseB_displacement": phaseB_disp,
    })
    running_k1 = new_k1


# Final
final_drift = running_k1
print(f"\n{'─'*60}")
print("STAGE 217 RESULT (slow PID, single pass, β_g):")
print('─'*60)
print(f"  T0:                {T0:.4f}")
print(f"  K=1 initial drift: {k1_initial:+.4f}")
print(f"\n  K=1 drift trajectory:")
running = k1_initial
print(f"    initial:                   K=1 Δ={running:+.4f}")
for d in diagnostics:
    delta = d["k1_drift"] - running
    print(f"    after {d['lever']:18s}: K=1 Δ={d['k1_drift']:+.4f}  "
          f"(this lever: {delta:+.4f}, max {lever['name']}={d['max_value_reached']:.4f})")
    running = d["k1_drift"]
print(f"\n  Total K=1 drift reduction: {k1_initial - final_drift:+.4f} nats "
      f"({100*(1 - final_drift/max(k1_initial, 1e-6)):.1f}% of initial drift)")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "k1_initial_drift": float(k1_initial),
        "k1_final_drift": float(final_drift),
        "total_reduction": float(k1_initial - final_drift),
        "n_body_params": int(n_body),
        "n_beta_g_params": int(n_beta),
        "n_other_levers": int(n_other),
        "phase_a_steps": PHASE_A_STEPS,
        "phase_b_steps": PHASE_B_STEPS,
        "body_lr": BODY_LR,
        "lever_lr": LEVER_LR,
        "beta_lr": BETA_LR,
        "pid_step_frac": PID_STEP_FRAC,
        "drift_target": DRIFT_TARGET,
        "drift_high": DRIFT_HIGH,
        "lever_diagnostics": diagnostics,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
