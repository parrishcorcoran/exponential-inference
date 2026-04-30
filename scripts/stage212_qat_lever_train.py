"""Stage 212 — PID adiabatic K=1 anneal with binary-enabled levers.

Per user directives:
  - ALWAYS PID (adiabatic descent — γ grows only when CE drift stays in band)
  - Always aware-trained (CE loss on OWT tokens; FP "memory" preserved
    via the γ-blend itself: student = γ·K1 + (1−γ)·FP smoothly)
  - Crude is fine (single-gate P controller, not full PID)
  - Use OWT corpus

Architecture (Stage 211 levers, all dormant at init):
  - Gated SubLN at o_proj/down_proj input  (gate α=0)
  - Per-head input scale on o_proj           (scale=1)
  - Per-output bias on every targeted Linear (init zero)
  - Per-channel residual stream gain         (gain=1)

Body weights wrapped with AdiabaticQuantizedLinear:
  W_eff = sign(W_fp) · (γ · α_g + (1 − γ) · |W_fp|)
  γ=0  → W_eff = W_fp  (lossless)
  γ=1  → W_eff = sign(W_fp) · α_g  (full K=1, per-group scale)

Continuous interpolation between FP and K=1, controlled by a single γ.

PID (crude, single gate):
  Every EVAL_EVERY steps: measure val CE.
  If drift_band[0] < (val_ce − T0) < drift_band[1]: γ += STEP_UP
  Else if drift > drift_band[1]:                    γ -= STEP_DOWN  (back off)
  Else (drift below band, recovery happening):      hold γ

Adam trains all levers continuously on OWT next-token CE.

Stop condition: γ ≥ 1 and drift stable, OR step budget exhausted.
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

# Line-buffered stdout so progress lands in logs immediately when piped through tee.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 16
N_CALIB_TOKENS = 4 * 128
BATCH_SIZE = 2
N_TRAIN_STEPS = 1200
EVAL_EVERY = 40
LR = 5e-4

# PID (crude P controller) parameters
DRIFT_TARGET = 0.05         # nats tolerance band upper edge
DRIFT_HIGH = 0.20           # if drift exceeds this, back off γ
GAMMA_STEP_UP = 0.04        # γ increment when drift in band
GAMMA_STEP_DOWN = 0.10      # γ decrement when drift too high
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0

RESULTS_PATH = Path("results/stage212_qat_lever_train.json")
LOG_PATH = Path("logs/stage212_train.log")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
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


# ─── Adiabatic K=1 wrapper ───
class AdiabaticQuantizedLinear(nn.Module):
    """Linear with γ-controlled adiabatic interpolation FP ↔ K=1.

    W_eff = sign(W_fp) · (γ · α_g + (1 − γ) · |W_fp|)

    γ=0: W_eff = W_fp (lossless).
    γ=1: W_eff = sign(W_fp) · α_g (per-group K=1).

    γ is a buffer (not parameter) — controlled externally by PID.
    α_g pre-computed and frozen (Bonsai-style per-128-group magnitude).
    Bias preserved (or zero-init if absent).
    """
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
        else:
            self.register_buffer("alpha",
                                 W_fp.abs().mean(dim=-1, keepdim=True).to(W_fp.dtype))
        self.group_size = group_size
        self.out_features, self.in_features = out, in_
        # γ as buffer (not trained; PID-controlled)
        self.register_buffer("gamma",
                             torch.tensor(0.0, dtype=W_fp.dtype))
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


# ─── SubLN (Stage 211 levers) ───
class SubLNLinear(nn.Module):
    """Gated SubLN wrapper for o_proj/down_proj inputs (or any Linear).

    NOTE: This wraps a downstream Linear — the wrapped Linear is
    AdiabaticQuantizedLinear (so SubLN feeds into γ-controlled body).
    """
    def __init__(self, wrapped_linear, num_heads=None, head_dim=None, eps=1e-6):
        super().__init__()
        self.wrapped = wrapped_linear   # AdiabaticQuantizedLinear or nn.Linear
        in_features = wrapped_linear.weight_fp.shape[1] if hasattr(wrapped_linear, "weight_fp") \
                       else wrapped_linear.weight.shape[1]
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


def install_residual_gains(model):
    n_layers = 0
    for layer in model.model.layers:
        hidden_size = layer.input_layernorm.weight.shape[0]
        layer.attn_gain = nn.Parameter(torch.ones(
            hidden_size, device=layer.input_layernorm.weight.device,
            dtype=layer.input_layernorm.weight.dtype))
        layer.mlp_gain = nn.Parameter(torch.ones(
            hidden_size, device=layer.input_layernorm.weight.device,
            dtype=layer.input_layernorm.weight.dtype))

        def new_forward(self, hidden_states, **kwargs):
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(hidden_states=x, **kwargs)
            x = residual + self.attn_gain * attn_out
            residual = x
            x = self.post_attention_layernorm(x)
            mlp_out = self.mlp(x)
            x = residual + self.mlp_gain * mlp_out
            return x

        layer.forward = types.MethodType(new_forward, layer)
        n_layers += 1
    return n_layers


def build_full_architecture(num_heads, head_dim, calib_ids):
    """Build model with Stage 211 levers + adiabatic-quantized body.

    Order of installation matters:
      1. Replace target Linears with AdiabaticQuantizedLinear (γ-controlled FP↔K1)
      2. Install residual gains (decoder-layer monkey-patch)
      3. Calibrate input RMS for o_proj/down_proj
      4. Wrap o_proj/down_proj with SubLN (which now wraps the AdiabaticQuantized)
    """
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    # Step 1: replace targeted Linears with AdiabaticQuantizedLinear (γ=0 init)
    n_quantized = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinear(mod)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n_quantized += 1

    # Step 2: residual gains
    n_layers = install_residual_gains(m)

    # Step 3: calibrate (γ=0 so this gives true FP RMS)
    rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))

    # Step 4: wrap o_proj and down_proj with SubLN
    # Need fresh parent_lookup since modules changed
    parent_lookup2 = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup2[full] = (mod, child_name)

    n_subln = 0; n_head_scaled = 0
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
        if is_o: n_head_scaled += 1

    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers,
                   n_subln=n_subln, n_head_scaled=n_head_scaled)


def is_lever_param(name):
    if any(t in name for t in ("subln_gate", "subln_gain", "h_scale",
                                "attn_gain", "mlp_gain")):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


def freeze_body_train_levers(model):
    train_p, train_count, frozen_count = [], 0, 0
    for name, p in model.named_parameters():
        if is_lever_param(name):
            p.requires_grad_(True)
            train_p.append(p); train_count += p.numel()
        else:
            p.requires_grad_(False)
            frozen_count += p.numel()
    return train_p, train_count, frozen_count


def set_gamma(model, gamma_value):
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinear):
            mod.gamma.fill_(gamma_value)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    batch = torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)
    return batch


def probe_levers(model):
    gates = [p.detach().float().item() for n, p in model.named_parameters() if "subln_gate" in n]
    bias_norms = [p.detach().float().norm().item()
                  for n, p in model.named_parameters() if "bias" in n and "norm" not in n]
    a_gains = [(p.detach().float() - 1.0).abs().mean().item()
               for n, p in model.named_parameters() if "attn_gain" in n]
    m_gains = [(p.detach().float() - 1.0).abs().mean().item()
               for n, p in model.named_parameters() if "mlp_gain" in n]
    return {
        "gate_mean": float(np.mean(gates)) if gates else 0.0,
        "gate_max":  float(np.max(np.abs(gates))) if gates else 0.0,
        "bias_norm_mean": float(np.mean(bias_norms)) if bias_norms else 0.0,
        "attn_gain_dev_mean": float(np.mean(a_gains)) if a_gains else 0.0,
        "mlp_gain_dev_mean": float(np.mean(m_gains)) if m_gains else 0.0,
    }


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus + model config...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 64:SEQ_LEN * 64 + 1_000_000].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)
print(f"  val tokens: {len(val_tokens)},  train tokens: {len(train_tokens)}")

# T0 baseline (vanilla FP)
print("\nMeasuring T0 (base FP)...")
m0 = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(m0, val_tokens)
cfg = m0.config
num_heads = cfg.num_attention_heads
head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
print(f"  T0 = {T0:.4f}")
del m0
import gc; gc.collect()


# Build full architecture
print("\nBuilding binary-enabled architecture with adiabatic body...")
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  Adiabatic body Linears: {install_stats['n_quantized']}")
print(f"  Residual gain layers:   {install_stats['n_residual_gain_layers']}")
print(f"  SubLN-wrapped Linears:  {install_stats['n_subln']} ({install_stats['n_head_scaled']} also head-scaled)")

# Verify γ=0 → lossless
ce_g0 = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
drift_g0 = ce_g0 - T0
print(f"\n  Lossless verify (γ=0): CE = {ce_g0:.4f}  Δ = {drift_g0:+.6f}  "
      f"({'✓' if abs(drift_g0) < 0.01 else 'distortion'})")

# Verify γ=1 → full K=1
set_gamma(model, 1.0)
ce_g1 = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
drift_g1 = ce_g1 - T0
print(f"  K=1 verify    (γ=1): CE = {ce_g1:.4f}  Δ = {drift_g1:+.4f}")
set_gamma(model, 0.0)

# Set up training
train_p, n_train, n_frozen = freeze_body_train_levers(model)
print(f"\nTrainable lever params: {n_train:,}  ({100*n_train/(n_train+n_frozen):.3f}%)")
print(f"Frozen body params:     {n_frozen:,}")
optimizer = torch.optim.Adam(train_p, lr=LR)
rng = np.random.default_rng(42)


# ─── PID adiabatic loop ───
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"\n{'─'*70}")
print(f"PID adiabatic anneal: γ band [drift<{DRIFT_TARGET:.2f} → +{GAMMA_STEP_UP}, "
      f"drift>{DRIFT_HIGH:.2f} → −{GAMMA_STEP_DOWN}]")
print(f"Train: {N_TRAIN_STEPS} steps, batch={BATCH_SIZE}, seq={SEQ_LEN}, lr={LR}")
print('─'*70)

current_gamma = 0.0
best_ce_at_gamma = {0.0: ce_g0}
history = [{"step": 0, "gamma": 0.0, "ce": ce_g0, "drift": drift_g0,
            "loss": None, "action": "init"}]
t_start = time.time()

set_gamma(model, current_gamma)
model.train()

for step in range(1, N_TRAIN_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    out = model(batch[:, :-1], use_cache=False)
    loss = F.cross_entropy(
        out.logits.float().reshape(-1, out.logits.size(-1)),
        batch[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS:
        val_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        probe = probe_levers(model)
        elapsed = time.time() - t_start

        # PID action
        if drift < DRIFT_TARGET:
            old_gamma = current_gamma
            current_gamma = min(current_gamma + GAMMA_STEP_UP, GAMMA_MAX)
            action = f"γ {old_gamma:.3f}→{current_gamma:.3f} (advance)"
        elif drift > DRIFT_HIGH:
            old_gamma = current_gamma
            current_gamma = max(current_gamma - GAMMA_STEP_DOWN, GAMMA_MIN)
            action = f"γ {old_gamma:.3f}→{current_gamma:.3f} (back off)"
        else:
            action = f"γ {current_gamma:.3f} (hold, in band)"
        set_gamma(model, current_gamma)

        print(f"  step {step:>4} γ={current_gamma:.3f}  "
              f"ce={val_ce:.4f} (Δ{drift:+.4f})  loss={loss.item():.3f}  "
              f"gate={probe['gate_mean']:.3f}  bias={probe['bias_norm_mean']:.3f}  "
              f"a_g={probe['attn_gain_dev_mean']:.3f}  m_g={probe['mlp_gain_dev_mean']:.3f}  "
              f"[{action}]  {elapsed:.0f}s")
        history.append({
            "step": step, "gamma": float(current_gamma),
            "ce": float(val_ce), "drift": float(drift),
            "loss": float(loss.item()), "action": action, **probe,
        })
        # Track best CE seen at each γ value (rough)
        γ_key = round(current_gamma, 2)
        if val_ce < best_ce_at_gamma.get(γ_key, float("inf")):
            best_ce_at_gamma[γ_key] = float(val_ce)

        model.train()

# Final state
print(f"\nFinal: γ = {current_gamma:.3f}")
final_ce = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
final_drift = final_ce - T0


# ─── Headline ───
print(f"\n{'─'*70}")
print("HEADLINE — PID adiabatic K=1 anneal with binary-enabled levers:")
print('─'*70)
print(f"  T0 (base FP):                {T0:.4f}")
print(f"  γ=0  (lossless start):       {ce_g0:.4f}  (Δ {drift_g0:+.6f})")
print(f"  γ=1  (raw K=1, no training): {ce_g1:.4f}  (Δ {drift_g1:+.4f})")
print(f"  Final γ={current_gamma:.3f}:   {final_ce:.4f}  (Δ {final_drift:+.4f})")
print()
if current_gamma >= 0.99:
    rel = (drift_g1 - final_drift) / drift_g1 * 100
    print(f"  Reached γ=1.0. K=1 drift recovered: {rel:.1f}% (from {drift_g1:+.3f} → {final_drift:+.3f})")
    if final_drift < 0.5:
        verdict = "STRONG: lever-only training crosses K=1 lossless. Body retraining may not be needed."
    elif final_drift < 2.0:
        verdict = "MODERATE: levers close most of K=1 gap. Full QAT (body via STE) likely closes rest."
    else:
        verdict = "PARTIAL: levers help but body retraining needed."
else:
    print(f"  Did NOT reach γ=1.0. Stuck at γ={current_gamma:.3f}.")
    if current_gamma > 0.5:
        verdict = "STALLED MID-ANNEAL: levers absorbing partial K=1 but can't reach full. Need more capacity / body STE."
    else:
        verdict = "STALLED EARLY: PID couldn't advance γ. Either drift band too tight or capacity insufficient."
print(f"\n  Verdict: {verdict}")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0_base_ce": float(T0),
        "ce_gamma_0": float(ce_g0),
        "drift_gamma_0": float(drift_g0),
        "ce_gamma_1_no_train": float(ce_g1),
        "drift_gamma_1_no_train": float(drift_g1),
        "final_gamma": float(current_gamma),
        "final_ce": float(final_ce),
        "final_drift": float(final_drift),
        "n_trainable_lever_params": int(n_train),
        "n_train_steps": N_TRAIN_STEPS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "drift_target_band": [0, DRIFT_TARGET],
        "drift_high_threshold": DRIFT_HIGH,
        "gamma_step_up": GAMMA_STEP_UP,
        "gamma_step_down": GAMMA_STEP_DOWN,
        "verdict": verdict,
        "install_stats": install_stats,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
