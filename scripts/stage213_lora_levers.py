"""Stage 213-A — LoRA adapters + more zero-loss levers + PID anneal.

Stage 212 saturated at γ=0.35 because the lever capacity (545K params,
mostly per-output bias) ran out — bias absorbs DC offset but not
structural K=1 error. This stage adds capacity:

NEW levers vs Stage 212:
  6. LoRA rank-32 on every body Linear  (B init zero → lossless, ~17M params)
  7. Logit temperature τ                (init 1.0 → lossless, 1 param)
  8. Per-channel attn residual offset   (init zero → lossless, 28×1024 params)
  9. Per-channel mlp residual offset    (init zero → lossless, 28×1024 params)

Combined with Stage 212's 5 levers (SubLN×2, h_scale, per-output bias,
per-channel residual gain), total trainable ≈ 18M params (~3% of model).

Same PID adiabatic γ-anneal: γ advances when drift < 0.05, holds in
[0.05, 0.20], backs off above 0.20.

Hypothesis test:
  - If γ pushes past 0.35 → capacity was the constraint. Adding more
    parameters helps. We may reach full K=1 with enough capacity.
  - If γ still saturates near 0.35 → capacity isn't the issue. Body
    weights are structurally fighting binary; need STE.

Compute on Mac MPS fp32: ~70 min for 5000 steps (slower than Stage 212
due to LoRA forward overhead).
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
N_TRAIN_STEPS = 5000
EVAL_EVERY = 50
LR = 5e-4
LORA_RANK = 32

DRIFT_TARGET = 0.05
DRIFT_HIGH = 0.20
GAMMA_STEP_UP = 0.05
GAMMA_STEP_DOWN = 0.10
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0

RESULTS_PATH = Path("results/stage213_lora_levers.json")
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


# ─── Adiabatic K=1 with built-in LoRA ───
class AdiabaticQuantizedLinearLoRA(nn.Module):
    """Adiabatic K=1 Linear + rank-r LoRA adapter.

    y = F.linear(x, W_eff, bias) + (x @ A.T) @ B.T
    W_eff = sign(W_fp) · (γ · α_g + (1 − γ) · |W_fp|)
    A: [rank, in_features], normal init
    B: [out_features, rank], zero init  → lossless at init.
    """
    def __init__(self, original_linear, group_size=GROUP_SIZE, rank=LORA_RANK):
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
        self.register_buffer("gamma", torch.tensor(0.0, dtype=W_fp.dtype))
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(
                out, device=W_fp.device, dtype=W_fp.dtype))
        # LoRA: A normal init, B zero init (→ lossless)
        self.lora_A = nn.Parameter(torch.empty(
            rank, in_, device=W_fp.device, dtype=W_fp.dtype))
        nn.init.normal_(self.lora_A, std=1.0 / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(
            out, rank, device=W_fp.device, dtype=W_fp.dtype))

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
        out = F.linear(x, W_eff, self.bias.to(x.dtype))
        # LoRA addition: x @ A.T → [..., rank], then @ B.T → [..., out]
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + lora_out


# ─── Stage 211/212 SubLN wrap ───
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


# ─── Logit temperature wrap ───
class TemperedLMHead(nn.Module):
    """Wraps lm_head: y = wrapped(x) / τ. Init τ=1 → lossless."""
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
    """Per-channel attn_gain, mlp_gain (×, init 1.0) AND attn_offset, mlp_offset (+, init 0)."""
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

    # Step 1: residual gains + offsets
    n_layers = install_residual_gains_and_offsets(m)

    # Step 2: calibrate (still nn.Linear at this point)
    rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    # Step 3: replace target Linears with AdiabaticQuantizedLinearLoRA
    n_quantized = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinearLoRA(mod, rank=LORA_RANK)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n_quantized += 1

    # Step 4: SubLN wrap on o_proj and down_proj
    parent_lookup2 = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup2[full] = (mod, child_name)

    n_subln = 0; n_head_scaled = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, AdiabaticQuantizedLinearLoRA): continue
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

    # Step 5: wrap lm_head with TemperedLMHead
    m.lm_head = TemperedLMHead(m.lm_head)

    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers,
                   n_subln=n_subln, n_head_scaled=n_head_scaled,
                   lora_rank=LORA_RANK)


def is_lever_param(name):
    if any(t in name for t in (
        "subln_gate", "subln_gain", "h_scale", "attn_gain", "mlp_gain",
        "attn_offset", "mlp_offset", "lora_A", "lora_B", "logit_tau"
    )):
        return True
    if "bias" in name and "norm" not in name:
        return True
    return False


def freeze_body_train_levers(model):
    train_p, train_count, frozen_count = [], 0, 0
    breakdown = {"bias": 0, "subln": 0, "h_scale": 0, "residual_gain": 0,
                 "residual_offset": 0, "lora": 0, "logit_tau": 0}
    for name, p in model.named_parameters():
        if is_lever_param(name):
            p.requires_grad_(True)
            train_p.append(p); train_count += p.numel()
            if "lora" in name: breakdown["lora"] += p.numel()
            elif "subln" in name: breakdown["subln"] += p.numel()
            elif "h_scale" in name: breakdown["h_scale"] += p.numel()
            elif "_gain" in name: breakdown["residual_gain"] += p.numel()
            elif "_offset" in name: breakdown["residual_offset"] += p.numel()
            elif "logit_tau" in name: breakdown["logit_tau"] += p.numel()
            elif "bias" in name: breakdown["bias"] += p.numel()
        else:
            p.requires_grad_(False)
            frozen_count += p.numel()
    return train_p, train_count, frozen_count, breakdown


def set_gamma(model, gamma_value):
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinearLoRA):
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
    out = {}
    gates = [p.detach().float().item() for n, p in model.named_parameters() if "subln_gate" in n]
    out["gate_mean"] = float(np.mean(gates)) if gates else 0.0
    bias_norms = [p.detach().float().norm().item()
                  for n, p in model.named_parameters() if "bias" in n and "norm" not in n]
    out["bias_norm_mean"] = float(np.mean(bias_norms)) if bias_norms else 0.0
    a_g = [(p.detach().float() - 1.0).abs().mean().item()
           for n, p in model.named_parameters() if "attn_gain" in n]
    out["attn_gain_dev"] = float(np.mean(a_g)) if a_g else 0.0
    a_o = [p.detach().float().norm().item()
           for n, p in model.named_parameters() if "attn_offset" in n]
    out["attn_offset_norm"] = float(np.mean(a_o)) if a_o else 0.0
    m_o = [p.detach().float().norm().item()
           for n, p in model.named_parameters() if "mlp_offset" in n]
    out["mlp_offset_norm"] = float(np.mean(m_o)) if m_o else 0.0
    lora_b_norm = [p.detach().float().norm().item()
                   for n, p in model.named_parameters() if "lora_B" in n]
    out["lora_B_norm_mean"] = float(np.mean(lora_b_norm)) if lora_b_norm else 0.0
    tau = [p.detach().float().item() for n, p in model.named_parameters() if "logit_tau" in n]
    out["logit_tau"] = float(tau[0]) if tau else 1.0
    return out


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 32].tolist()
train_tokens = corpus[SEQ_LEN * 32:SEQ_LEN * 32 + 1_000_000].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)
print(f"  val={len(val_tokens)}  train={len(train_tokens)}", flush=True)

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

print("\nBuilding enabled architecture (LoRA + extra levers)...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)

ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}", flush=True)

set_gamma(model, 1.0)
ce_g1 = lm_ce(model, val_tokens)
drift_g1 = ce_g1 - T0
print(f"  γ=1 verify: ce={ce_g1:.4f} Δ={drift_g1:+.4f}", flush=True)
set_gamma(model, 0.0)

train_p, n_train, n_frozen, breakdown = freeze_body_train_levers(model)
print(f"\nTrainable lever params: {n_train:,}  ({100*n_train/(n_train+n_frozen):.2f}% of model)")
print(f"  Breakdown:")
for k, v in breakdown.items():
    print(f"    {k:18s} {v:>12,}")
print(f"Frozen body: {n_frozen:,}", flush=True)

optimizer = torch.optim.Adam(train_p, lr=LR)
rng = np.random.default_rng(42)

print(f"\n{'─'*60}")
print(f"PID anneal: {N_TRAIN_STEPS} steps batch={BATCH_SIZE} seq={SEQ_LEN}")
print('─'*60, flush=True)

current_gamma = 0.0
history = [{"step": 0, "gamma": 0.0, "ce": ce_g0, "drift": drift_g0, "loss": None}]
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
        val_ce = lm_ce(model, val_tokens)
        drift = val_ce - T0
        probe = probe_levers(model)

        if drift < DRIFT_TARGET:
            old_g = current_gamma
            current_gamma = min(current_gamma + GAMMA_STEP_UP, GAMMA_MAX)
            action = f"γ {old_g:.2f}→{current_gamma:.2f} +"
        elif drift > DRIFT_HIGH:
            old_g = current_gamma
            current_gamma = max(current_gamma - GAMMA_STEP_DOWN, GAMMA_MIN)
            action = f"γ {old_g:.2f}→{current_gamma:.2f} −"
        else:
            action = f"γ {current_gamma:.2f} hold"
        set_gamma(model, current_gamma)

        elapsed = time.time() - t_start
        print(f"  step {step:>4} γ={current_gamma:.2f}  "
              f"ce={val_ce:.4f} Δ={drift:+.4f}  loss={loss.item():.3f}  "
              f"bias={probe['bias_norm_mean']:.2f} loraB={probe['lora_B_norm_mean']:.2f} "
              f"attnO={probe['attn_offset_norm']:.2f} mlpO={probe['mlp_offset_norm']:.2f} "
              f"τ={probe['logit_tau']:.3f} gate={probe['gate_mean']:.3f}  "
              f"[{action}]  {elapsed:.0f}s", flush=True)
        history.append({"step": step, "gamma": float(current_gamma),
                        "ce": float(val_ce), "drift": float(drift),
                        "loss": float(loss.item()), "action": action, **probe})
        model.train()

print(f"\nFinal γ = {current_gamma:.2f}")
final_ce = lm_ce(model, val_tokens)
final_drift = final_ce - T0

print(f"\n{'─'*60}")
print("STAGE 213-A RESULT (LoRA + more levers):")
print('─'*60)
print(f"  T0             : {T0:.4f}")
print(f"  γ=0 (lossless) : {ce_g0:.4f}  Δ={drift_g0:+.6f}")
print(f"  γ=1 raw K=1    : {ce_g1:.4f}  Δ={drift_g1:+.4f}")
print(f"  Final γ={current_gamma:.2f}  : {final_ce:.4f}  Δ={final_drift:+.4f}")
print(f"  Stage 212 ref  : γ=0.35, Δ=+0.161  (capacity 545K)")
print(f"  Stage 213-A    : γ={current_gamma:.2f}, Δ={final_drift:+.4f}  (capacity {n_train:,})")
print(f"  Steps = {N_TRAIN_STEPS}, train tokens = {N_TRAIN_STEPS*BATCH_SIZE*SEQ_LEN}")

if current_gamma > 0.40:
    verdict = (f"BREAKTHROUGH: γ pushed past Stage 212's 0.35 ceiling. "
               f"Capacity was the constraint — adding LoRA + more levers helps.")
elif current_gamma >= 0.35:
    verdict = (f"MATCHED: γ reached same ceiling as Stage 212 despite {n_train//545_272}× more capacity. "
               f"Body weights are the constraint — STE needed.")
else:
    verdict = (f"REGRESSED: γ saturated below Stage 212's 0.35. Optimization issue — "
               f"perhaps LR too high for new lever group.")
print(f"\n  Verdict: {verdict}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "ce_gamma_0": float(ce_g0), "drift_gamma_0": float(drift_g0),
        "ce_gamma_1_no_train": float(ce_g1), "drift_gamma_1_no_train": float(drift_g1),
        "final_gamma": float(current_gamma),
        "final_ce": float(final_ce), "final_drift": float(final_drift),
        "n_trainable_lever_params": int(n_train),
        "lever_breakdown": breakdown,
        "n_train_steps": N_TRAIN_STEPS,
        "lr": LR, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
        "lora_rank": LORA_RANK,
        "stage212_reference_final_gamma": 0.35,
        "stage212_reference_final_drift": 0.161,
        "verdict": verdict,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
