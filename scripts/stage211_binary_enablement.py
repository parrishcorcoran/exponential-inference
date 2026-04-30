"""Stage 211 — INSTALL binary-enabling structural levers (lossless at init).

User's directive: "We need to add them in now if they are zero loss so that
they are the compensation we need."

This stage doesn't test whether a single lever moves K=1 — Stage 210
showed that's pinned to the Gaussian floor at init regardless. This
stage INSTALLS the binary-enabling architecture as the new permanent
base. Every lever is initialized to identity (lossless at FP, Δ=0).
QAT in subsequent stages will OPEN the gates and adjust the gains to
compensate for binary-weight error.

Five levers, all gated/identity-init:
  1. Gated SubLN at o_proj input         (gate α=0 → identity)
  2. Gated SubLN at down_proj input      (gate α=0 → identity)
  3. Per-head input scale on o_proj      (scale=1 → identity)
  4. Per-output bias on every targeted Linear (init zero → identity)
  5. Per-channel residual stream gain    (gain=1 → identity)

γ for SubLN is pre-calibrated to per-channel input RMS so that when
QAT opens the gate (α > 0), the normalization is sensibly scaled.

Verification: Δ at FP must be ≈ 0 — confirms all levers truly identity.
K=1 projection drift must equal vanilla K=1 drift — confirms levers
contribute nothing at init (correct dormant behavior).

Output: a binary-enabled architecture ready for Stage 212 (QAT loop).
"""
import json
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 32
N_CALIB_TOKENS = 4 * 128
RESULTS_PATH = Path("results/stage211_binary_enablement.json")
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
    return sum(losses) / max(len(losses), 1)


# ─── Lever 1+2: gated SubLN-wrapped Linear ───
class SubLNLinear(nn.Module):
    """Linear with gated SubLN on input + optional per-head scale.

    Gated SubLN: y = (1 - α) * x + α * subln(x), where subln(x) = γ * x / RMS(x).
    At init α=0, output = x → trivially lossless.
    QAT can grow α toward 1 to compensate for binary-weight noise.

    Per-output bias initialized to original (or zero if absent).
    Per-head scale init to 1.0 (lossless).
    """
    def __init__(self, original_linear, num_heads=None, head_dim=None, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        out_features, in_features = self.weight.shape
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(
                out_features, device=self.weight.device, dtype=self.weight.dtype))
        # SubLN: γ init=1 (per-channel), gate α init=0 (scalar)
        self.subln_gain = nn.Parameter(torch.ones(
            in_features, device=self.weight.device, dtype=self.weight.dtype))
        self.subln_gate = nn.Parameter(torch.zeros(
            (), device=self.weight.device, dtype=self.weight.dtype))
        self.eps = eps
        if num_heads is not None and head_dim is not None:
            assert in_features == num_heads * head_dim, (
                f"o_proj in={in_features} but num_heads*head_dim={num_heads*head_dim}")
            self.h_scale = nn.Parameter(torch.ones(
                num_heads, device=self.weight.device, dtype=self.weight.dtype))
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
        # Gated SubLN: lossless at α=0, full normalization at α=1
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        normed = self.subln_gain * x / rms
        x = (1.0 - self.subln_gate) * x + self.subln_gate * normed
        return F.linear(x, self.weight, self.bias.to(x.dtype))


def calibrate_input_rms(model, calib_ids, target_suffixes):
    """Forward pass on calib_ids, recording average RMS at each target Linear's input.
    Returns dict[name → 1D tensor of per-channel RMS].
    """
    rms_sums = {}  # name → tensor
    counts = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp):
            x = inp[0].detach().float()
            # per-channel RMS: mean over batch, seq → per channel
            mean_sq = x.pow(2).mean(dim=tuple(range(x.dim() - 1)))   # [in_features]
            rms = mean_sq.sqrt()
            if name not in rms_sums:
                rms_sums[name] = rms.clone()
                counts[name] = 1
            else:
                rms_sums[name] += rms
                counts[name] += 1
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in target_suffixes):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))

    with torch.no_grad():
        model(calib_ids, use_cache=False)

    for h in hooks: h.remove()

    return {name: (rms_sums[name] / counts[name]).cpu()
            for name in rms_sums}


# ─── Lever 5: per-channel residual gain via decoder-layer monkey-patch ───
def install_residual_gains(model):
    """Add per-channel attn_gain and mlp_gain to each decoder layer.
    Patch forward to apply gain to residual contributions. Init = 1.0 → lossless."""
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


# ─── K=1 projection (replaces body weights with sign × per-group α) ───
def project_body_to_k1(model):
    """Replace body Linear weights with sign(W) * α_g per group of 128.
    Operates on .weight in place. Bias and any wrapping (SubLN, residual gain) preserved."""
    n = 0
    for name, mod in model.named_modules():
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        if not hasattr(mod, "weight"): continue
        W = mod.weight.data
        out, in_ = W.shape
        if in_ % GROUP_SIZE != 0: continue
        Wg = W.float().reshape(out, in_ // GROUP_SIZE, GROUP_SIZE)
        alpha = Wg.abs().mean(dim=-1, keepdim=True)
        Wq = (torch.sign(Wg) * alpha).reshape(out, in_).to(W.dtype)
        mod.weight.data.copy_(Wq)
        n += 1
    return n


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading val tokens + model...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ─── BASELINE: T0 ───
print("\nMeasuring T0 (base FP)...")
model = fresh_model()
T0 = lm_ce(model, val_tokens)
print(f"  T0 = {T0:.4f}")

# Identify head structure for o_proj per-head scale
cfg = model.config
num_heads = cfg.num_attention_heads
head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
print(f"  num_attention_heads={num_heads}, head_dim={head_dim}, "
      f"o_proj in={num_heads * head_dim}")
del model; import gc; gc.collect()


# ─── Vanilla K=1 baseline (no enablement) ───
print(f"\n{'─'*70}")
print("BASELINE: vanilla model + K=1 projection (no enablement)")
print('─'*70)
m_vanilla = fresh_model()
n_proj = project_body_to_k1(m_vanilla)
print(f"  Projected {n_proj} body weights to K=1")
ce_vanilla_k1 = lm_ce(m_vanilla, val_tokens)
drift_vanilla = ce_vanilla_k1 - T0
print(f"  CE: {T0:.4f} → {ce_vanilla_k1:.4f}  (Δ {drift_vanilla:+.4f})")
del m_vanilla; gc.collect()


# ─── Binary-enabled: install all levers, calibrate, verify Δ≈0 ───
print(f"\n{'─'*70}")
print("BINARY-ENABLED: install SubLN + head-scale + bias + residual-gain")
print('─'*70)
m = fresh_model()

# Lever 5: residual gains (init 1.0, trivially lossless)
n_layers = install_residual_gains(m)
print(f"  Installed per-channel residual gain on {n_layers} decoder layers")

# Calibrate input RMS for o_proj and down_proj
print(f"  Calibrating input RMS on {N_CALIB_TOKENS} tokens...")
rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))
print(f"  Calibrated {len(rms_table)} input RMS profiles")

# Wrap o_proj and down_proj with SubLN (lossless via calibrated γ)
parent_lookup = {}
for name, mod in m.named_modules():
    for child_name, child_mod in mod.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (mod, child_name)

n_subln = 0
n_head_scaled = 0
for name, mod in list(m.named_modules()):
    if not isinstance(mod, nn.Linear): continue
    is_o_proj = name.endswith("o_proj")
    is_down_proj = name.endswith("down_proj")
    if not (is_o_proj or is_down_proj): continue
    if name not in rms_table: continue
    gain = rms_table[name].to(device=mod.weight.device, dtype=mod.weight.dtype)

    nh, hd = (num_heads, head_dim) if is_o_proj else (None, None)
    new_layer = SubLNLinear(mod, num_heads=nh, head_dim=hd)
    # Pre-set γ to calibrated per-channel RMS so when QAT opens gate, γ is sensible
    with torch.no_grad():
        new_layer.subln_gain.data.copy_(gain)
    parent, child_attr = parent_lookup[name]
    setattr(parent, child_attr, new_layer)
    n_subln += 1
    if is_o_proj: n_head_scaled += 1
print(f"  Wrapped {n_subln} layers with SubLN ({n_head_scaled} also have per-head scale)")

# Lever 4: zero-init per-output bias on remaining (non-wrapped) targeted Linears
n_bias_added = 0
for name, mod in list(m.named_modules()):
    if not isinstance(mod, nn.Linear): continue
    if not any(name.endswith(s) for s in TARGET_NAMES): continue
    # Already wrapped with SubLNLinear above? skip
    # (SubLNLinear has bias by construction)
    if mod.bias is not None: continue
    new_bias = nn.Parameter(torch.zeros(
        mod.weight.shape[0], device=mod.weight.device, dtype=mod.weight.dtype))
    mod.bias = new_bias
    n_bias_added += 1
print(f"  Added zero-init per-output bias to {n_bias_added} additional linears")

# Verify lossless at FP
ce_enabled_fp = lm_ce(m, val_tokens)
drift_enabled_fp = ce_enabled_fp - T0
print(f"\n  FP check: Δ = {drift_enabled_fp:+.6f}  "
      f"({'≈ LOSSLESS ✓' if abs(drift_enabled_fp) < 0.01 else 'distortion present'})")

# K=1 project body weights inside the binary-enabled architecture
print(f"\n  K=1 projecting body weights inside binary-enabled architecture...")
n_proj2 = project_body_to_k1(m)
print(f"  Projected {n_proj2} body weights")
ce_enabled_k1 = lm_ce(m, val_tokens)
drift_enabled_k1 = ce_enabled_k1 - T0
print(f"  CE: {T0:.4f} → {ce_enabled_k1:.4f}  (Δ {drift_enabled_k1:+.4f})")


# ─── HEADLINE ───
print(f"\n{'─'*70}")
print("HEADLINE — does binary-enablement reduce the K=1 catastrophe?")
print('─'*70)
print(f"  Vanilla   + K=1: Δ = {drift_vanilla:+.4f} nats")
print(f"  Enabled   + K=1: Δ = {drift_enabled_k1:+.4f} nats")
gain = drift_vanilla - drift_enabled_k1
rel_gain = gain / drift_vanilla * 100 if abs(drift_vanilla) > 1e-3 else 0
print(f"  Improvement:     Δ = {gain:+.4f} nats  ({rel_gain:+.1f}% of vanilla drift)")

if abs(gain) < 1e-3:
    verdict = ("DORMANT (expected): all levers gated/identity-init at FP, "
               "so K=1 drift unchanged. Capacity available for QAT to use. "
               "Architecture is now binary-enabled and lossless.")
elif rel_gain > 30:
    verdict = "STRONG: structure absorbs K=1 noise even at init. QAT should converge fast."
elif rel_gain > 10:
    verdict = "MILD: some structural friction reduced even at init. Worth pursuing in QAT."
elif rel_gain > 0:
    verdict = "WEAK: minor effect at init. Structure isn't the dominant constraint."
else:
    verdict = "INVERTED: enabled-architecture made K=1 worse. Recipe pivot."
print(f"\n  Verdict: {verdict}")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0_base_ce": float(T0),
        "ce_vanilla_k1": float(ce_vanilla_k1),
        "drift_vanilla_k1": float(drift_vanilla),
        "ce_enabled_fp": float(ce_enabled_fp),
        "drift_enabled_fp": float(drift_enabled_fp),
        "ce_enabled_k1": float(ce_enabled_k1),
        "drift_enabled_k1": float(drift_enabled_k1),
        "improvement_nats": float(gain),
        "improvement_rel_pct": float(rel_gain),
        "n_subln_wrapped": int(n_subln),
        "n_head_scaled": int(n_head_scaled),
        "n_bias_added": int(n_bias_added),
        "n_residual_gain_layers": int(n_layers),
        "lossless_at_fp": bool(abs(drift_enabled_fp) < 0.01),
        "verdict": verdict,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
