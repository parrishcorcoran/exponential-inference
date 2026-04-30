"""Stage 209: per-group Hadamard rotation — directly attack binary friendliness.

K=1 binary quantization operates per-128-group: each group's 128 weights
become 128 sign bits + 1 scalar. The intra-row CV that determines K=1
quality is COMPUTED PER GROUP. So rotating WITHIN each group (smoothing
the 128 weights toward the group mean) directly attacks the K=1
quality metric.

Mechanism:
  - For Linear with in_dim divisible by 128:
    - Reshape weight to [out, n_groups, 128]
    - For each group, multiply by 128x128 Hadamard: W'[i, g] = W[i, g] @ H_128
    - Reshape back to [out, in]
  - Input compensation: reshape x to [..., n_groups, 128], apply Hadamard
    per group, reshape back. Lossless because H @ H^T = I.

This is more granular than Stage 208's per-Linear rotation because the
mixing happens within each group of 128 (= the K=1 grouping). Each
output element is now a Hadamard-mixed sum of the 128 inputs in its group.

Hadamard mixing should reduce within-group magnitude variance because
each output position becomes an average-like combination of all 128
inputs, smoothed regardless of original outliers.

Predicted: per-group rotation reduces intra-row CV more than per-Linear
rotation did (Stage 208's −0.022). Maybe −0.10 to −0.20.

Verify Δ=0 + measure intra-row CV / K=1 residual reduction.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage209_per_group_rotation.json")
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


def hadamard_matrix(n, device, dtype=torch.float32):
    """Sylvester-Hadamard, normalized so H @ H^T = I."""
    if n == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    H_half = hadamard_matrix(n // 2, device, dtype)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1),
    ], dim=0)
    return H


class PerGroupRotatedLinear(nn.Module):
    """Linear with per-group Hadamard rotation (group_size=128).

    Each group of 128 input dims gets the SAME 128x128 Hadamard rotation.
    Weights rotated correspondingly: W_new[i, g, :] = W[i, g, :] @ H_128.
    Input compensation: x_new[g, :] = x[g, :] @ H_128. Net: F.linear(x_new, W_new) == F.linear(x, W).
    """
    def __init__(self, original_linear, group_size=128):
        super().__init__()
        W = original_linear.weight.data.float()
        out_features, in_features = W.shape
        if in_features % group_size != 0:
            # Fallback: use original weight, no rotation (will be skipped)
            self.weight = nn.Parameter(W.to(original_linear.weight.dtype))
            self.bias = original_linear.bias
            self.group_size = group_size
            self.n_groups = 0  # signals no rotation
            self.register_buffer("H", torch.eye(1, device=W.device, dtype=W.dtype))
            return
        n_groups = in_features // group_size

        # Build normalized Hadamard
        H_normalized = hadamard_matrix(group_size, W.device, torch.float32) / np.sqrt(group_size)
        # Apply Hadamard to each group of 128 cols
        W_grouped = W.reshape(out_features, n_groups, group_size)
        W_rotated_grouped = W_grouped @ H_normalized   # broadcast: H acts on last dim
        W_rotated = W_rotated_grouped.reshape(out_features, in_features)

        self.weight = nn.Parameter(W_rotated.to(original_linear.weight.dtype))
        self.bias = original_linear.bias
        self.group_size = group_size
        self.n_groups = n_groups
        self.in_features = in_features
        self.register_buffer("H", H_normalized.to(original_linear.weight.dtype))

    def forward(self, x):
        if self.n_groups == 0:
            return F.linear(x, self.weight,
                            self.bias.to(x.dtype) if self.bias is not None else None)
        batch_shape = x.shape[:-1]
        x_grouped = x.reshape(*batch_shape, self.n_groups, self.group_size)
        x_rotated = x_grouped @ self.H   # per-group Hadamard on input
        x_rotated_flat = x_rotated.reshape(*batch_shape, self.in_features)
        return F.linear(x_rotated_flat, self.weight,
                        self.bias.to(x.dtype) if self.bias is not None else None)


def get_effective_weight(mod):
    """Return effective body weight matrix for measurement."""
    if isinstance(mod, nn.Linear):
        return mod.weight.detach().float()
    if hasattr(mod, "weight") and hasattr(mod, "H"):
        # PerGroupRotatedLinear: weight is W with Hadamard applied
        return mod.weight.detach().float()
    if hasattr(mod, "W_unit"):
        return mod.W_unit.detach().float()
    return None


def measure_body_intra_row_cv(model):
    cv_all = []
    k1_errs = []
    n_measured = 0
    for name, mod in model.named_modules():
        if not any(t in name for t in TARGET_NAMES): continue
        W = get_effective_weight(mod)
        if W is None or W.dim() != 2: continue
        out_features, in_features = W.shape
        if in_features % GROUP_SIZE != 0: continue
        n_groups = in_features // GROUP_SIZE
        grouped = W.reshape(out_features, n_groups, GROUP_SIZE)
        abs_w = grouped.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        cv = (abs_w.std(dim=-1) / mean_abs.squeeze(-1)).cpu().numpy().flatten()
        cv_all.extend(cv.tolist())
        scales = mean_abs
        W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
        rel_err = ((W - W_q).norm() / W.norm().clamp(min=1e-8)).item()
        k1_errs.append(rel_err)
        n_measured += 1
    return {
        "intra_row_cv_mean": float(np.mean(cv_all)) if cv_all else 0.0,
        "k1_residual_mean": float(np.mean(k1_errs)) if k1_errs else 0.0,
        "n_measured": n_measured,
    }


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading val tokens + model...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# Baseline
print("\nMeasuring T0 (base FP)...")
model = fresh_model()
T0 = lm_ce(model, val_tokens)
init = measure_body_intra_row_cv(model)
print(f"  T0 = {T0:.4f}")
print(f"  intra-row CV: {init['intra_row_cv_mean']:.4f}")
print(f"  K=1 residual: {init['k1_residual_mean']:.4f}")
print(f"  modules measured: {init['n_measured']}")
del model
import gc; gc.collect()


# ─── OP: Per-group Hadamard rotation ───
print(f"\n{'─'*70}")
print("OP: Per-group Hadamard rotation (128x128 within each group)")
print('─'*70)
m = fresh_model()
target_mods = [(n, mod) for n, mod in m.named_modules()
               if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
parent_lookup = {}
for name, mod in m.named_modules():
    for child_name, child_mod in mod.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (mod, child_name)

rotated_count = 0
for name, mod in target_mods:
    if mod.weight.shape[1] % GROUP_SIZE != 0: continue
    new_layer = PerGroupRotatedLinear(mod, GROUP_SIZE)
    parent, child_attr = parent_lookup[name]
    setattr(parent, child_attr, new_layer)
    rotated_count += 1
print(f"  Rotated {rotated_count}/{len(target_mods)} linears")

ce = lm_ce(m, val_tokens)
drift = ce - T0
metrics = measure_body_intra_row_cv(m)
print(f"  Δ: {drift:+.6f}  ({'LOSSLESS ✓' if abs(drift) < 1e-3 else 'lossy'})")
print(f"  intra-row CV: {init['intra_row_cv_mean']:.4f} → {metrics['intra_row_cv_mean']:.4f}  "
      f"(Δ {metrics['intra_row_cv_mean'] - init['intra_row_cv_mean']:+.4f})")
print(f"  K=1 residual: {init['k1_residual_mean']:.4f} → {metrics['k1_residual_mean']:.4f}  "
      f"(Δ {metrics['k1_residual_mean'] - init['k1_residual_mean']:+.4f})")
print(f"  modules measured: {metrics['n_measured']}")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0_base_ce": float(T0),
        "init": init,
        "post_per_group_rotation": metrics,
        "drift": float(drift),
        "is_lossless": bool(abs(drift) < 1e-3),
        "n_rotated": rotated_count,
        "n_total_linears": len(target_mods),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
