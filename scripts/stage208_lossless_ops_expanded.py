"""Stage 208: expanded lossless ops library — focus on ops that move
binary-friendliness, not just losslessness.

Stage 207 confirmed 3 ops are lossless but only SmoothQuant moved any
metric (norm_max). None moved intra-row CV — the metric that predicts
K=1 quality.

Stage 208 adds:
  4. Random orthogonal rotation per Linear (with input compensation)
  5. Hadamard rotation (special structure: power-of-2 dim, ±1 entries)
  6. Per-group magnitude factoring (Bonsai-style: α per 128-group)

The rotation ops should reduce intra-row CV — they mix outlier values
across positions in each row, smoothing the distribution.

Per-group magnitude factoring puts each group's mean into a stored
scale, leaving body with uniform-mean groups. Equivalent to Bonsai's
per-group structure but as lossless reparameterization.
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
RESULTS_PATH = Path("results/stage208_lossless_ops_expanded.json")
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


def measure_body_intra_row_cv(model):
    """Body intra-row CV — the binary-friendliness metric."""
    cv_all = []
    k1_errs = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        if not any(t in name for t in TARGET_NAMES): continue
        W = mod.weight.detach().float()
        out_features, in_features = W.shape
        if in_features % GROUP_SIZE != 0: continue
        n_groups = in_features // GROUP_SIZE
        grouped = W.reshape(out_features, n_groups, GROUP_SIZE)
        abs_w = grouped.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        cv = (abs_w.std(dim=-1) / mean_abs.squeeze(-1)).cpu().numpy().flatten()
        cv_all.extend(cv.tolist())
        # K=1 residual
        scales = mean_abs
        W_q = (torch.sign(grouped) * scales).reshape(out_features, in_features)
        rel_err = ((W - W_q).norm() / W.norm().clamp(min=1e-8)).item()
        k1_errs.append(rel_err)
    return {
        "intra_row_cv_mean": float(np.mean(cv_all)) if cv_all else 0.0,
        "k1_residual_mean": float(np.mean(k1_errs)) if k1_errs else 0.0,
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
del model
import gc; gc.collect()


results = {"T0_base_ce": float(T0), "init_metrics": init, "ops_tested": []}


def test_op(op_name, op_func, max_drift_for_lossless=1e-3):
    print(f"\n{'─'*70}")
    print(f"OP: {op_name}")
    print('─'*70)
    m = fresh_model()
    op_func(m)
    ce = lm_ce(m, val_tokens)
    drift = ce - T0
    metrics = measure_body_intra_row_cv(m)
    is_lossless = abs(drift) < max_drift_for_lossless
    verdict = "LOSSLESS ✓" if is_lossless else "lossy"
    print(f"  Δ: {drift:+.6f}  ({verdict})")
    print(f"  intra-row CV: {init['intra_row_cv_mean']:.4f} → {metrics['intra_row_cv_mean']:.4f}  "
          f"(Δ {metrics['intra_row_cv_mean'] - init['intra_row_cv_mean']:+.4f})")
    print(f"  K=1 residual: {init['k1_residual_mean']:.4f} → {metrics['k1_residual_mean']:.4f}  "
          f"(Δ {metrics['k1_residual_mean'] - init['k1_residual_mean']:+.4f})")
    results["ops_tested"].append({
        "name": op_name,
        "drift": float(drift),
        "is_lossless": bool(is_lossless),
        "intra_row_cv_before": init['intra_row_cv_mean'],
        "intra_row_cv_after": metrics['intra_row_cv_mean'],
        "k1_residual_before": init['k1_residual_mean'],
        "k1_residual_after": metrics['k1_residual_mean'],
    })
    del m
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


# ─── OP 4: Random orthogonal rotation (per body Linear, with input compensation) ───
class RotatedLinear(nn.Module):
    """Linear with input rotation + weight col rotation: lossless reparam.

    For W·x with rotation R (orthogonal):
      W_new = W·R, x_new = R^T·x
      W_new · x_new = W·R·R^T·x = W·x ✓
    """
    def __init__(self, original_linear, R):
        super().__init__()
        # Apply R to weight cols (which is the input dim)
        W_rotated = original_linear.weight.data.float() @ R.float().to(original_linear.weight.device)
        self.weight = nn.Parameter(W_rotated.to(original_linear.weight.dtype))
        self.bias = original_linear.bias
        # Save R^T for input rotation
        self.register_buffer("R_T", R.T.to(original_linear.weight.dtype).contiguous())

    def forward(self, x):
        x_rotated = x @ self.R_T   # equivalent to R^T · x for batched
        return F.linear(x_rotated, self.weight,
                        self.bias.to(x.dtype) if self.bias is not None else None)


def op_random_rotation(m):
    """Apply per-Linear random orthogonal rotation, lossless via input compensation."""
    target_mods = [(n, mod) for n, mod in m.named_modules()
                   if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    torch.manual_seed(42)
    for name, mod in target_mods:
        in_dim = mod.weight.shape[1]
        # QR on CPU (MPS lacks linalg.qr), then move to device
        A = torch.randn(in_dim, in_dim, device="cpu", dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        Q = Q.to(mod.weight.device)
        new_layer = RotatedLinear(mod, Q)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)


test_op("Random orthogonal rotation per Linear (with input compensation)", op_random_rotation)


# ─── OP 5: Hadamard rotation (special: power-of-2 dim, ±1 entries) ───
def hadamard_matrix(n, device, dtype=torch.float32):
    """Return normalized Hadamard matrix H/sqrt(n) where H is Sylvester-Hadamard.
    n must be a power of 2."""
    if n == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    # Recursive Sylvester construction
    H_half = hadamard_matrix(n // 2, device, dtype)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1),
    ], dim=0)
    return H


def op_hadamard_rotation(m):
    """Apply Hadamard rotation per-Linear (only on power-of-2 input dims)."""
    target_mods = [(n, mod) for n, mod in m.named_modules()
                   if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    rotated_count = 0
    for name, mod in target_mods:
        in_dim = mod.weight.shape[1]
        # Check power of 2
        if in_dim & (in_dim - 1) != 0:
            continue
        H = hadamard_matrix(in_dim, mod.weight.device, torch.float32)
        H_normalized = H / np.sqrt(in_dim)
        new_layer = RotatedLinear(mod, H_normalized)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        rotated_count += 1
    print(f"    (rotated {rotated_count}/{len(target_mods)} linears with power-of-2 in-dim)")


test_op("Hadamard rotation (per-Linear, power-of-2 dims only)", op_hadamard_rotation)


# ─── OP 6: Per-group magnitude factoring (Bonsai-style α per group) ───
class PerGroupAlphaLinear(nn.Module):
    """Linear with per-group α extracted as separate trainable scalar per group.

    For each row of W:
      Split into n_groups of group_size weights.
      For each group, factor: W_group = α_group · unit_W_group
      α_group = mean(|W_group|)  (or row L2/sqrt(group_size))
      Store unit_W_group (unit-mean-magnitude) and α_group separately.

    Forward: F.linear(x, weight) where weight is reconstructed from unit_W and α.
    """
    def __init__(self, original_linear, group_size=128):
        super().__init__()
        W = original_linear.weight.data.float()
        out_features, in_features = W.shape
        if in_features % group_size != 0:
            # Fallback: per-row only
            rn = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_unit = nn.Parameter((W / rn).to(original_linear.weight.dtype))
            self.alpha_per_group = nn.Parameter(rn.squeeze(-1).to(torch.float32))
            self.group_size = in_features
            self.n_groups = 1
        else:
            n_groups = in_features // group_size
            grouped = W.reshape(out_features, n_groups, group_size)
            mean_abs = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
            unit_grouped = grouped / mean_abs
            self.W_unit = nn.Parameter(unit_grouped.reshape(out_features, in_features)
                                       .to(original_linear.weight.dtype))
            self.alpha_per_group = nn.Parameter(mean_abs.squeeze(-1).to(torch.float32))
            self.group_size = group_size
            self.n_groups = n_groups
        self.bias = original_linear.bias
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        # Reconstruct weight: unit_W * α_per_group (broadcast)
        W_reconstructed = (self.W_unit.float().reshape(self.out_features, self.n_groups, self.group_size)
                           * self.alpha_per_group.float().unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return F.linear(x, W_reconstructed.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


def op_per_group_magnitude(m):
    """Apply per-group α factoring to each body Linear."""
    target_mods = [(n, mod) for n, mod in m.named_modules()
                   if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    for name, mod in target_mods:
        new_layer = PerGroupAlphaLinear(mod, GROUP_SIZE)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)


test_op("Per-group magnitude factoring (Bonsai-style α per 128-group)", op_per_group_magnitude)


# ─── Save ───
print(f"\n{'='*70}")
print("EXPANDED LOSSLESS OPS LIBRARY")
print('='*70)
print(f"  init: intra-row CV={init['intra_row_cv_mean']:.4f}  K=1 err={init['k1_residual_mean']:.4f}")
print()
print(f"  {'op':<60} {'lossless':>10} {'CV change':>10} {'K1 change':>10}")
for op in results["ops_tested"]:
    cv_delta = op['intra_row_cv_after'] - op['intra_row_cv_before']
    k1_delta = op['k1_residual_after'] - op['k1_residual_before']
    lossless_mark = "✓" if op['is_lossless'] else "✗"
    print(f"  {op['name'][:60]:<60} {lossless_mark:>10} {cv_delta:>+10.4f} {k1_delta:>+10.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
