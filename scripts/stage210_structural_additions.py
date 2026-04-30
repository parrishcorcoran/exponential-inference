"""Stage 210 — Plan A: structural additions probe.

The library-only path is closed at g=128 (Stage 209: CV ≈ 0.754, Gaussian
floor). All remaining probability mass lives in STRUCTURAL ADDITIONS —
new degrees of freedom that don't just rearrange existing capacity but
add per-group/per-row capacity to absorb K=1 reconstruction error.

Bonsai's representation:
    W ≈ sign(W) * α_g + β_g    (per-group scale α_g, per-group bias β_g)

The Stage 207-209 "K1 residual" measurements were against:
    W ≈ sign(W) * α_g          (NO bias term)

Per-group bias β_g captures the DC offset within each group of 128 —
something pure binary fundamentally cannot. The closed-form least-squares
optimum per group is:
    α_g = mean(|W_g|)
    β_g = mean(W_g - sign(W_g) * α_g)

Predictions:
  - Per-group bias should drop K1 residual from ~0.60 toward ~0.30.
    The DC offset is large because binary forces a bimodal distribution
    while real weights are unimodal-Gaussian-ish.
  - Per-output bias (single bias per row, lossless trivially) probably
    moves K1 residual very little — the row-level mean is already small.
  - The combination of α_g + β_g is exactly what Bonsai uses, so we
    should see K1 residual at this level approach what Bonsai achieves.

Stage A measurement protocol:
  1. Verify Δ=0 for zero-init bias additions on Linear (lossless sanity)
  2. Measure K1 residual under three K=1 representations:
     - bare:        W ≈ sign(W) * α_g                  (Stage 209's measure)
     - +row_bias:   W ≈ sign(W) * α_g + β_row          (per-row)
     - +group_bias: W ≈ sign(W) * α_g + β_g            (per-group, Bonsai)
     - both:        W ≈ sign(W) * α_g + β_g + β_row    (full structural)
  3. Report intra-row CV (unchanged - structural doesn't affect signs)
  4. Report effective bits/weight at each level

Bit budget at group=128:
  bare:        1 + 16/128                = 1.125 bits
  +row_bias:   1 + 16/128 + 16/in_dim    = 1.125 + tiny
  +group_bias: 1 + (16+16)/128           = 1.250 bits  (Bonsai)
  both:        1 + (16+16)/128 + 16/in_dim ≈ 1.250 bits
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
RESULTS_PATH = Path("results/stage210_structural_additions.json")
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


class ZeroBiasLinear(nn.Module):
    """Linear with zero-init per-output bias added (or replacing existing bias).

    Lossless by construction: if bias starts at zero, F.linear(x, W, 0)
    is identical to F.linear(x, W). This stage verifies that adding
    bias capacity is in fact Δ=0 before we start using it for real.
    """
    def __init__(self, original_linear):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        out_features = self.weight.shape[0]
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(
                out_features, device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias.to(x.dtype))


def k1_residual_bare(W):
    """K=1 with per-group scale only: W ≈ sign(W) * α_g."""
    out, in_ = W.shape
    g = in_ // GROUP_SIZE
    Wg = W.reshape(out, g, GROUP_SIZE)
    alpha = Wg.abs().mean(dim=-1, keepdim=True)
    Wq = torch.sign(Wg) * alpha
    return ((Wg - Wq).norm() / Wg.norm().clamp(min=1e-8)).item()


def k1_residual_row_bias(W):
    """K=1 with per-row bias: W ≈ sign(W) * α_g + β_row.
    Closed-form: β_row = mean over row of (W - sign(W) * α_g).
    """
    out, in_ = W.shape
    g = in_ // GROUP_SIZE
    Wg = W.reshape(out, g, GROUP_SIZE)
    alpha = Wg.abs().mean(dim=-1, keepdim=True)
    residual = Wg - torch.sign(Wg) * alpha
    beta_row = residual.reshape(out, -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    Wq = torch.sign(Wg) * alpha + beta_row
    return ((Wg - Wq).norm() / Wg.norm().clamp(min=1e-8)).item()


def k1_residual_group_bias(W):
    """K=1 with per-group bias (Bonsai): W ≈ sign(W) * α_g + β_g.
    Closed-form: β_g = mean(W_g - sign(W_g) * α_g) per group.
    """
    out, in_ = W.shape
    g = in_ // GROUP_SIZE
    Wg = W.reshape(out, g, GROUP_SIZE)
    alpha = Wg.abs().mean(dim=-1, keepdim=True)
    residual = Wg - torch.sign(Wg) * alpha
    beta_g = residual.mean(dim=-1, keepdim=True)
    Wq = torch.sign(Wg) * alpha + beta_g
    return ((Wg - Wq).norm() / Wg.norm().clamp(min=1e-8)).item()


def k1_residual_both(W):
    """K=1 with both per-group and per-row bias.
    β_g absorbs group DC, then β_row absorbs row-level remainder.
    """
    out, in_ = W.shape
    g = in_ // GROUP_SIZE
    Wg = W.reshape(out, g, GROUP_SIZE)
    alpha = Wg.abs().mean(dim=-1, keepdim=True)
    residual = Wg - torch.sign(Wg) * alpha
    beta_g = residual.mean(dim=-1, keepdim=True)
    residual2 = residual - beta_g
    beta_row = residual2.reshape(out, -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
    Wq = torch.sign(Wg) * alpha + beta_g + beta_row
    return ((Wg - Wq).norm() / Wg.norm().clamp(min=1e-8)).item()


def measure_k1_residuals(model):
    bare_errs, row_errs, group_errs, both_errs = [], [], [], []
    intra_row_cvs = []
    n_measured = 0
    for name, mod in model.named_modules():
        if not any(t in name for t in TARGET_NAMES): continue
        if isinstance(mod, (nn.Linear, ZeroBiasLinear)):
            W = mod.weight.detach().float()
        else:
            continue
        if W.dim() != 2: continue
        out_features, in_features = W.shape
        if in_features % GROUP_SIZE != 0: continue

        bare_errs.append(k1_residual_bare(W))
        row_errs.append(k1_residual_row_bias(W))
        group_errs.append(k1_residual_group_bias(W))
        both_errs.append(k1_residual_both(W))

        Wg = W.reshape(out_features, in_features // GROUP_SIZE, GROUP_SIZE)
        abs_w = Wg.abs()
        mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
        cv = (abs_w.std(dim=-1) / mean_abs.squeeze(-1)).cpu().numpy().flatten()
        intra_row_cvs.extend(cv.tolist())
        n_measured += 1

    return {
        "k1_bare_mean":        float(np.mean(bare_errs)) if bare_errs else 0.0,
        "k1_row_bias_mean":    float(np.mean(row_errs)) if row_errs else 0.0,
        "k1_group_bias_mean":  float(np.mean(group_errs)) if group_errs else 0.0,
        "k1_both_mean":        float(np.mean(both_errs)) if both_errs else 0.0,
        "intra_row_cv_mean":   float(np.mean(intra_row_cvs)) if intra_row_cvs else 0.0,
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


# ─── Baseline ───
print("\nMeasuring T0 (base FP)...")
model = fresh_model()
T0 = lm_ce(model, val_tokens)
init = measure_k1_residuals(model)
print(f"  T0 = {T0:.4f}")
print(f"  intra-row CV:        {init['intra_row_cv_mean']:.4f}")
print(f"  K=1 bare:            {init['k1_bare_mean']:.4f}")
print(f"  K=1 + row bias:      {init['k1_row_bias_mean']:.4f}")
print(f"  K=1 + group bias:    {init['k1_group_bias_mean']:.4f}  (Bonsai-style)")
print(f"  K=1 + both:          {init['k1_both_mean']:.4f}")
print(f"  modules measured:    {init['n_measured']}")
del model
import gc; gc.collect()


# ─── OP: Zero-init per-output bias addition (Δ=0 sanity) ───
print(f"\n{'─'*70}")
print("OP: Zero-init per-output bias on every targeted Linear (lossless sanity)")
print('─'*70)
m = fresh_model()
target_mods = [(n, mod) for n, mod in m.named_modules()
               if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
parent_lookup = {}
for name, mod in m.named_modules():
    for child_name, child_mod in mod.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (mod, child_name)

added_count = 0
for name, mod in target_mods:
    new_layer = ZeroBiasLinear(mod)
    parent, child_attr = parent_lookup[name]
    setattr(parent, child_attr, new_layer)
    added_count += 1
print(f"  Added zero-init bias to {added_count}/{len(target_mods)} linears")

ce = lm_ce(m, val_tokens)
drift = ce - T0
metrics = measure_k1_residuals(m)
print(f"  Δ: {drift:+.6f}  ({'LOSSLESS ✓' if abs(drift) < 1e-3 else 'lossy'})")
print(f"  K=1 bare:            {init['k1_bare_mean']:.4f} → {metrics['k1_bare_mean']:.4f}")
print(f"  K=1 + row bias:      {init['k1_row_bias_mean']:.4f} → {metrics['k1_row_bias_mean']:.4f}")
print(f"  K=1 + group bias:    {init['k1_group_bias_mean']:.4f} → {metrics['k1_group_bias_mean']:.4f}")
print(f"  K=1 + both:          {init['k1_both_mean']:.4f} → {metrics['k1_both_mean']:.4f}")


# ─── Headline: K1 residual reduction from structural additions ───
drop_row   = init['k1_bare_mean'] - init['k1_row_bias_mean']
drop_group = init['k1_bare_mean'] - init['k1_group_bias_mean']
drop_both  = init['k1_bare_mean'] - init['k1_both_mean']

rel_drop_row   = drop_row   / init['k1_bare_mean'] * 100
rel_drop_group = drop_group / init['k1_bare_mean'] * 100
rel_drop_both  = drop_both  / init['k1_bare_mean'] * 100

print(f"\n{'─'*70}")
print("HEADLINE — K=1 residual reduction from structural additions:")
print('─'*70)
print(f"  bare         → +row bias:     Δ = {drop_row:+.4f}  ({rel_drop_row:+.1f}%)")
print(f"  bare         → +group bias:   Δ = {drop_group:+.4f}  ({rel_drop_group:+.1f}%)  [Bonsai]")
print(f"  bare         → +both:         Δ = {drop_both:+.4f}  ({rel_drop_both:+.1f}%)")
print()
if rel_drop_group > 30:
    print(f"  ✓ STRUCTURAL HYPOTHESIS LIVE: per-group bias drops K1 residual {rel_drop_group:.1f}%")
    print(f"    → Stage B (group-size sweep) and Stage C (full preconditioning + K=1) are GO.")
elif rel_drop_group > 10:
    print(f"  ⚠ MILD effect: per-group bias drops K1 residual only {rel_drop_group:.1f}%")
    print(f"    → Structural additions help but not dramatic. Continue to Stage B.")
else:
    print(f"  ✗ STRUCTURAL HYPOTHESIS WEAK: per-group bias drops K1 residual only {rel_drop_group:.1f}%")
    print(f"    → Recipe pivot needed. Library + structure both near floor.")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0_base_ce": float(T0),
        "init": init,
        "post_zero_bias": metrics,
        "lossless_drift": float(drift),
        "is_lossless": bool(abs(drift) < 1e-3),
        "n_added": added_count,
        "n_total_linears": len(target_mods),
        "rel_drop_row_bias_pct": float(rel_drop_row),
        "rel_drop_group_bias_pct": float(rel_drop_group),
        "rel_drop_both_pct": float(rel_drop_both),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
