"""Stage 185: trend diagnostic — what compensating mechanisms grow as
precision shrinks?

Hypothesis (user's intuition, 2026-04-29): as you take bits away from
weights, the model has to amplify *everything else* to overcome the
rising relative noise floor. If true, we should see a coordinated rise
across multiple axes (weight magnitudes, RMSNorm gains, embedding
amplitudes, lm_head amplitudes) — not just one knob.

Three models on the precision spectrum:
  - Qwen3-0.6B        FP16            (full precision baseline)
  - BitNet b1.58 2B   ternary {-γ,0,γ} (1.58 bits per weight)
  - Bonsai-8B-1bit    binary  ±scale   (1.0 bits per weight)

Different sizes, but per-element / per-row metrics normalize that out.

Read-only — no training, no forward passes. Just walks each model's
state_dict and aggregates statistics.

Axes measured per model:
  1. Body row-norm distribution (mean, CV, max/min) per projection type
  2. Per-element RMS amplitude (row_norm / sqrt(in_features)) — dim-invariant
  3. RMSNorm gain distribution (mean, CV)
  4. Embedding row-norm distribution
  5. LM head row-norm distribution
  6. Total "amplitude budget" — sum of |w| / n_params

For Bonsai, body weights are packed binary (uint32 popcount needed);
norms, embeddings, lm_head remain FP.
"""
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


RESULTS_PATH = Path("results/stage185_compensation_trend.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def categorize_param(name):
    """Return (category, projection_subtype). category in:
       body | norm | embed | lm_head | other"""
    n = name.lower()
    if "embed_tokens" in n or "embed" in n and "lm_head" not in n:
        return "embed", None
    if "lm_head" in n:
        return "lm_head", None
    if "norm" in n and "weight" in n:
        return "norm", None
    if any(t in n for t in TARGET_NAMES):
        for t in TARGET_NAMES:
            if t in n:
                return "body", t
    return "other", None


def stats(arr):
    if len(arr) == 0:
        return None
    a = np.asarray(arr)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "cv": float(a.std() / max(a.mean(), 1e-12)),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
    }


def analyze_model_weights(state_dict, label):
    """Walk state_dict, compute compensation axes. state_dict values
    must be torch tensors of the EFFECTIVE weights (post-quant for
    body if applicable). Returns dict."""
    body_norms = defaultdict(list)        # per-projection-type row L2 norms
    body_rms = defaultdict(list)          # row_norm / sqrt(in_features)
    body_in_features = {}
    norm_gains = []                       # all RMSNorm weight values
    embed_norms = []
    lm_head_norms = []
    total_abs_sum = 0.0
    total_n = 0

    for name, tensor in state_dict.items():
        if tensor.dim() == 0:
            continue
        category, proj_type = categorize_param(name)
        t = tensor.detach().float()

        if category == "body" and proj_type is not None and t.dim() == 2:
            row_norms = t.norm(dim=-1).cpu().numpy()
            body_norms[proj_type].extend(row_norms.tolist())
            in_feat = t.shape[1]
            body_in_features[proj_type] = in_feat
            body_rms[proj_type].extend((row_norms / math.sqrt(in_feat)).tolist())
        elif category == "norm":
            norm_gains.extend(t.flatten().cpu().numpy().tolist())
        elif category == "embed" and t.dim() == 2:
            embed_norms.extend(t.norm(dim=-1).cpu().numpy().tolist())
        elif category == "lm_head" and t.dim() == 2:
            lm_head_norms.extend(t.norm(dim=-1).cpu().numpy().tolist())

        total_abs_sum += float(t.abs().sum().item())
        total_n += int(t.numel())

    body_overall = []
    for v in body_norms.values():
        body_overall.extend(v)

    return {
        "label": label,
        "body_overall": stats(body_overall),
        "body_per_projection": {k: stats(v) for k, v in body_norms.items()},
        "body_per_projection_rms": {k: stats(v) for k, v in body_rms.items()},
        "body_in_features": body_in_features,
        "norm_gains": stats(norm_gains),
        "embed_norms": stats(embed_norms),
        "lm_head_norms": stats(lm_head_norms),
        "amplitude_budget_per_param": total_abs_sum / max(total_n, 1),
        "total_n_params": total_n,
    }


# ─── 1. Qwen3-0.6B (FP) ─────────────────────────────────────────────
print("=" * 70)
print("Loading Qwen3-0.6B (FP16 baseline)")
print("=" * 70)
model_qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float32, low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()
sd_qwen = {n: p.data for n, p in model_qwen.named_parameters()}
qwen_axes = analyze_model_weights(sd_qwen, "Qwen3-0.6B FP")
del model_qwen, sd_qwen
gc.collect()


# ─── 2. BitNet b1.58 ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Loading BitNet b1.58 2B-4T")
print("=" * 70)
try:
    model_bitnet = AutoModelForCausalLM.from_pretrained(
        "microsoft/bitnet-b1.58-2B-4T", dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()
    sd_bitnet_master = {n: p.data for n, p in model_bitnet.named_parameters()}

    # Compute "effective" body weights via ternary quantization (matches
    # what the deployed model uses). Other params (norms, embed, lm_head)
    # stay FP.
    sd_bitnet_eff = {}
    for n, t in sd_bitnet_master.items():
        cat, _ = categorize_param(n)
        if cat == "body" and t.dim() == 2:
            gamma = t.abs().mean()
            t_eff = gamma * torch.clamp(torch.round(t / gamma.clamp(min=1e-8)), -1, 1)
            sd_bitnet_eff[n] = t_eff
        else:
            sd_bitnet_eff[n] = t

    bitnet_axes = analyze_model_weights(sd_bitnet_eff, "BitNet b1.58 (effective)")
    bitnet_master_axes = analyze_model_weights(sd_bitnet_master, "BitNet b1.58 (master FP)")
    del model_bitnet, sd_bitnet_master, sd_bitnet_eff
    gc.collect()
except Exception as e:
    print(f"  failed to load BitNet: {e}")
    bitnet_axes = None
    bitnet_master_axes = None


# ─── 3. Bonsai-8B-1bit (MLX format, packed binary) ───────────────────
print("\n" + "=" * 70)
print("Loading Bonsai-8B-1bit (binary, MLX format)")
print("=" * 70)


def popcount_uint32(x):
    """Vectorized popcount for uint32 tensor."""
    x = x.long() & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24) & 0xFF


def decode_bonsai_layer_to_effective(weight_packed, scales, biases, group_size=128):
    """Decode MLX-packed Bonsai 1-bit weights to effective FP tensor.
       weight_packed: [out, in/32] uint32 (each uint32 packs 32 binary weights)
       scales:        [out, in/group_size] FP — per-group magnitude
       biases:        [out, in/group_size] FP — per-group offset
       Effective weight per element = bias + (bit==1 ? scale : 0). With
       symmetric encoding: bit==1 → scale, bit==0 → -scale (when bias≈0).
       This matches diag_bonsai_hypersphere.py decoding."""
    out_features = weight_packed.shape[0]
    in_packed = weight_packed.shape[1]
    in_features = in_packed * 32
    n_groups = in_features // group_size

    # Unpack: bit b of uint32 → 0 or 1. Total in_features bits per row.
    # We don't need the full FP matrix to compute row norms — just popcount
    # per group. But we DO need it to compute amplitude budget. Still
    # avoid materialising 8B * 4 = 32GB.
    # Compromise: compute row norms directly via popcount, and approximate
    # amplitude budget from per-group statistics.

    # Per-group bits set:
    # Reshape packed [out, n_groups, group_size/32] → popcount → sum
    pack_per_group = group_size // 32
    wp = weight_packed.view(out_features, n_groups, pack_per_group)
    bits_set_per_group = popcount_uint32(wp).sum(dim=-1).float()  # [out, n_groups]
    bits_unset_per_group = group_size - bits_set_per_group

    # Sum-of-squares per group:  bits_set * (scale+bias)^2 + bits_unset * (bias)^2
    # Effective row norm = sqrt(sum_groups of sum_sq_per_group)
    sum_sq = bits_set_per_group * (scales + biases).pow(2) + \
             bits_unset_per_group * biases.pow(2)
    row_norm_sq = sum_sq.sum(dim=-1)
    row_norms = row_norm_sq.sqrt()

    # Sum of |w| per group = bits_set * |scale + bias| + bits_unset * |bias|
    sum_abs = bits_set_per_group * (scales + biases).abs() + \
              bits_unset_per_group * biases.abs()
    total_abs = float(sum_abs.sum().item())

    return row_norms.cpu().numpy(), total_abs, in_features, out_features * in_features


def analyze_bonsai_directly(model_repo="mlx-community/Bonsai-8B-1bit"):
    """Walk Bonsai's safetensors directly, compute axes."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    print(f"  downloading/locating {model_repo}...")
    local = snapshot_download(model_repo)
    files = sorted([f for f in os.listdir(local) if f.endswith(".safetensors")])
    print(f"  found {len(files)} safetensor file(s)")

    body_norms = defaultdict(list)
    body_rms = defaultdict(list)
    body_in_features = {}
    norm_gains = []
    embed_norms = []
    lm_head_norms = []
    total_abs_sum = 0.0
    total_n = 0

    # Track packed body layers to combine .weight + .scales + .biases
    body_packed = defaultdict(dict)  # layer_prefix → {"weight":..., "scales":..., "biases":...}

    for fname in files:
        path = os.path.join(local, fname)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dim() == 0:
                    continue

                # Detect Bonsai's MLX-style packed format: tensor names like
                # "...q_proj.weight" with dtype uint32 → packed binary
                # Companion ".scales" and ".biases" alongside.
                if any(tname in key for tname in TARGET_NAMES) and \
                   (key.endswith(".weight") or key.endswith(".scales") or key.endswith(".biases")):
                    prefix = key.rsplit(".", 1)[0]
                    suffix = key.rsplit(".", 1)[1]
                    body_packed[prefix][suffix] = t
                    continue

                cat, proj = categorize_param(key)
                tt = t.float() if t.dtype != torch.uint32 else t
                if cat == "norm":
                    norm_gains.extend(tt.flatten().cpu().numpy().tolist())
                    total_abs_sum += float(tt.abs().sum().item())
                    total_n += int(tt.numel())
                elif cat == "embed" and tt.dim() == 2:
                    embed_norms.extend(tt.norm(dim=-1).cpu().numpy().tolist())
                    total_abs_sum += float(tt.abs().sum().item())
                    total_n += int(tt.numel())
                elif cat == "lm_head" and tt.dim() == 2:
                    lm_head_norms.extend(tt.norm(dim=-1).cpu().numpy().tolist())
                    total_abs_sum += float(tt.abs().sum().item())
                    total_n += int(tt.numel())
                else:
                    total_abs_sum += float(tt.abs().sum().item())
                    total_n += int(tt.numel())

    # Decode body packed groups
    for prefix, parts in body_packed.items():
        if "weight" not in parts:
            continue
        proj_type = next((t for t in TARGET_NAMES if t in prefix), None)
        if proj_type is None:
            continue
        weight = parts["weight"]
        scales = parts.get("scales", None)
        biases = parts.get("biases", None)
        if scales is None:
            print(f"  warning: {prefix} has packed weight but no scales — skipping")
            continue
        if biases is None:
            biases = torch.zeros_like(scales)
        try:
            row_norms, abs_sum, in_features, n_total = decode_bonsai_layer_to_effective(
                weight, scales.float(), biases.float(), group_size=128
            )
            body_norms[proj_type].extend(row_norms.tolist())
            body_rms[proj_type].extend((row_norms / math.sqrt(in_features)).tolist())
            body_in_features[proj_type] = in_features
            total_abs_sum += abs_sum
            total_n += n_total
        except Exception as e:
            print(f"  failed to decode {prefix}: {e}")

    body_overall = []
    for v in body_norms.values():
        body_overall.extend(v)

    return {
        "label": "Bonsai-8B-1bit (effective)",
        "body_overall": stats(body_overall),
        "body_per_projection": {k: stats(v) for k, v in body_norms.items()},
        "body_per_projection_rms": {k: stats(v) for k, v in body_rms.items()},
        "body_in_features": body_in_features,
        "norm_gains": stats(norm_gains),
        "embed_norms": stats(embed_norms),
        "lm_head_norms": stats(lm_head_norms),
        "amplitude_budget_per_param": total_abs_sum / max(total_n, 1),
        "total_n_params": total_n,
    }


try:
    bonsai_axes = analyze_bonsai_directly()
except Exception as e:
    print(f"  failed to load/decode Bonsai: {e}")
    bonsai_axes = None
gc.collect()


# ─── Side-by-side report ────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPENSATION TREND ACROSS PRECISION SPECTRUM")
print("=" * 80)


def fmt(x, fmtspec="{:>9.4f}"):
    if x is None:
        return f"{'—':>9}"
    return fmtspec.format(x)


def get(axes, *path):
    cur = axes
    for p in path:
        if cur is None or p not in cur:
            return None
        cur = cur[p]
    return cur


axes_set = [
    ("Qwen3-0.6B FP", qwen_axes),
    ("BitNet master FP", bitnet_master_axes),
    ("BitNet effective", bitnet_axes),
    ("Bonsai effective", bonsai_axes),
]

print(f"\n{'AXIS':<40} " + " ".join(f"{label:>20}" for label, _ in axes_set))
print("-" * (42 + 21 * len(axes_set)))


def row(name, *vals, fmtspec="{:>20.4f}"):
    cells = []
    for v in vals:
        cells.append(("{:>20}").format("—") if v is None else fmtspec.format(v))
    print(f"{name:<40} " + " ".join(cells))


# Body row-norm overall
print("\n[Body weight row-norm distribution]")
row("  mean", *(get(a, "body_overall", "mean") for _, a in axes_set))
row("  CV",   *(get(a, "body_overall", "cv")   for _, a in axes_set))
row("  max",  *(get(a, "body_overall", "max")  for _, a in axes_set))

# Per-element RMS amplitude (dim-invariant)
print("\n[Per-element RMS amplitude  =  row_norm / sqrt(in_features)]")
print("   (dim-invariant — direct comparison across model widths)")
for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    row(f"  {proj}", *(get(a, "body_per_projection_rms", proj, "mean") for _, a in axes_set))

# RMSNorm gains
print("\n[RMSNorm gain distribution]")
row("  mean", *(get(a, "norm_gains", "mean") for _, a in axes_set))
row("  CV",   *(get(a, "norm_gains", "cv")   for _, a in axes_set))
row("  max",  *(get(a, "norm_gains", "max")  for _, a in axes_set))

# Embedding row norms
print("\n[Embedding row-norm distribution]")
row("  mean", *(get(a, "embed_norms", "mean") for _, a in axes_set))
row("  CV",   *(get(a, "embed_norms", "cv")   for _, a in axes_set))

# LM head row norms
print("\n[LM head row-norm distribution]")
row("  mean", *(get(a, "lm_head_norms", "mean") for _, a in axes_set))
row("  CV",   *(get(a, "lm_head_norms", "cv")   for _, a in axes_set))

# Amplitude budget (mean |w| per param)
print("\n[Total amplitude budget per parameter  =  Σ|w| / N]")
row("  mean |w|", *(get(a, "amplitude_budget_per_param") for _, a in axes_set))


# ─── Trend interpretation ────────────────────────────────────────────
print("\n" + "=" * 80)
print("INTERPRETATION HINTS")
print("=" * 80)

# Compare Qwen → BitNet eff → Bonsai eff (the precision descent)
qwen_rms = get(qwen_axes, "body_overall", "mean")
bitnet_eff_rms = get(bitnet_axes, "body_overall", "mean")
bonsai_rms = get(bonsai_axes, "body_overall", "mean")
print(f"\nBody row-norm trend (FP → ternary → binary):")
if qwen_rms and bitnet_eff_rms and bonsai_rms:
    print(f"  Qwen  {qwen_rms:.3f}  →  BitNet  {bitnet_eff_rms:.3f}  →  Bonsai  {bonsai_rms:.3f}")
    print(f"  ratio Bonsai/Qwen = {bonsai_rms/qwen_rms:.2f}×")

q_norm = get(qwen_axes, "norm_gains", "mean")
b_norm = get(bitnet_master_axes, "norm_gains", "mean")
bo_norm = get(bonsai_axes, "norm_gains", "mean")
print(f"\nRMSNorm gain trend:")
if q_norm and b_norm and bo_norm:
    print(f"  Qwen  {q_norm:.3f}  →  BitNet  {b_norm:.3f}  →  Bonsai  {bo_norm:.3f}")

q_emb = get(qwen_axes, "embed_norms", "mean")
b_emb = get(bitnet_master_axes, "embed_norms", "mean")
bo_emb = get(bonsai_axes, "embed_norms", "mean")
print(f"\nEmbedding row-norm trend:")
if q_emb and b_emb and bo_emb:
    print(f"  Qwen  {q_emb:.3f}  →  BitNet  {b_emb:.3f}  →  Bonsai  {bo_emb:.3f}")

q_lm = get(qwen_axes, "lm_head_norms", "mean")
b_lm = get(bitnet_master_axes, "lm_head_norms", "mean")
bo_lm = get(bonsai_axes, "lm_head_norms", "mean")
print(f"\nLM-head row-norm trend:")
if q_lm and b_lm and bo_lm:
    print(f"  Qwen  {q_lm:.3f}  →  BitNet  {b_lm:.3f}  →  Bonsai  {bo_lm:.3f}")

print(f"\nLook for: do norms/embeds/lm_head amplify in a coordinated way as bits drop,")
print(f"  or does only the body weight magnitude rise? Coordinated rise = compensation")
print(f"  budget redistributes across DOF. Body-only rise = magnitude is the only knob.")


# ─── Save ────────────────────────────────────────────────────────────
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "qwen": qwen_axes,
        "bitnet_master": bitnet_master_axes,
        "bitnet_effective": bitnet_axes,
        "bonsai_effective": bonsai_axes,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
