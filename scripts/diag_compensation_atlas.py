"""Compensation Atlas: comprehensive comparison of every measurable
compensation axis between Qwen3 (FP base) and Bonsai (1-bit derived).

Bonsai = Qwen3-8B that's been compressed to 1-bit and continues to work
at 89% benchmarks. Whatever Bonsai DID to its compensation channels,
that's the recipe trace we can read off the deployed weights.

For each axis: Qwen value, Bonsai value, direction (UP/DOWN/SAME),
significance (ESSENTIAL/MODEST/IGNORED).

This is read-only. No training. No forward passes. Just walk the
state_dicts and compute statistics.

Output:
  - results/compensation_atlas.json (raw data)
  - docs/compensation_atlas.md (human-readable atlas)
"""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


RESULTS_PATH = Path("results/compensation_atlas.json")
DOC_PATH = Path("docs/compensation_atlas.md")
GROUP_SIZE = 128
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BONSAI_PATH = Path(
    "/Users/abundancemachine/.cache/huggingface/hub/"
    "models--prism-ml--Bonsai-8B-mlx-1bit/snapshots/"
    "019934f87a61a654e3960ea22f53688e0d2c49ba"
)


def categorize_param(name):
    n = name.lower()
    if ("embed_tokens" in n or "embed" in n) and "lm_head" not in n:
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


def stats(arr, percentiles=(1, 5, 50, 95, 99, 99.9)):
    if len(arr) == 0:
        return None
    a = np.asarray(arr)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "cv": float(a.std() / max(abs(a.mean()), 1e-12)),
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        **{f"p{int(p*10)/10}": float(np.percentile(a, p)) for p in percentiles},
    }


def popcount_uint32(x):
    x = x.long() & 0xFFFFFFFF
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24) & 0xFF


def decode_bonsai_layer(weight_packed, scales, biases, group_size=128):
    """Return per-row metrics: row_norm, intra-row scale CV, intra-row bias CV."""
    out_features = weight_packed.shape[0]
    in_packed = weight_packed.shape[1]
    in_features = in_packed * 32
    n_groups = in_features // group_size
    pack_per_group = group_size // 32

    wp = weight_packed.view(out_features, n_groups, pack_per_group)
    bits_set = popcount_uint32(wp).sum(dim=-1).float()
    bits_unset = group_size - bits_set

    sum_sq = bits_set * (scales + biases).pow(2) + bits_unset * biases.pow(2)
    row_norms = sum_sq.sum(dim=-1).sqrt()

    # Intra-row variation: how much do scales/biases differ across the n_groups in a row?
    # CV per row of the 32 scales
    scales_per_row_cv = (scales.std(dim=-1) / scales.abs().mean(dim=-1).clamp(min=1e-8)).cpu().numpy()
    biases_per_row_cv = (biases.std(dim=-1) / biases.abs().mean(dim=-1).clamp(min=1e-8)).cpu().numpy()

    return {
        "row_norms": row_norms.cpu().numpy(),
        "scales_flat": scales.flatten().cpu().numpy(),
        "biases_flat": biases.flatten().cpu().numpy(),
        "scales_per_row_cv": scales_per_row_cv,
        "biases_per_row_cv": biases_per_row_cv,
        "in_features": in_features,
    }


def measure_qwen(checkpoint="Qwen/Qwen3-0.6B"):
    """Walk Qwen3 FP weights, measure all axes."""
    print(f"\nLoading {checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()

    body_per_proj_row_norms = defaultdict(list)
    body_per_proj_intra_row_cv = defaultdict(list)
    body_per_proj_in_features = {}
    norm_gains = []
    embed_norms = []
    lm_head_norms = []
    biases_present = 0
    biases_magnitude = []
    total_abs_sum = 0.0
    total_n = 0

    for name, p in model.named_parameters():
        if p.dim() == 0: continue
        cat, proj = categorize_param(name)
        t = p.detach().float()

        if cat == "body" and proj is not None and t.dim() == 2:
            row_norms = t.norm(dim=-1).cpu().numpy()
            body_per_proj_row_norms[proj].extend(row_norms.tolist())
            body_per_proj_in_features[proj] = t.shape[1]
            # Intra-row CV: within each row, group into chunks of GROUP_SIZE,
            # measure std/mean of |w| per group, then CV of those group means
            if t.shape[1] % GROUP_SIZE == 0:
                n_groups = t.shape[1] // GROUP_SIZE
                grouped = t.reshape(t.shape[0], n_groups, GROUP_SIZE)
                group_means = grouped.abs().mean(dim=-1)  # [out, n_groups]
                # CV of group means within each row
                cv_per_row = (group_means.std(dim=-1) / group_means.mean(dim=-1).clamp(min=1e-8)).cpu().numpy()
                body_per_proj_intra_row_cv[proj].extend(cv_per_row.tolist())
        elif cat == "norm":
            norm_gains.extend(t.flatten().cpu().numpy().tolist())
        elif cat == "embed" and t.dim() == 2:
            embed_norms.extend(t.norm(dim=-1).cpu().numpy().tolist())
        elif cat == "lm_head" and t.dim() == 2:
            lm_head_norms.extend(t.norm(dim=-1).cpu().numpy().tolist())

        if "bias" in name and "embed" not in name:
            biases_present += 1
            biases_magnitude.extend(t.flatten().cpu().numpy().tolist())

        total_abs_sum += float(t.abs().sum().item())
        total_n += int(t.numel())

    body_overall = []
    body_intra_row_cv_all = []
    for v in body_per_proj_row_norms.values():
        body_overall.extend(v)
    for v in body_per_proj_intra_row_cv.values():
        body_intra_row_cv_all.extend(v)

    # Top-K outlier RMSNorm gains
    arr = np.array(norm_gains)
    top10_idx = np.argsort(np.abs(arr))[-10:][::-1]
    top10_outliers = [{"value": float(arr[i])} for i in top10_idx]

    return {
        "label": checkpoint,
        "body_overall_row_norms": stats(body_overall),
        "body_per_proj_row_norms": {k: stats(v) for k, v in body_per_proj_row_norms.items()},
        "body_in_features": body_per_proj_in_features,
        "body_intra_row_cv_overall": stats(body_intra_row_cv_all),
        "body_intra_row_cv_per_proj": {k: stats(v) for k, v in body_per_proj_intra_row_cv.items()},
        "norm_gains": stats(norm_gains),
        "norm_gains_top10": top10_outliers,
        "embed_norms": stats(embed_norms) if embed_norms else None,
        "lm_head_norms": stats(lm_head_norms) if lm_head_norms else None,
        "biases_present": biases_present,
        "biases_magnitude": stats(biases_magnitude) if biases_magnitude else None,
        "amplitude_per_param": total_abs_sum / max(total_n, 1),
        "total_n_params": total_n,
    }


def measure_bonsai():
    """Walk Bonsai's safetensors, decode binary body, compute axes."""
    from safetensors import safe_open

    print(f"\nLoading Bonsai from {BONSAI_PATH}")
    files = sorted([f for f in os.listdir(str(BONSAI_PATH)) if f.endswith(".safetensors")])

    body_per_proj_row_norms = defaultdict(list)
    body_per_proj_scale_cv = defaultdict(list)  # Bonsai's per-group scales — within-row CV
    body_per_proj_in_features = {}
    body_scales_all = []
    body_biases_all = []
    norm_gains = []
    embed_norms = []
    lm_head_norms = []
    total_abs_sum = 0.0
    total_n = 0

    # Group keys by prefix (weight + scales + biases triplets are packed binary)
    all_keys_by_prefix = defaultdict(dict)
    for fname in files:
        path = os.path.join(str(BONSAI_PATH), fname)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith((".weight", ".scales", ".biases")):
                    prefix = key.rsplit(".", 1)[0]
                    suffix = key.rsplit(".", 1)[1]
                    all_keys_by_prefix[prefix][suffix] = (path, key)
                else:
                    all_keys_by_prefix[key]["__solo__"] = (path, key)

    file_handles = {}
    def get_tensor(path, key):
        if path not in file_handles:
            file_handles[path] = safe_open(path, framework="pt").__enter__()
        return file_handles[path].get_tensor(key)

    try:
        for prefix, parts in all_keys_by_prefix.items():
            if "weight" in parts and "scales" in parts:
                # Packed binary body or lm_head
                weight = get_tensor(*parts["weight"])
                scales = get_tensor(*parts["scales"]).float()
                biases = (get_tensor(*parts["biases"]).float()
                          if "biases" in parts else torch.zeros_like(scales))

                proj_type = next((t for t in TARGET_NAMES if t in prefix), None)
                if proj_type is not None:
                    decoded = decode_bonsai_layer(weight, scales, biases, GROUP_SIZE)
                    body_per_proj_row_norms[proj_type].extend(decoded["row_norms"].tolist())
                    body_per_proj_scale_cv[proj_type].extend(decoded["scales_per_row_cv"].tolist())
                    body_per_proj_in_features[proj_type] = decoded["in_features"]
                    body_scales_all.extend(decoded["scales_flat"].tolist())
                    body_biases_all.extend(decoded["biases_flat"].tolist())
                    total_abs_sum += float(scales.abs().sum().item() + biases.abs().sum().item())
                    total_n += int(scales.numel() + biases.numel())
                elif "lm_head" in prefix.lower():
                    decoded = decode_bonsai_layer(weight, scales, biases, GROUP_SIZE)
                    lm_head_norms.extend(decoded["row_norms"].tolist())
            else:
                # Solo tensor: norm, embed, etc.
                if "__solo__" in parts:
                    path, key = parts["__solo__"]
                elif "weight" in parts:
                    path, key = parts["weight"]
                else:
                    continue
                t = get_tensor(path, key)
                if t.dim() == 0 or t.dtype == torch.uint32:
                    continue
                tt = t.float()
                cat, _ = categorize_param(key)
                if cat == "norm":
                    norm_gains.extend(tt.flatten().cpu().numpy().tolist())
                elif cat == "embed" and tt.dim() == 2:
                    embed_norms.extend(tt.norm(dim=-1).cpu().numpy().tolist())
                elif cat == "lm_head" and tt.dim() == 2:
                    lm_head_norms.extend(tt.norm(dim=-1).cpu().numpy().tolist())
                total_abs_sum += float(tt.abs().sum().item())
                total_n += int(tt.numel())
    finally:
        for h in file_handles.values():
            h.__exit__(None, None, None)

    body_overall = []
    scale_cv_overall = []
    for v in body_per_proj_row_norms.values():
        body_overall.extend(v)
    for v in body_per_proj_scale_cv.values():
        scale_cv_overall.extend(v)

    arr = np.array(norm_gains)
    top10_idx = np.argsort(np.abs(arr))[-10:][::-1]
    top10_outliers = [{"value": float(arr[i])} for i in top10_idx]

    return {
        "label": "Bonsai-8B-1bit (effective)",
        "body_overall_row_norms": stats(body_overall),
        "body_per_proj_row_norms": {k: stats(v) for k, v in body_per_proj_row_norms.items()},
        "body_in_features": body_per_proj_in_features,
        # Bonsai's "intra-row CV" is the CV of its 32 per-group scales within each row.
        # This is the analog of intra-row weight magnitude CV in Qwen.
        "body_intra_row_cv_overall": stats(scale_cv_overall),
        "body_intra_row_cv_per_proj": {k: stats(v) for k, v in body_per_proj_scale_cv.items()},
        "body_per_group_scales": stats(body_scales_all),
        "body_per_group_biases": stats(body_biases_all),
        "norm_gains": stats(norm_gains),
        "norm_gains_top10": top10_outliers,
        "embed_norms": stats(embed_norms) if embed_norms else None,
        "lm_head_norms": stats(lm_head_norms) if lm_head_norms else None,
        "biases_present": 0,  # Bonsai doesn't use FP biases on linears separately
        "amplitude_per_param": total_abs_sum / max(total_n, 1),
        "total_n_params": total_n,
    }


qwen_axes = measure_qwen()
import gc
gc.collect()

bonsai_axes = measure_bonsai()


# ─── Print comparison atlas ───
print("\n" + "=" * 100)
print("COMPENSATION ATLAS — Qwen3-0.6B FP vs Bonsai-8B-1bit")
print("=" * 100)


def fmt_dir(qwen_val, bonsai_val, threshold_ratio=1.15):
    """UP / DOWN / SAME / N/A with magnitude indicator."""
    if qwen_val is None or bonsai_val is None:
        return "  N/A "
    if abs(qwen_val) < 1e-8:
        return f"  N/A "
    ratio = bonsai_val / qwen_val
    if ratio > threshold_ratio:
        return f"  ↑×{ratio:.2f}"
    if ratio < 1 / threshold_ratio:
        return f"  ↓×{ratio:.2f}"
    return f"  ≈{ratio:.2f}"


def row(label, qwen_val, bonsai_val, fmt="{:.4f}"):
    qstr = fmt.format(qwen_val) if qwen_val is not None else "—"
    bstr = fmt.format(bonsai_val) if bonsai_val is not None else "—"
    direction = fmt_dir(qwen_val, bonsai_val)
    print(f"  {label:<45} {qstr:>14} {bstr:>14}   {direction}")


print(f"\n  {'AXIS':<45} {'Qwen3-0.6B':>14} {'Bonsai-8B':>14}   direction")
print("  " + "-" * 90)

print("\n[BODY ROW-NORM DISTRIBUTION]")
row("  body row-norm mean",
    qwen_axes["body_overall_row_norms"]["mean"],
    bonsai_axes["body_overall_row_norms"]["mean"])
row("  body row-norm CV (across rows)",
    qwen_axes["body_overall_row_norms"]["cv"],
    bonsai_axes["body_overall_row_norms"]["cv"])
row("  body row-norm max",
    qwen_axes["body_overall_row_norms"]["max"],
    bonsai_axes["body_overall_row_norms"]["max"])

print("\n[INTRA-ROW SHAPE — KEY METRIC]")
row("  intra-row CV (within-row magnitude spread)",
    qwen_axes["body_intra_row_cv_overall"]["mean"],
    bonsai_axes["body_intra_row_cv_overall"]["mean"])

print("\n[PER-PROJECTION ROW-NORM (mean)]")
for proj in TARGET_NAMES:
    qm = qwen_axes["body_per_proj_row_norms"].get(proj, {}).get("mean")
    bm = bonsai_axes["body_per_proj_row_norms"].get(proj, {}).get("mean")
    row(f"  {proj}", qm, bm)

print("\n[RMSNORM GAIN DISTRIBUTION]")
row("  norm gain mean",
    qwen_axes["norm_gains"]["mean"], bonsai_axes["norm_gains"]["mean"])
row("  norm gain CV (across all gains)",
    qwen_axes["norm_gains"]["cv"], bonsai_axes["norm_gains"]["cv"])
row("  norm gain max (outlier)",
    qwen_axes["norm_gains"]["max"], bonsai_axes["norm_gains"]["max"])
row("  norm gain p99.9",
    qwen_axes["norm_gains"].get("p99.9"), bonsai_axes["norm_gains"].get("p99.9"))

print("\n[EMBEDDING ROW-NORM]")
row("  embed mean",
    qwen_axes["embed_norms"]["mean"] if qwen_axes["embed_norms"] else None,
    bonsai_axes["embed_norms"]["mean"] if bonsai_axes["embed_norms"] else None)
row("  embed CV",
    qwen_axes["embed_norms"]["cv"] if qwen_axes["embed_norms"] else None,
    bonsai_axes["embed_norms"]["cv"] if bonsai_axes["embed_norms"] else None)

print("\n[LM HEAD ROW-NORM]")
row("  lm_head mean",
    qwen_axes["lm_head_norms"]["mean"] if qwen_axes["lm_head_norms"] else None,
    bonsai_axes["lm_head_norms"]["mean"] if bonsai_axes["lm_head_norms"] else None)
row("  lm_head CV",
    qwen_axes["lm_head_norms"]["cv"] if qwen_axes["lm_head_norms"] else None,
    bonsai_axes["lm_head_norms"]["cv"] if bonsai_axes["lm_head_norms"] else None)

print("\n[BONSAI PER-GROUP STRUCTURE — only Bonsai has these]")
if "body_per_group_scales" in bonsai_axes and bonsai_axes["body_per_group_scales"]:
    s = bonsai_axes["body_per_group_scales"]
    print(f"  per-group scale: mean={s['mean']:.4f}  CV={s['cv']:.3f}  max={s['max']:.3f}")
if "body_per_group_biases" in bonsai_axes and bonsai_axes["body_per_group_biases"]:
    b = bonsai_axes["body_per_group_biases"]
    print(f"  per-group bias:  mean={b['mean']:.4f}  CV={b['cv']:.3f}  max={b['max']:.3f}")

print("\n[BIAS PRESENCE]")
print(f"  Qwen biases on linears: {qwen_axes['biases_present']} tensors")
print(f"  Bonsai biases on linears: integrated as per-group biases (not separate)")

print("\n[TOTAL AMPLITUDE BUDGET]")
row("  amplitude per param",
    qwen_axes["amplitude_per_param"], bonsai_axes["amplitude_per_param"])


# ─── Save raw data ───
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "qwen": qwen_axes,
        "bonsai": bonsai_axes,
    }, f, indent=2)
print(f"\nSaved raw data: {RESULTS_PATH}")


# ─── Generate atlas markdown ───

def direction_label(qwen_val, bonsai_val):
    if qwen_val is None or bonsai_val is None or abs(qwen_val) < 1e-8:
        return "N/A", 1.0
    ratio = bonsai_val / qwen_val
    if ratio > 1.5: return "**UP** (essential)", ratio
    if ratio > 1.15: return "**up** (modest)", ratio
    if ratio < 1/1.5: return "**DOWN** (essential)", ratio
    if ratio < 1/1.15: return "**down** (modest)", ratio
    return "≈ same", ratio


with open(DOC_PATH, "w") as f:
    f.write("# Compensation Atlas: Qwen3 → Bonsai\n\n")
    f.write("**Source:** Read-only inspection of Qwen3-0.6B FP and Bonsai-8B-1bit weights.\n")
    f.write("**Purpose:** Identify which compensation axes are *load-bearing* in trained low-bit models.\n\n")
    f.write("Bonsai = Qwen3-8B compressed to 1-bit (89% benchmark retention). What its weights look like ")
    f.write("vs FP Qwen3 reveals which axes the training procedure used to compensate for binary precision.\n\n")
    f.write("**Reading the table:**\n")
    f.write("- **UP/DOWN essential**: ratio >1.5× or <0.67× — Bonsai meaningfully changed this axis\n")
    f.write("- **up/down modest**: ratio 1.15-1.5× — small adjustment\n")
    f.write("- **≈ same**: not a load-bearing compensation channel\n\n")

    f.write("## Direct comparison\n\n")
    f.write("| axis | Qwen | Bonsai | ratio | direction |\n")
    f.write("|---|---:|---:|---:|---|\n")

    rows = [
        ("body row-norm mean", qwen_axes["body_overall_row_norms"]["mean"], bonsai_axes["body_overall_row_norms"]["mean"]),
        ("body row-norm CV", qwen_axes["body_overall_row_norms"]["cv"], bonsai_axes["body_overall_row_norms"]["cv"]),
        ("body row-norm max", qwen_axes["body_overall_row_norms"]["max"], bonsai_axes["body_overall_row_norms"]["max"]),
        ("**intra-row magnitude CV** (shape)", qwen_axes["body_intra_row_cv_overall"]["mean"], bonsai_axes["body_intra_row_cv_overall"]["mean"]),
        ("RMSNorm gain mean", qwen_axes["norm_gains"]["mean"], bonsai_axes["norm_gains"]["mean"]),
        ("RMSNorm gain CV", qwen_axes["norm_gains"]["cv"], bonsai_axes["norm_gains"]["cv"]),
        ("RMSNorm gain max (outlier)", qwen_axes["norm_gains"]["max"], bonsai_axes["norm_gains"]["max"]),
        ("embedding row-norm mean", qwen_axes["embed_norms"]["mean"] if qwen_axes["embed_norms"] else None, bonsai_axes["embed_norms"]["mean"] if bonsai_axes["embed_norms"] else None),
        ("embedding row-norm CV", qwen_axes["embed_norms"]["cv"] if qwen_axes["embed_norms"] else None, bonsai_axes["embed_norms"]["cv"] if bonsai_axes["embed_norms"] else None),
        ("lm_head row-norm mean", qwen_axes["lm_head_norms"]["mean"] if qwen_axes["lm_head_norms"] else None, bonsai_axes["lm_head_norms"]["mean"] if bonsai_axes["lm_head_norms"] else None),
        ("amplitude per param", qwen_axes["amplitude_per_param"], bonsai_axes["amplitude_per_param"]),
    ]
    for label, q, b in rows:
        if q is None or b is None:
            f.write(f"| {label} | {'—' if q is None else f'{q:.4f}'} | {'—' if b is None else f'{b:.4f}'} | — | tied/unmatched |\n")
            continue
        direction, ratio = direction_label(q, b)
        f.write(f"| {label} | {q:.4f} | {b:.4f} | {ratio:.2f}× | {direction} |\n")

    f.write("\n## Per-projection breakdown — body row-norm mean\n\n")
    f.write("| projection | Qwen | Bonsai | ratio | direction |\n")
    f.write("|---|---:|---:|---:|---|\n")
    for proj in TARGET_NAMES:
        q = qwen_axes["body_per_proj_row_norms"].get(proj, {}).get("mean")
        b = bonsai_axes["body_per_proj_row_norms"].get(proj, {}).get("mean")
        if q is None or b is None:
            continue
        direction, ratio = direction_label(q, b)
        f.write(f"| {proj} | {q:.4f} | {b:.4f} | {ratio:.2f}× | {direction} |\n")

    f.write("\n## Bonsai-only structures (no Qwen analog)\n\n")
    if bonsai_axes.get("body_per_group_scales"):
        s = bonsai_axes["body_per_group_scales"]
        f.write(f"- **Per-group scales** (32 per row × N rows × N layers): mean={s['mean']:.4f}, CV={s['cv']:.3f}, max={s['max']:.3f}\n")
    if bonsai_axes.get("body_per_group_biases"):
        b = bonsai_axes["body_per_group_biases"]
        f.write(f"- **Per-group biases** (32 per row × N rows × N layers): mean={b['mean']:.4f}, CV={b['cv']:.3f}, max={b['max']:.3f}\n")
    f.write("\nThese are the *additional* FP DOFs Bonsai uses beyond what FP Qwen has. Each layer has ")
    f.write("32 scales + 32 biases per row of 4096 — they hold the per-group magnitude and offset that ")
    f.write("a single sign bit can't carry.\n\n")

    f.write("## Compensation channels classified\n\n")
    f.write("Based on the direction column, group the axes by load-bearing status:\n\n")
    f.write("### ESSENTIAL compensation channels (Bonsai changed strongly)\n")
    f.write("- *(filled in below based on data — see direction column above)*\n\n")
    f.write("### MODEST changes\n")
    f.write("- *(filled in based on data)*\n\n")
    f.write("### IGNORED / ≈ SAME\n")
    f.write("- *(filled in based on data)*\n\n")
    f.write("### NEW channels (Bonsai-only structures)\n")
    f.write("- Per-group scales (32 per row)\n")
    f.write("- Per-group biases (32 per row)\n\n")

    f.write("## Implications for our pipeline\n\n")
    f.write("**Channels Bonsai used heavily** (the load-bearing compensation):\n")
    f.write("1. *(automated reading from data)*\n\n")
    f.write("**Channels we should also coax in our PID** (currently unused or under-used):\n")
    f.write("- Add per-group bias as separate trainable parameter (currently computed from master)\n")
    f.write("- Active per-row α coaxing (we have it, just not on PID)\n\n")
    f.write("**Channels probably redundant** (Bonsai didn't use them, or they correlate with other axes):\n")
    f.write("- *(based on direction = ≈ same)*\n\n")

    f.write("## Reproducibility\n\n")
    f.write("```bash\n")
    f.write("python scripts/diag_compensation_atlas.py\n")
    f.write("```\n")

print(f"Saved atlas doc: {DOC_PATH}")
