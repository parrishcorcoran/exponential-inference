"""Stage 175a: per-head structure in o_proj of BitNet b1.58 (ternary).

We measured Bonsai 1-bit (PC1=35%) and Qwen3-0.6B FP base (PC1=43%).
Question: does ternary {-1, 0, +1} preserve the per-head selection
pattern, or does the threshold-based ternary destroy it?

BitNet stores FP master weights, applies absmean-threshold ternary at
forward. Compute the ternary effective weights, then reshape by head
structure, PCA across n_heads.

Predict: ternary preserves the structure (high PC1) because it preserves
the relative magnitude pattern between heads. The "0" state collapses
small weights toward zero but doesn't break per-head correlation.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig


CHECKPOINT = "1bitLLM/bitnet_b1_58-large"
RESULTS_PATH = Path("results/stage175a_head_structure_bitnet.json")


print(f"Loading {CHECKPOINT}...")
cfg = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
n_heads = cfg.num_attention_heads
hidden = cfg.hidden_size
head_dim = hidden // n_heads
print(f"  hidden={hidden}, n_heads={n_heads}, head_dim={head_dim}")

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).eval()


def ternary_effective(W):
    """BitNet b1.58 ternary: gamma = mean(|W|), W_q = clip(round(W/gamma), -1, 1).
    Effective forward weight = gamma * W_q."""
    gamma = W.abs().mean()
    W_q = torch.clamp(torch.round(W / gamma), -1, 1)
    return gamma * W_q


# Aggregate per-(row, head) magnitudes from ternary-effective o_proj
master_per_head = []   # FP master magnitudes
ternary_per_head = []  # ternary effective magnitudes
matched = 0
for name, mod in model.named_modules():
    if "o_proj" in name and hasattr(mod, "weight") and mod.weight is not None and mod.weight.ndim == 2:
        W_master = mod.weight.detach().float()
        if W_master.shape[1] % n_heads != 0:
            continue
        actual_head_dim = W_master.shape[1] // n_heads

        # Master per-head
        W_m_reshaped = W_master.reshape(W_master.shape[0], n_heads, actual_head_dim)
        master_per_head.append(W_m_reshaped.norm(dim=-1).cpu().numpy())

        # Ternary effective
        W_t = ternary_effective(W_master)
        W_t_reshaped = W_t.reshape(W_t.shape[0], n_heads, actual_head_dim)
        ternary_per_head.append(W_t_reshaped.norm(dim=-1).cpu().numpy())
        matched += 1

print(f"\nMatched {matched} o_proj layers.")
master_agg = np.concatenate(master_per_head, axis=0)
ternary_agg = np.concatenate(ternary_per_head, axis=0)
print(f"  master shape: {master_agg.shape}")
print(f"  ternary shape: {ternary_agg.shape}")


def analyze(name, agg):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    sample = agg[np.random.permutation(agg.shape[0])[:2000]]

    # Per-group
    group_means = sample.mean(axis=0)
    group_stds = sample.std(axis=0)
    cv_of_means = group_means.std() / max(group_means.mean(), 1e-8)
    print(f"  CV of per-head means: {cv_of_means:.4f}")

    # Correlation
    corr = np.corrcoef(sample.T)
    off_diag = corr[~np.eye(sample.shape[1], dtype=bool)]
    print(f"  cross-row corr mean: {off_diag.mean():.4f}")

    # PCA
    centered = sample - sample.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S**2) / (S**2).sum()
    print(f"  PC1: {explained[0]*100:.1f}%")
    print(f"  Top 3 PCs: {[f'{e*100:.1f}%' for e in explained[:3]]}")

    return {
        "cv_of_means": float(cv_of_means),
        "cross_row_corr": float(off_diag.mean()),
        "pc1": float(explained[0]),
        "pc2": float(explained[1]),
        "pc3": float(explained[2]),
    }


master_stats = analyze("BitNet b1.58 large — FP MASTER o_proj", master_agg)
ternary_stats = analyze("BitNet b1.58 large — TERNARY EFFECTIVE o_proj", ternary_agg)

print(f"\n{'='*70}\nCROSS-MODEL COMPARISON: per-head structure of o_proj\n{'='*70}")
print(f"  Model                                  PC1     cv_means    cross_row_corr")
print(f"  Qwen3-0.6B FP base                     42.8%   0.020       0.399")
print(f"  Bonsai-8B 1-bit effective              35.3%   0.017       0.323")
print(f"  BitNet b1.58 large FP MASTER           {master_stats['pc1']*100:.1f}%   {master_stats['cv_of_means']:.4f}      {master_stats['cross_row_corr']:.3f}")
print(f"  BitNet b1.58 large TERNARY EFFECTIVE   {ternary_stats['pc1']*100:.1f}%   {ternary_stats['cv_of_means']:.4f}      {ternary_stats['cross_row_corr']:.3f}")

print(f"\n  Verdict on whether ternary preserves head structure:")
delta_pc1 = (master_stats['pc1'] - ternary_stats['pc1']) / master_stats['pc1']
print(f"    PC1 change from ternary: {(ternary_stats['pc1']-master_stats['pc1'])*100:+.1f} percentage points ({delta_pc1*100:+.1f}% relative)")
if abs(delta_pc1) < 0.15:
    print(f"    ternary PRESERVES structure (PC1 changed <15% relative)")
else:
    print(f"    ternary alters structure significantly")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "master_fp": master_stats,
        "ternary_effective": ternary_stats,
        "comparison": {
            "qwen3_06b_fp": {"pc1": 0.428, "cv_of_means": 0.020, "cross_row_corr": 0.399},
            "bonsai_1bit": {"pc1": 0.353, "cv_of_means": 0.017, "cross_row_corr": 0.323},
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
