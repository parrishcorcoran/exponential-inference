"""Stage 173: Verify head-alignment hypothesis on Bonsai o_proj.

Hypothesis: Bonsai's o_proj has 32 per-128-weight groups per row. With
group_size=128 = head_dim of Qwen3-8B, each group corresponds to one
attention head's contribution.

If true, we should see:
  1. Some groups (head indices) have systematically larger scales than
     others — heads have different "importance"
  2. The per-group scale pattern is correlated across rows (heads that
     matter for one output dim tend to matter for many)
  3. The principal component of (rows × groups) captures meaningful
     head-importance structure

If false, the per-group pattern would be random/uniform, suggesting the
group_size=128 alignment is coincidental, not load-bearing.

Diagnostic outputs:
  - Per-group-index mean and std of scales (averaged across rows)
  - Cross-row correlation by group index
  - PCA of the (out_dim × n_groups) scale matrix
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


CHECKPOINT_PATH = Path(
    "/Users/abundancemachine/.cache/huggingface/hub/"
    "models--prism-ml--Bonsai-8B-mlx-1bit/snapshots/"
    "019934f87a61a654e3960ea22f53688e0d2c49ba/model.safetensors"
)
RESULTS_PATH = Path("results/stage173_bonsai_head_alignment.json")
NUM_HEADS_QWEN3_8B = 32  # Qwen3-8B attention head count


# Read all o_proj scales
with safe_open(str(CHECKPOINT_PATH), framework="pt") as f:
    keys = [k for k in f.keys() if k.endswith("o_proj.scales")]
    print(f"Found {len(keys)} o_proj.scales tensors")
    o_scales_per_layer = {}
    for k in keys:
        layer_idx = int(k.split(".")[2])  # model.layers.{N}.self_attn.o_proj.scales
        scales = f.get_tensor(k).float().abs()  # [out, n_groups] = [4096, 32]
        o_scales_per_layer[layer_idx] = scales

# Aggregate across layers — concatenate rows from all layers
all_scales = torch.cat([s for s in o_scales_per_layer.values()], dim=0)  # [n_layers*4096, 32]
print(f"\nAggregated o_proj scales: {tuple(all_scales.shape)}  (rows × groups)")

n_rows, n_groups = all_scales.shape
print(f"  n_rows: {n_rows:,}")
print(f"  n_groups: {n_groups}  (matches num_heads={NUM_HEADS_QWEN3_8B}: {n_groups==NUM_HEADS_QWEN3_8B})")


# ─── Test 1: per-group mean / std ───
print(f"\n{'='*70}")
print("TEST 1: are some groups (heads) systematically larger?")
print(f"{'='*70}")
group_means = all_scales.mean(dim=0)  # [n_groups]
group_stds = all_scales.std(dim=0)
print(f"  Per-group mean (across all rows):")
for g in range(n_groups):
    print(f"    group {g:>3} (head {g}): mean={group_means[g].item():.5f}  std={group_stds[g].item():.5f}")

mean_of_means = group_means.mean().item()
std_of_means = group_means.std().item()
cv_of_means = std_of_means / mean_of_means
print(f"\n  CV of per-group means: {cv_of_means:.4f}")
print(f"  If CV ≈ 0: groups are uniform → no head structure")
print(f"  If CV >> 0: some heads are systematically larger → head structure exists")


# ─── Test 2: cross-row correlation by group ───
print(f"\n{'='*70}")
print("TEST 2: is per-group magnitude correlated across rows?")
print(f"{'='*70}")
# Correlate row pattern of (group magnitudes) with all other rows
# Sample 1000 rows for tractability, compute mean correlation across (group_idx, group_idx)
sample = all_scales[torch.randperm(n_rows)[:2000]].numpy()  # [2000, 32]
# Pearson correlation across rows for each pair of groups
group_corr = np.corrcoef(sample.T)  # [32, 32] correlation between group indices

off_diagonal = group_corr[~np.eye(n_groups, dtype=bool)]
print(f"  Mean off-diagonal correlation across groups: {off_diagonal.mean():.4f}")
print(f"  Max off-diagonal correlation: {off_diagonal.max():.4f}")
print(f"  Min off-diagonal correlation: {off_diagonal.min():.4f}")
print(f"  If near 0: groups are independent (no head structure)")
print(f"  If high: groups are correlated (rows that use head X tend to use head Y too — implausible)")
print(f"  Negative or near-zero suggests heads work somewhat independently — expected if they're real heads")


# ─── Test 3: PCA — does first PC capture per-head importance? ───
print(f"\n{'='*70}")
print("TEST 3: PCA — does the leading axis capture meaningful head structure?")
print(f"{'='*70}")
# Center
centered = sample - sample.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
explained_variance = (S ** 2) / (S ** 2).sum()
print(f"  Top 5 explained variance ratios:")
for i in range(min(5, len(S))):
    print(f"    PC{i+1}: {explained_variance[i]:.4f} ({explained_variance[i]*100:.1f}%)")

print(f"\n  PC1 loadings (per group, i.e., per head):")
pc1 = Vt[0]
sorted_groups = np.argsort(np.abs(pc1))[::-1]
for rank, g in enumerate(sorted_groups[:10]):
    print(f"    group {g} (head {g}): {pc1[g]:.4f}  (rank {rank+1})")


# ─── Test 4: per-layer comparison (do all layers have same head structure?) ───
print(f"\n{'='*70}")
print("TEST 4: do different layers have similar per-head importance?")
print(f"{'='*70}")
per_layer_group_means = []
for layer_idx in sorted(o_scales_per_layer.keys()):
    s = o_scales_per_layer[layer_idx]
    per_layer_group_means.append(s.mean(dim=0).numpy())  # [32]
per_layer_group_means = np.array(per_layer_group_means)  # [n_layers, 32]

# Correlation between layers in their per-head importance pattern
inter_layer_corr = np.corrcoef(per_layer_group_means)  # [n_layers, n_layers]
off_diag = inter_layer_corr[~np.eye(len(inter_layer_corr), dtype=bool)]
print(f"  Mean correlation of per-group importance across layers: {off_diag.mean():.4f}")
print(f"  If high (close to 1): same heads are 'important' in every layer → universal head importance")
print(f"  If low (close to 0): heads have layer-specific importance → no universal pattern")


# ─── Verdict ───
print(f"\n{'='*70}\nVERDICT\n{'='*70}")
verdict = []
if cv_of_means > 0.1:
    verdict.append(f"  ✓ Groups have systematic per-index variation (CV={cv_of_means:.3f})")
    verdict.append(f"    → Some heads are consistently larger than others")
else:
    verdict.append(f"  ✗ Groups are roughly uniform (CV={cv_of_means:.3f})")
    verdict.append(f"    → No clear per-head importance pattern at this aggregation")

if explained_variance[0] > 0.15:
    verdict.append(f"  ✓ PC1 captures {explained_variance[0]*100:.0f}% of variance — strong leading axis")
    verdict.append(f"    → The 32-group scales have low-rank structure consistent with per-head meaning")
else:
    verdict.append(f"  ✗ PC1 captures only {explained_variance[0]*100:.0f}% — no dominant axis")

if off_diag.mean() > 0.3:
    verdict.append(f"  ✓ Layers share head-importance pattern (corr={off_diag.mean():.3f})")
    verdict.append(f"    → Some heads are universally important across layers")
else:
    verdict.append(f"  ✗ Layers have independent head-importance patterns (corr={off_diag.mean():.3f})")
    verdict.append(f"    → Heads matter differently in different layers (still consistent with head structure)")

print("\n".join(verdict))

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "n_rows": n_rows,
        "n_groups": n_groups,
        "qwen3_8b_n_heads": NUM_HEADS_QWEN3_8B,
        "groups_match_heads": n_groups == NUM_HEADS_QWEN3_8B,
        "per_group_mean": group_means.cpu().numpy().tolist(),
        "per_group_std": group_stds.cpu().numpy().tolist(),
        "cv_of_per_group_means": cv_of_means,
        "pc1_loadings": pc1.tolist(),
        "explained_variance_top5": explained_variance[:5].tolist(),
        "inter_layer_correlation_mean": float(off_diag.mean()),
        "inter_layer_correlation_min": float(off_diag.min()),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
