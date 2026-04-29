"""Stage 174: Per-head structure in o_proj of FP base Qwen3 models.

Bonsai's o_proj quantization showed per-row × per-head importance with
low-rank structure (PC1 = 35% of variance). Question: is this structure
already in the FP base model, or does Bonsai's per-group quantization
create it?

For each base Qwen3 model, reshape o_proj to [out, n_heads, head_dim]
and compute per-(row, head) magnitude. Then PCA across the n_heads
dimension. If similar low-rank structure (PC1 > 25%), the per-head
selection pattern is intrinsic to pretrained transformers, not an
artifact of quantization.

Tests:
  Qwen3-0.6B  (hidden=1024, n_heads=16, head_dim=64)
  Qwen3-4B    (hidden=2560, n_heads=32, head_dim=128 if equal head_dim;
              else compute from config)
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig


CHECKPOINTS = ["Qwen/Qwen3-0.6B"]  # 4B/14B/Bonsai compared separately
RESULTS_PATH = Path("results/stage174_head_structure_fp_base.json")


def analyze_o_proj_head_structure(model_id):
    print(f"\n{'='*70}\n{model_id}\n{'='*70}")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = hidden // n_heads
    print(f"  hidden={hidden}, n_heads={n_heads}, head_dim={head_dim}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    ).eval()

    # Aggregate per-row × per-head magnitudes across all layers
    all_per_head = []
    matched = 0
    for name, mod in model.named_modules():
        if "o_proj" in name and hasattr(mod, "weight") and mod.weight is not None and mod.weight.ndim == 2:
            W = mod.weight.detach().float()
            if W.shape[1] % n_heads != 0:
                continue
            actual_head_dim = W.shape[1] // n_heads
            W_reshaped = W.reshape(W.shape[0], n_heads, actual_head_dim)
            per_head_mag = W_reshaped.norm(dim=-1)  # [out, n_heads]
            all_per_head.append(per_head_mag.cpu().numpy())
            matched += 1
    print(f"  matched {matched} o_proj layers")

    aggregated = np.concatenate(all_per_head, axis=0)  # [n_rows_total, n_heads]
    print(f"  aggregated shape: {aggregated.shape}")

    # Per-group (per-head) means
    group_means = aggregated.mean(axis=0)  # [n_heads]
    group_stds = aggregated.std(axis=0)
    cv_of_means = group_means.std() / group_means.mean()
    print(f"  per-head mean (across rows): {group_means.mean():.4f} ± {group_means.std():.4f}")
    print(f"  CV of per-head means: {cv_of_means:.4f}")

    # Cross-row correlation
    sample = aggregated[np.random.permutation(aggregated.shape[0])[:2000]]
    corr = np.corrcoef(sample.T)
    off_diag = corr[~np.eye(n_heads, dtype=bool)]
    print(f"  mean off-diag correlation: {off_diag.mean():.4f}")

    # PCA
    centered = sample - sample.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()
    print(f"  PC1 explained variance: {explained[0]:.4f} ({explained[0]*100:.1f}%)")
    print(f"  Top 3 PCs: {[f'{e*100:.1f}%' for e in explained[:3]]}")

    # Per-layer head correlation
    per_layer_means = np.array([m.mean(axis=0) for m in all_per_head])  # [n_layers, n_heads]
    inter_layer_corr = np.corrcoef(per_layer_means)
    inter_off = inter_layer_corr[~np.eye(len(inter_layer_corr), dtype=bool)]
    print(f"  cross-layer head-importance correlation: {inter_off.mean():.4f}")

    del model
    return {
        "model": model_id,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "n_rows_total": int(aggregated.shape[0]),
        "per_head_mean_overall": float(group_means.mean()),
        "per_head_cv_of_means": float(cv_of_means),
        "cross_row_corr_mean": float(off_diag.mean()),
        "pc1_explained": float(explained[0]),
        "pc2_explained": float(explained[1]),
        "pc3_explained": float(explained[2]),
        "cross_layer_head_corr": float(inter_off.mean()),
    }


results = []
for ckpt in CHECKPOINTS:
    try:
        r = analyze_o_proj_head_structure(ckpt)
        results.append(r)
    except Exception as e:
        print(f"  ERROR on {ckpt}: {e}")


# Compare across models
print(f"\n{'='*70}\nCOMPARISON: per-head structure in o_proj across models\n{'='*70}")
print(f"\n  Bonsai-8B 1-bit (already measured):")
print(f"    cv_of_means: 0.017")
print(f"    cross_row_corr: 0.323")
print(f"    PC1: 35.3%")
print(f"    cross_layer_corr: -0.005")

for r in results:
    print(f"\n  {r['model']}:")
    print(f"    cv_of_means: {r['per_head_cv_of_means']:.4f}")
    print(f"    cross_row_corr: {r['cross_row_corr_mean']:.4f}")
    print(f"    PC1: {r['pc1_explained']*100:.1f}%")
    print(f"    cross_layer_corr: {r['cross_layer_head_corr']:.4f}")

print(f"\n{'='*70}\nINTERPRETATION\n{'='*70}")
print(f"  Hypothesis: per-row head-selection structure pre-exists in FP weights")
print(f"  (would mean Bonsai's quantization preserved an existing structure,")
print(f"   not created it).")
print(f"")
print(f"  Look for FP-base PC1 > 25% to support hypothesis.")

with open(RESULTS_PATH, "w") as f:
    json.dump({"models": results,
               "bonsai_reference": {"pc1": 0.353, "cv_of_means": 0.017,
                                    "cross_row_corr": 0.323}}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
