"""Test whether the Stage 77 carry-flip phase-conjugate pair structure exists
in the WEIGHT rows (not just attribution patterns).

Stage 77 found pair-structure (cos ≈ -1 or cos ≈ 0) in token attribution
clusters — the "5+5 hologram hypothesis". User wants to know if the same
structure exists in weight rows of our model. If so, it's exploitable for
sub-bit quantization (paired rows share signs).

Method:
  For each Linear in FP Qwen3-0.6B:
    1. Take weight matrix W [out, in]
    2. Optionally L2-normalize rows (we want directional structure, not magnitude)
    3. K-means cluster the rows, sweep K ∈ [2, 4, 8, 16, 32]
    4. Compute cosine matrix between centroids
    5. Count anti-pairs (cos < -0.5) and orthogonal pairs (|cos| < 0.2)
    6. Compare to a random-baseline (shuffled rows) to gauge significance

Output: per-Linear and per-K stats. If we see consistent anti-pair structure
beyond random baseline, the holographic hypothesis applies to weights.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from transformers import AutoModelForCausalLM

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


CHECKPOINT = "Qwen/Qwen3-0.6B"
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
K_VALUES = [2, 4, 8, 16, 32]
N_RANDOM_BASELINE_TRIALS = 5
RESULTS_PATH = Path("results/carry_flip_weight_structure.json")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.float32   # need f32 for K-means stability
elif torch.backends.mps.is_available():
    device = "cpu"; dtype = torch.float32   # K-means on cpu, simpler
else:
    device = "cpu"; dtype = torch.float32


def cosine_matrix(centroids):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    unit = centroids / norms
    return unit @ unit.T


def pair_stats(cos_mat):
    """Count anti-pairs (cos < -0.5) and orthogonal pairs (|cos| < 0.2),
    excluding diagonal."""
    n = cos_mat.shape[0]
    off_diag = cos_mat[~np.eye(n, dtype=bool)]
    anti = int(((cos_mat < -0.5) & (~np.eye(n, dtype=bool))).sum() // 2)
    ortho = int(((np.abs(cos_mat) < 0.2) & (~np.eye(n, dtype=bool))).sum() // 2)
    pos = int(((cos_mat > 0.5) & (~np.eye(n, dtype=bool))).sum() // 2)
    return {
        "anti_pairs_below_neg_0.5": anti,
        "ortho_pairs_below_abs_0.2": ortho,
        "pos_pairs_above_0.5": pos,
        "off_diag_min": float(off_diag.min()),
        "off_diag_max": float(off_diag.max()),
        "off_diag_mean": float(off_diag.mean()),
    }


def random_baseline(W, K, n_trials=N_RANDOM_BASELINE_TRIALS, seed=42):
    """Random-shuffled-row baseline: cluster a row-shuffled version of W,
    compute pair stats. Average over trials. Tests whether the original
    structure is significant beyond what random gives."""
    rng = np.random.default_rng(seed)
    anti_counts = []
    ortho_counts = []
    for _ in range(n_trials):
        W_shuf = W[rng.permutation(W.shape[0])]
        km = KMeans(n_clusters=K, n_init=4, random_state=int(rng.integers(0, 1<<30)))
        km.fit(W_shuf)
        cos_mat = cosine_matrix(km.cluster_centers_)
        stats = pair_stats(cos_mat)
        anti_counts.append(stats["anti_pairs_below_neg_0.5"])
        ortho_counts.append(stats["ortho_pairs_below_abs_0.2"])
    return {
        "anti_mean": float(np.mean(anti_counts)),
        "anti_std": float(np.std(anti_counts)),
        "ortho_mean": float(np.mean(ortho_counts)),
        "ortho_std": float(np.std(ortho_counts)),
    }


def analyze_linear(name, W_tensor, K_values=K_VALUES):
    """For a single Linear's weight matrix, sweep K and report structure stats."""
    W = W_tensor.detach().cpu().float().numpy()   # [out, in]
    out_f, in_f = W.shape
    print(f"\n--- {name}  shape=[{out_f}, {in_f}] ---", flush=True)

    # L2-normalize each row → directional analysis (magnitude factored out)
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    W_unit = W / norms

    per_K_results = []
    for K in K_values:
        if K >= out_f: break
        km = KMeans(n_clusters=K, n_init=10, random_state=42)
        km.fit(W_unit)
        # silhouette can be expensive for large out_f; sample if needed
        if out_f > 5000:
            sample_idx = np.random.default_rng(42).choice(out_f, 5000, replace=False)
            sil = silhouette_score(W_unit[sample_idx], km.labels_[sample_idx])
        else:
            sil = silhouette_score(W_unit, km.labels_)
        cos_mat = cosine_matrix(km.cluster_centers_)
        actual_stats = pair_stats(cos_mat)
        baseline_stats = random_baseline(W_unit, K)

        # Significance: how many sigmas above baseline mean?
        anti_z = ((actual_stats["anti_pairs_below_neg_0.5"] -
                   baseline_stats["anti_mean"]) /
                  max(baseline_stats["anti_std"], 0.5))
        ortho_z = ((actual_stats["ortho_pairs_below_abs_0.2"] -
                    baseline_stats["ortho_mean"]) /
                   max(baseline_stats["ortho_std"], 0.5))

        per_K_results.append({
            "K": K,
            "silhouette": float(sil),
            "actual": actual_stats,
            "baseline": baseline_stats,
            "anti_z_above_baseline": float(anti_z),
            "ortho_z_above_baseline": float(ortho_z),
        })
        print(f"  K={K:>3}  sil={sil:+.3f}  "
              f"anti={actual_stats['anti_pairs_below_neg_0.5']:>3} "
              f"(baseline {baseline_stats['anti_mean']:.1f}±{baseline_stats['anti_std']:.1f}, "
              f"z={anti_z:+.2f})  "
              f"ortho={actual_stats['ortho_pairs_below_abs_0.2']:>3} "
              f"(baseline {baseline_stats['ortho_mean']:.1f}±{baseline_stats['ortho_std']:.1f}, "
              f"z={ortho_z:+.2f})  "
              f"cos_range=[{actual_stats['off_diag_min']:+.2f}, {actual_stats['off_diag_max']:+.2f}]",
              flush=True)

    # Best K by silhouette
    best = max(per_K_results, key=lambda r: r["silhouette"])
    print(f"  best K={best['K']}  silhouette={best['silhouette']:+.3f}", flush=True)
    return {
        "name": name,
        "shape": [out_f, in_f],
        "per_K": per_K_results,
        "best_K": best["K"],
        "best_silhouette": best["silhouette"],
    }


print(f"device={device}", flush=True)
print(f"Loading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
).eval()

# Sample a few Linears representative of each type, plus all 7 types in layer 0
# For full coverage on Mac, run on a subset (each KMeans takes ~5-15s on 1024-d rows)
print("\nSelecting representative Linears...", flush=True)
target_modules = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    # Only target the 7 attention/MLP types
    if not any(name.endswith(s) for s in TARGET_NAMES): continue
    target_modules.append((name, mod))

# Strategy: test 2 layers (layer 0 and layer ~middle) × 7 types = 14 Linears
# For Qwen3-0.6B with 28 layers, take layer 0 and layer 14
selected = []
for name, mod in target_modules:
    if "layers.0." in name or "layers.13." in name:
        selected.append((name, mod))

print(f"  Selected {len(selected)} Linears for analysis", flush=True)
for name, _ in selected:
    print(f"    {name}", flush=True)

results = []
for name, mod in selected:
    res = analyze_linear(name, mod.weight)
    results.append(res)


# ─── Aggregate signal ───
print(f"\n{'═'*60}")
print(f"AGGREGATE: pair-structure significance across {len(results)} Linears")
print('═'*60)
# Average z-score across all (linear, K) pairs
all_anti_z = []
all_ortho_z = []
for r in results:
    for pk in r["per_K"]:
        all_anti_z.append(pk["anti_z_above_baseline"])
        all_ortho_z.append(pk["ortho_z_above_baseline"])
print(f"  anti-pair z-score:   mean={np.mean(all_anti_z):+.2f}  std={np.std(all_anti_z):.2f}  "
      f"max={np.max(all_anti_z):+.2f}", flush=True)
print(f"  ortho-pair z-score:  mean={np.mean(all_ortho_z):+.2f}  std={np.std(all_ortho_z):.2f}  "
      f"max={np.max(all_ortho_z):+.2f}", flush=True)

# Verdict
mean_anti_z = float(np.mean(all_anti_z))
mean_ortho_z = float(np.mean(all_ortho_z))
print(f"\n  VERDICT:")
if mean_anti_z > 1.5 or mean_ortho_z > 1.5:
    print(f"    → STRUCTURE PRESENT: weights show pair-clustering above random baseline.")
    print(f"      This means the carry-flip 5+5 hologram hypothesis applies to weights.")
    print(f"      Exploitable: paired rows could share sign+flip-bit, sub-bit encoding possible.")
else:
    print(f"    → No significant pair-structure detected (z < 1.5 mean).")
    print(f"      Weight rows do NOT show holographic phase-conjugate clustering.")
    print(f"      Stage 77's pattern was specific to attribution; weights are different.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "K_values": K_VALUES,
        "n_random_baseline_trials": N_RANDOM_BASELINE_TRIALS,
        "linears_analyzed": [r["name"] for r in results],
        "results": results,
        "aggregate": {
            "anti_z_mean": mean_anti_z,
            "anti_z_max": float(np.max(all_anti_z)),
            "ortho_z_mean": mean_ortho_z,
            "ortho_z_max": float(np.max(all_ortho_z)),
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
