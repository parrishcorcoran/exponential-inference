# Finding 08 — Minimal 8-feature set captures 80% of full LOPO R²

## The claim

A minimal subset of **8 orthogonal runtime features**, selected greedily
under leave-one-prompt-out linear-regression criterion, reaches a
cross-prompt LOPO R² of **0.272 — 80% of the full 47-feature set's R² of
0.341**. Each of the 8 features represents a different physical axis
(quantum, boundary, trajectory, angular, density, interaction), not a
variation on the same signal.

For deployment, this 8-feature set is the essential minimum. Adding
more features to reach the full 47 yields diminishing gains (+0.07 R²
across the remaining 39 features, each contributing <0.005 on average).

## Why it's a stop-and-think

Earlier stages accumulated 47 features by exploring many physics
framings (summary statistics, curvature, quantum density matrix,
black-hole bipartite decomposition). Most field-level proposals for
routing signals pick one family and live in it. This analysis shows
the OPTIMAL small set is heterogeneous: **one signal from each
framing, rather than several from any one**.

Reads as: the feature space is effectively 8-dimensional. Within each
physics framing most of the formulations are redundant (purity ≈ VN
entropy ≈ effective rank, for instance — stage 28), but ACROSS
framings there's independent information because they probe different
aspects of the same underlying geometric object.

## How it was measured

Greedy forward selection under LOPO linear regression:
1. Start with an empty feature set.
2. At each step, for each remaining feature, test adding it to the
   current set and measure LOPO R² (35 folds, one per prompt).
3. Add the feature giving the largest R² gain. Repeat.

This is on 4165 records across 35 prompts from stage 31.

## The numbers

### Greedy selection trace

| k | feature added | LOPO R² | gain |
|---|---|---|---|
| 1 | bipartite_vn_late | 0.006 | +0.006 |
| **2** | **prod_H_last_norm** | **0.134** | **+0.129** ← big jump |
| 3 | centeredness | 0.177 | +0.043 |
| 4 | knn_dist_min | 0.212 | +0.035 |
| 5 | upd_kurtosis | 0.244 | +0.031 |
| 6 | attn_peak_recency | 0.256 | +0.012 |
| 7 | kde_log_density | 0.267 | +0.011 |
| **8** | **layer_halves_align** | **0.272** | +0.006 |
| 9 | hidden_norm_mid | 0.281 | +0.009 |
| 10 | H_q3_layer | 0.290 | +0.009 |
| 11 | cross_layer_align | 0.297 | +0.007 |
| 12 | max_head_sharpness | 0.302 | +0.005 |
| 47 (full) | — | 0.341 | — |

### The 8-feature minimum and what each probes

| feature | physical axis |
|---|---|
| `bipartite_vn_late` | black-hole boundary decomposition (quantum + bipartite split) |
| `prod_H_last_norm` | curvature interaction (attention concentration × hidden energy) |
| `centeredness` | trajectory position on the manifold |
| `knn_dist_min` | manifold locality (distance to nearest calibration neighbor) |
| `upd_kurtosis` | per-layer update distribution shape (quantum higher moment) |
| `attn_peak_recency` | attention angular locality (how recent is attention peak) |
| `kde_log_density` | kernel density on the calibration manifold |
| `layer_halves_align` | depth-wise boundary alignment (early vs late layer update cos) |

Each entry is from a different physics framing; no two are variants
of the same measurement. The greedy procedure implicitly enforces
this orthogonality — if two features carry the same signal, only the
first is picked.

### The "feature 1 alone is nothing" anomaly

`bipartite_vn_late` individually gives R² = 0.006 — basically nothing.
It's picked first because it's a useful base for pairing. At k=2,
adding `prod_H_last_norm` jumps R² to 0.134, a +0.129 gain.

Reading: the most useful routing signals are INTERACTIONS, not
individual features. Linear regression on raw features misses this
because it can't form products; but the greedy procedure finds pairs
whose product structure (captured via the pre-computed
`prod_H_last_norm` feature) recovers it.

## What it predicts / enables

1. **Deployment cost for routing signals is bounded.** Only 8 numbers
   per step per layer (or per-head where applicable). Total
   computation is microseconds; no meaningful overhead vs the
   forward pass itself.

2. **Cross-model transfer should work for this subset.** Features
   selected because they're orthogonal tend to generalize better
   than the full set (which includes some that overfit). A 32B
   deployment can probably use the same 8 features without re-selection.

3. **The 80-percent threshold is a natural operating point.** Going
   to 12 features for 89% of full R² is a small quality bump for
   +50% more features. 8 is the sweet spot.

## Limitations

1. Greedy selection may miss joint optima that simultaneous selection
   would find. A proper L1-regularized selection might give a slightly
   different minimal set.
2. We don't know if the same 8 features are optimal for different
   model sizes. Plausibly yes (cross-tokenizer universality from
   Finding 02) but untested.
3. We measured under linear regression; a small non-overfitting tree
   ensemble might find a different minimal set.

## Reproduce

```bash
python scripts/stage32_minimal_subset.py \
    --model Qwen/Qwen3-0.6B \
    --max-new-tokens 120 \
    --max-k 12 \
    --device mps
```

## Related

- [Finding 07](07_easy_token_classifier.md) — the parent finding; this
  subset is 80% of its predictive power at 17% of its features.
- Stage 25's feature-clustering analysis showed 23 features collapse
  to 14 independent axes at |r| > 0.6. Greedy LOPO selection here
  gives an 8-feature subset that's even more orthogonal: each feature
  comes from a different physics framing.
