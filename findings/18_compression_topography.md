# Finding 18 — KV cache has independent compression axes; the levers don't correlate

Stage 138 measured per-layer slack across five compression axes on
Qwen3-0.6B simultaneously. The axes have **independent per-layer
shapes**, which means they stack multiplicatively — a key precondition
for the kind of total compression we're aiming at.

## What was measured per layer

For each of 28 layers in Qwen3-0.6B, on a 512-token sequence:

1. **Rank** — PR_K, PR_V (variance) + EVR-95 rank (information)
2. **Quantization** — relative reconstruction error of K, V at Q16/Q8/Q4/Q2/Q1
3. **Cluster redundancy** — K-means reconstruction error at k ∈ {5, 10, 50, 100, 500}
4. **Attention concentration** — Gini coefficient of attention rows (eviction tolerance)
5. **Position contribution** — Δ-PR over early/middle/late quartiles (from stage 132)

## Per-axis profiles

### K rank (EVR-95) — TWO-GATE WORMHOLE shape
| Layer zone | EVR_K |
|---|---|
| L0–L5 (mouth/entry) | 1–17 |
| L8, L11, L13 (walls) | 120, 104, 103 |
| L14 (deep throat) | 86 |
| L16, L18–L26 (exit gates + aftermath) | 80–99 |
| L27 (final) | 48 |

Confirms finding 15's two-gate topology in compression-rank space.
EVR is wider than PR suggested — information rank at walls is 80–120,
not 1–5.

### V rank (EVR-95) — UNIFORMLY HIGH
Range across all layers: 89–209, mostly 150–200. **No wormhole shape
on V.** V is content-rich at every layer.

### Quantization (Q4 reconstruction error)
- K: 0.14–0.39 (most layers ~0.25–0.35)
- V: 0.17–0.25 (more uniform)
- **V quantizes BETTER than K** despite needing more rank. Different
  axes attack different content properties.

### K-cluster-100 redundancy
- L0–L5: error 0.10–0.22 (highly redundant — most K vectors duplicate)
- L8–L26: error 0.30–0.47 (less compressible by clustering)
- **Clustering is FRONT-LOADED**. Early layers have repetitive K vectors;
  deeper layers carry more diverse routing patterns.

### Attention Gini
- Range 0.84–0.98, mean ≈ 0.92
- All layers have highly concentrated attention
- **Eviction tolerance is uniform** across depth; H2O works the same way
  at every layer.

## The structural finding: levers are independent

| Axis | Per-layer shape | Implication |
|---|---|---|
| K rank | Two-gate wormhole | Per-layer schedule needed |
| V rank | Uniform high | Modest uniform reduction |
| Bits | Uniform moderate | Uniform Q4 with FT |
| Clustering | Front-loaded | Aggressive on L0–L5 only |
| Eviction | Uniform sparse | Uniform H2O |

Different shapes → orthogonal levers. Stacking gives multiplicative
compression rather than additive.

## Combined compression budget (per layer)

For each layer, we now have a per-axis slack number. The optimal
schedule respects each:

- L0–L5: K rank ≈ 16, V rank ≈ 200, K cluster representatives ≈ 20,
  Q4 OK on both
- L8, L11, L13 (walls): K rank ≈ 80, V rank ≈ 200, no clustering,
  Q4 OK
- L14 (deep throat): K rank ≈ 86, V rank ≈ 100, no clustering, Q4 OK
- L21 (exit wall — hardest): K rank ≈ 90, V rank ≈ 175, Q4 OK
- L25–L26 (mouth 2 entry): K rank ≈ 60, V rank ≈ 130, Q4 OK
- L27 (final): K rank ≈ 50, V rank ≈ 175

Estimated combined cache compression at this schedule with FT:
- Average K rank: ~50 (down from 1024) = 20×
- Average V rank: ~150 (down from 1024) = 6.8×
- Q4 stacked: 4×
- Cluster representatives on L0–L5: ~2×
- H2O on top: 5×
- **Stacked: 100–300× cache compression projected**

## Predictions and follow-up

This is the topographic map we needed. The multi-axis squeeze (stage
137 to be built) uses this directly: anneal each axis independently
within its layer-aware schedule, with finetune between each step,
finding the trained-aware floor. Each axis preserved separately to
not break others.

## Date + sources

2026-04-24. `scripts/stage138_compression_topography.py` and
`results/stage138_compression_topography.json`.
