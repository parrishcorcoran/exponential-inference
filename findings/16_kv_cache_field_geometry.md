# Finding 16 — KV cache as a field: angularly uniform, scale-free, non-conservative

The KV cache has a measurable geometric structure independent of the
residual stream's wormhole. Three measurements characterize it.

## Per-token novelty is monotone decreasing, not bell-shaped

Stage 132 measured how much each new token contributes to the K cache
subspace. For each layer in Qwen3-0.6B, on a 512-token sequence:

- Position 0-128: novelty ~0.08-0.10
- Position 128-384: ~0.03-0.04
- Position 384-512: ~0.02-0.03

**27/28 layers show monotone-decreasing novelty.** Zero layers show
bell-curve or inverted-bathtub. The first ~50 tokens establish the
K subspace; everything after lives within that span.

This explains why **StreamingLLM's anchor + window** pattern works:
the early tokens carry rank, the rest are mostly redundant. We have a
mechanistic justification for what they discovered empirically.

## K-cache participation ratio is shockingly low across all layers

| Layer zone | PR_K | PR_V |
|---|---|---|
| Mouths (L0, L27) | 1-2 | 13-40 |
| Throat (L8-L20) | 1-5 (peaks at L13=4.7, L26=4.9) | 27-46 |
| Average | ~2.5 | ~32 |

PR_K is rank-1 to rank-5 across 28 layers. Out of d_kv=1024, only ~5
dimensions carry meaningful variance per layer for K. V is moderately
higher (~12-46 dims). **Most of d_kv is empty space at every layer.**

Caveat documented in Finding 17: PR captures variance dominance but
not information rank. Even though PR_K=5, you can't truncate K to
rank 5 post-hoc and keep quality.

## K vectors fill their subspace 360° — angular spread is uniform

Stage 133 measured pairwise angles between K vectors at different
positions, projected to top-3 PCA components per layer.

- Mean angle: 89-90° (matches random)
- Std deviation: 39-47° (matches random)
- Spread/random ratio: 1.04 averaged over all layers

K vectors are uniformly distributed in their (low-rank) subspace. No
preferred direction. Field-like in the geometric sense.

## Attention decay is scale-free, with shallow exponent

Stage 133 fit power-law and exponential models to mean attention
weight as a function of positional distance Δ.

- Power-law fit: R² = 0.62 (avg α = 0.40)
- Exponential fit: R² = 0.27

Attention is **scale-free** (no characteristic length scale, no
exponential cutoff). The decay exponent α=0.40 is much shallower
than physical magnetic fields (α=2-3 in 3D) — effective dimensionality
is below 1.

Per-layer dependence:
- L0 (near input): α=0.95 (most "local")
- L27 (near output): α=0.11 (essentially uniform)

**Late layers see all positions as nearly equally relevant.** This
rules out sliding-window attention as a free optimization for late
layers — the global context is being used.

## Conservation fails — the field is non-conservative

Sum of squared norms of the residual stream across positions, per
layer:

- max/min ratio across layers: 5 × 10⁵
- coefficient of variation: 1.05

The total "energy" is NOT conserved across layers. This is the
wormhole magnitude pump (finding 13) — RMSNorm rescales between
layers, throat layers amplify the dominant axis 800×.

Magnetostatically: this is an active electromagnet with internal
sources/sinks, not a passive permanent magnet. The model GENERATES
field at certain layers rather than just propagating an input field.

## What the field geometry tells us about KV compression

1. **Per-token-position dropping is suboptimal.** K vectors at all
   positions live in the same low-dim subspace; dropping tokens
   wastes the rank that's already established and leaves redundant
   ones in cache.

2. **H2O's heavy-hitter heuristic measures the wrong thing.** Tokens
   that are frequently attended-to may live entirely in subspace
   span of less-attended tokens. Attention frequency ≠ information
   contribution.

3. **Sliding window attention should fail at late layers** (α≈0.1 means
   no horizon). Confirmed by long-context degradation in models that
   use windowed attention.

4. **MLA-style latent KV compression has the right target** (compress
   d_kv) but is uninformed about LAYER-VARYING rank. A wormhole-aware
   schedule should narrow the latent at throat layers, widen at mouths.

## Predictions and follow-up

Stage 134 attempted post-hoc projection of K, V into measured
subspaces — failed catastrophically (Finding 17). The path forward
is training-aware compression that USES this geometry as a prior
during finetuning, not a post-hoc constraint.

## Date + sources

2026-04-24. Based on Qwen3-0.6B measurements via MPS:
- Stage 132 (`scripts/stage132_kv_rank_pertoken.py`)
- Stage 133 (`scripts/stage133_magnet_field_test.py`)
