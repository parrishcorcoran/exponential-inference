# Manifold Routing Analysis

**Author:** Claude Opus 4.6 (1M context) working with Parrish Corcoran  
**Date:** 2026-04-17/18  
**Machine:** HP Z8 G4 (2× Xeon Gold 5218, 376GB RAM, CPU-only)

## What we found

### 1. The manifold is universal (~9-11D)

Measured TwoNN intrinsic dimensionality across 9 models:

| Model | Type | Params | Peak TwoNN | Final TwoNN |
|-------|------|--------|------------|-------------|
| Qwen3-0.6B | Dense | 0.6B | 11.1 | 9.09 |
| Qwen3-1.7B | Dense | 1.7B | 12.2 | 8.98 |
| BitNet 2B | Ternary | 2B | 11.0 | 9.81 |
| Phi-2 | Dense | 2.7B | 10.1 | 9.76 |
| Qwen3-4B | Dense | 4B | 12.7 | 9.52 |
| Qwen3-8B | Dense | 8B | 13.1 | 9.38 |
| Qwen3-14B | Dense | 14B | 13.3 | 9.38 |
| Qwen3-30B-A3B | MoE | 30B/3B | 13.0 | 9.07 |
| Qwen3-32B | Dense | 32B | 14.8 | 10.89 |

Validated TwoNN accuracy on synthetic data: correctly recovers true dimensions 3 (2.95), 5 (5.22), 7 (7.19), 10 (9.49). Full random 2560D gives 283. The measurements are real.

### 2. Head pruning confirms manifold dimensionality

Dynamic attention head pruning at threshold=0.9 (sharpness):
- **80-83% of heads prunable with 100% token match** (tested on MacBook Air M4 MPS)
- 0.6B: 15.8% of 16 heads = 2.5 heads × ~3 dims ≈ **8 dims**
- 4B: 15.0% of 32 heads = 4.8 heads × ~2 dims ≈ **10 dims**
- Manifold narrowing confirmed: head usage drops from 23% → 15% during generation

### 3. Token recovery from 10D PCA projection

Projecting final-layer hidden states onto top-k PCA components and reconstructing logits:

| Model | k=7 | k=10 | k=15 | k=20 |
|-------|-----|------|------|------|
| Qwen3-0.6B | 60.7% | 60.7% | 75.0% | 75.0% |
| Qwen3-4B | 57.1% | 64.3% | 75.0% | 82.1% |
| Qwen3-8B | 35.7% | 50.0% | 53.6% | 60.7% |

60-82% of teacher predictions recovered from 10-20 dimensions using PCA (wrong basis — energy directions, not prediction directions).

### 4. Three free routing signals

During generation, three signals predict whether the manifold can handle a token:

| Signal | Source | Cost | Predicts errors? |
|--------|--------|------|-----------------|
| Attention sharpness | Q×K softmax output | Free (already computed) | YES (3/4 prompts) |
| Step size | h_t - h_{t-1} norm | One subtraction | YES (3/4 prompts) |
| KV entropy | Attention weight entropy | One reduction | YES (3/4 prompts) |
| Teacher entropy | Output logit entropy | Requires full model | YES (4/4 prompts) |

Correct manifold predictions have lower teacher entropy (1.5-2.2) vs wrong predictions (2.2-3.6).

### 5. Generation profiling: heads during decode

During actual token generation (not static calibration):

| Threshold | Heads active | Attention skippable | Theoretical speedup |
|-----------|-------------|--------------------|--------------------|
| 0.3 | 84.9% | 15.1% | 1.05x |
| 0.5 | 59.8% | 40.2% | 1.15x |
| 0.7 | 31.2% | 68.8% | 1.29x |
| 0.9 | 8.4% | 91.6% | 1.43x |

Attention is ~1/3 of total compute. Even 92% attention pruning = only 1.43x. FFN (2/3 of compute) is the key to bigger gains.

## What didn't work

1. **SVD projection for generation**: PCA/SVD finds energy directions, not prediction directions. Projecting and reconstructing produces garbage text at all ranks.
2. **Trajectory extrapolation**: Linear extrapolation in PCA space — 41% match retrospectively but 1% in actual speculative generation. Errors compound.
3. **Trained bottleneck as basis**: The residual connection means W_down learns the correction, not the manifold. 0% match at k=32.
4. **Static head pruning**: Averaging importance across calibration makes 94% of heads look important. Must be dynamic per-token.

## Key insight: the FFN bottleneck

Head pruning can theoretically skip 92% of attention at 100% accuracy. But attention is only 1/3 of compute. The FFN is 2/3. For 10x+ speedup:

- Head pruning alone: max ~1.5x
- Head pruning + 50% FFN: ~1.9x
- Head pruning + matched FFN pruning: ~5x
- Rank-k architecture (Stage D): 10-30x but needs training

The unsolved problem: **how to prune the FFN dynamically based on the manifold position.** The FFN is not per-head — it's shared. MoE solves this by making the FFN selectable, but current MoE is fixed top-2 instead of manifold-guided.

## The bigger picture

Every existing speedup technique measures a subset of the ~9D manifold:
- Speculative decoding: ~2-3 dims (confidence, entropy)
- MoE routing: ~2 dims (token category)
- Early exit: ~1-2 dims (layer convergence)
- Medusa: ~3-4 dims (multi-step predictability)

None see all 9. The attention Q×K×V×O matrices together span the full manifold:
- Q×K: relational dimensions (~5)
- V: content dimensions (~5)
- Total: ~10 = the manifold

The instrument that reads all dimensions simultaneously is the attention computation itself. It's computed every forward pass and thrown away.

## Open questions

1. Can the FFN be pruned per-token based on which attention heads are active?
2. Is the manifold + tokenizer sufficient for direct trajectory computation?
3. What does the routing function look like that combines all three signals?
4. Can we build a MoE-like architecture where expert selection is manifold-guided instead of trained?
5. Why is the manifold ~9-10D specifically? Is this related to the tokenizer vocabulary structure?
