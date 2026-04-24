# Finding 13 — The activation bathtub: residual stream collapses to rank-1 in middle layers, not in weights

## Claim

Across Qwen3-0.6B (measured), and predicted to hold across all standard-architecture LMs, the per-layer hidden state traces a "bathtub" shape in participation ratio (PR):

- **Early layers (L0-2)**: high PR (~30-70) — the model fans out token embeddings into many feature channels
- **Middle layers (L3-L22, the "dead zone")**: PR collapses to ~1 — the residual stream carries a single dominant direction, with magnitude growing 800× through depth
- **Late layers (L22-L27)**: PR rises slowly — features demux back into multiple directions
- **Final layer (L28)**: PR spikes from ~4 to ~27 — dramatic re-expansion for LM head projection

This shape is an **emergent property of activations**, NOT of the weights. Layer weights have roughly uniform PR (~200-700 across all depths). The bathtub arises from the cumulative effect of flat-rank weights acting on an accumulating residual.

## Supporting data (stage 111)

| scale | what's measured | bathtub seen? | Pearson r to scale 1 |
|---|---|---|---|
| 1 — bulk hidden states, many inputs | PR per layer | **yes** | — |
| 2 — weight singular value PR per matrix | weight's effective rank | **no** (flat) | +0.33 |
| 3 — single-sequence trajectory | PR of one input's positions | **yes** | **+0.73** |

Activations are fractal across scales: per-sequence trajectory matches bulk manifold shape with Pearson r = 0.73. Weights do not follow this pattern.

## Physical mechanism

The residual stream undergoes three phases:

**Phase 1 — Fan out (L0-2)**: token embeddings are high-dimensional semantic objects. Early attention + MLP spread information across many feature directions simultaneously. High PR.

**Phase 2 — Transit (L3-L22)**: features are superimposed into a single dominant direction. Like bundling all streams into one carrier. Information is preserved in magnitude/phase trajectory along that direction, not in separate channels. PR = 1.

**Phase 3 — Fan in (L22-L28)**: the single direction gets demultiplexed back into separable components for the LM head projection. PR rises.

The magnitude curve supports this: `||h||` grows monotonically 0.8 → 680 across L0-27, then drops to 103 at L28 after the final RMSNorm. The residual stream is literally accumulating into one direction, then getting decomposed and normalized at the end.

This is a statistical-mechanics prediction: a random walk in residual space with bounded per-step operators will concentrate onto the dominant direction over time. Active edges break this concentration because the input (embedding) and output (vocabulary) projections pull the representation off the dominant axis temporarily.

## Why this matters for compression

**Stage 109 layer-skip asymmetry follows directly**:

- Skipping any active layer (L0-2 or L22-L28): catastrophic (1000×+ worse ppl)
- Skipping any middle layer (L3-L22): broken but not catastrophic (100-1000× worse)

You can't remove middle layers post-hoc even though they look rank-1 in the bulk manifold measurement. The rank-1 stream is carrying a specific magnitude trajectory that downstream layers need. Low PR ≠ no work.

**Stage 112+ implication (not yet tested)**: a compression architecture that respects the bathtub would:

1. Full-rank weights in edge layers (L0-2, L22-28): ~8 layers
2. Rank-1 or near-rank-1 weights in middle layers (L3-L22): ~20 layers
3. Preserve residual magnitude trajectory through the middle

Expected parameter savings if weights in the middle can compress to rank-32 (from rank-1024): `(20 × 32/1024 + 8 × 1) / 28 = ~31%` of full weight storage while respecting role structure. Untested.

## Relation to Anthropic's superposition

This is superposition observed from the other side. Elhage/Olah 2022 predicted features pack into directions of a high-dimensional space. The bathtub shows those directions **collapse to a single axis during transit** and re-spread at the edges. Superposition is depth-varying.

## Cross-model test needed

We have bathtub confirmed on Qwen3-0.6B (stage 111) and implicitly on Qwen3-4B, 14B, 32B via our manifold catalog (results/Qwen_Qwen3-*_manifold.json, though those files are partially damaged from a disk-full incident). Need:

- Formal multi-model confirmation of bathtub at same relative depth
- Phi-2, Mistral, LLaMA, etc. — is bathtub universal or Qwen-family-specific?
- Encoder-decoder models (T5, mBART): does the bathtub appear differently when encoder and decoder are explicit?

## Date + sources

2026-04-23. Stage 111 fractal test. Supporting: stages 107, 109, Finding 10, Finding 12.
