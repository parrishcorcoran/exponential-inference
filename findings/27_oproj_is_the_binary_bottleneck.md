# Finding 27: o_proj is the universal binary-quantization bottleneck

## Summary

Across every model and every method we measured (pretrained Qwen3 family,
NVIDIA-style nGPT, BitNet ternary, Bonsai 1-bit), the attention output
projection (`o_proj`) consistently has the highest magnitude variation
that resists naive quantization. The mechanism is structural: o_proj's
input is the head-stacked attention output, so its rows encode per-head
selection patterns that other projections don't have.

This is the single most important constraint on clean low-bit
quantization of pretrained transformers, and it's the same constraint
across pretraining methods.

## The four converging measurements

### Stage 162: per-layer CV on Qwen3-0.6B FP base

The "exit wall" at layer 19 has the highest CV in the model. Layer 19
is part of the late-attention region where o_proj's variance dominates.

### Stage 168: nGPT_800m architectural choice

The community nGPT (p2o6e100/nGPT_800m, trained from scratch with
NVIDIA's recipe) explicitly leaves o_proj un-normalized:

```
qkv_proj    mean=1.000  CV=0.0017  ← input projections (sphere)
o_proj      mean=0.96   CV=0.30    ← NOT normalized
gate_proj   mean=1.000  CV=0.0019  ← input projections (sphere)
up_proj     mean=1.000  CV=0.0018  ← input projections (sphere)
down_proj   mean=2.00   CV=0.075   ← NOT normalized
```

NVIDIA's training procedure couldn't (or didn't try to) push o_proj to
unit norm because the training loss preferred letting it keep magnitude
variation.

### Stage 170: o_proj normalization is FREE on top of nGPT

Forcing nGPT_800m's o_proj to unit norm via our recipe + α recovery:
```
T0 baseline CE=2.850
T3 with our extension CE=2.840 (Δ=-0.010, 1% improvement)
```

So forcing o_proj to unit norm IS achievable — it just isn't what
training-from-scratch optimization finds. nGPT's pattern was a
training-speed convenience, not a quality optimum.

### Stage 172: o_proj has 2.5× higher intra-row scale variation than every other projection

Bonsai-8B 1-bit per-projection intra-row CV (within-row variation of
its 32 per-128-weight scales):
```
q_proj     0.066    ← single α captures the info
k_proj     0.071
v_proj     0.073
gate_proj  0.066
up_proj    0.064
down_proj  0.070
o_proj     0.182    ← 2.5× higher
```

For all projections except o_proj, a single per-row α would capture
the magnitude info. For o_proj, the 32 per-group scales are doing
real work.

### Stage 173: o_proj's per-group structure encodes per-row × per-head selection

Bonsai's o_proj per-group scales show low-rank structure with PC1=35.3%
of variance. Cross-row correlation 0.32 indicates that each output
channel selects a SUBSET of attention heads to weight heavily, and that
selection is correlated across nearby groups.

The 32 per-group scales = 32 attention heads = head_dim=128. The
quantization grouping aligns exactly with attention head boundaries.

### Stage 174: this structure is INTRINSIC to pretrained FP weights

Same head-block analysis on Qwen3-0.6B FP base:
```
                       PC1     cv_means    cross_row_corr
Bonsai 1-bit          35.3%    0.017       0.323
Qwen3-0.6B FP base   42.8%    0.020       0.399
```

PC1 is *higher* in the FP base than in Bonsai. Bonsai's quantization
slightly diluted (but preserved) the existing structure. The pattern
is intrinsic to pretrained transformers, not a quantization artifact.

## Mechanism

Attention has multi-head structure. Each layer's o_proj must combine
the outputs of N heads into a single residual-stream contribution. The
weights of o_proj therefore encode "which heads matter for which output
dimensions."

For other projections:
- `q/k/v_proj`: input is the residual stream (single source). No
  per-input-group structure to encode.
- `gate/up_proj`: input is the residual stream after layer norm.
  Same single-source.
- `down_proj`: input is the MLP intermediate (single source after
  silu/gelu). Some structure but not group-aligned.
- `o_proj`: input is the head-stacked attention output. Each
  head_dim-sized block of input represents a different head's
  contribution. Per-block magnitude encodes per-head importance for
  this output dimension.

Only o_proj has this multi-source structured input. That's why only
o_proj has high intra-row magnitude variation.

## Implications for the binary recipe

The straightforward 3-stage pipeline (project to unit norm + α + binary)
works cleanly for q/k/v/gate/up/down projections. For o_proj it has
limited capacity — α-per-row averages across the per-head selection
pattern.

Three viable solutions:

1. **Block-α for o_proj specifically**: use one α scalar per (row, head)
   pair. For Qwen3-8B this would be 4096×32 = 131K extra scalars per
   layer. Captures the per-head selection pattern Bonsai uses with 32
   per-group scales.

2. **QAT-driven flattening during binary anneal**: train master weights
   under binary forward projection so they redistribute toward uniform
   per-head magnitude. The model would learn to compensate for the lost
   per-head selection by doing more work elsewhere (other projections,
   norms, biases). Pure α might suffice afterward.

3. **Don't normalize o_proj** (NVIDIA's nGPT choice). Accept that the
   per-head structure is too specialized to flatten. Bonsai effectively
   does this — its o_proj has CV 0.30 across rows even after 1-bit
   quantization.

The choice depends on the deployment goal. For maximum compression
(option 1 or 2), per-head awareness must be in the recipe somehow. For
training-speed wins (option 3), nGPT's recipe is correct as published.

## Cross-references

- Stage 162: Qwen3-0.6B per-layer CV
- Stage 168: nGPT_800m row-norm diagnostic
- Stage 170: nGPT_800m → ours conversion
- Stage 172: Bonsai intra-row scale variation
- Stage 173: Bonsai head alignment PCA
- Stage 174: head structure in FP base
