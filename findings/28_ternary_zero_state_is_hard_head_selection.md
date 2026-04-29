# Finding 28: BitNet ternary's "0" state IS hard per-head selection

## Summary

The BitNet b1.58 → BitNet 1.0 quality gap (ternary works at LM scale,
binary doesn't) is mechanistically explained by what happens to the
attention output projection's per-head structure under each
quantization scheme.

Across our measurements:
- Qwen3-0.6B FP base o_proj per-head PC1: **42.8%**
- BitNet b1.58 FP master o_proj PC1:      **37.4%**
- BitNet b1.58 *ternary effective* PC1:   **73.3%**  ← amplified
- Bonsai-8B 1-bit o_proj PC1:             **35.3%**  (slightly diluted)

**Ternary projection AMPLIFIES the per-head selection pattern. Binary
projection PRESERVES OR DILUTES it.** This difference explains why
ternary matches FP at LM scale and binary doesn't.

## The mechanism

Pretrained transformers naturally encode per-row × per-head importance
in `o_proj` (Stage 174 measured PC1=43% intrinsic to FP weights). Each
output channel of `o_proj` cares about a *subset* of the attention
heads, and the "off" heads contribute near-zero magnitude.

What each quantization scheme does to this structure:

### Pure FP (no quant)
Magnitude variation across head boundaries encodes per-head importance.
Each head's contribution = `||W_row[head_dim*h : head_dim*(h+1)]||`.
Some heads have small contributions (~0), others large.

### Ternary {-1, 0, +1} × γ
Threshold-based: weights below `0.7 × abs(W).mean()` go to 0; rest go
to `±γ`. The "off" heads — whose weights are mostly small — get their
weights collapsed to all-zero. The "on" heads — whose weights are
mostly large — get their weights crystallized to ±γ.

Net effect: per-head selection becomes BINARY-PER-HEAD. A head is
either "fully on" (contributes magnitude γ) or "fully off"
(contributes 0). This is HARDER selection than FP magnitudes encode,
which produced PC1=73%.

### Binary {-1, +1} (no zero)
Sign quantization with a single per-row scale. Every weight is `±α`.
There's NO "off" state — the head must be assigned a sign even if its
weights were mostly small. This DESTROYS the per-head selection
mechanism and replaces it with arbitrary sign patterns. PC1 drops
slightly (35% in Bonsai vs 42% in FP base) because the structure is
preserved at the magnitude level (per-row scales) but the per-head
"off" capability is gone.

### Binary + per-group scale (Bonsai)
Bonsai's per-128-weight groups (= head_dim) gets separate FP scales,
so it CAN have small scale per group → effectively "off" head, even
though every individual weight is ±sign. This is why Bonsai works at
all (11% loss) instead of being totally broken: the per-group scale
recovers some of the per-head selection.

## Why ternary works at LM scale

BitNet b1.58 trained the master weights with ternary forward (QAT).
The master weights drift to a configuration where:
- Heads that should be "off" for a particular output channel have
  their corresponding weights all small (collapse to 0 under threshold)
- Heads that should be "on" have large weights (become ±γ)

This produces PC1=73% — the model has been TRAINED INTO a state where
per-head selection is sharper than FP. The ternary structure isn't
fighting the model; it's a closer match to the per-head importance
the model wants to encode.

## Why pure binary breaks at LM scale

BitNet 1.0 (binary, no zero state) quality dropped because:
- Each weight must be ±α — no "off" mechanism per weight
- To express "head h is off for output i", the row's per-head block
  must have weights that average out to zero
- This requires a specific sign pattern that's harder to find via
  gradient descent than ternary's "all-zero" easy escape
- Quality penalty grows with model scale because attention's per-head
  structure becomes more important

## Implications for our recipe

If we want pure binary on a pretrained transformer to work, we need
either:

1. **Per-head block-α** (one scalar per head per row, like Bonsai's
   per-128-group scales). Provides explicit "off" per head. ~1.5×
   the parameter count of plain α but recovers the head-selection
   capability.

2. **QAT during binary anneal** that flattens per-head importance
   first. The master weights move to a configuration where every head
   contributes ~uniformly to every output channel. After flattening,
   pure binary is sufficient because there's no per-head selection
   to encode. Cost: model loses some specialization, but α-only is
   then enough.

3. **Use ternary instead of binary** for our compound recipe. The
   "0" state IS the head-selection mechanism. 1.58 bits/weight is
   only 0.58 bits more than binary, but functionally captures
   per-head structure for free.

The cleanest path is probably (3) for the first compound — match
BitNet b1.58 exactly but on top of our nGPT-shape conversion. The
o_proj's per-head structure aligns with ternary's "0" state, and our
recipe contributes the unit-norm normalization that nGPT validates
at smaller scale.

## Stages this finding draws from

- Stage 173: Bonsai's o_proj has per-row × per-head selection structure (PC1 35%)
- Stage 174: structure is intrinsic to FP base (PC1 43% in Qwen3-0.6B)
- Stage 175a: **ternary AMPLIFIES the structure (PC1 73% in BitNet ternary)**
- Stage 175b: structure weakens with model scale (PC1 25% in Qwen3-4B)
