# Finding 14 — Universal throat geometry, private mouth-2 decoder

Three converging measurements (stages 121, 122c, 123b) nail down a
cross-model structure:

**The residual stream's throat coordinate system is universal across
same-tokenizer models. The mouth-2 decoder calibrated on top of it is
model-private and brittle.**

This splits the wormhole into a shared geometric part and a private
functional part, and immediately explains speculative decoding,
Medusa, and early exit.

## The three measurements

### Stage 121 — throat coords align cross-model (linear R²)

Direct linear-regression alignment of per-sentence mean states between
Qwen3-0.6B and Qwen3-1.7B at five normalized depths:

| pos  | R²   | interpretation |
|------|------|---|
| 0.10 | 0.08 | early mouth — looks private |
| 0.25 | 0.94 | throat entry — nearly universal |
| 0.50 | 0.91 | deep throat — nearly universal |
| 0.75 | 0.68 | throat exit |
| 0.90 | 0.21 | late mouth — looks private |

The throat-entry R² = 0.94 is the headline. The low mouth R² was
initially read as "mouths are private." That reading was wrong and
corrected by stage 122c.

### Stage 122c — PCA-aligned CCA reveals shared manifold at all depths

Per-token states (N = 2031 tokens) projected to top-200 PCA
components per model, then canonical-correlation-analyzed:

| pos  | % PCs > 0.7 | cliff @ 0.5 (real / shuffle) |
|------|-------------|-----------------|
| 0.10 | 72%         | rank 165 / 21   |
| 0.25 | 72%         | rank 170 / 19   |
| 0.50 | 71%         | rank 172 / 19   |
| 0.75 | 78%         | rank 177 / 19   |
| 0.90 | 70%         | rank 167 / 17   |

Shuffle baseline: ~10× below real signal. Roughly **100 of small's
top-200 principal directions match large's at corr > 0.9 at every
depth including the mouths**. The alignment is uniform across depth,
not concentrated at the throat.

Reconciliation with 121: the mouths share a subspace; each model
places its coordinate *axes* differently inside that subspace. Linear
regression on raw coords at the mouth failed because it tried to hit
all d_large dims; PCA-aligned CCA sees past the rotation. The
apparent "mouth privacy" from 121 was a coordinate-frame artifact,
not a subspace-content difference.

### Stage 123b — PCA-subspace transplant fails anyway

Given the aligned subspace, the obvious test: inject large's throat
state into small's forward pass through a learned linear map, then
measure next-token NLL on held-out sentences.

Setup:
  - Fit A (k×k, k ∈ {50, 100, 200}) mapping large's top-k PCs to
    small's top-k PCs.
  - Test generalization: **R²_test = 0.998** at every k.
  - At inference, decompose small's own throat into shared + private
    components. Replace only the shared component with A-mapped
    large throat. Sweep α ∈ {0, 0.5, 1}.

Baseline small: NLL = 3.5581 (PPL = 35.09)
Baseline large: NLL = 3.1194 (PPL = 22.63)

| k   | α=0   | α=0.5 | α=1.0 |
|-----|-------|-------|-------|
| 50  | 3.558 | 3.590 | 3.715 |
| 100 | 3.558 | 3.665 | 3.961 |
| 200 | 3.558 | 3.686 | 4.068 |

α=0 exactly matches baseline (hook sanity). Every α > 0 degrades
monotonically. **Larger aligned k → more damage**, the opposite of
what a "more information helps" story would predict.

Crucially: **A generalizes nearly perfectly (R²_test = 0.998).**
The reconstruction of small's throat from large's throat is
essentially lossless. Yet mouth 2 still breaks.

## Why the transplant fails (the new claim)

**Mouth 2 is not a smooth linear inverse. It is a brittle nonlinear
decoder with very narrow tolerance.** A 0.2% reconstruction error at
L14 compounds through 14 more nonlinear layers into a ~0.4 nat NLL
jump. Each model's mouth 2 is calibrated tightly to the specific
throat trajectory its own L14 produces — not to the geometric
coordinate system.

Analogy: two lab instruments with physically identical measurement
axes but different personal calibration curves. You can translate
between their coordinate readings (A works), but you can't read one
instrument's output on the other's display.

So the wormhole decomposes as:

- **Shared**: throat coordinate system, manifold structure, mouth-1
  diffusion path, mouth-2 rough decomposition direction
- **Private**: mouth-2 fine-grained decoder calibration curve

## Consequences for known inference techniques

### Speculative decoding / draft models

Standard account: "draft shares knowledge with target."
Wormhole account: **draft shares the throat coord system; disagrees
on mouth-2 calibration.** Acceptance rate = throat-direction agreement,
modulated by mouth-2 divergence.

Predictions:
1. Acceptance rate has a **hard ceiling** set by mouth-2 divergence.
   No amount of knowledge distillation can break through it —
   because you're distilling through the noisy mouth-2 output, not
   the throat.
2. **Draft depth > draft width** for acceptance. Reaching throat
   alignment happens within 2–3 layers of throat. Parameter-matched
   shallow drafts should beat parameter-matched deep ones.
3. **Train drafts to match target's throat, not target's logits.**
   Current distillation optimizes KL on softmax output — that pipes
   the training signal through mouth-2 noise. Training the draft's
   L_throat state to match target's L_throat (via MSE / CCA) should
   raise the acceptance ceiling. Untested but follows directly.

### Medusa heads

Medusa heads live within a single model's own throat distribution.
The brittleness measured in 123b doesn't apply because they're
decoding their native throat. Wormhole reframes them as **parallel
unbinders of superimposed futures from one throat state** — a
specific implementation of a more general technique.

### Early exit

Same story — early-exit probes use the native throat, so the
decoder calibration matches by construction. Wormhole predicts
**per-token variable throat saturation**: easy tokens saturate the
rank-1 axis early (exit at L10), hard tokens require the full
mouth-2 unbind sequence.

## Consequences for compression

Stage 120's shape-aware squeeze init (edges Q10 / inner Q8 / throat
Q6) is now independently justified: **edges must stay near full
precision because mouth 2 is brittle**. The throat is a universal
coord system — perturb it and another model's decoder will match
imperfectly; it's robust to quantization noise that stays within
its own distribution.

Scale-aware compression budget prediction: **mouth-2 brittleness
scales with d_model** because the decoder has more dims to
calibrate. Bigger models have more tolerant throats and more
brittle mouths — consistent with Strix's 14B compression results
allowing 80% throat cuts at strict edge precision.

## Consequences for cross-model distillation

Raw throat transfer between models **does not work**. You need either:

1. **Throat-matched retraining**: after coord transplant, fine-tune
   small's mouth-2 on (large-throat → ground-truth logits) pairs.
   Essentially a distillation variant that targets the throat
   interface, not the logit output.
2. **Mouth-2 adapter**: a learned small module that adapts large's
   throat output into the coord frame small's mouth 2 expects. This
   is the minimal intervention and preserves the existing mouth 2.
3. **Full re-distillation via throat-MSE**: the draft-training
   suggestion above, applied as a general cross-model transfer
   protocol.

Untested. Direct extensions from 123b.

## Testable predictions

| # | claim | test |
|---|---|---|
| 1 | Cross-family (different tokenizer) models have lower throat alignment | CCA between Qwen3-0.6B and Llama-3-8B — predict < 50% of PCs match |
| 2 | Draft acceptance rate correlates with throat R² across model pairs | Measure on standard spec-decode setups |
| 3 | Throat-MSE-trained drafts outperform logit-KL drafts at matched size | Train one pair each way, compare acceptance |
| 4 | Throat caching works within a single model | Precompute throat for shared prefixes, skip mouth 1 + throat traversal |
| 5 | Mouth-2 width scales with d_model; throat calibration tolerance inversely | Repeat 123b at 14B, predict larger NLL damage per perturbation magnitude |

## Citations

- Stage 121: `scripts/stage121_cross_model_alignment.py`, `results/stage121_alignment.json`
- Stage 122c: `scripts/stage122c_pca_cca.py`, `results/stage122c_pca_cca.json`
- Stage 123b: `scripts/stage123b_pca_transplant.py`, `results/stage123b_pca_transplant.json`
- Finding 13: wormhole topology (bathtub → wormhole reframing)
- Stage 120: shape-aware squeeze (3.6× on 0.6B at teacher quality)

## Date

2026-04-24. Based on measurements taken on Qwen3-0.6B and Qwen3-1.7B,
same tokenizer family, via Apple MPS. Strix 14B measurements pending.
