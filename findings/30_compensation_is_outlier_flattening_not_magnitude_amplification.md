# Finding 30: Compensation across the precision spectrum is outlier-flattening, not magnitude amplification

**Status**: established (Stage 185, 2026-04-29). Single read-only
diagnostic across three checkpoints. Replicable; no training involved.

## Claim

When we compare trained low-bit transformers to their FP equivalents,
the dominant geometric signature is *not* an across-the-board rise in
magnitudes. It is the **collapse of outlier RMSNorm gain channels**.
Body row norms only go up by ~2.5×. RMSNorm gain *maxima* drop by up
to two orders of magnitude.

This reframes what "compensation" means at low bit-width and changes
what a successful anneal pipeline has to do.

## Three-model trend

Read-only walk of state_dicts. Per-element RMS amplitude is dim-invariant
(`row_norm / sqrt(in_features)`) so model width doesn't bias the comparison.

| axis | Qwen3-0.6B FP | BitNet b1.58 (eff) | Bonsai-8B-1bit (eff) |
|---|---:|---:|---:|
| body row-norm mean | 0.97 | 2.32 | 2.56 |
| body row-norm CV | 0.32 | 0.31 | 0.28 |
| per-elem RMS (o_proj) | 0.025 | 0.061 | 0.035 |
| per-elem RMS (down_proj) | 0.026 | 0.065 | 0.031 |
| **RMSNorm gain max** | **192** | **1.01** | **34** |
| RMSNorm gain mean | 2.68 | 0.47 | 0.73 |
| RMSNorm gain CV | 1.54 | 0.37 | 1.21 |
| Embedding row-norm mean | 0.93 | 2.30 | 1.10 |
| LM head row-norm mean | (tied) | (tied) | 1.58 |
| Total |w| per parameter | 0.022 | 0.045 | 0.034 |

The user's prior intuition was that "magnitude needs to go WAY up
exponentially" as bits drop. The data says otherwise: the bulk
magnitudes only roughly double, while the *tail* of the RMSNorm
distribution collapses dramatically.

## Mechanism

**Outlier feature channels** are a well-known phenomenon in Qwen/Llama
family models: a small number of residual-stream channels get
amplified by per-channel RMSNorm gains in the 100–250 range. The body
weights then interact with these massively-amplified channels through
correspondingly-sized weight values. This concentrates dynamic range
in a few specific dimensions.

Low-bit weight quantization breaks this mechanism because the body
weights cannot represent the wide range of values needed. The model
must either:

1. **Train it out from scratch** (BitNet path): start with the
   constraint, learn a flat geometry where every channel sits in a
   bounded range, no outliers.
2. **Try to quench it post-hoc** (Bonsai path): partially succeed —
   some outliers reduced, but not eliminated; max gain still 34×.

Bonsai's incomplete quenching plausibly explains its quality plateau:
binary weights cannot deliver the precise values that 34× outlier
channels need to function correctly.

## Why this matters for our anneal pipeline

Our pipeline (Stages 169, 170, 180, 184) projects body row norms to
unit length and adds α-bridge per channel. **It does nothing about
RMSNorm gains.** When we anneal Qwen3 → nGPT-shape, the 192× outlier
RMSNorm channels survive untouched.

This predicts:
- Our recipe will plateau short of fully matching BitNet-quality
  binary models, even with extensive QAT, unless we also dampen
  RMSNorm gain outliers during anneal.
- The Stage 184 plateau gap of +3.85 nats with compensation-only
  training is consistent with the model being unable to recover the
  outlier-channel function under binary weights.
- Our nGPT-conversion of Qwen3-14B (currently running on Strix) will
  likely also need an outlier-flattening step to reach the bottom
  of the binary quality envelope.

The cleanest fix is to add a stage that progressively constrains the
RMSNorm gain distribution — for example, regularizing toward
gain CV ≤ 0.5 or capping max gain at some quantile during anneal.
This costs FP precision in the residual stream but matches the
geometry that BitNet achieves naturally.

## Cross-reference: what this is NOT

- It is **not** evidence that body weight magnitudes don't matter.
  They roughly double, which is real. It just isn't the dominant
  signal.
- It is **not** evidence that BitNet is "better" than Bonsai. BitNet
  was trained from scratch with 4T tokens; Bonsai is a PTQ derivative
  with much less training. The geometric difference is consistent
  with their training procedures, not a quality verdict.
- It does **not** falsify the perturbation-tolerance story (Finding
  29). Unit-norm rows still bound per-row error from bit flips.
  Outlier flattening is a *complementary* mechanism: it bounds
  per-channel residual-stream amplitudes.

## Next tests

- **Stage 186** (proposed): measure RMSNorm gain distribution before
  and after our anneal stages. Confirms the outliers survive our
  current pipeline.
- **Stage 187** (proposed): add a regularizer to the anneal that
  penalizes RMSNorm gain CV; rerun α-recovery on Qwen3-0.6B and see
  if the binary plateau drops below +3.85 nats.
- Cross-check on Qwen3-4B and Qwen3-14B (Strix) once available — is
  the outlier scale consistent across sizes? Outlier max usually
  *grows* with model scale, which would make the problem worse at
  frontier scale, not better.

## Files

- `scripts/stage185_compensation_trend_across_precision.py`
- `results/stage185_compensation_trend.json`

## Related findings

- Finding 25: hypersphere shape of pretrained weights (Qwen body row-norm CV)
- Finding 27: o_proj is the binary bottleneck (per-row-per-head structure)
- Finding 28: ternary's "0" state is hard head selection
- Finding 29: hypersphere is a forced choice, not natural attractor
