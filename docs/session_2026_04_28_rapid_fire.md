# Session 2026-04-28: Rapid-fire diagnostic batch (stages 162–178)

## Context

Strix's Qwen3-14B nGPT conversion is running (~2-3 days). While it
runs, this session ran a chain of mostly-analytical stages on Mac M4
to fill out the methodology and characterize the binary-quantization
problem.

14 stages, 14 commits in ~6 hours. Most were diagnostic (no training).

## Headline findings

### 1. The α-bridge mechanism is theoretically grounded and validated

- **Stage 169 T2**: `α = row_norm` reproduces base CE *exactly*
  (Δ = +0.000). Validates the math identity.
- **Stage 169 T3**: training α 500 steps from row_norm init produces
  Δ = −0.121 nats (12% perplexity improvement) over base.
- **Stage 178**: trained α drifts only 0.2% std from row_norm init.
  Theory holds: α at row_norm is essentially the optimum, training
  finds tiny per-channel refinement.

**The decomposition `W = unit-direction × per-channel-magnitude` is a
linear identity. The per-channel magnitude is recoverable from row_norm
of the original weights. Adding α as a separate trainable parameter
unlocks small per-channel optimization that gradient descent finds.**

### 2. Our recipe is a strict generalization of nGPT

- **Stage 168**: nGPT (community 800M) only normalizes input
  projections (qkv, gate, up). Output projections (o_proj, down_proj)
  retain magnitude variation.
- **Stage 170**: extending nGPT_800m to OUR full normalization
  (forcing o_proj and down_proj to unit norm too) costs Δ = −0.010
  nats — **strictly improves nGPT** at 0 quality cost.

**nGPT's input-only normalization was a training-speed convenience,
not a quality optimum. Our recipe reaches a more constrained geometry
than nGPT, with no quality penalty, with the bonus of being binary-
quantization ready.**

### 3. o_proj is the universal binary bottleneck (Finding 27)

Across every model and every method:
- **Stage 162-163**: Qwen3 0.6B/4B per-layer CV — late layers bumpy
- **Stage 164**: Bonsai PTQ flattens middle layers but not edges
  (2.98× spread)
- **Stage 168**: nGPT specifically does NOT normalize o_proj
- **Stage 172**: Bonsai's o_proj has 2.5× higher intra-row scale
  variation than every other projection
- **Stage 173-175**: o_proj's variation aligns with attention head
  boundaries; per-row × per-head selection structure with low-rank
  PC1 = 25-43% across models

**The attention output projection is the structural bottleneck for
low-bit quantization across all models we measured. It encodes
per-row × per-head selection that other projections don't have.**

### 4. Ternary's "0" state is hard per-head selection (Finding 28)

- **Stage 175a**: ternary projection of BitNet's o_proj AMPLIFIES the
  per-head structure: PC1 = 37% (FP master) → **73% (ternary effective)**.
- Binary post-hoc (Bonsai): PC1 dilutes from 43% → 35%.

**The "0" state in ternary IS the head-selection mechanism. Heads
whose weights are mostly small collapse to all-zero (effectively
"off"). Heads with large weights crystallize to ±γ ("on"). Pure binary
has no "off" state — explains why BitNet 1.58 works at LM scale while
BitNet 1.0 didn't.**

### 5. Wall is at 3-bit, not at binary

- **Stage 171** (progressive quant with α refinement):
  - 8-bit: Δ = −0.137 (free, even improves)
  - 4-bit: Δ = +0.422 (small cost)
  - **3-bit: Δ = +3.877 (WALL — 10× jump)**
  - 2-bit, 1.58-bit, 1-bit: all Δ ≈ +13–15 nats (uniform broken regime)

**Below 4-bit, post-hoc α refinement cannot escape the catastrophic
basin. QAT (master-weight training under quantization) is required.**

### 6. Per-head structure WEAKENS with model size

- **Stage 175b**: Qwen3-4B FP base PC1 = 25.1% (vs 0.6B's 42.8%)
- More heads (16 → 32) distribute magnitude info, dilute structure

**Our binary recipe should land cleaner at frontier scale than at toy
scale. The o_proj bottleneck is a small-model phenomenon — Strix's 14B
and eventual 32B should hit lower conversion costs.**

### 7. Naive binary baseline (no preconditioning) is catastrophic

- **Stage 167**: post-hoc binary on Qwen3-0.6B without our recipe:
  Δ = +13.34 nats. Total model collapse.
- vs Bonsai (per-group quantization) at +0.7 nats / 11% benchmark drop
- vs Strix's nGPT τ=1.0 (no quant) at ~0 nats

**The gap our recipe must close: from +13 catastrophic baseline to ~0
or below. Each stage of the pipeline (project, α, binary QAT) closes
some of that gap.**

## Methodology contributions identified

1. **Synthetic τ=1.0 + α-recovery** (Stage 169) is a complete pipeline
   producing strictly better-than-base models on Qwen3-0.6B.

2. **Progressive quantization with α refinement** (Stage 171) gives
   8-bit and 4-bit quantization for free or very cheap.

3. **Per-head awareness for o_proj** (Findings 27, 28) is necessary
   for binary at LM scale. Three viable mechanisms: block-α aligned
   to head_dim, QAT-driven flattening, or use ternary instead.

## What's still open

- Strix's 14B nGPT conversion (running 1-2 days): validates Stage 1
  at non-toy scale
- Stage 2 (α-recovery on 14B baked checkpoint) once 14B finishes
- Stage 3 (binary QAT) — need to design the QAT-during-anneal recipe;
  post-hoc doesn't work below 4-bit
- Cross-family universality (Llama, Phi, Gemma) — diagnostic stages
  candidate for next session

## Files added in this session

### Scripts
- `scripts/stage162_per_layer_cv.py` — per-layer CV diagnostic
- `scripts/stage163_per_layer_cv_4b.py` — same on 4B
- `scripts/stage164_per_layer_cv_bonsai.py` — same on Bonsai 1-bit
- `scripts/stage167_post_hoc_binary_baseline.py` — naive Q1 baseline
- `scripts/stage169_alpha_recovery_then_quant.py` — full T0–T5 pipeline
- `scripts/stage170_convert_ngpt800m_to_ours.py` — extend nGPT
- `scripts/stage171_progressive_quantization.py` — graduated bit reduction
- `scripts/stage172_bonsai_intra_row_scale_variation.py` — within-row CV
- `scripts/stage173_bonsai_head_alignment.py` — per-head PCA
- `scripts/stage174_head_structure_fp_base.py` — same on FP base
- `scripts/stage175a_head_structure_bitnet.py` — BitNet head structure
- `scripts/stage175b_head_structure_qwen3_4b.py` — Qwen3-4B head structure
- `scripts/stage178_alpha_theory_validation.py` — α drift after training
- `scripts/diag_bitnet_row_norms.py` — utility
- `scripts/diag_bonsai_hypersphere.py` — utility

### Results
- 14 result JSONs in `results/stage16*-stage17*.json`
- Plot files where applicable

### Findings
- `findings/27_oproj_is_the_binary_bottleneck.md`
- `findings/28_ternary_zero_state_is_hard_head_selection.md`

## Hardware time used

Mac M4 base (16GB):
- ~6 hours of mostly-diagnostic compute
- Most stages ran in 5-30 minutes (analysis only)
- Stages 169, 170, 171, 178 trained α for 500-1500 steps each
- Total Mac compute: probably ~3-4 hours of actual GPU time

Background activity:
- Strix continues 14B nGPT conversion (independent)
- Z8 had finished CV scaling and PID experiments (pulled at session start)
