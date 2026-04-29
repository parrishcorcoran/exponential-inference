# Finding 26: Qwen3-14B nGPT conversion — preregistered predictions

## Status

**Running on Strix as of 2026-04-28.** Estimated wall-clock 2–3 days for full
τ=0 → τ=1.0 sweep with progressive unfreezing schedule.

This finding is registered BEFORE the result lands so we can honestly
distinguish "predicted correctly" from "post-hoc rationalized."

## Setup

- Model: Qwen/Qwen3-14B (14.8B params, hidden_size=5120, 28 layers)
- Recipe: same as Strix's 0.6B run that produced τ=1.0 at +0.031 nats
  - Forward-time projection of weight rows toward unit L2 norm
  - FP master weights, STE backward
  - τ schedule: 0.10 → 1.00 in 10 drops of 0.1 each
  - 2000 fine-tune steps per drop, thermostat at +0.5 CE
  - **Progressive unfreezing**: norm-only for τ < 0.2, full body for τ ≥ 0.2
- Training data: same 50M-token pre-tokenized OWT cache used on Mac/0.6B
- Hardware: Strix Halo (RDNA 3.5, 128GB unified, ~30 TFLOPS bf16 effective)

## Why this experiment is novel

Strix already did pretrained → nGPT conversion on Qwen3-0.6B (commit 359c193).
This is the first replication on a 14B model — the size class where:
1. nGPT's training-from-scratch speedup claims have not been independently validated
2. Z8's CV scaling diagnostic shows attention projections are already half as
   spherical as 0.6B (q_proj CV 0.222 vs 0.448) — predicts the conversion
   should be cheaper to do here
3. The 14B class is the smallest size where industry inference deployment
   actually cares about the geometry (0.6B is too small to ship; 14B is the
   first "real" frontier-deployment-relevant size)

## Predictions (registered before result)

### Quality cost at τ=1.0

- **Most likely (60% confidence): <0.5%** perplexity above baseline
  - Reasoning: Qwen3-14B starts with attention CV roughly half of 0.6B's,
    so half as much "magnitude variance" needs to be absorbed during the
    projection. Expect proportional reduction in conversion cost.
- **Possible (25% confidence): <0.2%** perplexity above baseline (effectively zero)
  - This would happen if the per-projection-type CV reductions (Q, K, O all
    halved or more from 0.6B) translate fully to lower conversion cost.
- **Unlikely (10% confidence): exactly 0% or below baseline**
  - Strix's 0.6B trend was still improving at step 2000 (could have gone
    below baseline with more steps). At 14B, the model has more capacity
    to compensate, so might land below baseline at the natural stopping
    point.
- **Very unlikely (5% confidence): >1% loss**
  - Would imply the conversion cost increases or stays flat with model size.
    Z8's CV scaling data argues against this.

### Trajectory features

- **Best τ point: τ ∈ [0.20, 0.40]** with quality below baseline
  - 0.6B peaked at τ=0.20. 14B's peak should be similar or slightly later
    because the model has more capacity to find improvements.
- **Free zone extends to: τ ∈ [0.7, 0.9]**
  - 0.6B's free zone went to τ=0.9. 14B's should be at least as long; may
    extend further because the conversion is cheaper.
- **Per-step recovery rate accelerates with τ**: 1.5–3× faster at τ=0.5 than
  τ=0.2 (the partial nGPT speedup signal)
  - 0.6B showed 4.6e-5 → 9.5e-5 = 2× across this range. Predict similar
    or stronger acceleration at 14B.

### Compute / runtime predictions

- **Total wall-clock**: 2–3 days on Strix.
  - 14B is ~25× more parameters than 0.6B. Strix forward+backward scales
    roughly linearly in params for the body, so per-step time ~25× longer.
    0.6B took ~few hours per drop; 14B should take ~1–2 days per drop. With
    progressive unfreezing reducing the early-stage cost, total should
    fit in 2–3 days.
- **Memory footprint at peak**: ~70 GB (model 28GB + AdamW state 28GB +
  activations 10GB + tokens 0.2GB) on Strix's 128GB.

### Failure modes to watch for

1. **Thermostat trips repeatedly mid-sweep** (e.g., at τ=0.50)
   - This would suggest the recipe has a wall at 14B that didn't exist at
     0.6B, contradicting the "easier at scale" prediction.
   - Most likely cause: progressive unfreezing schedule fired too late.
     Could try unfreezing earlier (τ=0.1).
2. **Per-step recovery rate DOES NOT accelerate with τ**
   - Would mean partial nGPT speedup doesn't transfer through scale.
     Important null result. Wouldn't kill the conversion result but would
     weaken the speedup-claim corroboration.
3. **τ=1.0 quality cost > 1% perplexity**
   - Would mean the conversion isn't cheaper at scale. Possible explanations:
     - 14B has different bottleneck layers than 0.6B
     - Some specific projections (Q at higher CV than expected at scale)
       resist conversion
     - The progressive unfreezing schedule needs tuning for 14B
4. **Strix runs out of memory or time**
   - 14B at full body is the largest training Strix has done.
     If memory issues arise, fall back to grad accumulation 16+ or
     reduce seq_len to 64. Falls back to 8-bit Adam or CPU optimizer
     offload via DeepSpeed if needed.

## What success looks like

τ=1.0 conversion completes with <0.5% quality loss, training-speedup signal
visible in per-step recovery rates, completed within 3 days. Becomes:

- The first 14B-scale validation of pretrained-to-nGPT conversion
- Strong evidence the recipe scales beyond toy-model regime
- Preserved checkpoint usable as substrate for Stage 2 (α recovery) and
  Stage 3 (binary quantization) experiments
- Direct comparison point against Bonsai-8B-mlx-1bit's 11% quality loss
  (different model size but same family scale)

## What failure looks like

τ=1.0 walls or costs >2% — would mean:
- The recipe needs significant adaptation for larger models
- The "easier at scale" hypothesis from CV scaling doesn't translate to
  conversion cost
- Need to investigate which specific projections or layers resist conversion

A null result here is still informative — it would tell us 14B has structure
0.6B doesn't, and we'd need targeted experiments to find it.

## Files

- Script: `scripts/pipeline_unit_norm_anneal.py` (env-var configurable)
- Launch command:
  ```bash
  CHECKPOINT="Qwen/Qwen3-14B" RUN_TAG="strix_14b" \
    STEPS_PER_DROP=2000 BATCH=1 GRAD_ACCUM=8 SEQ_LEN=128 \
    LR=2e-5 TOKEN_CACHE="data/owt_tokens_50M.pt" \
    .venv/bin/python scripts/pipeline_unit_norm_anneal.py
  ```
- Expected results location: `results/pipeline_magnitude_anneal_strix_14b.json`
- Expected checkpoints: `checkpoints/Qwen_Qwen3-14B/magnitude_anneal_strix_14b_*.pt`

## Cross-references

- Finding 25: original 0.6B conversion methodology
- Strix commit 359c193: 0.6B result (τ=1.0 at +0.031 nats)
- Strix commit 98cb4ce: nGPT geometry measurement across scale (the CV
  diagnostic that motivates 14B as the next scale)
- Bonsai-8B-mlx-1bit: external comparison point at same family
