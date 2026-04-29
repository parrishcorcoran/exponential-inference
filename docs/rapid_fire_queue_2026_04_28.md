# Rapid-fire stage queue — 2026-04-28

While 14B nGPT conversion runs on Strix (2-3 days), this is the queue of
small experiments to chain through on Mac. Each ~30 min – 2 h, each numbered.

Pattern: small script, runs to completion, produces a tagged JSON in
`results/`, optional plot in `results/`, brief one-liner finding committed
back to repo. Goal is data accumulation, not deep individual experiments.

## Stage 162: Per-layer CV profile on Qwen3-0.6B base

**Question:** Does CV vary across the 28 layers? Are some layers naturally
more spherical than others?
**Why it matters:** Identifies which layers will be cheapest/hardest to
convert. Sets up per-layer schedules later.
**Cost:** 10 min Mac.
**Output:** `results/stage162_per_layer_cv.json` + plot of CV vs layer index.

## Stage 163: Per-layer CV profile on Qwen3-4B base

Same as 162, larger model. Tests whether layer-CV pattern is universal or
size-dependent.
**Cost:** 20 min Mac.
**Output:** `results/stage163_per_layer_cv_4b.json`

## Stage 164: Per-layer CV profile on Bonsai-8B 1-bit (effective weights)

Same diagnostic on the binary model we downloaded. Compares layer profile
to the FP base — does PTQ flatten some layers more than others?
**Cost:** 5 min Mac (model already downloaded).
**Output:** `results/stage164_per_layer_cv_bonsai.json`

## Stage 165: α-recovery dry run on Qwen3-0.6B base (sanity)

Take the FP Qwen3-0.6B base, freeze body, add per-channel α (initialized to
1.0), train α only for 500 steps on OWT. Should converge to roughly
identity α (since base weights already have natural magnitudes). Tests the
recovery mechanism itself before applying to the post-conversion case.
**Cost:** ~30 min Mac (only 65K trainable, fast).
**Output:** `results/stage165_alpha_dry_run.json` + α distribution plot.

## Stage 166: α-recovery on a synthetic unit-norm Qwen3-0.6B

Take Qwen3-0.6B base, project all rows to unit norm in stored weights
(emulate Strix's bake without the anneal), then run α-recovery. Tests if
α can recover a projected-without-training model. Predict: large quality
gain because we're recovering magnitude information.
**Cost:** ~30 min Mac.
**Output:** `results/stage166_alpha_recovery_synthetic.json`

## Stage 167: Mac binary projection sanity test

Take Qwen3-0.6B base, post-hoc binary quantize (sign × per-channel α from
mean(|W|)), measure val CE. Establishes our own baseline for "post-hoc
binary on standard model." Compare to Bonsai's 11% quality drop on its
larger model.
**Cost:** ~10 min Mac.
**Output:** `results/stage167_post_hoc_binary_baseline.json`

## Stage 168: Cross-family CV — Llama-3-1B (or smallest available)

Run the row-norm diagnostic on Llama-3 family. Test if "natural attractor
near CV ~0.30" is family-specific or universal across pretrained transformers.
**Cost:** ~15 min Mac (~2GB download).
**Output:** `results/stage168_llama3_cv.json`

## Stage 169: Cross-family CV — Phi-3 mini

Same as 168 on Phi-3 family.
**Cost:** ~15 min Mac.
**Output:** `results/stage169_phi3_cv.json`

## Stage 170: Cross-family CV — Gemma-2-2B

Same as 168 on Gemma family.
**Cost:** ~15 min Mac.
**Output:** `results/stage170_gemma2_cv.json`

## Stage 171: Magnitude-vs-CV scatter

Aggregate across all measured models (0.6B, 1.7B, 4B, 8B, 14B, 32B from
Z8, plus Bonsai, plus cross-family from 168-170, plus BitNet). Scatter of
mean row norm vs CV. Look for clustering / family-specific lines.
**Cost:** ~5 min (post-process existing data).
**Output:** `results/stage171_magnitude_cv_scatter.png` + `.json`

## Stage 172: Bonsai per-projection magnitude analysis

We have row norms by type for Bonsai. Compare distribution of per-group
scales (from `.scales` tensor) to per-channel α we'd use. How much
information is in the per-group structure that single-α-per-channel would lose?
**Cost:** ~15 min.
**Output:** `results/stage172_bonsai_scale_analysis.json`

## Stage 173: Compute the "rotation" between base W and its projected version

For Qwen3-0.6B, for each linear, compute W_base vs W_projected (unit-norm).
Express the difference as a rotation (singular vectors that change direction)
vs a scale (singular value magnitudes that change). Decompose what's lost
in the projection.
**Cost:** ~30 min.
**Output:** `results/stage173_projection_decomposition.json`

## Stage 174: Half-anneal (τ=0.5 only) with norm-only on Qwen3-1.7B

Run a partial anneal — single-stage projection to τ=0.5, just norm-only,
500 steps. Tests if there's any benefit from extending the recipe to 1.7B
quickly. If the model reacts the same way as 0.6B, predicts 14B will land
similarly.
**Cost:** ~2 hours Mac (might be tight on memory; could need streaming
data, which we already have).
**Output:** `results/stage174_qwen3_1.7B_half_anneal.json`

## Stage 175: Stage-2-script written and ready

Not a measurement — a deliverable. Write `scripts/pipeline_alpha_recovery.py`
that takes a τ=1.0 baked checkpoint, freezes body, adds α, trains for N
steps. Ready to drop on Strix's 0.6B τ=1.0 checkpoint immediately.
**Cost:** 30 min coding.
**Output:** committed script.

## Stage 176: Stage-3-script written and ready

Same — write `scripts/pipeline_binary_qat.py` that takes the α-recovered
checkpoint, anneals binary projection on top with thermostat. Ready to run
post-Stage 2.
**Cost:** 30 min coding.
**Output:** committed script.

## Stage 177: Downstream eval harness

Write `scripts/eval_downstream.py` that loads any checkpoint and runs:
LM val CE, MMLU subset, HellaSwag subset, GSM8K subset. Apples-to-apples
quality comparison ready for when the binary 0.6B is done. Compare against
Bonsai-8B's 70.5 average score.
**Cost:** 1 hour coding.
**Output:** committed script + `results/stage177_eval_baseline.json`.

## Stage 178: Quick reservoir-baseline sanity on Qwen3-0.6B

Take Qwen3-0.6B, replace ALL body weights with random Gaussian (untrained),
keep only embed/lm_head/norms trainable. Train embed/lm_head/norms only for
1000 steps. Establishes the "no body training" floor. If our converted
+ binary + α model gets within X of this, we know we're in good company;
if much worse, body weights matter.
**Cost:** ~1 hour Mac.
**Output:** `results/stage178_reservoir_baseline.json`

## Stage 179: PID controller replication on Mac

Strix's PID multi-axis (commit 27ab2de) tested on 4B. Replicate the
controller logic on Qwen3-0.6B with two axes (magnitude shrink + V rank
reduction). Validates the methodology at small scale before pushing to 14B/32B.
**Cost:** ~2 hours Mac.
**Output:** `results/stage179_pid_replication.json`

## Stage 180: Speedup test setup — fine-tune on a downstream task

Take Qwen3-0.6B base and run 500 steps of fine-tune on a small downstream
task (say arithmetic / GSM8K subset). Record convergence curve. To be
compared with: same fine-tune on the τ=1.0 baked checkpoint. The
*comparison* is what tests the partial nGPT speedup claim transferring
to converted models.
**Cost:** ~1 hour Mac.
**Output:** `results/stage180_baseline_finetune_curve.json`

## Stage 181: Same fine-tune on τ=1.0 baked Qwen3-0.6B

Pair to 180. Take Strix's τ=1.0 baked checkpoint, fine-tune on same task,
same hyperparameters. Compare convergence. Direct test of "nGPT-shape
trains faster downstream."
**Cost:** ~1 hour Mac (after pulling Strix's baked checkpoint).
**Output:** `results/stage181_ngpt_finetune_curve.json`

---

## Execution order (by Mac runnable, fastest first)

**Quick (< 30 min each, can chain rapidly):**
162, 163, 164, 167, 171, 172

**Medium (~30 min each):**
165, 166, 168, 169, 170, 173

**Coding (no compute):**
175, 176, 177

**Longer (1-2 hours):**
174, 178, 179, 180, 181

**Total Mac compute:** ~10-15 hours of background work. Spread across the
2-3 days Strix is running 14B, this fills the wait time productively.

---

## Commit pattern

After each stage:
```
git add scripts/stage<N>*.py results/stage<N>*.json findings/<finding-N>.md
git commit -m "Stage <N>: <one-line headline finding>"
git push
```

Headline finding example (good): "Stage 162: bottom 8 layers of Qwen3-0.6B are
50% more spherical than top 8 layers — predicts asymmetric anneal cost."

Headline finding example (bad): "Stage 162: ran the diagnostic." (No insight.)
