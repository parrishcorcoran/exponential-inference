# Finding 29: The hypersphere is a forced choice, not the natural attractor of binary QAT

**Status**: established (Stage 183, 2026-04-29). Single experiment, single
seed, 300 steps. Strong null on the gradual-drift hypothesis; doesn't rule
out late-training phase transitions.

## Claim

When binary quantization is applied in the forward pass (Bonsai-style
per-128-group absmax) with FP master weights trainable underneath via
straight-through estimator, **the master weights do not drift toward the
unit hypersphere**, even as the loss recovers from total collapse.

The hypersphere geometry that our recipe imposes is therefore a *forced
choice we make for perturbation-tolerance reasons*, not a basin that
gradient descent on the binary objective finds on its own.

## Evidence (Stage 183)

Setup: Qwen3-0.6B, all 196 body linears patched with binary STE forward,
all 440M body params trainable, full AdamW, no unit-norm projection, 300
gradient steps, master row-norm CV measured every 50 steps.

| step | row-norm CV | mean row-norm | val CE |
|---:|---:|---:|---:|
| 0   | 0.3228 | 0.969 | 16.36 |
| 50  | 0.3228 | 0.969 | 7.72 |
| 100 | 0.3227 | 0.969 | 7.12 |
| 150 | 0.3226 | 0.970 | 6.70 |
| 200 | 0.3225 | 0.970 | 6.43 |
| 250 | 0.3224 | 0.970 | 6.31 |
| 300 | 0.3224 | 0.970 | 6.27 |

ΔCV = −0.0004 across 300 steps. ΔCE = −10.09 nats over the same window.
The model is in active descent; the master is being shaped; row-norm
uniformity is *not* the direction of motion.

## Mechanism (working hypothesis)

Under binary STE, the master's gradient routes through the quantized
forward, which depends on master only via:

1. **Sign content** — which weights are positive vs negative in each
   group of 128. A sign flip changes the binary code without changing
   the row L2 norm.
2. **Per-group scale** — Bonsai's per-group scale is `mean(|w|)` over
   the 128 weights in that group. The master can rebalance |w| *across
   groups within a row* without changing the row total.

Both are invisible to row-norm CV. The descent we see is presumably
happening through these two channels, plus whatever residual gradient
leaks through unrelated FP paths (norm gains, embedding, lm_head — those
were left untrained in this run).

## Cross-reference: NVIDIA's own ablations support this

Loshchilov et al. (2024, Table 5) report that replacing nGPT's
per-channel learnable scaling factors (α_A, α_M, s_qk, s_u, s_v) with
*single fixed scalars* costs ≤ 0.3% val loss. **The hypersphere
geometry, not the scaling DOFs, is what does the work.** Combined with
Stage 183's null, the picture coheres: gradient descent does not
spontaneously prefer the unit-norm constraint; the constraint is doing
something we have to put there by hand.

## Why we still want the hypersphere

The agent who mined the nGPT literature surfaced one near-miss in the
original paper: nGPT models show dramatically lower attention condition
numbers (Loshchilov Fig. 5). That is the closest thing to a
"perturbation tolerance" claim — and it is the obvious reason
unit-norm rows would survive a 1-bit quantizer:

> Each row of a unit-norm matrix contributes a bounded amount to the
> output. A bit-flip in a unit-norm row is a bounded perturbation. A
> bit-flip in a CV=0.32 model is unbounded — high-magnitude rows blow
> up disproportionately.

This converts our story from "gradient descent likes the hypersphere"
(which Stage 183 falsifies) to "the hypersphere is the only
distribution under which the quantizer's per-row error is uniformly
bounded" (which is a clean geometric claim, untested).

## What this changes about our recipe

1. **Don't expect the hypersphere to emerge from QAT alone.** It must
   be imposed (project rows to unit norm) or trained-toward via an
   explicit loss term. Our pipeline's Stage 1 (anneal to nGPT-shape)
   is therefore load-bearing — without it, QAT will optimize toward
   some other geometry.

2. **Per-channel α may be overkill.** NVIDIA's Table 5 shows scalar
   α suffices. Our 440K-param α-bridge is buying very little; the
   geometry does the work. Future stages should test scalar α vs
   per-channel α at matched binary CE.

3. **The justification is perturbation-tolerance, not optimization
   landscape.** When we publish, lead with "bounded per-row error
   under bit-flips" rather than "natural attractor."

## Caveats and open questions

- **300 steps is short.** Final CE 6.27 vs base ~3.0 — model is far
  from a local minimum. A late phase transition where CV collapses
  remains possible, just not observed here.
- **CV is one summary.** Sign-flip count, per-group scale CV (within
  each row), and PC1 of per-head structure could all be moving while
  CV holds flat. Worth measuring on the saved master.
- **Single seed, single model.** Replicate on Qwen3-4B and at longer
  horizon before treating as conclusive.

## Next test (recommended by the nGPT-literature agent)

Per-row perturbation tolerance vs row-norm CV. Take a CV=0.32
checkpoint and a CV→0 checkpoint at matched CE; inject Gaussian noise
into each row at a sweep of scales; measure ΔCE. If the
perturbation-tolerance story is correct, the unit-norm model should
degrade roughly linearly with noise scale while CV=0.32 degrades
super-linearly. No training required, just forward passes.

This converts Stage 183's loss-curve evidence into a *causal* geometric
claim: the hypersphere isn't where optimization wants to go, it's where
binary quantizers can survive.

## Files

- `scripts/stage183_binary_qat_natural_drift.py`
- `results/stage183_binary_qat_natural_drift.json`

## Related findings

- Finding 25 (magnitude anneal to nGPT-shape) — Stage 1 of the recipe
- Finding 27 (o_proj is the binary bottleneck) — where structure lives
- Finding 28 (ternary's "0" is hard head selection) — companion mech
