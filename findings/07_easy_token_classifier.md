# Finding 07 — Token-difficulty routing signals (honest LOPO numbers)

## The claim

Free runtime signals derived from the forward pass — attention
entropy, hidden-state norms, per-layer update magnitudes, trajectory
geometry, quantum density-matrix statistics, black-hole-inspired
bipartite boundary measures — **do carry real token-difficulty
signal**. On 35 diverse prompts tested under leave-one-prompt-out
cross-validation, 47 such features predict output entropy at
**linear LOPO R² = 0.341**, which is **78% of what the full final
hidden-state (PCA-64) can predict under the same validation** (0.437).

The signal is genuine but smaller than a naive random-split test
would show: random 80/20 gives 0.474, LOPO (true cross-prompt) gives
0.341, so ~28% of the apparent predictive power is prompt-specific
and doesn't generalize.

## Why it's a stop-and-think

Under naive random-split validation, signal routing looks easy. Under
honest LOPO validation:
- **Reasoning prompts fail to generalize** (LOPO R² = 0.21 vs 0.48
  for free-form). Routing that works everywhere else breaks on
  reasoning tokens.
- **The signal is real** (label shuffle → R² ≈ 0).
- **MLP overfits catastrophically on OOD prompts** (LOPO R² < 0
  when train/test are different prompts). Linear regression is the
  honest deployment form.

This is the answer to "does the dynamic-routing claim actually work?"
Yes, with a LOPO R² of 0.34, directional signal is real. But the
routing should be LINEAR, should use an 8-feature minimal set
(Finding 08), and reasoning prompts need a different mechanism.

## How it was measured

Collected 4165 records (35 prompts × 120 tokens) across four prompt
categories (factual, reasoning, free_form, ambiguous). At each decode
step: extracted 47 runtime features from attention weights, hidden
states, and per-layer update vectors; labelled with output_entropy
and logit_margin; full hidden state h_final also stored for ceiling
computation.

Three validations:

1. **Random 80/20** — naive baseline.
2. **Leave-one-prompt-out (LOPO)** — 35 folds, each holds out an
   entire prompt. Reports mean R² across folds.
3. **Leave-one-category-out (LOCO)** — 4 folds, each holds out an
   entire category (e.g., all reasoning prompts).

Linear regression is primary; small MLP (16-hidden, 2-layer) also
tested for generalization parity.

## The numbers

### Random vs LOPO across feature tiers (linear regression)

| feature set | random R² | LOPO R² | overfit gap |
|---|---|---|---|
| summary (17) | 0.270 | 0.123 | 0.147 |
| + curvature (28) | 0.366 | 0.261 | 0.105 |
| + quantum (36) | 0.402 | 0.279 | 0.123 |
| **+ structural (47)** | **0.474** | **0.341** | **0.133** |
| h_final PCA-64 (ceiling) | 0.606 | 0.437 | 0.169 |

LOPO coverage of h_final ceiling: **78%**.

### Per-category LOPO R² (47 features)

| category | LOPO R² | n prompts |
|---|---|---|
| free_form | **+0.476** | 9 |
| ambiguous | +0.356 | 8 |
| factual | +0.326 | 9 |
| reasoning | **+0.207** | 9 |

### Leave-one-CATEGORY-out (structural features)

| held out | R² |
|---|---|
| free_form | +0.453 |
| ambiguous | +0.392 |
| factual | +0.374 |
| **reasoning** | **+0.199** |

Reasoning held-out always gives the worst cross-category score,
confirming a systematic feature-difficulty mismatch for those tokens.

## Interpretation

Three readings:

1. **The signal is real but modest.** LOPO R² = 0.34 means our
   features explain ~34% of output-entropy variance on truly unseen
   prompts. Enough for directional routing (run full compute at
   predicted-hard tokens, cheap compute at predicted-easy tokens),
   not enough for quality-critical per-token decisions.

2. **The MLP is the wrong deployment form.** Random 80/20: MLP big
   R² = 0.58, small = 0.41. LOPO: MLP big = −0.17, small = −0.98.
   MLP memorizes prompt-specific patterns. Use linear regression in
   deployment; it's honest.

3. **Reasoning prompts are structurally different.** The mechanism
   is probably that reasoning tokens have "computation" states
   (mid-reasoning, uncertain, branching) and "commitment" states
   (final answer) that our features don't distinguish. The whole
   feature set is calibrated to predict committed outputs.

## What it predicts

1. **Deployable linear routing gets you modest but real compute
   savings.** At LOPO R² = 0.34, a 30% reduction on 40% of tokens
   with low predicted difficulty, plus full compute on the other
   60%, is a realistic target. Check empirically: the routing will
   miss some easy tokens and occasionally route hard tokens to cheap
   paths.

2. **Reasoning tokens need a different signal.** Chain-of-thought
   style prompts may require tracking WHICH reasoning step is
   active, not just local token difficulty.

3. **Combining with teacher's own confidence (when available)** via
   speculative decoding should compose: the linear signal cheaply
   flags high-confidence tokens, the teacher verifies occasionally.

## Limitations

1. 35 prompts is modest; 100+ would give tighter bounds.
2. One model tested (Qwen3-0.6B); cross-model consistency unverified.
3. Linear regression is simple; a small non-overfitting gradient-
   boosted model could pick up another few points of R².
4. We predict output_entropy, not task quality. A task-quality
   label would be a harder target.

## Reproduce

```bash
# Signal collection + full LOPO analysis
python scripts/stage31_expanded_lopo.py --model Qwen/Qwen3-0.6B \
    --max-new-tokens 120 --device mps

# Validation that random-split would overstate
python scripts/stage30_validation.py --model Qwen/Qwen3-0.6B \
    --device mps

# Minimal essential subset
python scripts/stage32_minimal_subset.py --model Qwen/Qwen3-0.6B \
    --max-k 12 --device mps
```

## Related

- [Finding 08](08_minimal_signal_subset.md) — an 8-feature orthogonal
  subset reaches 80% of the full-47-feature LOPO R². For deployment
  that's the essential minimum.
- Stage 30 noted smaller-than-random LOPO R² but with only 6 prompts
  was noisy. Stage 31 replaces it with 35 prompts and gives the
  honest estimate of 0.341.
