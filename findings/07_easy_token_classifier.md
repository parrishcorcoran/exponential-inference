# Finding 07 — Easy-token routing signals are real but weak

## The claim

Free runtime signals — attention entropy, hidden-state norm, step size —
**do correlate with token-level uncertainty**, but the correlations are
MODEST. Combined linear-regression R² of 0.15–0.29 across two model
sizes means ~15–30% of variance in output entropy is explainable from
these signals alone.

This is strong enough to say **cheap routing is directionally viable**
(we can identify some easy tokens) but weaker than needed to drive
aggressive per-token compute tiering without additional sophistication.

## Why it's a stop-and-think

The "dynamic per-token compute" pitch depends on being able to identify
easy vs hard tokens cheaply. A naive reading of the manifold framework
predicts this should be clean — easy tokens are deep in a basin, hard
tokens are near saddles, free signals should separate them clearly.

The measurement tells a more tempered story: **two universal signals
emerge** (last-layer attention entropy, hidden norm), but neither is a
standalone decision tool. Combined linear regression explains only
15–30% of variance. The gap matters: routing based on these signals
will mis-classify roughly 70–85% of tokens' difficulty relative to
what an oracle would say.

## What the signals mean

### Universal (consistent across 0.6B and 1.7B)

- **attn_entropy_last_layer**: r ≈ 0.25–0.35 vs output entropy. When
  the final layer's attention is spread across many cached tokens
  (high entropy), the output is more uncertain. Makes physical sense:
  final attention is what shapes the logit distribution.
- **hidden_norm**: r ≈ 0.24–0.29 vs output entropy. Larger-norm
  hidden states correspond to more uncertain outputs — consistent
  with "higher energy → more frustrated → harder commit" physics
  intuition.

### Model-dependent

- **step_size**: strong on 1.7B (|r| ≈ 0.21), noise on 0.6B (|r| < 0.03).
  The signal is real but not universal — probably depends on
  training-dynamics specifics.
- **centeredness**: weak, inconsistent sign across models.
- **dH_dt** (saddle detector from stage F): noise for single-step
  token-level prediction at this scale. Possibly useful for
  multi-token trajectory prediction (not tested here).

## How it was measured

### Protocol (stage 24, `scripts/stage24_easy_token_classifier.py`)

1. Load teacher model. Pick 6 diverse prompts.
2. For each prompt, generate 150 tokens. At each step collect:
   - **Features (free from forward pass):**
     - `attn_entropy_{mean, max, min, first_layer, last_layer}` — from
       eager attention weights.
     - `dH_dt` — change in mean attention entropy from prev step.
     - `hidden_norm` — L2 norm of final hidden state.
     - `step_size` — ||h_t - h_{t-1}|| / ||h_{t-1}||.
     - `centeredness` — distance from calibration-final mean.
   - **Labels:**
     - `logit_margin` — top-1 logit minus top-2 logit.
     - `output_entropy` — entropy of softmax distribution.
     - `log_p_top1` — log probability of top-1 token.
3. Compute Pearson correlation (feature, label) for each combination.
4. Fit linear regression of each label on all features combined.
5. Sort tokens into easy (top 30% margin) vs hard (bottom 30%), compare
   feature means.

## The numbers

### Qwen3-0.6B (894 records)

| feature | r(margin) | r(out_ent) | r(log_p) |
|---|---|---|---|
| attn_entropy_mean | +0.209 | -0.108 | +0.167 |
| attn_entropy_max | +0.029 | +0.046 | +0.007 |
| attn_entropy_min | -0.015 | +0.137 | -0.071 |
| attn_entropy_first_layer | +0.055 | -0.095 | +0.090 |
| **attn_entropy_last_layer** | **-0.185** | **+0.347** | **-0.254** |
| dH_dt | -0.030 | +0.074 | -0.060 |
| **hidden_norm** | **-0.136** | **+0.287** | **-0.244** |
| step_size | +0.004 | -0.020 | -0.023 |
| centeredness | -0.033 | +0.080 | -0.099 |

Combined linear R²: **logit_margin 0.166, output_entropy 0.286, log_p_top1 0.221.**

### Qwen3-1.7B (894 records)

| feature | r(margin) | r(out_ent) | r(log_p) |
|---|---|---|---|
| attn_entropy_mean | -0.047 | +0.090 | -0.059 |
| attn_entropy_max | -0.188 | +0.125 | -0.108 |
| attn_entropy_min | -0.099 | +0.140 | -0.112 |
| attn_entropy_first_layer | +0.048 | -0.009 | +0.013 |
| **attn_entropy_last_layer** | **-0.217** | **+0.253** | **-0.195** |
| dH_dt | -0.020 | +0.035 | -0.022 |
| **hidden_norm** | **-0.125** | **+0.240** | **-0.217** |
| **step_size** | **+0.243** | **-0.211** | **+0.167** |
| centeredness | +0.192 | -0.011 | -0.000 |

Combined linear R²: **logit_margin 0.188, output_entropy 0.150, log_p_top1 0.108.**

### Easy-vs-hard classification on 0.6B (top 30% margin vs bottom 30%)

| feature | easy_mean | hard_mean | ratio |
|---|---|---|---|
| attn_entropy_mean | 0.4084 | 0.3888 | 1.050 |
| attn_entropy_last_layer | 0.1888 | 0.2036 | 0.927 |
| hidden_norm | 103.54 | 107.96 | 0.959 |
| step_size | 0.8351 | 0.8441 | 0.989 |
| centeredness | 89.23 | 92.85 | 0.961 |

Even the "best" features differ by <10% between easy and hard tokens.
No clean separation.

## Why not stronger?

Several hypotheses for why the signals don't explain more variance:

1. **Linear regression is too restrictive.** A learned classifier with
   non-linear combinations might do substantially better.
2. **The signals we tested are coarse.** Per-head sharpness
   distributions, multi-step trajectory shape, or cross-layer
   entropy patterns might carry more predictive power.
3. **Token-level difficulty is not monolithic.** The "easy/hard"
   distinction may fragment into multiple distinct difficulty modes
   (end-of-clause vs beginning-of-word vs list continuation, etc.)
   that no single signal captures.
4. **Per-token prediction is intrinsically hard.** The model's
   committed top-1 might depend on specific content that free
   signals don't see.

## What it predicts

If this level of prediction is typical, then:

1. **Binary decisions (use cheap path or full path) are viable** at
   ~70% correctness. Good enough for cost reduction but not for
   quality-critical routing.
2. **Graded rank selection** (k as a function of signals) should
   work similarly: the rank scales with predicted difficulty with
   ~30% noise.
3. **Learned classifier is worth trying.** If a small MLP on these
   features reaches R² > 0.5, the routing story strengthens
   substantially.

## Limitations

1. Two models tested; broader validation needed.
2. Only linear regression tried; a learned non-linear classifier
   might do much better.
3. Only tested on generated text, not held-out text.
4. 894 records is a modest sample; statistical noise non-trivial.

## Reproduce

```bash
python scripts/stage24_easy_token_classifier.py \
    --model Qwen/Qwen3-0.6B \
    --max-new-tokens 150 \
    --device mps \
    --out results/stage24_easy_token_qwen3_0.6b.json
```

## Related

- [Finding 06](06_rsb_descent_profiles.md) — the entropy-profile zoo.
  Those are macro-patterns across entire generations; this finding
  looks at token-by-token prediction.
- [Finding 04](04_head_pruning_redundancy.md) — attention sharpness as
  a routing signal works for head selection at ~100% accuracy; for
  token-level difficulty prediction it's ~30% accurate. Different
  questions, different signal strengths.
- `docs/research_context.md` § "All-dynamic principle" — this finding
  tempers expectations. All-dynamic compute needs this signal to be
  stronger OR needs richer features / learned classifiers.
