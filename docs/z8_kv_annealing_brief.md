# Z8 project brief: continuous KV rank annealing

## The question

Does continuous rank annealing preserve quality that discrete rank truncation destroys?

## Context

Discrete rank truncation of KV projections is lossy — we've shown
this repeatedly. Stage 38 (rank-128 post-hoc, 8× compression) diverged
at token 1 despite KV directions being preserved. Stage 97 (same rank +
full fine-tune) only partially recovered. Stage 98 (QAT ternary weights)
is recovering — val_ppl down from 1.6M to 3103 in 100 steps — which
validates that gradient-aware compression works when gradient flows
through the constraint.

The obvious next question: can we apply the same "gradient-aware"
principle to rank reduction via a **smooth schedule** instead of
discrete jumps?

## The proposal

Replace each attention layer's `k_proj` and `v_proj` with a module
whose effective rank is a continuous variable ("slider"), controlled
externally. Use SVD factorization as the underlying representation:

    W = U @ diag(S) @ V
    mask[i] = sigmoid((slider - i) / temperature)
    effective_W = U @ diag(S * mask) @ V

At step 0, slider = full rank (d), so mask ≈ all ones — equivalent to
the pretrained model. At step T, slider = target (e.g., 16), so only
top-16 singular directions are active. Between, smooth interpolation.

**The model sees a continuously-tightening rank constraint, not a
discrete jump. Gradient has something meaningful to work with at every
step.**

Optionally anneal temperature from soft to hard across training, so
the mask becomes sharper (closer to a true rank-r projection) as
training progresses.

## Why this might be genuinely novel

Adjacent published work:
- **AdaLoRA** (Zhang 2023): dynamic LoRA rank during fine-tune, but
  for adapters, not base KV
- **DyLoRA** (Valipour 2022): multi-rank training, not annealing
- **DeepSeek MLA**: learned KV compression but trained from scratch
  at fixed latent dim c=512, no annealing
- **Progressive pruning**: weight-magnitude-based, not singular values

**First things to verify:**
1. Run a careful lit search — "KV rank annealing," "continuous rank
   scheduling," "singular-value masking" in the LM context. If there's
   a direct hit, note it and adjust framing.
2. If no direct hit: proceed with the assumption that the specific
   combination (KV-only, autoregressive LM, fine-tune from pretrained,
   continuous SVD-mask annealing) is open.

## Monitors required (the "when does it break" question)

Track at every step:
- Loss and gradient norm
- Effective rank per layer (count of sv with mask > 0.5)
- NaN/Inf detection

Track every N steps (eval cadence):
- Val perplexity on a held-out set
- Attention output cosine to the unmodified teacher on fixed probe prompts
  (so we can see when attention diverges, not just when output does)

Breakpoint criteria (auto-save checkpoint, log, optionally pause):
- Val ppl rises >2× above rolling minimum
- Gradient norm exceeds 5× its rolling average
- Loss fails to descend for 500 consecutive steps
- NaN/Inf in loss or weights

The POINT of the monitors: we want to know at what `slider` value the
model breaks. That tells us the real rank floor for continuous annealing.

## Suggested initial configuration

- Model: Qwen3-0.6B (fast iteration) or Qwen3-4B if 0.6B is too noisy
- KV projection only (don't touch Q, O, or MLP — isolate the axis)
- Start rank: full (d_kv = 1024 for 0.6B, 1024 for 4B)
- Target rank: 16
- Anneal steps: 5000
- Total steps: 6000 (1000 of stabilization at target rank)
- Start temperature: 4.0 (soft mask, wide sigmoid)
- End temperature: 0.5 (sharp mask, near-hard truncation)
- LR: 5e-5
- Data: wikitext-2 is fine for smoke; move to wikitext-103 if patterns look real

## Expected deliverables

1. A script that implements the above (Z8 can pattern-match from
   `scripts/stage103_full_pipeline.py` — there's a `LowRankLinear`
   class there as a starting reference).
2. Results JSON with full training history including slider / effective
   rank / val_ppl at each eval step.
3. A plot or table showing: **slider value vs val_ppl**. The knee of
   that curve is the answer — where does continuous annealing break?
4. Written comparison to our discrete stage 97 result at the same
   final rank. Did continuous get further?

## Coordination with Strix

Strix is running **Qwen Halo** (scripts/stage103_full_pipeline.py) —
discrete round-robin compression covering KV + weights + embedding +
early exit + Medusa. That's the broad stack.

Z8's job is narrow: one axis (KV rank), one technique (continuous
annealing), one question (does it beat discrete at same target rank).

Z8's result feeds into Qwen Halo's Phase 3: if continuous annealing
beats discrete, Phase 3's KV schedule changes from discrete jumps to
a continuous slider.

## Stop condition

Run until one of:
- Final target rank reached (slider stabilized at target for 1000
  steps without quality degradation)
- Breakpoint monitor triggers and doesn't recover
- 48 hours of compute

Whichever first. Report what happened.

## Written writeup requirement

If continuous annealing works (val_ppl within 10-20% of teacher at
rank-16), write up as a short note:
- Method
- Monitor trajectory (which metric moved first as rank dropped)
- Final results vs discrete baseline
- Pointer to the checkpoint for Qwen Halo integration

That note becomes a component of the larger Qwen Halo paper narrative
or stands alone as a shorter submission.
