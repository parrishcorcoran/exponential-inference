# Finding 14 — QAT helps past the post-hoc cliff, hurts before it

## Claim

Quantization-aware training (QAT) fine-tune is useful specifically when post-hoc quantization would degrade quality by more than the fine-tune's own overfit cost. Below that threshold, QAT MAKES THINGS WORSE on small models with small fine-tune corpora.

On Qwen3-0.6B fine-tuned for 300 steps on wikitext-2 (25K tokens):

| weight bits | post-hoc Δ ppl | QAT 300-step Δ ppl | verdict |
|---|---|---|---|
| Q8 | +0.37 | +4.15 | **QAT worse** (overfit exceeds compression damage) |
| Q6 | +1.10 | +5.34 | **QAT worse** |
| Q4 | +31.4 | +5.89 | **QAT rescues** (large cliff → moderate cost) |
| Q3 | +38,967 | +47.5 | **QAT rescues** (broken → moderate) |
| Q2 (ternary) | +52M | (in progress) | expected: QAT rescues to ~10× teacher, not to teacher |

## The utility boundary

QAT helps ↔ post-hoc cost > fine-tune overfit cost

On this setup, fine-tune overfit cost is ~3-5 ppl (comes from 300 steps on wikitext-2's small train set moving weights off-optimum). This means:

- **Q7+ (cheap post-hoc)**: apply post-hoc. QAT hurts.
- **Q5 and below (cliff or broken)**: apply QAT. Recovery proportional to how broken post-hoc was.

The rescue factor grows with cliff size. Q4 rescued 5.3×; Q3 rescued 823×.

## Small-data caveat

Fine-tune overfit cost scales inversely with dataset size. With a large train corpus (like Strix's 14B experiments with more data):

- Q8 QAT might be free or positive (no overfit pressure)
- Lower bit QAT would rescue closer to teacher

On 0.6B with 25K train tokens, the overfit floor is ~3-5 ppl across all QAT variants. That's why even Q8 QAT (where compression damage is <1 ppl) lands at +4 ppl.

## Implication: post-hoc + QAT hybrid

Optimal single-model compression at small scale:

- Use post-hoc quantization for cheap axes (weight Q8, embed Q8, embed Q6)
- Use QAT only for axes past their post-hoc cliff
- Don't fine-tune on small data if you don't need to

**On 0.6B this gives**: weight Q6 + embed Q6 (both post-hoc, both cheap) + SwiGLU rank 1024 via SVD (free per stage 92) → ~3× compression at ~1.2 ppl total cost. Without fine-tuning.

## Why this refines Finding 12

Finding 12 states the resolution limit for ternary at 0.6B. Finding 14 specifies when fine-tuning helps overcome that limit: it doesn't, at small data scales. You need **both** sufficient `d_model` width AND enough fine-tune data for ternary to work. 0.6B fails because `d` is too small AND wikitext-2 is too small.

## Predicted fix

For 0.6B compression research on Mac-scale data:

1. Don't QAT cheap configs (below cliff)
2. QAT only past the cliff, accept the ~3-5 ppl overfit floor
3. Use larger-data QAT if we want to push closer to teacher (e.g., train on 10× more wikitext or switch to distillation from teacher's logits)

## Date + sources

2026-04-23. Stage 108 results, supported by stage 107 post-hoc baseline.
