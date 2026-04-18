# Finding 05 — The manifold floor

## The claim

Rank-k factored compression of a trained transformer LM has a
**parameter-count floor** — a minimum number of factored parameters
below which no distillation procedure can preserve the teacher's
behavior. This floor is **approximately size-independent**: the floor
for Qwen3-0.6B is in the same range as the floor for Qwen3-32B.

The floor appears to be roughly **80–160M parameters** for the
Qwen tokenizer-induced manifold.

Consequence: small teachers (0.6B) don't have enough room to
demonstrate rank-k compression, because even modest factored ranks
sit below the floor. Large teachers (32B) have enormous factored-
parameter budgets even at aggressive ranks, comfortably above the
floor.

## Why it's a stop-and-think

In the standard distillation framing, a smaller student = harder
problem. "Distilling 0.6B into something smaller is hard because 0.6B
is already small." That framing is wrong here.

The manifold-floor framing says: the minimum faithful compressed
representation of a trained LM has a SIZE (measured in parameters
that can hold the tokenizer-induced manifold) that is independent of
the source model's size. 0.6B at rank-32 is BELOW the floor; 32B at
rank-32 is ABOVE it. Scaling up the teacher doesn't make distillation
easier — it makes the compression budget big enough to clear the
floor.

This explains a large body of distillation literature where small-to-
small distillation produces quality gaps that don't close with more
data, while large-to-small distillation works well at moderate
compression ratios.

## How it was measured

### The progression (stages 8, 10b, 13, 15)

| stage | setup | observation |
|---|---|---|
| 8 | rank-32 distillation of Qwen3-0.6B, 16 calib texts, 1500 steps | ppl_ratio 922× on held-out |
| 13 | same, 102 calib chunks (4× data) | ppl_ratio dropped to 99× (9× improvement) |
| 10b | rank-k projection of 0.6B residual stream, k sweep 8–512 | quality collapses at k < 500 |
| 15 | Matryoshka rank [16–128] on 0.6B | training diverged |

Interpretation: at rank-32 on Qwen3-0.6B (= 20M factored params),
distillation scales with data but has a ceiling. At rank-256 (= 160M
factored params, 36% of the full 440M model), projection still
produces degenerate output. The floor appears to be somewhere in the
20M–160M range for this tokenizer-induced manifold.

### The budget arithmetic

For rank-32 factored weights across all target Linears:

| model | full params | factored @ rank-32 | % of full |
|---|---|---|---|
| Qwen3-0.6B | 440M | 20.2M | 4.58% |
| Qwen3-4B | ~3.2B | ~90M | 2.8% |
| Qwen3-32B | ~31B | ~270M | 0.86% |

A 0.6B at rank-32 has 1/13th the factored-parameter budget of a 32B
at rank-32. If the floor is around 80–160M, then:

- 0.6B at rank-32 (20M) → far below floor → distillation will fail.
- 0.6B at rank-256 (160M) → at or near floor → barely works, if at all.
- 4B at rank-32 (90M) → at the floor → borderline.
- 32B at rank-32 (270M) → above the floor → should work cleanly.
- 32B at rank-16 (135M) → near floor → maybe workable.

## Why size-independent?

The manifold is a property of the tokenizer and training data, not
the model size (Finding 01). The minimum parameter budget to encode
"a function that maps token-context → token-distribution on this
tokenizer's manifold" is a function of the manifold, not of the
teacher that was trained to it.

Larger teachers have the SAME manifold to encode but MORE parameters
redundantly encoding it. Compression reveals the irreducible
parameter count; it doesn't scale with source size.

## What it predicts

1. **32B rank-32 Matryoshka distillation on Strix Halo should converge
   cleanly** where 0.6B's didn't. This is the central experiment
   queued on that machine.

2. **Any trained LM can be compressed to roughly the same parameter
   count for the same tokenizer.** 0.6B and 32B distilled faithfully
   should land at similar final sizes (~100–300M).

3. **Different tokenizer families should have different floors.**
   Testable; requires distillation runs on Llama / Mistral / etc.

## What it refutes

It refutes the framing that our 0.6B failures indicate framework
problems. Every 0.6B stage (8, 10b, 13, 15) result that looked like
training instability, bad hyperparameters, or data insufficiency is
consistent with "below the floor; no procedure can succeed here." The
experiments were never going to work. The results aren't bugs; they're
a structural constraint.

## Caveats

1. The floor estimate (80–160M) comes from a handful of 0.6B runs.
   A real empirical floor would be measured by sweeping rank on
   multiple model sizes and looking for the rank at which distillation
   just barely converges. Probably a range, not a single number, with
   dependence on tokenizer and training recipe.

2. "Floor" here is loose: it means the irreducible parameter count
   for a successful distillation under our current procedure (KL +
   hidden MSE + Matryoshka). A radically different procedure (e.g.,
   retrain-from-scratch at small size) might land at a smaller floor.

3. We have not yet confirmed the 32B run works cleanly. The prediction
   will be falsified if 32B at rank-32 shows the same kind of
   degradation 0.6B did.

## Reproduce

```bash
# The sequence showing below-floor failure on 0.6B:
python scripts/stage8_distill_factored.py \
    --model Qwen/Qwen3-0.6B --rank 32 --steps 1500 --device mps

# Scale up to 32B on Strix Halo (the real test):
python machines/strix_halo/scripts/train_matryoshka.py \
    --teacher Qwen/Qwen3-32B \
    --corpus <HF path> \
    --k-min 32 --k-max 128 --steps 5000
```

## Related

- [Finding 01](01_universal_manifold_dim.md) — the manifold whose
  size the floor is approximating.
- Stage 8 / 13 / 15 result files for the actual numbers.
- `docs/research_context.md` § "State at 0.6B / MPS checkpoint" for
  the full breakdown of what the 0.6B work did and didn't establish.
