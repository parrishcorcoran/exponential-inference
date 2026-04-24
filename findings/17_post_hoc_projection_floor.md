# Finding 17 — Post-hoc subspace projection has no rank floor on small models

Three independent measurements (stages 119, 124b, 134) confirm the
same pattern: **post-hoc projection of any kind into a low-rank
subspace fails catastrophically on Qwen3-0.6B**, regardless of which
quantity is projected.

## The pattern

| Stage | What was projected | Result |
|---|---|---|
| 119 | Activations onto top-k PCA dims at throat | Smooth degradation; no clean floor |
| 124b | Activations across all throat layers, fine-grained k sweep | Δloss climbs continuously from k=1024 down; no cliff |
| 134 | KV cache (K and V) per-layer subspaces | Even rank_K=20 / rank_V=200 (95% EVR) gives PPL > 1M |

In all three, the model is broken by even mild rank truncation
post-hoc. There is no measurable "natural rank" below which we
cannot squeeze.

## Why the wormhole geometry suggested a floor that isn't there

Stages 111, 117, 132 all measured **participation ratio** (variance-
weighted effective rank) and found it shockingly low: PR ≈ 1 in the
throat for both 0.6B and 14B residual streams, PR_K ≈ 1-5 for KV
cache at 0.6B.

PR is the right metric for *variance concentration*. It is the wrong
metric for *information rank*. The two diverge:

- **Variance** is dominated by the top singular values. A long tail of
  small singular values barely affects variance.
- **Information** is carried in the LONG TAIL of small singular values
  for token-level distinctions. Each token's specific identity rides
  on the small dimensions; the dominant axis carries shared magnitude.

So PR=1 means "one direction has 99% of variance" — true. It does
NOT mean "rank 1 captures the signal." The remaining 1% of variance
holds the disambiguating information.

This is the same pattern as **information theory**: high-entropy
signals can have most of their energy in low-frequency modes while
encoding their actual content in high-frequency residual.

## What this rules out, what survives

**Rules out:**
- Post-hoc activation rank reduction at the throat (stage 124b)
- Post-hoc weight rank reduction at small scale (stage 126's quick
  attempt) without long fine-tuning
- Post-hoc KV cache subspace projection (stage 134)
- Anything claiming "the throat is rank-1, just project there"

**Survives:**
- Strix's 14B rank-3 throat with quality improvement — because Strix
  uses ANNEALING + fine-tuning, not post-hoc projection
- Stage 120's shape-aware squeeze (3.6× on 0.6B) — annealed slowly
  with fine-tuning between every step
- Stage 118's KV rank annealing with fine-tuning — slowly tightened
  over many steps

The unifying observation: **finetuning lets the model redistribute
information into the surviving dimensions**. Post-hoc projection
demands the model immediately operate in a subspace it was never
trained to use. Neither holds without retraining.

## The scale dependence

Strix's 14B rank-3 throat works WITHOUT annealing (stage 119 LASER
result). Why?

Hypothesis from Finding 14's "scale-dependent compressibility":
- Bigger models have more "slack" — over-parameterization that
  amounts to noise, removable via low-rank constraint
- Smaller models like 0.6B have nearly-tight parameterization;
  every dim is working
- The crossover scale where post-hoc rank cuts work without retraining
  is somewhere between 1.7B and 7B (untested precisely)

For shipping at any scale below the crossover, **training-aware
compression is mandatory**.

## Practical lesson for compression engineering

The pattern is now well-established. Any compression schedule we
design for sub-7B models must include:

1. **Initial measurement** — use PR or EVR to identify TARGET ranks per
   layer. This is the WHERE.
2. **Slow anneal** — multiplicative rank reduction with fine-tuning
   between each step. This is the HOW.
3. **Per-axis tolerance** — different axes (rank, bits, MLP width, KV
   subspace) have independent slack; anneal each separately.
4. **Quality gate** — back off when fine-tuning can no longer recover.
   The floor emerges from training dynamics, not from spectral analysis
   of the un-trained-into shape.

Strix's stage 117 + stage 119 + stage 120 demonstrate this protocol.
Z8's 0.6B finetune training showed convergence to 1.5-1.8× teacher
PPL at much lower ranks than post-hoc would allow.

## What I should have caught earlier

Stage 132's PR=1-5 measurement was honestly reported but I let it
lead me to predict rank-5 post-hoc projection would work. Stage 134
is the third confirmation that this prediction was wrong. The pattern
should have been clear after stage 124b. Logging this for future
work: **PR is a description of variance distribution, not a recipe
for compression rank**. Don't conflate them again.

## Date + sources

2026-04-24. Stages 119, 124b, 134 (`scripts/stage119_*.py`,
`scripts/stage124b_*.py`, `scripts/stage134_*.py`). Findings 13, 14,
16 establish context.
