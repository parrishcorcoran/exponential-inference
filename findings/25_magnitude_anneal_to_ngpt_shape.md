# Finding 25: Unit-norm anneal — pretrained transformer → nGPT-shape

## Context: complementary to Strix's magnitude anneal (commit 8c46c53)

Strix already showed (on Qwen3-4B, commit 8c46c53) that **multiplicative magnitude shrinkage** is a real third compression axis: 13% global magnitude removal *improves* quality, 17% is near-baseline, wall hits at 20%. That's compression along *total energy* — every weight × shrink factor, fine-tune norms.

This finding tests a *different* operation that also targets weight magnitude: **uniformizing magnitude across rows** rather than shrinking it globally. Each linear's rows get projected toward unit L2 norm. Total magnitude doesn't necessarily decrease — it gets redistributed.

The two are orthogonal:
- **Strix's anneal**: scales magnitude → 0 along the global axis
- **This anneal**: equalizes per-row magnitude → 1 along the geometric axis (puts the model on the unit hypersphere)

Both are forward-time projections with FP master weights and norm-fine-tune absorption. Different shape, same recipe.

## Why this specific shape

The end-state of unit-norm-row geometry is the architecture of nGPT (Loshchilov et al., NVIDIA, arXiv 2410.01131, Oct 2024) — but nGPT trains from scratch. Nobody has shown the conversion path from a normal pretrained checkpoint, per a 327-cite literature pass run separately.

Possible second-order benefits if the anneal lands clean:
1. **nGPT-style training speedup transfers** to the converted model on subsequent fine-tunes. nGPT reports 4–20×; we'd capture a fraction depending on which other nGPT pieces (embedding norm, residual re-norm, eigen-LR) get added next.
2. **Sub-ternary quantization becomes natural** because per-row magnitude info is *gone*, not just constrained. Direction-only weights are 1-bit cheap.
3. **Stacks with Strix's anneal** — equalize first, then shrink globally. Could push past 20% energy removal because the variance was the load-bearing part.

## What we measured on Mac (Qwen3-0.6B)

**Diagnostic** (`scripts/diag_row_norms.py`): the base model is *already* nearly spherical.
```
Total rows: 344,064 across 196 linears
mean: 0.969    median: 0.912    cv: 0.32
p1: 0.39       p99: 2.03        spread (p1–p99): ~5×
```
Pretraining nearly selects unit-norm rows on its own. Closing the remaining gap is small.

**Anneal early signal** (`scripts/pipeline_unit_norm_anneal.py`, τ=0 → 1 in 10 drops of 0.1, 2000 steps each, streaming OWT):
```
DROP τ=0.10:
  step 250/2000:  train=3.93  val=3.73  d=-0.18   ← below base
  step 500/2000:  train=3.94  val=3.84  d=-0.07   ← below base
```
Val CE drops *below* base under mild projection pressure. Geometry isn't fighting the LM loss.

## How Strix should run this on Qwen3-4B

(Independent of the magnitude-shrink anneal Strix already finished. This is the geometric variant.)

**Step 1: diagnostic.**
```bash
CHECKPOINT="Qwen/Qwen3-4B" .venv/bin/python scripts/diag_row_norms.py
```
Predict: 4B's row-norm CV is *lower* than 0.6B's (bigger pretrained models converge to more uniform geometry). Compare directly.

**Step 2: smoke test.**
```bash
CHECKPOINT="Qwen/Qwen3-4B" RUN_TAG="strix_4b" SMOKE=1 \
  .venv/bin/python scripts/pipeline_unit_norm_anneal.py
```

**Step 3: full anneal.**
```bash
CHECKPOINT="Qwen/Qwen3-4B" RUN_TAG="strix_4b" \
  STEPS_PER_DROP=2000 BATCH=1 GRAD_ACCUM=8 SEQ_LEN=128 LR=2e-5 \
  .venv/bin/python scripts/pipeline_unit_norm_anneal.py
```
Checkpoints: `checkpoints/Qwen_Qwen3-4B/magnitude_anneal_strix_4b_*.pt` (saved every 2000-step block).
Results: `results/pipeline_magnitude_anneal_strix_4b.json`.

**Memory:** 4B bf16 ~8GB params + grad/Adam ~32GB. Strix Halo's 128GB unified should be comfortable.

**Key signal to look for:** does **per-drop recovery accelerate as τ grows**? If later drops (τ=0.7, 0.8, 0.9) recover faster than earlier drops (τ=0.1, 0.2, 0.3), that's partial nGPT-speedup showing up empirically — a transfer of nGPT's training-speedup claim to a pretrained checkpoint via anneal, by us.

## Composing with Strix's magnitude shrink

If both anneals work independently, the natural follow-up is the **stacked** experiment:

1. First: unit-norm anneal (this one) — get the model to the sphere
2. Then: magnitude shrink (Strix's recipe) — see how much further we can compress past 20% on a sphere-converted model

Hypothesis: Strix's 20% wall might move because the per-row variance that this anneal removes was *itself* part of what failed at 20%. If true, sphere conversion + shrink gives more total compression than shrink alone.

## Caveats

- Mac result is at τ=0.10, 500 steps. Need full 0.1→1.0 sweep before claiming the conversion holds.
- Only one piece of nGPT's full geometry (weight rows). Embedding norm, residual re-norm, eigen-LR are separate follow-ups.
- 327-cite negative search doesn't mean novel forever — just unbuilt at the time of writing.

## Files

- `scripts/pipeline_unit_norm_anneal.py` — env-var configurable forward-time row projection anneal
- `scripts/diag_row_norms.py` — measures base row-norm distribution
- `results/diag_row_norms.json` + `.png` — Mac result on Qwen3-0.6B base
- `results/pipeline_magnitude_anneal_smoke.json` — Mac smoke test
- `results/pipeline_magnitude_anneal.json` — Mac in-flight, will be filled in when run completes
