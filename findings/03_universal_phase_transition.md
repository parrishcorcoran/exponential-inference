# Finding 03 — Universal phase transition at layer 0→1

## The claim

Every transformer LM we have measured has its **largest per-layer
basis rotation at the embedding-to-first-transformer-layer boundary**
(layer 0 → layer 1). The normalized depth of this phase transition is
0.00 in every measured model, regardless of size, tokenizer family,
or architecture.

The rotation at this single transition is uniform across all k
principal directions (the principal cosines are tightly clustered),
meaning layer 1 applies a **uniform global reorientation** from the
embedding basis to a task-oriented feature basis. It is not a
selective "preserve these features, rotate those" projection.

## Why it's a stop-and-think

Deep networks are often depicted as having distributed processing —
each layer contributes a bit, no one layer is "the important one."
This finding says otherwise for the question of basis orientation: a
single layer (the first) does nearly all of the rotation from
embedding space to feature space. Subsequent layers make gradual
refinements.

Two implications:

1. **Layer 1 is structurally special.** It's not just "the first
   layer" — it's the phase boundary between the lookup-table world
   (embedding lookups) and the dynamics world (contextualized
   computation). Every design choice that treats all layers
   equivalently is mis-modeling this boundary.

2. **The embedding basis is not the computation basis.** Layer 0's
   state is literally `embed_tokens[token_id]`. Layer 1 forward is
   the rotation that makes the representation computable. Calibration
   for factored weights MUST happen post-layer-1; the embedding basis
   itself is the wrong basis for the stack to operate in.

## How it was measured

### Protocol (stage 20)

1. For each model, compute per-layer basis `P_i` at rank 32 from
   calibration activations (20 paragraphs, ~200 tokens per model).
2. Compute adjacent-layer subspace overlap
   `overlap(P_i, P_{i+1})` for every i.
3. Find the index with the minimum overlap and its fraction-of-depth.

### Layer-1 rotation character (stage 21)

Compute principal cosines between the text-weighted embedding basis
`P_embed` and the layer-1 activation basis `P_act[1]` via the SVD of
`P_embedᵀ P_act[1]`. If the rotation is uniform, all k principal
cosines are similar. If it is selective, the cosines split into
"preserved" (near 1) and "rotated" (near 0).

## The numbers

### Phase transition location

| model | layers | min-overlap layer pair | min overlap | frac-depth |
|---|---|---|---|---|
| Qwen3-0.6B | 28 | 0 → 1 | 0.188 | 0.00 |
| Qwen3-1.7B | 28 | 0 → 1 | 0.144 | 0.00 |
| Phi-2 | 32 | 0 → 1 | 0.226 | 0.00 |

Phase-transition-depth spread: **0.00**. Universal location.

Source: `results/stage20_cross_model_basis.json`.

### Layer-1 rotation character

| model | embed→layer-1 overlap | top-5 principal cosines | spread |
|---|---|---|---|
| Qwen3-0.6B | 0.188 | 0.35 0.33 0.32 0.30 0.28 | 0.35 |
| Qwen3-1.7B | 0.144 | 0.27 0.25 0.25 0.23 0.23 | 0.27 |
| Phi-2 | 0.226 | 0.43 0.40 0.38 0.37 0.35 | 0.43 |

The top-5 cosines are closely clustered in each model; the spread
(max cosine minus min cosine across all 32 directions) is moderate.
The rotation preserves none of the embedding directions strongly
(no principal cosine near 1), and rotates all of them by similar
amounts. Uniform, not selective.

Source: `results/stage21_curve_shape.json`.

## What it predicts / enables

1. **Matryoshka Pprior for new models:** for any new transformer LM,
   we expect the biggest basis rotation to be at layer 1. Calibration
   for factored weights can be focused most heavily there.

2. **Shared layer-1 rotation across models in a family?** (Untested
   but predicted.) If the universal phase-transition rotation is also
   a family-level invariant, we could pre-compute a layer-1 rotation
   for each tokenizer family and reuse it across models of different
   sizes, bypassing calibration at that layer.

3. **Architectural hypothesis:** layer 1's weights should look more
   "structural" (closer to a simple rotation matrix) than deeper
   layers' weights. Worth inspecting directly.

## Limitations

1. Measured on 3 models; the layer-0→1 location is very clean but
   broader sampling would strengthen.
2. The adjacent-layer-overlap metric is one way to locate a phase
   transition. Other metrics (e.g., spectral gap across layers, change
   in TwoNN) could give different answers and should be cross-checked.
3. We haven't tested whether encoder or encoder-decoder architectures
   (T5, BART) have the same property. Decoder-only is the only tested
   class.

## Reproduce

```bash
python scripts/stage20_cross_model_basis.py \
    --models "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,microsoft/phi-2" \
    --rank 32 --device mps
```

## Related

- [Finding 02](02_universal_rotation_curve.md) — the rotation profile
  across subsequent layers is smooth; the phase transition at 0→1 is
  the one exception.
- [Finding 01](01_universal_manifold_dim.md) — the dimension of the
  basis being rotated is ~9–11 universally.
