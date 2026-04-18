# Finding 02 — Universal rotation curve shape

## The claim

The per-layer basis rotation — how fast the rank-k subspace of
hidden-state activations rotates from layer to layer — has the **same
curve shape across tokenizer families and model sizes** when
normalized to [0, 1] depth. Pairwise Pearson correlation of the
normalized adjacent-layer-overlap curves exceeds **r > 0.97** on every
pair tested, including cross-tokenizer.

Different models differ in the **offset and magnitude** of the curve
(Phi-2 rotates more gradually than Qwen, larger models rotate more in
total), but the **shape** is a universal transformer-LM constant.

## Why it's a stop-and-think

The per-layer subspace rotation is a concrete geometric property of a
trained transformer. One could expect every model to have its own
idiosyncratic rotation schedule — different tokenizers, different
training data, different architectures, different optima. Instead,
all three measured models produce curves that are statistically
indistinguishable in shape.

Two implications follow:

1. **The rotation schedule is a property of the NEXT-TOKEN-PREDICTION
   PROBLEM, not the specific model that solves it.** Any transformer
   solving next-token prediction on broad text ends up with roughly
   the same per-layer basis-rotation profile.

2. **Calibration can be compressed.** If the shape is known in advance,
   the per-layer basis for a new model doesn't need dense measurement —
   sparse measurement plus interpolation along the known curve suffices.

Combined with Finding 01 (universal dim), this gives us BOTH the
dimension AND the shape of the per-layer rotation schedule for free.
Only the offset and magnitude require per-model calibration.

## How it was measured

### Protocol (stage 19, 20, 21)

1. Load a model. Tokenize a ~20-chunk calibration corpus.
2. For each layer `i`, collect input activations → `X_i` of shape `[N, d]`.
3. For each layer, compute the rank-k PCA basis `P_i` as the top-k
   eigenvectors of `X_iᵀ X_i` (centered).
4. Compute adjacent-layer subspace overlap
   `subspace_overlap(P_i, P_{i+1}) = ||P_iᵀ P_{i+1}||_F / √k`.
5. Plot overlap vs normalized depth `i / (n_layers - 2)`.
6. Compare curves across models by resampling to a common 20-point
   normalized grid and computing Pearson r + MAD.

### The random baseline

Two random rank-k subspaces of `d`-dim have expected overlap
`√(k/d)`. At k=32, d=1024: expected 0.177. Empirical (from 30 random
draws): 0.179. So anything > ~0.18 is meaningful alignment.

## The numbers

### Rotation curve endpoints across three models

| model | tokenizer | layers | hidden | phase@0→1 | mean adjacent | first-vs-last |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | Qwen | 28 | 1024 | 0.188 | 0.856 | 0.287 |
| Qwen3-1.7B | Qwen | 28 | 2048 | 0.144 | 0.847 | 0.238 |
| Phi-2 | CodeGen | 32 | 2560 | 0.226 | 0.935 | 0.120 |

Source: `results/stage20_cross_model_basis.json`,
`results/stage21_curve_shape.json`.

### Cross-model curve similarity (normalized-depth resampled)

| pair | Pearson r | mean absolute diff |
|---|---|---|
| Qwen3-0.6B vs Qwen3-1.7B (same tokenizer, different size) | 0.984 | 0.0224 |
| Qwen3-0.6B vs Phi-2 (different tokenizer, different scale) | 0.990 | 0.0752 |
| Qwen3-1.7B vs Phi-2 (different tokenizer) | 0.978 | 0.0835 |

All three pairs: r > 0.97. The mean-absolute-diff term is slightly
larger cross-tokenizer (~0.08) vs within-tokenizer (~0.02), reflecting
the offset difference. But the SHAPE correlation is universal.

### Shape description

The curve (per-layer adjacent-overlap as a function of normalized
depth in [0, 1]):

- **depth ≈ 0**: sharp drop to ~0.14–0.23 (the phase transition,
  Finding 03).
- **depth ≈ 0.1–0.8**: rises to plateau at 0.93–0.98. Most adjacent
  layers share ~95% of their k-dim subspace.
- **depth ≈ 0.9–1.0**: slight decline to 0.85–0.95.

Described as a function `f(normalized_depth)`, it starts at ~0.18 and
quickly rises to a plateau near 0.94.

## What it predicts

1. Any transformer LM not yet measured should produce a curve matching
   this shape when resampled. Testable on GPT-2, Llama-3, Mistral, T5
   with the same protocol.

2. The rotation curve is a "prior" we can use to allocate calibration
   effort: spend more on the phase-transition layer (0→1) and the
   final layers (0.8–1.0); spend less on the mid-stack (0.2–0.7) where
   neighbors are near-identical.

3. Training a fresh model from scratch should produce a hidden-state
   representation whose layer-by-layer rotation, once measured,
   follows this same curve. If it doesn't, the model is undertrained
   or has some architectural anomaly.

## The offset matters too

While the SHAPE is universal, the OFFSET differs:
- Phi-2 has overall higher overlaps (mean 0.935): more gradual rotation.
- Qwen models at 0.85: more abrupt transitions.

Offset correlates loosely with architecture / training setup, not
with size or tokenizer alone. Worth more investigation with additional
models.

## Limitations

1. Three models is not a large sample. The r > 0.97 finding is
   compelling but stronger confirmation needs 8–10 cross-family
   measurements.
2. We used one corpus; the rotation profile depends moderately on
   corpus (Finding 01 references similar corpus sensitivity).
3. Phi-2 uses CodeGen tokenizer, which is descended from GPT-2 BPE.
   A truly distant tokenizer (e.g., T5 SentencePiece, GPT-NeoX BPE,
   ByT5 byte-level) would be a harder test.

## Reproduce

```bash
# Requires Qwen3-0.6B, Qwen3-1.7B, Phi-2 cached
python scripts/stage21_curve_shape.py \
    --models "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,microsoft/phi-2" \
    --rank 32 --device mps
```

## Related

- [Finding 03](03_universal_phase_transition.md) — the consistent
  steep drop at depth 0 is universally at layer 0→1.
- [Finding 01](01_universal_manifold_dim.md) — the DIMENSION of the
  basis being rotated is also universal.
- Parallel Procrustes-alignment analysis in
  `analysis/manifold_training/` confirms >90% correlation on first 9
  dimensions across Qwen3 sizes using a different method.
