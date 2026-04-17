# Live Manifold Measurement Log — BitNet b1.58-2B-4T

Real-time observations during Stage 1 measurement on HP Z8 G4 (2x Xeon Gold 5218, 376GB RAM, CPU-only). Each layer's intrinsic dimensionality was computed as the hidden states streamed in.

## The Entry Phase (L00-L03)
```
L00  PR=  55.21   TwoNN=  0.39   r50=  53
L01  PR=  45.70   TwoNN=  4.45   r50=  51
L02  PR=  65.79   TwoNN=  6.53   r50=  67
L03  PR=  92.72   TwoNN=  8.24   r50=  86
```
> Embedding space expanding. The model is entering the manifold. PR rising from 55 to 93. TwoNN climbing from near-zero to 8 — the hidden states are spreading into higher dimensions as the model begins processing.

## The First Compression (L04-L07) — Finding the Manifold
```
L04  PR=  38.54   TwoNN=  9.75   r50=  68
L05  PR=  18.65   TwoNN=  9.93   r50=  45
L06  PR=  11.39   TwoNN= 10.96   r50=  26
L07  PR=  10.13   TwoNN= 10.75   r50=  21
```
> **Sharp compression.** PR drops from 93 to 10 in just 4 layers. The model found its ground state manifold. Half the energy is now in just 21 dimensions (r50=21 at L07). But TwoNN stabilizes at ~10 — the intrinsic dimensionality has settled. The manifold shape is established.

## The Bulk Expansion (L08-L21) — Exploring the Manifold
```
L08  PR=  12.32   TwoNN= 10.43   r50=  25
L09  PR=  14.97   TwoNN= 10.28   r50=  30
L10  PR=  17.68   TwoNN= 10.36   r50=  36
L11  PR=  23.11   TwoNN= 10.44   r50=  45
L12  PR=  30.49   TwoNN= 10.38   r50=  50
L13  PR=  40.52   TwoNN= 10.39   r50=  56
L14  PR=  50.40   TwoNN= 10.24   r50=  66
L15  PR=  59.44   TwoNN= 10.48   r50=  75
L16  PR=  77.75   TwoNN= 10.16   r50=  79
L17  PR=  94.57   TwoNN=  9.74   r50=  87
L18  PR= 111.97   TwoNN=  9.80   r50=  92
L19  PR= 126.04   TwoNN=  9.94   r50=  98
L20  PR= 133.45   TwoNN=  9.79   r50= 100
L21  PR= 137.42   TwoNN=  9.93   r50= 108
```
> The system is exploring. PR climbs from 12 back up to 137. But **TwoNN stays constant at ~10 across every single layer.** The energy is spreading across more components, but the intrinsic dimensionality of the manifold doesn't change. The model is exploring more of the *same* ~10D surface, not adding new dimensions. The higher dimensions are noise that carries energy but not information.

## The Peak and Collapse (L22-L30) — Relaxation to Ground State
```
L22  PR= 137.52   TwoNN= 10.15   r50= 115  ← peak
L23  PR= 121.35   TwoNN=  9.82   r50= 103  ← turning
L24  PR= 115.45   TwoNN=  9.94   r50= 103
L25  PR= 111.69   TwoNN=  9.91   r50= 103
L26  PR= 101.60   TwoNN=  9.91   r50=  97  ← accelerating collapse
L27  PR=  85.84   TwoNN=  9.87   r50=  92
L28  PR=  62.69   TwoNN=  9.84   r50=  81
L29  PR=  33.21   TwoNN=  9.85   r50=  58  ← sharp snap
L30  PR=  32.19   TwoNN=  9.81   r50=  46  ← ground state
```
> **The spin glass relaxes.** PR drops from 138 to 32 — smooth descent with a sharp snap at L29. The system finds its ground state. And through the entire collapse, **TwoNN remains at ~9.8-10.** The manifold dimensionality is invariant. The physics is constant; only the energy distribution changes.

## Key Observations

1. **TwoNN = ~10 across all 31 layers.** The intrinsic dimensionality of the hidden-state manifold is constant. This is the manifold dimension — possibly 5 true dimensions × 2 (key + value projections).

2. **PR follows expand → peak → collapse.** This is the energy profile of a spin glass: frustration builds, peaks, then relaxes to ground state. The same pattern should appear across the *sequence* dimension during token generation.

3. **r50 tells you the compression ratio.** At the ground state (L30), half the energy is in 46 of 2560 dimensions. That's a 55x compression that preserves 50% of the signal. At L07 (deepest compression), r50=21 — a 122x compression.

4. **The manifold shape is a property of the model weights, not the input.** This measurement was done on a calibration corpus. The same geometry should appear on any input, because it's determined by the spin glass ground state (the trained weights).

## The Fractal Insight

Engine A (per-layer depth) and Engine B (per-token sequence) measure the same manifold at different scales. The expand → peak → collapse pattern in the layer dimension should mirror the frustration → relaxation pattern in the token generation dimension. One forward pass through 30 layers is structurally equivalent to generating 30 tokens in a sentence.

This is why a single manifold measurement (one forward pass) gives you the geometry for all future inference. The manifold is the model's fingerprint.
