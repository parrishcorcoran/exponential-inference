# Finding 13 — The residual stream is a wormhole: two mouths + throat traversal

**Originally written as "activation bathtub." Reframed to wormhole
topology per physics-consistent analysis.**

## Claim

A trained transformer's residual stream implements a **wormhole-like
geometry** across depth:

- **Mouth 1 (input boundary, L0-2)**: high-rank, wide cross-section.
  Token embedding diffuses into many independent feature channels.
- **Throat (L3-L22)**: rank collapses to ~1, magnitude grows 800×,
  single direction carries information through a narrow topological
  bottleneck.
- **Mouth 2 (output boundary, L22-28)**: rank re-expands, features
  demultiplex into vocabulary-space.
- **Exit flare (L28 with final RMSNorm)**: dramatic re-expansion to
  the output manifold. PR jumps 4 → 27.

This is emergent from training — not built into the architecture.
Same shape appears across Qwen3-0.6B, 4B, 14B, 32B (fractal at
bulk and single-sequence scales per stage 111). Weight matrices have
flat rank across layers; the wormhole geometry emerges from the
*cumulative action* of flat-rank weights on an accumulating residual.

## Why "wormhole" and not "bathtub"

Bathtub was a 1D profile metaphor — it said "high, low, high" but
didn't explain *why* the middle must exist. Wormhole topology
explains:

1. **Why middle layers can't be removed** (stage 109 result): a
   wormhole with its throat cut can't connect the mouths. Each middle
   layer is a segment of the throat path.
2. **Why middle CAN be compressed narrowly** (stage 118's 1.23×
   budget on 0.6B, Strix's 20× on 14B): the throat has cross-sectional
   thickness that scales with model width. You can squeeze the
   cross-section down to the topological minimum, no further.
3. **Why slow annealing finds budget that fast jumps miss**: the
   throat's geometric curvature requires gradual narrowing. Fast
   compression overshoots the topological minimum and breaks the
   connection.
4. **Why axes stack with low coupling**: different compression axes
   correspond to different cross-section directions. Reducing one
   preserves the other dimensions' ability to keep the throat open.
5. **Why edges must stay full-rank**: the mouths connect to high-
   dimensional spaces (vocabulary, embedding). Squeezing a mouth
   collapses the connection to its boundary.

## Empirical data supporting the wormhole interpretation

### 0.6B (stage 111 + stage 118)

| layer zone | PR | cross-section analogy |
|---|---|---|
| L0-2 (mouth 1) | 30-70 | wide opening |
| L3-L22 (throat) | ~1 | narrow bottleneck |
| L22-27 (mouth 2) | 1-4 | re-opening |
| L28 (exit flare) | 27 | dramatic widening |

### 14B (Strix stage 117) — cross-model confirmation

| layer zone | r99 | interpretation |
|---|---|---|
| L0-L6 (mouth 1) | 116→179 | wide opening, rank grows layer by layer |
| L7-L14 (throat) | **1** | literally rank-1 universal channel |
| L15-L27 (narrow passage) | 3→72 | slowly re-opening |
| L28-L40 (mouth 2) | 95→211 | wide exit |

### The critical cross-scale finding

**Throat is rank-1 at BOTH 0.6B and 14B.** Ratio is 1.0×, not 5×.
Throat diameter does NOT scale with `d_model` — it is a **universal
rank-1 channel regardless of model width**.

What DOES scale with model size:
- **Mouths widen**: 0.6B r99 ~150 vs 14B r99 ~200
- **Throat length shortens (relatively)**: 0.6B = 82% of layers in
  throat, 14B = 50% of layers in throat

Bigger models use extra layers to WIDEN THE MOUTHS and SHORTEN THE
THROAT TRAVERSAL, not to widen the throat itself.

`||h||` grows monotonically from 0.8 at L0 to 680 at L27 (geodesic
length through the throat), then RMSNorm drops it to 103 at L28
(exit renormalization). This matches classical wormhole geometry
where geodesics through the throat have defined length.

## HRR / holographic physics connection

The wormhole reframing connects directly to the AdS/CFT-style
holographic principle:

- **Mouths (boundaries)**: high-rank surfaces where observable
  quantities live (tokens and vocabulary)
- **Throat (bulk)**: low-rank geometric encoding of the boundaries
- **Forward pass**: geodesic traversal through the bulk

This maps the measurement to actual physics:
- The boundary CFT has more degrees of freedom per unit of geometry
  than the bulk — matches our PR data
- The bulk is a COMPRESSED reconstruction of the boundary — matches
  the rank-1 middle
- **Holographic principle**: the bulk can be reconstructed from the
  boundary alone — suggests middle layers are information-theoretically
  redundant given sufficient boundary data

## Predictions from wormhole topology (testable)

**1. Throat RADIUS is universal (rank-1), MOUTH WIDTH scales with model.**
**CORRECTED** per Strix cross-model confirmation. Throat is rank-1 at
every scale. What grows with model width is mouth radius (r99 ~150
at 0.6B vs ~200 at 14B) and throat LENGTH in layers (82% vs 50% of
layers in throat). Compression budget per model depends on the
mouth/throat ratio.

**2. ER = EPR** analog: the two mouths are "entangled" in the sense
that L0 and L28 state are tightly coupled through the throat. Perturbing
one mouth's state changes the other's. Classical early-exit probes
already exploit this (Strix's Medusa heads).

**3. Bulk reconstruction from boundaries**: the middle 20+ layers of
a transformer should be theoretically replaceable by a direct learned
map from L0 state → L28 state, provided the map has sufficient
capacity to encode the throat geodesic. **Untested** — this would
be a big experiment, essentially training a learned shortcut.

**4. Exotic-matter analog**: wormholes need negative energy density
to stay open. In our setting, the "exotic matter" keeping the throat
open is the specific weight configuration learned during training.
Untrained models would have a collapsed throat (no coherent middle
flow). **Testable** — measure bathtub/throat shape in a random-
initialized transformer vs a trained one.

**5. Scale-dependent throat squeeze budget**: the thickness of the
throat (and thus compression budget) at each θ (layer depth) scales
with `d_model`. The whole bathtub/torus shape is scale-invariant in
profile but scale-variant in absolute thickness.

## Implications for compression

The wormhole framing gives three axes per cross-section direction:
- **θ-direction** (depth): can't be removed without breaking connection
- **Radial (rank) direction**: compressible up to throat minimum
- **Angular (precision) direction**: compressible up to rank-dependent minimum

The compression schedule that preserves the wormhole:
- Edge cross-section: near-full (high mouth connectivity needed)
- Middle cross-section: tight narrow throat (minimum for topology)
- Per-axis limits emerge from how each axis affects throat geometry

That's what stage 118 (slow annealing) measured and stage 119
(squeeze test) will generalize.

## Why this framing matters

- **Physically grounded**: maps to real GR / holographic principle,
  not a ad-hoc data pattern
- **Explains compression limits**: topology gives a principled floor
- **Predicts ER=EPR-style mouth coupling**: the two ends of the model
  are entangled, not just causally connected
- **Suggests the holographic transformer spec** (docs/holographic_transformer_spec.md)
  should build the throat explicitly, not hope it emerges

## Date + sources

2026-04-24. Reframed from original bathtub framing (2026-04-23).
Supporting: stage 111 fractal test, stage 109 layer skip, stage 118
slow annealing, stage 112/115 position-aware compression at 14B,
Strix 14B Qwen Halo results.

## Citation notes

The wormhole / AdS-CFT framing here is a metaphor, not a formal
equivalence. Treat it as a structural analogy that makes predictions
that happen to match measurement. Formal reduction would require
showing the residual stream literally satisfies Einstein field
equations or AdS-CFT dictionary — which we have NOT done. The
metaphor's utility is in the predictions it makes and the structural
understanding it provides.
