# Finding 22 — Topology revision: no universal wormhole. Cavities and walls per layer, per axis.

**Supersedes the global "wormhole" framing in findings 13, 15, 20, 21.**
**Falsifies G2 ("Wormhole on Qwen3-14B").**

## Claim

The "wormhole" topology — clean rank-1 throat between two mouths — was
overstated. What's actually true:

1. **0.6B-class autoregressive language transformers** show a wormhole-like
   shape on the K-rank axis only (Finding 18: K wormhole, V uniform,
   clustering front-loaded, eviction uniform). Even on 0.6B it was never
   a single global wormhole — it was per-axis topography.

2. **Qwen3-14B has no wormhole.** Per-layer cavity anneal
   (`scripts/pipeline_step1_5b_perlayer_cavity.py`,
   `results/pipeline_step1_5b_cavity.json`) drove K rank per layer down
   to its individual tolerance and converged to a noisy distribution
   (range 211–400, mean ~342) with no clean throat. Random cavities at
   L16 and L34. No global low-middle pattern. Earlier 14B "throat at
   L7-L14" claim (finding 13, Strix stage 117) was a measurement
   artifact, not a structural feature.

3. **Cross-domain transformers (protein ESM-2 150M, whale 6L) show
   completely different topologies** — protein has high-rank middle
   bulge, whale is essentially flat. The wormhole shape is not universal
   across architectures or training regimes.

## What's actually true about per-layer compressibility

Each axis has its own per-layer landscape — a **topography of cavities
(compressible) and walls (resistant)**. The shape varies by:
- model scale (small models cluster cavities in the middle; big models
  scatter them)
- axis (K is non-uniform, V is uniform on 0.6B)
- architecture (autoregressive vs masked, dense vs MoE, full-precision
  vs ternary)

## Evidence

### 14B per-layer K rank after cavity anneal (Strix, 2026-04-25)

```
L 0  380   L10  265   L20  384   L30  380
L 1  361   L11  384   L21  364   L31  276
L 2  276   L12  265   L22  361   L32  400
L 3  276   L13  327   L23  324   L33  380
L 4  380   L14  345   L24  380   L34  211  ← deepest cavity
L 5  361   L15  364   L25  307   L35  342
L 6  400   L16  251   L26  342   L36  380
L 7  364   L17  265   L27  380   L37  400
L 8  364   L18  345   L28  361   L38  361
L 9  345   L19  384   L29  380   L39  342
```

Range 211–400. No throat. Random cavities and walls. Final PPL 41.6 with
coherent generation. This is a **landscape**, not a wormhole.

### ESM-2 150M (protein) — opposite shape

PR peaks in middle (388 at L6), low at embed (18.8). Magnitude pumps
slowly throughout, drops only at final norm. **No bottleneck. No throat.**
Different domain, different topology.

### Whale 6L (3.4M, autoregressive on cetacean codas)

PR essentially flat (116–144). Magnitude flat (0.71× pump). Too small
to develop any structure.

## Reframing for the project

| Old framing | New framing |
|---|---|
| "Wormhole topology" | "Per-axis compression topography" |
| "Mouth + throat + mouth" | "Walls (resist) + cavities (compress)" |
| "Compress toward the throat" | "Anneal per-layer to find each layer's tolerable rank" |
| "Universal across architectures" | "Specific to 0.6B-class autoregressive language K-axis. Not universal." |
| "Resonant geometry" | (dropped — no evidence) |

## What this means for the pipeline

The compression strategy already shifted in practice (pipeline_step1_5b
is per-layer cavity anneal — exactly what this revision predicts). What
needs to change is the **language** in docs and the **assumptions** baked
into other levers:

- **C category levers** (per-layer schedule): rename from "wormhole shape"
  to "cavity/wall schedule". The shape is data, not assumption.
- **F1/F2 (slow anneal + thermostat)**: validated. The thermostat finds
  the cavity-and-wall landscape automatically.
- **G2 (14B wormhole)**: marked falsified.
- **G1 (0.6B wormhole)**: re-marked as "K-axis topography on 0.6B" — the
  shape is real on K but it's not the whole model.

## What survives

- Per-axis compressibility is real (Finding 18).
- Per-layer thermostat anneal works on small AND large models.
- Some layers tolerate dramatic compression; others don't. Random per
  model, but reproducible per fixed model + axis.
- The compression budget exists; we just stop framing it as a single
  geometric object.

## What dies

- "The universe is one fractal loop manifesting via resonance" — empirically
  falsified by protein and whale measurements.
- "All transformers have the same wormhole" — falsified by 14B.
- "Compress toward the throat" — replaced by per-layer cavity search.

## Date + sources

2026-04-25.
- `results/pipeline_step1_5b_cavity.json` (Strix, 14B K cavity anneal)
- `/tmp/protein_wormhole.json` (ESM-2 150M, this session)
- `/tmp/whale_wormhole.json` (whale 6L, this session)
- `scripts/pipeline_step1_5b_perlayer_cavity.py` (the cavity anneal that
  did the test)
