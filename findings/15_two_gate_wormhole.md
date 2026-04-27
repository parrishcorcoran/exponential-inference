# Finding 15 — Two-gate wall structure on Qwen3-0.6B (formerly "two-gate wormhole")

> **⚠ PARTIALLY SUPERSEDED by Finding 22 (2026-04-25).** The two-gate
> wall pattern on 0.6B's K-rank axis is real (per-layer rank floors
> measured). The "wormhole" framing is dropped — this is **K-axis
> topography on 0.6B**, not a universal geometry. 14B has noisy cavities
> and walls with no global pattern; protein and whale models have
> entirely different topology.

Per-layer rank floor measurements on Qwen3-0.6B reveal that the
residual-stream wormhole isn't a uniform tunnel — it has a **two-gate
structure** with a sparse interior corridor and transitional walls at
each end.

## Measurement

Stage 127 annealed per-layer weight rank sequentially on layers 2–25
of Qwen3-0.6B (skipping mouths). At each layer: activation-aware SVD
(ASVD whitening), seed from 95% cumulative EVR, geometric anneal
(×0.85) until marginal Δ loss > 0.05 nat, back off, freeze, move on.
L2–L25 processed in ~90 minutes on MPS.

Result:

| L | q/k/v/o rank floor | Absolute rank range | Zone label |
|---|---|---|---|
| 2 | 2/2/2/3 | 2–3 | entry approach |
| 3 | 14/11/14/16 | 11–16 | entry approach |
| 4 | 1/1/1/1 | 1 | cavity |
| **5** | 141/**222**/187/222 | **141–222** | **entry wall** |
| 6 | 1/1/1/1 | 1 | cavity |
| 7 | 85/118/118/134 | 85–134 | wall |
| 8 | 1/1/1/1 | 1 | cavity |
| 9 | 1/1/1/1 | 1 | cavity |
| 10 | 1/1/1/1 | 1 | cavity (triple) |
| 11 | 62/79/89/90 | 62–90 | wall |
| 12 | 4/4/4/5 | 4–5 | soft wall |
| 13 | 6/8/9/10 | 6–10 | soft wall |
| 14 | 4/5/5/6 | 4–6 | soft wall |
| 15 | 1/1/1/1 | 1 | cavity |
| 16 | 23/30/39/41 | 23–41 | wall |
| 17 | 19/27/33/34 | 19–34 | wall |
| 18 | 11/14/18/20 | 11–20 | soft wall |
| 19 | 126/161/200/206 | 126–206 | BIG wall |
| 20 | 137/174/182/232 | 137–232 | BIG wall |
| **21** | **408**/566/549/**729** | **408–729** | **exit wall — hardest in model** |
| 22 | 2/3/3/4 | 2–4 | soft |
| 23 | 1/1/1/1 | 1 | cavity |
| 24 | 1/1/1/1 | 1 | cavity |
| 25 | 106/155/158/257 | 106–257 | wall (approach to mouth 2) |

## Topology

The transformer's "throat" is not a uniform tunnel. It has:

- **Entry gate (L5)** — first major wall. 141–222 rank required.
- **Interior corridor (L6–L18)** — sparse; four cavities (L6, L8–L10, L15)
  interspersed with moderate walls (L7, L11) and soft walls (L12–L14, L16–L18).
- **Exit gate (L19–L21)** — three-layer wall complex peaking at **L21**,
  which needs 408–729 rank out of 1024. The hardest layer in the model.
- **Exit buffer (L22–L24)** — soft wall + two cavities.
- **Mouth 2 approach (L25+)** — wall returns.

The classical wormhole metaphor has a single narrow throat. Ours has
**two narrowing gates with a long sparse corridor between them.**

## Reinterpretation

The two-gate structure implies the transformer learns **two distinct
representational transforms**, not one:

1. **Entry transform (L2–L5)**: expand token embeddings into the
   throat coordinate system. Compresses to ~140-220 rank at the gate,
   then drops to rank 1 immediately after (L6).
2. **Exit transform (L19–L21)**: reverse operation — demultiplex the
   throat's rank-1 representation back toward vocabulary-bound features.
   Peaks at **L21 requiring near-full rank** — this is the
   demultiplexing layer.

The corridor between (L6–L18) does *refinement* on the rank-1 axis:
cavities are amplifiers; soft walls are small adjustments. Consistent
with the "rank-1 throat" claim from finding 13 — but with structural
detail that 13 hid.

## Predictions the two-gate view makes

1. **L21 ablation breaks the model more than any other single layer.**
   Its uniqueness as the hardest wall says it's the demultiplexing
   bottleneck. Stage 109-style layer removal should hit harder at L21
   than at L14.

2. **Removing cavity layers costs less than removing walls.** Predict
   PPL impact correlates with rank floor per layer. Specifically L8–L10
   triple cavity should be removable with a looped replacement.

3. **Two "mouth 1 / mouth 2" labels are wrong.** The model has four
   functional zones:
   - Embedding boundary (L0–L1)
   - Entry gate (L2–L5)
   - Interior corridor (L6–L18) — the actual "throat" per variance rank
   - Exit gate (L19–L21)
   - Output boundary (L22–L27)

4. **Cross-model universality**: if the two-gate structure is
   universal, Strix's 14B should show the same pattern at proportional
   depths (≈12.5% for entry gate, ≈70–75% for exit gate in 40 layers
   = L5 and L28–L30). Currently Strix's wormhole schedule places
   throat at L7–L14 and mouths at L0–L6/L28–L39. Our data suggests
   Strix's "throat" label ALREADY CONTAINS the two-gate structure;
   layers L7–L14 are mostly the entry gate + first part of corridor.

5. **Compression implication**: The exit gate (L19–L21 in 0.6B) needs
   near-full rank. Any compression schedule that treats the entire
   "throat" uniformly will either over-compress L21 (breaks model) or
   under-compress the corridor (leaves budget on the table). Stage
   120's shape-aware squeeze correctly preserves mouths but didn't know
   about the exit gate — it used moderate compression in what we now
   call the exit gate, which is likely why 3.6× was the ceiling.

## Experimental proposal (stage 128, next)

Replace the cavity sequences (L8–L10 triple, L23–L24 pair) with a
small looped block. If transformer depth is redundant refinement
between computational gates, looped iteration should recover quality
at a fraction of the compute. Concrete test:

- Drop L8, L9, L10 (three cavities) from forward.
- Insert a single learned 1-layer block at L8's position.
- Loop it K times before passing to L11.
- Measure PPL recovery vs K.

If quality recovers at small K, this is direct evidence for the
"cavities as loops" interpretation and an immediate ~10% compute
reduction on 0.6B. Scales better on larger models with longer
corridors.

## Date + sources

2026-04-24. Measurements from `scripts/stage127_layerwise_anneal_06b.py`
on Qwen3-0.6B via MPS (lr=5e-5 not used; this was pure post-hoc
ASVD factorization with per-layer anneal, no fine-tuning). Full data
in `results/stage127_layerwise_anneal.json`.

## Citation notes

Per-layer rank floors with this granularity have not been previously
reported in the SVD-LLM / ASVD / SliceGPT / FWSVD literature — those
works apply a uniform rank budget or activation-weighted importance,
not a per-layer anneal to marginal failure. The *two-gate topology*
observation is new as far as we have checked, though the
"non-uniform importance" claim is compatible with Mixture-of-Depths
(Raposo 2024, arXiv:2404.02258) and ShortGPT (Men 2024,
arXiv:2403.03853) findings that middle layers are individually less
important. Neither publishes per-layer rank floors or names a two-gate
structure.
