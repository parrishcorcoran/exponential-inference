# Compression Levers — Complete Catalog

All compression axes we've identified across the project. Status legend:
✓ confirmed working in measurements
⏳ in progress
🔲 proposed / TODO

## A. Density levers — per-element compression

### Weights
- ✓ A1. **Weight bits** (Q-bit width per weight) — stage 120, BitNet exists
- ✓ A2. **Weight rank** (SVD factorization of attention/MLP matrices) — Strix stage 119, stage 135
- ✓ A3. **Activation rank** (post-hoc projection) — stage 124b/134 (works only with FT)

### KV cache
- ✓ A4. **K projection rank** (W_K factorization) — stage 135 (4× at quality on 0.6B)
- ✓ A5. **V projection rank** (W_V factorization) — stage 135
- 🔲 A6. **K bits per cache entry** (KIVI-style quantization on cached K)
- 🔲 A7. **V bits per cache entry** (KIVI-style on cached V)
- 🔲 A8. **K vs V differential compression** (K rank-5, V rank-50 — we measured the asymmetry)

## B. Size levers — count reduction

### Model structure
- ✓ B1. **Layer count** — drop layers (cavities cheap individually, stage 128)
- 🔲 B2. **Attention head count** — fewer Q heads (existing technique, untested in our framework)
- 🔲 B3. **KV head count** (GQA → MQA narrowing) — design-time choice usually
- 🔲 B4. **MLP intermediate width** (d_ffn reduction)
- 🔲 B5. **Vocabulary trimming** (LM head row removal for unused tokens)

### KV cache
- 🔲 B6. **Token eviction** (H2O / StreamingLLM-style — we have a better principle)
- 🔲 B7. **Cluster consolidation** (K-means representatives instead of per-token)
- 🔲 B8. **Sliding window** (keep only recent N tokens)
- 🔲 B9. **Token-frequency aware lookup** (common tokens share precomputed slots)

## C. Per-layer schedule — wormhole shape applied per axis

Each density/size lever above can be **per-layer scheduled** to follow
the wormhole topology (aggressive at cavities, conservative at walls).

- ✓ C1. **Per-layer rank schedule** (K and weights) — stage 127, 137 in progress
- ✓ C2. **Per-layer bit schedule** — stage 120 squeeze
- 🔲 C3. **Per-layer MLP width schedule**
- 🔲 C4. **Per-layer KV head count schedule** (novel — narrow at throat, wide at mouths)
- 🔲 C5. **Per-layer cluster count schedule**
- 🔲 C6. **Per-layer K vs V differential schedule** (since K and V have different rank profiles)

## D. Per-position adaptive — within sequence

- ✓ D1. **Per-position certainty signal** — stage 139 (entropy-keyed, replaces H2O)
- ✓ D2. **Per-position rank schedule** — stage 132 measured monotone novelty
- 🔲 D3. **Certainty-driven per-token precision** (combine D1 with bit-budget per position) — TODO, the H2O replacement
- 🔲 D4. **Per-token early exit** (depth varies by certainty)
- 🔲 D5. **Adaptive head/MLP gating** (skip heads/MLPs at high-certainty positions)
- 🔲 D6. **Adaptive context window** (high-certainty positions need less history)

## E. Information-flow / decode-side architectural

- ✓ E1. **Speculative decoding / Medusa heads** (existing, multi-token decode)
- 🔲 E2. **Throat-located heads** (vs standard final-layer Medusa — stage 130 found L21 sweet spot)
- 🔲 E3. **KV-Medusa heads** (predict future K/V from throat, NEW unlock)
- 🔲 E4. **Wide KV-Medusa** (20-50 heads enabled by KV compression density)
- 🔲 E5. **Throat caching** (precompute K/V for shared prefixes, skip mouth 1)
- 🔲 E6. **HRR superposition cache** (one C vector per layer, requires cleanup memory)
- 🔲 E7. **Joint K-V binding** (shared subspace for K and V)
- 🔲 E8. **Cross-layer KV correlation** (same token's KV across layers is correlated)
- 🔲 E9. **Cross-model throat reuse** (precompute large-model throat for small-model decode)

## F. Methodology / training-aware

- ✓ F1. **Slow anneal with finetune** — the recurring signal (stages 117, 119, 120, 135)
- ✓ F2. **Thermostat policy** (try a step on any axis, accept if quality holds) — stage 120, 137
- ✓ F3. **LASER denoising** (low-rank constraint can IMPROVE quality on big models) — Strix stage 119
- 🔲 F4. **Per-axis independent anneal then combine** (build the multi-axis squeeze methodically)
- 🔲 F5. **Cross-axis coupling discovery** (do certain axes interact?)

## G. Cross-architecture validation

- ✓ G1. **Wormhole on Qwen3-0.6B** — stage 111
- ✓ G2. **Wormhole on Qwen3-14B** — Strix stage 117
- ✓ G3. **Wormhole on BitNet b1.58 2B** — stage 142 (sharper, magnitude-driven)
- 🔲 G4. **Wormhole on LLaMA family** — untested
- 🔲 G5. **Wormhole on AlphaFold / protein models** — untested (transferability claim)
- 🔲 G6. **Wormhole on multimodal models** (Vision Transformer, etc.)

---

## Summary count

| Category | Confirmed | TODO | Total |
|---|---|---|---|
| Density (A) | 5 | 3 | 8 |
| Size (B) | 1 | 8 | 9 |
| Per-layer schedule (C) | 2 | 4 | 6 |
| Per-position adaptive (D) | 2 | 4 | 6 |
| Information-flow (E) | 1 | 8 | 9 |
| Methodology (F) | 3 | 2 | 5 |
| Cross-arch validation (G) | 3 | 3 | 6 |
| **TOTAL** | **17 confirmed** | **32 TODO** | **49 levers** |

We've measured/built 17. Another 32 are proposed, designed, or in flight.

## Highest-priority TODOs (by impact × novelty)

1. 🔲 **D3: Certainty-driven adaptive precision** — direct H2O replacement, stacks with everything
2. 🔲 **E4: Wide KV-Medusa (20-50 heads)** — enabled by compression, biggest decode multiplier
3. 🔲 **A8: K vs V differential compression** — exploits asymmetry we measured
4. 🔲 **B7: Cluster consolidation** — front-loaded redundancy from stage 138
5. 🔲 **E5: Throat caching** — long-prompt workload speedup
6. 🔲 **B9: Token-frequency aware lookup** — likely massive on common-token-heavy text
7. 🔲 **A6 + A7: KV cache bit quantization** (KIVI-style) — orthogonal to rank, stacks
8. 🔲 **C4: Per-layer KV head count schedule** — wormhole-shape on a new axis
9. 🔲 **D5: Adaptive head/MLP gating tied to certainty** — combines D1 with E1 ideas
10. 🔲 **E2: L21 (exit-gate) Medusa** — engineering improvement on standard Medusa

## Validated stacks

What we have evidence for combining:
- F1 + A2 + Strix's 14B = LASER effect (rank-3 attention IMPROVES quality)
- F1 + F2 + (A1, A4, A5) on 14B = stage 120's 3.6× compression at quality
- F1 + (A4, A5) on 0.6B = stage 135's 4× KV compression
- C1 + (A2, A4, A5) projected = stage 137 (running) — multi-axis squeeze

## Untouched stacks (predicted to multiply)

- E4 × A4-A8 = wide KV-Medusa enabled by compression (50× decode throughput target)
- D3 × E1 × Compressed cache = full adaptive decode (300× ceiling)

## Date

2026-04-24. Maintain as work progresses. Update status flags as items
land or new levers are discovered.
