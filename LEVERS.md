# Compression Levers — Complete Catalog

All compression axes we've identified across the project. Status legend:
✓ confirmed working in measurements
⏳ in progress
🔲 proposed / TODO
✗ falsified / superseded

**Topology framing (revised 2026-04-25, see Finding 22):** drop "wormhole."
Each axis has its own per-layer **topography of cavities (compressible)
and walls (resistant)**. Shape varies by model, scale, and axis. The
strategy is per-layer thermostat anneal — find each layer's tolerable
rank/bits/cluster-count individually. No global throat assumption.

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

## C. Per-layer cavity-and-wall schedule — apply each axis per-layer

Each density/size lever above can be **per-layer scheduled** based on
that axis's measured per-layer tolerance (cavities compress hard, walls
hold rank). The shape is **measured, not assumed** — Finding 22 retired
the global-wormhole assumption. Some axes are uniform, some are noisy,
some are bimodal.

- ✓ C1. **Per-layer rank schedule** (K and weights) — stage 127, 137; 14B cavity anneal (pipeline_step1_5b)
- ✓ C2. **Per-layer bit schedule** — stage 120 squeeze
- 🔲 C3. **Per-layer MLP width schedule**
- 🔲 C4. **Per-layer KV head count schedule** (novel — narrow at cavities, wide at walls)
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
- 🔲 E2. **Cavity-located heads** (place Medusa where K-axis is most compressible — stage 130 found L21 sweet spot on 0.6B)
- 🔲 E3. **KV-Medusa heads** (predict future K/V from compressed cache, NEW unlock)
- 🔲 E4. **Wide KV-Medusa** (20-50 heads enabled by KV compression density)
- 🔲 E5. **Prefix caching** (precompute K/V for shared prefixes — formerly "throat caching")
- 🔲 E6. **HRR superposition cache** (one C vector per layer, requires cleanup memory)
- 🔲 E7. **Joint K-V binding** (shared subspace for K and V)
- 🔲 E8. **Cross-layer KV correlation** (same token's KV across layers is correlated)
- 🔲 E9. **Cross-model KV reuse** (precompute large-model K/V for small-model decode)

## F. Methodology / training-aware

- ✓ F1. **Slow anneal with finetune** — the recurring signal (stages 117, 119, 120, 135, pipeline_step1_5b)
- ✓ F2. **Thermostat policy** (try a step on any axis, accept if quality holds) — stage 120, 137, pipeline_step1_5b
- ✓ F3. **LASER denoising** (low-rank constraint can IMPROVE quality on big models) — Strix stage 119
- 🔲 F4. **Per-axis independent anneal then combine** (build the multi-axis squeeze methodically)
- 🔲 F5. **Cross-axis coupling discovery** (do certain axes interact?)

## G. Cross-architecture / cross-domain validation

- ✓ G1. **K-axis topography on Qwen3-0.6B** — stage 111, finding 18. K is wormhole-like ON the K axis only; V is uniform; full model is per-axis topography.
- ✗ G2. **~~Wormhole on Qwen3-14B~~** — **falsified 2026-04-25 (Finding 22)**. 14B cavity anneal converged to noisy 211–400 K-rank distribution with no clean throat. Earlier "L7-L14 throat" claim was measurement artifact.
- ✓ G3. **K-axis topography on BitNet b1.58 2B** — stage 142 (sharper, magnitude-driven; K has cavity-and-wall structure)
- 🔲 G4. **Topography on LLaMA family** — untested
- ✗ G5. **~~Wormhole on protein models~~** — **measured 2026-04-25, no wormhole.** ESM-2 150M shows high-rank middle bulge (PR peaks at L6=388), opposite of 0.6B language K-axis. Different topology entirely.
- ✗ G6. **~~Wormhole on whale-trained transformer~~** — **measured 2026-04-25, no wormhole.** 6L 3.4M-param model, PR flat 116–144 across all layers.
- 🔲 G7. **Topography on Vision Transformers** — untested
- 🔲 G8. **Topography across MoE experts** — partial (finding 21, MoE shows K-axis cavity-and-wall on Granite)

---

## Summary count

| Category | Confirmed | TODO | Falsified | Total |
|---|---|---|---|---|
| Density (A) | 5 | 3 | 0 | 8 |
| Size (B) | 1 | 8 | 0 | 9 |
| Per-layer schedule (C) | 2 | 4 | 0 | 6 |
| Per-position adaptive (D) | 2 | 4 | 0 | 6 |
| Information-flow (E) | 1 | 8 | 0 | 9 |
| Methodology (F) | 3 | 2 | 0 | 5 |
| Cross-arch validation (G) | 2 | 3 | 3 | 8 |
| **TOTAL** | **16 confirmed** | **32 TODO** | **3 falsified** | **51 levers** |

Three "wormhole exists everywhere" claims falsified (G2 14B, G5 protein,
G6 whale). Two new validation slots opened (G7 ViT, G8 MoE).

## Highest-priority TODOs (revised 2026-04-26)

After 14B cavity-anneal evidence, prioritize **topology-independent levers
that stack with per-layer cavity anneal**:

0. 🔲 **NEXT: Nonlinear Q probe at L15** — Finding 23 dual-layer sweep
   showed Q peaks at L15 with linear cos 0.41 (vs K/V at L14 with 0.83/0.55).
   Linear is a lower bound. Train 2-layer MLP probe (same as KV-Medusa K/V
   heads) on h[L15] for Q at offset +1, 200 gradient steps. If cos jumps
   to 0.7+, build full KVQ-Medusa with K/V at L14 and Q at L15.
1. 🔲 **D3: Certainty-driven adaptive precision** — direct H2O replacement, no topology assumption, stacks with everything
2. 🔲 **A6 + A7: KV cache bit quantization** (KIVI-style) — orthogonal to rank, stacks
3. 🔲 **A8: K vs V differential compression** — exploits asymmetry we measured
4. 🔲 **E4: Wide KV-Medusa (20-50 heads)** — enabled by compression, biggest decode multiplier
5. 🔲 **E3: KV-Medusa heads** (predict K/V) — paired with E4
6. 🔲 **B7: Cluster consolidation** — front-loaded redundancy from stage 138
7. 🔲 **B6: Eviction (certainty-driven)** — adaptive, doesn't need throat
8. 🔲 **C4: Per-layer KV head count schedule** — apply cavity-anneal to a new axis
9. 🔲 **B9: Token-frequency aware lookup** — likely massive on common-token-heavy text
10. 🔲 **E5: Prefix caching** — long-prompt workload speedup

## Validated stacks

What we have evidence for combining:
- F1 + A2 + Strix's 14B = LASER effect (rank-3 attention IMPROVES quality)
- F1 + F2 + (A1, A4, A5) on 14B = stage 120's 3.6× compression at quality
- F1 + (A4, A5) on 0.6B = stage 135's 4× KV compression
- C1 + (A2, A4, A5) projected = stage 137 (running) — multi-axis squeeze
- C1 + A4 on 14B = pipeline_step1_5b cavity anneal (PPL 41.6, coherent)

## Untouched stacks (predicted to multiply)

- E4 × A4-A8 = wide KV-Medusa enabled by compression (50× decode throughput target)
- D3 × E1 × Compressed cache = full adaptive decode (300× ceiling)

## Date

2026-04-25. Revised after Finding 22 (topology revision). Update status
flags as items land or new levers are discovered.
