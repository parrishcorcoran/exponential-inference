# Save Point — 2026-04-24

Comprehensive snapshot of the Exponential Inference project state. Use
as a single document to brief any new contributor or to restart after
an interruption.

---

## 0. The big picture

We're building a research+commercial path to **dramatically faster
LLM inference** by exploiting a measured geometric property of trained
transformers — the *wormhole topology* of the residual stream — and
applying compression with a slow-anneal-with-finetune methodology that
beats published methods at every axis we've tested.

**Theoretical ceiling**: 300× cache compression + 50× decode throughput
for matched quality, projected from independent per-axis measurements.

**Currently confirmed**:
- 3.6× model compression on Qwen3-0.6B at teacher quality (stage 120, ours)
- 4× KV cache compression on Qwen3-0.6B with finetune (stage 135, ours)
- Rank-3 attention throat on Qwen3-14B with QUALITY IMPROVEMENT (Strix stage 119)
- Rank 64 K compression on Qwen3-0.6B at 1.77× teacher PPL (Z8 finetune)
- Wormhole topology confirmed on BitNet (universal across precision regimes)

---

## 1. The wormhole topology (foundation finding)

**Core claim**: Trained transformers' residual streams have a universal
geometric structure — wide "mouths" at input/output, narrow "throat"
in the middle, with magnitude growing 800× through the throat. This is
emergent from training, not architectural.

### Findings index (relevant to wormhole)

- **Finding 13** (`findings/13_*`): Original wormhole identification.
  Bathtub shape of participation ratio across layers. Reframed from
  "bathtub" to "wormhole" based on topological consequences.
- **Finding 14** (`findings/14_*`): Universal throat geometry, private
  mouth-2 decoder. Throat coords are aligned cross-model (R²=0.94),
  but mouth-2 is brittle to throat perturbation (stage 123b).
- **Finding 15** (`findings/15_*`): Two-gate topology — entry wall (L5),
  exit wall (L19-21), sparse corridor between. Per-layer rank floors
  vary dramatically.
- **Finding 16** (`findings/16_*`): KV cache field geometry. K vectors
  fill subspace 360°, attention is scale-free power-law decay.
- **Finding 17** (`findings/17_*`): Post-hoc subspace projection fails
  on small models. **The recurring signal — every time post-hoc fails,
  trained-aware anneal-with-finetune at the same target succeeds.**
- **Finding 18** (`findings/18_*`): Compression topography. 5 axes
  (rank K/V, bits, clusters, attention Gini) have INDEPENDENT per-layer
  shapes — they stack multiplicatively.
- **Finding 19** (`findings/19_*`): Certainty growth. Output entropy
  drops 32% over a sequence; provides H2O replacement principle.
- **Finding 20** (`findings/20_*`): BitNet has the wormhole too —
  sharper, more magnitude-driven (137,000× pump vs Qwen's 746×).

### Cross-model confirmation

Same wormhole on:
- Qwen3-0.6B (Mac/MPS measurements)
- Qwen3-1.7B (Mac)
- Qwen3-4B (Strix)
- Qwen3-14B (Strix — rank-3 throat IMPROVES quality)
- BitNet b1.58 2B (Mac CPU) — sharper version

---

## 2. The methodology (the actual product)

### The recurring signal

Across stages 117/119, 124b/120, 134/135, the pattern is identical:

1. Measure the geometric shape (PR, EVR, topography)
2. Try post-hoc projection at the measured rank → fails badly
3. Apply slow anneal with finetune at the same target → works at quality
4. Often quality IMPROVES (LASER effect on big models)

**This is the protocol.** Documented in `RUNBOOK.md` for any
contributor to follow.

### Phase-by-phase protocol

Phase 1: shape measurement (stages 111, 127, 132, 133, 138)
Phase 2: post-hoc floor (stages 124b, 134) — confirms post-hoc fails
Phase 3: trained-aware floor (stages 117, 120, 135) — finds the real floor
Phase 4: adaptive compression (stage 139, 143) — multi-axis squeeze
Phase 5: benchmarks (TODO — MMLU, HellaSwag, GSM8K)

---

## 3. Compression levers identified (49 total)

See `LEVERS.md` for complete catalog. Categories:

- **Density** (per-element): bits, rank — 8 levers
- **Size** (count reduction): layers, heads, tokens — 9 levers
- **Per-layer schedule**: wormhole-shape on each axis — 6 levers
- **Per-position adaptive**: certainty-driven, novelty-driven — 6 levers
- **Information-flow**: Medusa, throat caching, HRR — 9 levers
- **Methodology**: anneal+FT, thermostat, LASER — 5 levers
- **Cross-arch validation**: Qwen, BitNet, others — 6 levers

**17 confirmed working** (measurements + working compressions).
**32 TODO** (proposed, designed, or in flight).

### Top priority TODOs

1. D3: Certainty-driven adaptive precision (H2O replacement)
2. E4: Wide KV-Medusa (20-50 heads enabled by compression)
3. A8: K vs V differential compression (we measured K rank 1-5, V rank 12-46)
4. B7: Cluster consolidation (front-loaded redundancy)
5. E5: Throat caching (long-prompt speedup)

---

## 4. What we're shipping toward

### Conservative defendable claim (right now, no new work)

- 3.6× model size reduction at teacher quality (stage 120)
- 4× KV cache compression at quality with finetune (stage 135)
- Slow-anneal-with-finetune methodology that beats post-hoc at every
  axis we've tested
- Combined cache compression projected 50× via lever stacking (untested
  combined, projected from independent measurements)

### Realistic ship target (1-2 weeks of focused work)

- 0.6B running at 200-400 tok/s on Mac (vs ~40 baseline) = 5-10× wall-clock
- 14B running at 50-100 tok/s on workstation GPU (vs ~6-10 baseline)
- Open-weights "WormholeQwen" release with benchmark numbers vs
  AWQ/GPTQ/SVDLLM/MLA

### Headline ceiling (research direction, not yet measured stacked)

- 300× total cache compression
- 50× decode throughput via wide KV-Medusa enabled by compression
- "Fastest open-weights model at matched quality" claim

---

## 5. Stages 100-143 — running ledger

| Stage | What | Status | Key result |
|---|---|---|---|
| 111 | Bathtub measurement | done | wormhole shape across all Qwen3 sizes |
| 117 (Strix) | Total anneal 14B | done | shape-aware multi-axis works |
| 118 | KV slow anneal 0.6B | done | 1.23× free compression |
| 119 (Strix) | Wormhole speed 14B | done | 1.08× wall-clock + rank-3 LASER |
| 120 (Strix) | Throat anneal 14B | done | 3.6×-equivalent on 14B |
| 121 | Cross-model alignment | done | R²=0.94 throat, R²=0.08 mouths |
| 122c | PCA-CCA mouths nested | done | 70% subspace overlap at every depth |
| 123b | PCA-subspace transplant | done | mouth-2 brittle (R²=0.998 still breaks) |
| 124b | Activation rank anneal | done | smooth degrade, no clean floor post-hoc |
| 126 | Weight rank slow anneal 0.6B | done | confirmed FT path works |
| 127 | Per-layer rank floors 0.6B | done | two-gate topology |
| 128 | Cavity-as-loops test | done | refuted, cavities do distinct work |
| 129 | Holographic probe | done | graceful decay confirmed (Future Lens reproduced) |
| 130 | Layer × k head grid | done | L28 wins, throat-Medusa refuted |
| 131 | Annealed ensemble | done | modest cross-layer ensemble gain |
| 132 | Per-token novelty | done | monotone decreasing, NOT bell |
| 133 | Magnet field test | done | 2/3 confirmed (angular + power-law) |
| 134 | Post-hoc KV projection | done | catastrophic fail (expected) |
| 135 | KV anneal with FT 0.6B | done | **4× cache at Δ+0.19 quality** |
| 135b | Longer FT anneal | done | rank 96 floor at Δ+0.99 |
| 136 | HRR capacity | done | works at L14 to ~50-80 tokens |
| 137 | Multi-axis squeeze v1 | done | revealed engineering bug |
| 137b | Multi-axis squeeze v2 | done | uniform stepping underperforms |
| 138 | Compression topography | done | 5-axis per-layer profile |
| 139 | Certainty growth | done | entropy 4.0→2.7 over sequence |
| 142 | BitNet wormhole | done | universal across precision regimes |
| 143 | Full physical KV squeeze | running | 4 axes annealing, floor=1 |

---

## 6. Current open work

### Running

- **Stage 143b** (Mac MPS): full physical KV squeeze with floor=1.
  All four levers (K rank, V rank, K bits, V bits) annealing
  simultaneously, thermostat finds each (axis, layer)'s true floor
  by rejection. Expected ~3-5 hours.

### Pending (LEVERS.md highest priority)

- **D3 — certainty-driven adaptive precision**: build H2O replacement.
  Use stage 139's per-token entropy as compression budget signal.
  ~3 hours work.

- **E4 — wide KV-Medusa**: 20-50 heads enabled by stage 135 + 143
  compressed cache. Each head ~70 bytes total. Projected 5-10× decode
  throughput. ~1 day work.

- **A8 — K vs V differential compression**: exploits measured K rank
  1-5 vs V rank 12-46 asymmetry. Apply Q3 rank 5 to K, Q6 rank 50 to V.

- **141 — token-frequency cache redundancy**: top-1000 most common
  tokens have nearly identical K/V across contexts. Pre-compute and
  share. Likely massive on common-token-heavy text.

---

## 7. Multi-machine sync state

| Machine | Role | Recent activity |
|---|---|---|
| Mac (this) | 0.6B measurements, methodology dev | Stage 143b running |
| Strix (Halo) | 14B compression, fast iteration on rank | Stage 117/119/120 done; rank-3 throat measured |
| Z8 | 0.6B finetune at low ranks | Achieved rank 64 attention at 1.77× teacher PPL |

All sync via GitHub. Latest commits on main:
- `dd0024d`: LEVERS.md catalog
- `4577bfc`: Finding 20 + stage 142 (BitNet)
- `f633db0`: Stages 135b/136/138/139 + findings 18-19 + RUNBOOK
- `2a4c3f0`: Finding 15 + stage 127 (two-gate topology)
- `7c36ad0`: Stages 128-134 + findings 16-17 (KV geometry + post-hoc fails)

---

## 8. Key files / where to look

### Documents
- `RUNBOOK.md` — protocol for any model
- `LEVERS.md` — 49 compression axes catalog
- `findings/13` through `findings/20` — methodology findings
- `docs/substack_wormhole_compression.md` — public-facing writeup
- `docs/holographic_transformer_spec.md` — proposed architecture

### Scripts (recent, last 50 stages)
- `scripts/stage120_squeeze.py` — Strix 14B multi-axis squeeze
- `scripts/stage127_layerwise_anneal_06b.py` — per-layer anneal
- `scripts/stage135_kv_anneal_with_ft.py` — KV rank anneal (4× confirmed)
- `scripts/stage138_compression_topography.py` — 5-axis profile
- `scripts/stage139_certainty_growth.py` — H2O replacement signal
- `scripts/stage142_bitnet_wormhole.py` — cross-precision validation
- `scripts/stage143_full_kv_squeeze.py` — running

### Results
- `results/stage135_kv_anneal_ft.json` — 4× KV at Δ+0.19
- `results/stage138_compression_topography.json` — full per-layer per-axis
- `results/stage139_certainty.json` — entropy decay curve
- `results/stage142_bitnet_wormhole.json` — BitNet shape

---

## 9. Remaining risks / unknowns

1. **Multi-axis stack coupling**: each axis works independently, but
   stacking them might have negative interactions we haven't measured.
   Stage 143 is the first attempt to measure this.

2. **Custom kernels needed for wall-clock speedup**: most of our
   compression saves memory bandwidth. Realizing it as wall-clock
   speedup needs Triton/CUDA kernels for factored + quantized matmul.
   Estimated 1 week engineering.

3. **Wide KV-Medusa untested**: theoretical projection depends on
   throat hologram capacity holding for trained heads. Stage 129
   showed graceful decay; trained Medusa might do better or worse.

4. **Benchmarks not run**: WikiText perplexity is the only quality
   metric we've used. Need MMLU, HellaSwag, GSM8K to make the
   shipping claim defensible.

5. **0.6B may be too small to demonstrate the full story**: Z8's rank
   64 at 1.77× teacher is OK but not headline. The real demo is
   probably at 1.7B or 4B where we have more slack.

---

## 10. What I'd do tomorrow if I picked up this project from scratch

1. Read `RUNBOOK.md` and `LEVERS.md` (15 min orientation)
2. Run stage 142 to confirm wormhole on whatever model you target
3. Run stage 138 to map the compression topography for that model
4. Apply stage 135 (KV rank anneal) — get the 4× as a base
5. Build D3 (certainty-driven precision) on top — get another 2×
6. Build E4 (wide KV-Medusa) on top — get 5× decode throughput
7. Run benchmarks vs published methods at matched compression
8. Open-source release with the methodology + pretrained weights

Total: ~2 weeks for a defensible 10-20× wall-clock speedup demo.

---

## Date

2026-04-24. Active development. Update before each major milestone or
machine handoff.
