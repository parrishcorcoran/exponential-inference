# Wormhole Compression Runbook

Apply our shape-aware annealed compression methodology to any
transformer LLM. Tested on Qwen3-0.6B (Mac/MPS) and Qwen3-14B
(workstation GPU). Should generalize to any dense decoder transformer.

## Prerequisites

- Pretrained model (HF transformers compatible)
- WikiText or similar calibration corpus
- 1 GPU for >1B models, MPS adequate for ≤0.6B
- Python venv: torch, transformers, datasets, numpy

## The protocol in one paragraph

For any model: (1) measure the per-layer wormhole shape via
participation ratio + EVR, (2) measure the per-axis compression
topography (rank/bits/clusters/Gini), (3) measure per-position certainty
growth, (4) anneal each compression axis with finetune to find the
TRUE per-axis floor, (5) combine the per-axis floors via the multi-axis
squeeze with finetune, (6) benchmark the result against published methods
at matched compression ratio.

## Phase 1: SHAPE MEASUREMENT (post-hoc, no training)

### Step 1.1 — Wormhole shape (per-layer rank)
- Run: `scripts/stage111_*.py` (or equivalent — measure PR per layer)
- Output: PR curve across all layers (the wormhole "bathtub")
- Look for: low-rank middle (throat), higher-rank ends (mouths)

### Step 1.2 — Two-gate verification (finding 15 style)
- Run: `scripts/stage127_layerwise_anneal_06b.py` (per-layer anneal)
- Output: per-layer rank floor for K, V, Q, O attention projections
- Look for: walls and cavities pattern (entry gate, corridor, exit gate)

### Step 1.3 — KV cache geometry (finding 16 style)
- Run: `scripts/stage132_kv_rank_pertoken.py` (per-token novelty)
- Run: `scripts/stage133_magnet_field_test.py` (angular spread + decay)
- Output: KV cache field structure
- Look for: monotone-decreasing per-token novelty, scale-free
  attention decay, high angular uniformity

### Step 1.4 — Compression topography (finding 18)
- Run: `scripts/stage138_compression_topography.py`
- Output: per-layer per-axis slack (rank/bits/clusters/Gini)
- Look for: independent shapes per axis (lever independence)

### Step 1.5 — Certainty growth (finding 19)
- Run: `scripts/stage139_certainty_growth.py`
- Output: per-position entropy / top-1 / Gini curves
- Look for: entropy drops 30%+, Gini grows 30%+ across sequence

## Phase 2: POST-HOC FLOOR (find the soft floor)

For each compression axis:

### Step 2.1 — Activation rank
- Run: `scripts/stage124b_rank_anneal.py`
- Anneal rank, no finetuning
- Output: smooth degradation curve per rank
- Floor: where Δ-loss exceeds threshold

### Step 2.2 — Weight rank (KV)
- Run: `scripts/stage134_kv_subspace_projection.py` for KV
- Output: post-hoc projection breaks at all ranks (expected)
- This is the "post-hoc fails" signal that points to phase 3

### Step 2.3 — Quantization
- Run on K, V at varying bit widths (Q8, Q4, Q2, Q1)
- Output: relative reconstruction error per layer
- Floor: where reconstruction error exceeds threshold

If post-hoc floors are reasonable: ship without finetuning.
If post-hoc floors are too high (typical for <7B models):
**proceed to phase 3 — this is the recurring signal.**

## Phase 3: TRAINED-AWARE FLOOR (slow anneal + finetune)

This is where the real compression lives.

### Step 3.1 — Single-axis anneal with finetune
- Pick one compression axis at a time
- Slowly reduce target (× 0.85 per step)
- Finetune ~80–200 steps between rank reductions
- Stop when finetuning can no longer recover

Reference: `scripts/stage135_kv_anneal_with_ft.py` (KV rank)
Reference: `scripts/stage120_*.py` (multi-axis squeeze, 14B)

Expected outcomes:
- 0.6B KV rank floor: ~256 (4× compression)
- 14B KV rank floor: ~3 (340× compression)
- Floors scale-dependently — bigger models compress harder

### Step 3.2 — Per-layer anneal
- Re-run anneal with per-layer rank targets (heterogeneous)
- Walls keep higher rank, cavities go lower
- Guided by phase 1 topography measurements

### Step 3.3 — Multi-axis squeeze
- All axes annealed simultaneously with finetune
- Use thermostat policy: try a step on any axis, accept if quality holds
- Each axis preserved separately to avoid breaking others
- Reference: stage 137 (to be built — multi-axis from per-layer floors)

## Phase 4: ADAPTIVE COMPRESSION (per-position)

### Step 4.1 — Certainty-driven precision
- Compute per-token entropy during inference
- Map entropy → compression budget (high entropy = full precision)
- Apply per-token compression based on the budget
- Reference: finding 19, stage 139 measurements

### Step 4.2 — Token-frequency aware (untested — see stage 141)
- Pre-compute K, V for top-1000 most common tokens
- Cache stores delta from precomputed values
- Repetitive content compresses 10–100×

## Phase 5: BENCHMARK

For matching ship credibility:

- WikiText perplexity (current)
- HellaSwag (zero-shot accuracy)
- MMLU (multi-task accuracy)
- GSM8K (reasoning)
- Compare against AWQ Q4, GPTQ Q4, SVDLLM, MLA

Target: match teacher within 2% PPL or zero-shot accuracy at matched
compression ratio, and beat published methods at that quality threshold.

## Sequence of stages we ran

| Stage | What it does | Status |
|---|---|---|
| 111 | Bathtub measurement (wormhole shape) | done |
| 117/120 (Strix) | Multi-axis squeeze on 14B | done — 14B rank-3 throat works |
| 118 | Slow anneal 0.6B KV | done — 1.23× compression |
| 124b | Activation rank anneal | done — no clean floor without FT |
| 126 | Weight rank slow anneal 0.6B | done — confirmed FT path |
| 127 | Per-layer rank floors 0.6B | done — two-gate topology |
| 128 | Cavity loop hypothesis | done — refuted |
| 129 | Holographic probe | done — graceful decay confirmed |
| 130 | Layer × k head grid | done — L28 wins |
| 131 | Annealed ensemble | done — modest ensemble gain |
| 132 | Per-token novelty | done — monotone decreasing |
| 133 | Magnet field test | done — 2/3 confirmed |
| 134 | Post-hoc KV projection | done — fails (expected) |
| 135 | KV anneal with FT | done — 4× at quality on 0.6B |
| 136 | HRR capacity | done — works at L14, weak elsewhere |
| 138 | Compression topography | done — five-axis profile |
| 139 | Certainty growth | done — H2O replacement principle |
| **137** | **Multi-axis squeeze (combine all axes)** | **next** |
| 140 | K-vs-V differential compression | proposed |
| 141 | Token-frequency cache redundancy | proposed |

## Key empirical findings (the recurring signal)

**Post-hoc projection fails on small models.** Variance rank ≠ information
rank. Token-disambiguating information lives in the long tail of small
singular values. Every time post-hoc fails, slow-anneal-with-finetune at
the same target succeeds. This pattern holds across: residual stream
rank (stage 124b → 120), attention rank (stage 119 → Strix 14B),
KV cache rank (stage 134 → 135), and is confirmed at three independent
compression axes.

**Bigger models tolerate more aggressive post-hoc compression.** Crossover
scale roughly between 1.7B and 7B. Below: must use slow anneal + FT.
Above: can skip finetuning. Universal protocol: always anneal + FT.

**Compression axes are independent.** Per-layer shapes differ across rank,
bits, clustering, and attention concentration. This means multi-axis
compression stacks multiplicatively, not additively.

**Certainty grows over a sequence.** Provides a principled per-position
compression signal that replaces H2O's attention-score heuristic.

## Findings index

- `findings/13_*.md` — Wormhole topology
- `findings/14_*.md` — Universal geometry, private decoder
- `findings/15_*.md` — Two-gate topology
- `findings/16_*.md` — KV cache field geometry
- `findings/17_*.md` — Post-hoc projection floor (the recurring signal)
- `findings/18_*.md` — Compression topography (independent levers)
- `findings/19_*.md` — Certainty growth (H2O replacement)

## Date

2026-04-24. Active development.
