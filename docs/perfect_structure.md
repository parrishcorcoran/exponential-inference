# Perfect Structure: Target Architecture for Binary-Quantized LLM

**Working target for the BetterBonsai recipe.** What we'd design from scratch
given everything Stages 169-204 have validated. Defines the endpoint we
anneal toward. Updated as experiments inform.

## The architecture

```
                Token IDs
                    │
                    ▼
      ┌─────────────────────────────┐
      │  Embedding (FP or INT8)     │  ← decoupled from lm_head
      │  separately scalable         │
      └─────────────────────────────┘
                    │
                    ▼
        For each transformer block:

      ┌─────────────────────────────┐
      │  RMSNorm (gain bounded)     │  ← gains capped, no outliers
      │  capped at ~5, mean ~1       │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │  Q/K/V/O Projection (binary) │  ← signs in 1 bit
      │  Per-128-group: scale + bias │  ← Bonsai-style + per-row α
      │  Per-row α (Stage 169 bridge)│
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │  SubLN (BitDistill insertion)│  ← extra norm before MHSA-out
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │  Residual add                │  ← optional eigen-LR α_A
      └─────────────────────────────┘

        ... similar for MLP block (gate/up/down) ...

                    │
                    ▼
      ┌─────────────────────────────┐
      │  Final RMSNorm              │
      └─────────────────────────────┘
                    │
                    ▼
      ┌─────────────────────────────┐
      │  LM head (FP or INT8)       │  ← decoupled, separately scaled
      │  + logit temperature         │
      └─────────────────────────────┘
                    │
                    ▼
                 Logits
```

## Body precision: ~1.26 bits/weight effective

```
Per group of 128 body weights:
  - 128 sign bits           (the binary content)
  - 1 FP16 scale s_g        (per-group magnitude)
  - 1 FP16 bias b_g         (per-group offset, gives 4 effective values)
Per row of 4096 (32 groups):
  - 1 FP16 α_r              (per-row magnitude bridge)

Total per row: 4096 + 32×16 + 32×16 + 16 = 5168 bits
                         = 1.262 bits/weight effective
```

Two-tier magnitude compensation: per-group `(s_g, b_g)` handles within-row
distribution, per-row `α_r` handles row-magnitude variation. Cleanly
orthogonal axes.

## Compensation channels — what holds the FP precision

| channel | role | typical magnitude in target |
|---|---|---|
| Per-group `s_g` (32/row) | within-row scale | learned per group |
| Per-group `b_g` (32/row) | within-row offset | learned per group |
| Per-row `α_r` (1/row) | row-magnitude bridge | matches original row L2 |
| RMSNorm gains | residual stream amplification | bounded ≈ [0.1, 5.0] |
| Embedding row-norms | input information density | full FP, possibly scaled |
| LM head row-norms | output projection | decoupled from embed |
| Logit temperature | softmax sharpness | learnable scalar |

## Architectural additions vs vanilla Qwen3

1. **Decouple embed and lm_head** (Qwen3 ties them — adds one trainable matrix)
2. **Insert SubLN** before MHSA-out and FFN-out (BitDistill's lever, ~12% capability)
3. **Add per-row α-bridge** to body linears (Stage 169 mechanism)
4. **Add per-group scales+biases** to body linears (Bonsai mechanism)
5. **Add logit temperature scalar** (separates lm_head magnitude from softmax sharpness)
6. **Optional**: per-layer eigen-LRs `α_A, α_M` on residual additions (nGPT)

## What we've validated (experiments)

- ✓ Magnitude anneal lossless (Stage 169 T2: Δ=0)
- ✓ α-bridge training improves below FP base (Stage 169 T3: Δ=−0.121)
- ✓ Sharpness anneal + body training improves below FP base (Stage 204: Δ=−0.37 at midpoint)
- ✓ Bonsai-style binary + per-row α is lossless re-encoding (Stage 180 T2=T1)
- ✓ Static blanket scaling on FP DOFs is anti-compensation (Stages 184-202)

## What we haven't validated (open questions)

- Combined magnitude + sharpness anneal (Stage 205 candidate)
- Embedding scale anneal (Stage 206 candidate)
- Decoupled lm_head + scaling (Stage 207 candidate)
- SubLN insertion impact on Qwen3 (BitDistill validated on their setup)
- Final K=1 binary quality after full recipe (Stage 208 candidate)
- Coherency / benchmark retention after each anneal (need eval pass)

## Comparison to baselines

| recipe | bits/weight | body shape | tied I/O | RMSNorm | quality |
|---|---:|---|---|---|---|
| Qwen3-0.6B FP16 | 16 | Gaussian | tied | 192× outliers | base |
| Bonsai-8B-1bit | 1.13 | bimodal-ish | tied | partial flatten (max 34) | 89% |
| BitNet b1.58 | 1.58 | ternary | tied | flat (max 1.01) | lossless from scratch (4T tokens) |
| BitDistill | 1.58 | ternary | tied | flat | lossless from pretrained (10B tokens CT) |
| **BetterBonsai (target)** | **~1.26** | **bimodal per-group** | **decoupled** | **flat (cap ~5)** | **≥ Qwen3 base, possibly above** |

## The recipe to reach this structure

```
Phase 0 — Architectural prep (one-shot):
  - Untie embed and lm_head (initialize lm_head = embed.clone())
  - Insert SubLN modules
  - Add per-row α-bridge to body linears (init α = row_norm)
  - Add per-group s_g, b_g trainable to body linears
  - Add logit temperature scalar
  - Add eigen-LRs (optional)

Phase 1 — Magnitude anneal (Stage 169 protocol):
  - Project body rows to unit norm
  - α-bridge restores effective magnitude (Δ=0)
  - Train α + per-group scales/biases briefly (Δ improves)

Phase 2 — Sharpness anneal (Stage 204 protocol):
  - PID-throttled cap descent on RMSNorm gains
  - Body trains to absorb each cap
  - Walk T from initial max down toward target (~5)
  - Drift stays bounded under PID

Phase 3 — Restore RMSNorm (your "give it back" insight):
  - Un-cap RMSNorm gains
  - Body keeps its bimodal shape
  - Optional brief refinement

Phase 4 — Quantize body to K=1 binary:
  - Apply Bonsai-style projection (signs + per-group s_g, b_g)
  - Body shape was already prepped; residual error small
  - α-bridge holds row magnitudes

Phase 5 — Final compensation training:
  - Body weights frozen at binary
  - All FP DOFs trainable: α, s_g, b_g, RMSNorm, embed, lm_head, temp
  - Train to match teacher distribution if available, or just CE
```

## The pitch

> First sub-1.58-bit lossless compression of pretrained LLMs at frontier
> scale. Lower bits than BitDistill (1.26 vs 1.58), open recipe, open
> weights, deployable on any modern GPU via standard binary kernel paths.
> Built by anneal-from-pretrained — no retraining cost.

## Status

This document is a **target spec**, not an achieved result. As of writing,
we've validated the magnitude and sharpness anneal mechanisms separately.
Combined recipe (Stages 205-208) is the next sequence. Each stage adds
or tests one component of the perfect structure.
