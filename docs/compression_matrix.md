# Compression Matrix — living document

Shared table where Mac/Strix/Z8 append data points as we measure them.
Every row is ONE compression config on ONE model, with val_ppl and cost.

## Format rules

- **One row per measurement.** Don't edit existing rows; add new ones.
- **Append to bottom.** Keeps history intact.
- **Cite source.** Include the stage/script number and a date.
- **All val_ppl on wikitext-2 validation** (for comparability), or
  note the eval set explicitly if different.
- **Cost bucket** per Strix's marginal cost convention:
  - `FREE_WIN` = val_ppl improves (< -0.1 delta)
  - `free` = |Δ| < 0.5 ppl
  - `cheap` = 0.5 ≤ Δ < 2
  - `moderate` = 2 ≤ Δ < 10
  - `expensive` = 10 ≤ Δ < 100
  - `broken` = Δ ≥ 100 or random output

## Teacher perplexities (reference baseline)

| model | params | teacher val_ppl | source |
|---|---|---|---|
| Qwen3-0.6B | 600M | 28.84 | stage 107 (2026-04-23) |
| Qwen3-4B | 4B | — | (TBD) |
| Qwen3-14B | 14B | 10.56 | Strix qwen_halo_full (2026-04-23) |

## Matrix

| model | axis | value | fine_tune | post_compress_ppl | final_ppl | Δ_ppl | cost | source |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | weight_bits | 8 | no | 29.21 | 29.21 | +0.37 | cheap | stage107 |
| Qwen3-0.6B | weight_bits | 6 | no | 29.94 | 29.94 | +1.10 | cheap | stage107 |
| Qwen3-0.6B | weight_bits | 4 | no | 60.20 | 60.20 | +31.4 | expensive | stage107 |
| Qwen3-0.6B | weight_bits | 3 | no | 38997 | 38997 | ∞ | broken | stage107 |
| Qwen3-0.6B | weight_bits | 2 | no | 52M | 52M | ∞ | broken | stage107 |
| Qwen3-0.6B | weight_bits | 1.58 (ternary) | yes (QAT) | 1.6M | ~302 | ~9× teacher | broken | stage98 |
| Qwen3-0.6B | embed_bits | 8 | no | 28.87 | 28.87 | +0.03 | free | stage107 |
| Qwen3-0.6B | embed_bits | 6 | no | 28.95 | 28.95 | +0.11 | free | stage107 |
| Qwen3-0.6B | embed_bits | 4 | no | 30.59 | 30.59 | +1.75 | cheap | stage107 |
| Qwen3-0.6B | embed_bits | 3 | no | 41.30 | 41.30 | +12.5 | expensive | stage107 |
| Qwen3-0.6B | embed_bits | 2 | no | 6717 | 6717 | ∞ | broken | stage107 |
| Qwen3-0.6B | d_ffn (naive) | 2048 | no | 142 | 142 | +113 | expensive | stage107 |
| Qwen3-0.6B | d_ffn (naive) | 1536 | no | 264 | 264 | +235 | expensive | stage107 |
| Qwen3-0.6B | d_ffn (naive) | 1024 | no | 1126 | 1126 | +1097 | broken | stage107 |
| Qwen3-0.6B | d_ffn (naive) | 768 | no | 2292 | 2292 | ∞ | broken | stage107 |
| Qwen3-0.6B | swiglu_rank (SVD) | 1024 | no | ~28.84 | ~28.84 | ~0 | free | stage92 |
| Qwen3-0.6B | swiglu_rank (SVD) | 512 | no | 4.6k ppl | — | broken | broken | stage92 |
| Qwen3-0.6B | kv_rank (posthoc) | any | no | random output | — | broken | broken | stage38 |
| Qwen3-0.6B | kv_rank (aware) | 64 | yes (300 steps) | 906k | ~1840 | ~55× teacher | broken | stage104c |
| Qwen3-14B | kv_rank | 5120→512 | yes (Qwen Halo) | 76 | 8.72 | -4.2 (of teacher 13.84) | FREE_WIN | Strix qwen_halo_annealed |
| Qwen3-14B | weight_bits | 8 | yes (Qwen Halo) | 9.59 | 9.80 | -0.34 (of teacher 13.84) | FREE_WIN | Strix qwen_halo_annealed |
| Qwen3-14B | embed_bits | 8 | yes (Qwen Halo) | 10.56 | 10.55 | -0.27 (of teacher 13.84) | FREE_WIN | Strix qwen_halo_annealed |
| Qwen3-14B | kv_rank | 5120→384 | yes (Qwen Halo) | 55.5 | 8.94 | -0.44 | FREE_WIN | Strix qwen_halo_annealed |
| Qwen3-14B | weight_bits | 6 | yes (Qwen Halo) | 44.4 | 34.5 | +0.91 (vs teacher 13.84) | cheap | Strix qwen_halo_annealed |
| Qwen3-14B | embed_bits | 6 | yes (Qwen Halo) | 36.8 | 38.6 | +1.02 | cheap | Strix qwen_halo_annealed |
| Qwen3-14B | kv_rank | 5120→256 | yes (Qwen Halo) | 19362 | 28.57 | +0.72 | cheap | Strix qwen_halo_annealed |
| Qwen3-14B | weight_bits | 4 | yes (Qwen Halo) | 1.5M | 1.6M | +11.7 | broken | Strix qwen_halo_annealed |
| Qwen3-0.6B | layer_skip | L27 only (final) | no | 49.4 | 49.4 | +19.8 | expensive | stage109 |
| Qwen3-0.6B | layer_skip | L25-27 (last 3) | no | 132 | 132 | +103 | broken | stage109 |
| Qwen3-0.6B | layer_skip | L0 only (first) | no | 1.75M | 1.75M | ∞ | catastrophic | stage109 |
| Qwen3-0.6B | layer_skip | L0-2 (first 3) | no | 380K | 380K | ∞ | catastrophic | stage109 |
| Qwen3-0.6B | layer_skip | L3-22 (all dead-zone, 20) | no | 88K | 88K | ∞ | catastrophic | stage109 |
| Qwen3-0.6B | layer_skip | every other in dead-zone (10) | no | 1375 | 1375 | ∞ | broken | stage109 |
| Qwen3-0.6B | layer_skip | L3-10 (first-half dead) | no | 770 | 770 | +741 | broken | stage109 |
| Qwen3-0.6B | layer_skip | L10-17 (second-half dead) | no | 152 | 152 | +122 | broken | stage109 |
| Qwen3-14B | uniform_Q8 | 8 bits all layers | no | 11.46 | 11.46 | +0.01 | free | stage112_14b |
| Qwen3-14B | uniform_Q6 | 6 bits all layers | no | 11.72 | 11.72 | +0.27 | free | stage112_14b |
| Qwen3-14B | uniform_Q4 | 4 bits all layers | no | 15.81 | 15.81 | +4.36 | moderate | stage112_14b |
| Qwen3-14B | uniform_Q3 | 3 bits all layers | no | 57414 | 57414 | ∞ | broken | stage112_14b |
| Qwen3-14B | uniform_Q2 | 2 bits (ternary) | no | 61M | 61M | ∞ | broken | stage112_14b |
| Qwen3-14B | position-aware | Q8-edge(w=7) + Q4-mid | no | 12.86 | 12.86 | +1.40 | cheap | stage112_14b |
| Qwen3-14B | position-aware | Q6-edge(w=7) + Q4-mid | no | 13.14 | 13.14 | +1.69 | cheap | stage112_14b |
| Qwen3-14B | position-aware | Q8-edge(w=7) + Q3-mid | no | 758 | 758 | +746 | broken | stage112_14b |
| Qwen3-14B | position-aware | Q8-edge(w=7) + Q2-mid | no | 387K | 387K | ∞ | broken | stage112_14b |
| Qwen3-14B | layer_skip | L0 only | no | 26268 | 26268 | ∞ | catastrophic | bathtub_profile |
| Qwen3-14B | layer_skip | middle (L13-25 avg) | no | ~14 | ~14 | -0.1 | FREE_WIN | bathtub_profile |
| Qwen3-14B | layer_skip | L39 (final) | no | 30.3 | 30.3 | +12.8 | expensive | bathtub_profile |
| Qwen3-14B | mlp_prune | 99% keep (global) | no | 17.2 | 17.2 | +1.5 | cheap | lever_matrix_partC |
| Qwen3-14B | mlp_prune | 90% keep (global) | no | 23.0 | 23.0 | +7.3 | moderate | lever_matrix_partC |
| Qwen3-14B | mlp_prune | 85% keep (global) | no | 31.1 | 31.1 | +15.4 | expensive | lever_matrix_partC |
| Qwen3-14B | head_angle | Givens rotation 0-90° | no | 15.7-19.3 | — | ~0 | gauge_symmetry | lever_matrix_partC |

## Paired interactions (axis × axis)

| model | axis A | axis B | combined final_ppl | Δ | verdict | source |
|---|---|---|---|---|---|---|
| Qwen3-14B | kv_rank=512 | weight_Q8 | 12.47 | +1.91 | shared_budget (8.13× ratio) | Strix orthogonality_verdict |
| Qwen3-14B | kv=512 + W8 + E8 | — | 11.54 | +1.00 | working compressed config | Strix qwen_halo_full |
| Qwen3-14B | kv=128 + W8 + E6 | — | 18.7 | +5.0 | optimal Strix marginal-cost pick | Strix 56191ff |
| Qwen3-14B | bathtub Q5-mid + MLP 95%-mid | — | 12.8 | +1.4 | additive orthogonal | stage115 |
| Qwen3-14B | bathtub Q6/Q4 + MLP 95%-mid | — | 14.5 | +3.1 | additive | stage115 |
| Qwen3-14B | bathtub Q6/Q4 + MLP 90%-mid | — | 15.3 | +3.9 | additive | stage115 |
| Qwen3-14B | bathtub Q5-mid + MLP 90% + E6 | — | 13.4 | +2.0 | **perfectly additive** (0.3+1.5+0.27≈2.0) | stage115 |

## Open questions

- ~~Is weight Q4 rescuable via QAT fine-tune on 0.6B?~~ YES, stage 108: +31→+5.9
- ~~Axis EXPANSION: does boosting heads compensate?~~ Lever matrix Part C: only at exact GQA divisors
- ~~Head angle rotation as lever?~~ NO — gauge symmetry, rotation invariant (Part C)
- Position-aware Q4-mid WITH QAT on middle layers — can we close the gap from +1.4 to ~0?
- Position-aware Q3-mid WITH QAT on 14B — does QAT rescue Q3 like it rescued Q4 on 0.6B?
- Per-layer variable KV rank (bathtub-aware): low rank for middle, full rank for edges
- Per-layer variable MLP pruning (bathtub-aware): aggressive prune middle, keep edges
- Stacked position-aware: Q4-mid weights + KV rank reduction + MLP pruning, all bathtub-shaped
- 4B data points for scaling law between 0.6B and 14B
- Q5 middle on 14B (between Q4 working and Q3 broken) — where exactly is the cliff?

## How to contribute a new row

1. Run your experiment, save results/<stage>.json
2. Append one or more rows to the table above
3. Link the source json in the source column
4. Commit + push to main
