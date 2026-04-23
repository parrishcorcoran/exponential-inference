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

## Paired interactions (axis × axis)

| model | axis A | axis B | combined final_ppl | Δ | verdict | source |
|---|---|---|---|---|---|---|
| Qwen3-14B | kv_rank=512 | weight_Q8 | 12.47 | +1.91 | shared_budget (8.13× ratio) | Strix orthogonality_verdict |
| Qwen3-14B | kv=512 + W8 + E8 | — | 11.54 | +1.00 | working compressed config | Strix qwen_halo_full |
| Qwen3-14B | kv=128 + W8 + E6 | — | 18.7 | +5.0 | optimal Strix marginal-cost pick | Strix 56191ff |

## Open questions

- Is weight Q4 rescuable via QAT fine-tune on 0.6B? (**stage 108 in progress**)
- Is embed Q3 rescuable via fine-tune on 0.6B? (**stage 108**)
- Can d_ffn naive shrink recover via fine-tune, or only SVD-rotated version? (**stage 108**)
- Axis EXPANSION: does boosting heads / α / layers compensate for aggressive compression? (**opposite lever test TBD**)
- 4B / 8B / 32B / 70B data points for scaling law fit
- KV compression on 0.6B with aware fine-tune: is there ANY rank that works?

## How to contribute a new row

1. Run your experiment, save results/<stage>.json
2. Append one or more rows to the table above
3. Link the source json in the source column
4. Commit + push to main
