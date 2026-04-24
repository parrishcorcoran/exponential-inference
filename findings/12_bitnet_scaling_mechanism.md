# Finding 12 — BitNet's scaling mechanism: width × ternary = fp-resolution via superposition

## Claim

Ternary weight quantization (BitNet b1.58) works at 8B+ and breaks at 0.6B **not** because "larger models are more robust" generically, but because **ternary's effective output resolution per-neuron scales linearly with model width `d` via superposition of many 3-level weights**, while fp16's resolution is constant. Below some critical width (~4096), ternary cannot match fp16 in high-rank layers.

Formally:
- Output value `y_i = α · Σ_{j=1..d} W_ternary[i,j] · x[j]`
- Resolution: up to `~d` distinct levels per output via superposition
- At `d = 1024` (Qwen3-0.6B): ~1024 levels — insufficient for high-rank transforms
- At `d = 4096` (8B-class): ~4096 levels — matches fp16 effective use

## Supporting data

| model | d_model | stage | post-hoc Q4 Δ | QAT ternary result |
|---|---|---|---|---|
| Qwen3-0.6B | 1024 | 98, 107, 108 | +31.4 cliff | floored at val_ppl 302 (9× teacher) |
| Qwen3-14B | 5120 | Strix Qwen Halo | at <teacher with 8.5× stack | working |
| BitNet 8B | ~4096 | prior work | — | matches fp16 baseline |

Our 0.6B results directly show the floor: stages 98 and 108 fine-tuning from fp16 cannot recover teacher quality with ternary alone. Strix's 14B results show the same pipeline tolerating weight Q8 + KV 512 + embed Q8 at teacher-equivalent perplexity.

## Mechanism via bathtub physics (Finding 13)

The residual stream has two regimes (Finding 13):

**Middle layers** (rank-1 bathtub bottom): the stream carries a single dominant direction plus growing magnitude. Ternary weights preserve direction perfectly at any `d` because sign is preserved at any precision; α handles magnitude. Ternary fits naturally.

**Edge layers** (active L0-2, L22-28): high-rank output, need to preserve many independent directions. Here, ternary's coarse per-weight precision only averages into fp-equivalent output if there are enough weights in the superposition. This is the CLT-like scaling: `O(√d)` error reduction requires sufficient `d`.

This predicts: ternary weights will fail on 0.6B mostly at the edges and tolerate the middle — consistent with stage 98 recovering to val_ppl 302 (direction survives, magnitude fails on edges).

## Predicted architecture

A size-adaptive precision scheme should work:
- **Edges (L0-2, L22-28)**: Q4 or Q6 (moderate precision, preserves rank-k output)
- **Middle bathtub (L3-L22)**: Ternary (rank-1 stream, all it needs is direction + α)

Parameter savings at 0.6B: 20 of 28 layers at ternary, 8 at Q4 → `(20 × 1.58 + 8 × 4) / (28 × 16) = 13%` of fp16 weight storage (~7.7× compression) while respecting layer-role precision needs.

**Untested.** This is the next concrete experiment.

## What the literature got wrong

BitNet b1.58's paper attributed scale-dependent results to "training dynamics" and "compute." Our framing: the scale dependence comes from **effective output resolution per neuron scaling linearly with `d`**. Specifically:

- `d = 1024` (0.6B): not enough terms in ternary sum to match fp16 output distribution
- `d = 4096+` (8B+): sufficient terms; ternary output indistinguishable from fp16

This is a testable prediction that **hybrid precision matched to layer role** would work at any scale — even below BitNet's 8B threshold.

## Related findings

- Finding 10 (holographic compressibility): the boundary/bulk partition this formalizes
- Finding 13 (bathtub): activation-native regime structure
- Stage 107: 0.6B post-hoc marginal cost matrix
- Stage 108: 0.6B QAT rescues Q4 from cliff (+31 → +5.9) but not ternary
- Strix Qwen Halo: 14B tolerates 8-10× stacked compression

## Date + sources

2026-04-23. Derived from stages 98/107/108 (Mac) + Qwen Halo results (Strix).

## Citations if this is written up

BitNet b1.58 — Ma 2024 (2402.17764) — the paper this reframes.
Our measurement pipeline — this repo, stages 72 (cross-size rank), 107, 111.
