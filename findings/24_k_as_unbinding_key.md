# Finding 24 — K is the literal unbinding key. Joint head+decoder training reads tokens off the K-manifold.

**Session:** 2026-04-26 (Mac MPS, Qwen3-0.6B vanilla; Stage 144–159 work).

**Companion:** Finding 23 (Z8 G4 closed-form layer sweep) measured the
*structural* K/V/Q vs layer profile and confirmed L14 is optimal for K, V
and L15 for Q. This finding tests what can be *done* with that signal —
joint head+decoder training to recover token identity from predicted K.

## Claim

The cached K-vector at layer 14 of Qwen3-0.6B is structurally an HRR
(Plate 1995) **unbinding key** — the address half of a (key, value)
bound state — and not just a learned attention weight. Three measurements
support this:

1. **K-prediction is uniform across a 10-token horizon at both 0.6B and
   14B scales.** Cos similarity 0.74–0.81 every offset, no decay. The
   address space is structured by symmetry/role, predictable from h_t.
2. **Substituted K, V at layer 14 preserves token output 96% of the
   time** (oracle drafts). The address book is functionally
   interchangeable when predicted instead of computed.
3. **Real K alone decodes to the correct token at 56.0% top-1** via a
   1M-parameter linear projection feeding the frozen LM head. The
   K-vector contains over half the token signature. Q at the same
   layer decodes at 61.5% top-1, 78.5% top-5.

The first time *predicted* K decodes to a token at meaningful accuracy
required **joint head + decoder training with combined MSE+CE loss on
a single offset.** The previous parallel/multi-offset training mode
produced 12–18% top-1; the focused 1×1 joint setup hit 21% top-1, 46%
top-5. The methodology, not the architecture, was the bottleneck.

## Context — what was known

- **Strix 14B (April 2026):** K-Medusa heads on KV-rank-256 substrate
  — cos_k 0.75–0.81 across 10 offsets, 99.1% top-1 acceptance with
  oracle drafts, projected 5.17× speedup. The novel result was that the
  K-prediction was *uniform* across offsets, not the usual decay.
- **Plate (1995, 2003 book) HRR / Kanerva SDM (1988) / Hopfield Networks
  Is All You Need (Ramsauer 2020):** softmax(QK^T/√d)V is the
  energy-minimization update of a continuous modern Hopfield network.
  K is the stored pattern's address; Q is the probe; V is the content.
- **Standard Medusa heads on Qwen3-0.6B:** 32% / 5% / 2.5% / 2.5% / 1.9%
  (head 1–5). Token prediction from h_t is too weak for spec decoding
  to clear baseline.

## What was measured this session

### M1. K-Medusa replicates at 0.6B

`scripts/pipeline_kv_medusa_06b.py` — 10 heads, 300 steps each,
focused training one offset at a time. cos_k stable in **0.71–0.77**
across all 10 offsets. cos_v 0.23–0.41 (V uniformly weaker — same
structural pattern as Strix's 14B). The K-uniformity finding is
scale-stable, not a 14B artifact.

### M2. KV substitution acceptance

`scripts/pipeline_kv_medusa_06b_token_test.py` — substitute layer-14
K, V at draft positions with predicted values; check whether top-1
output token matches the no-substitution baseline. **96.0% mean
acceptance across 10 offsets, 50 anchor positions per offset.** Strong
evidence that predicted K, V is a *functionally correct address* even
at cos 0.75.

### M3. Standard Medusa fails token-side; KV doesn't rescue it

The combined token-Medusa + KV-Medusa pipeline gives essentially zero
KV gain (`pipeline_kv_medusa_06b_combined.py`). Diagnosis: when draft
tokens are wrong, the **Q at that position is corrupted by the wrong
input embedding propagating through layers 0–13**. KV substitution
fixes the address book and content store, but Q is the probe, and a
wrong probe unbinds the wrong filler. Conditional KV-Medusa
(`pipeline_kv_medusa_06b_combined_cond.py`) confirmed: even when KV
heads condition on the candidate token, the Q-path corruption
dominates.

### M4. K-decoder real-K ceiling: 56.0%, Q-decoder real-Q ceiling: 61.5%

`pipeline_kv_medusa_06b_unbind.py` — 1M-param linear projection,
1000 steps on real cached K → frozen LM head. **Real K → token
top-1 = 0.560 on val.** Q-decoder analog
(`pipeline_kv_medusa_06b_q_only.py`): **Real Q → token top-1 = 0.615,
top-5 = 0.785.** Q is the more informative single stream — consistent
with HRR (the probe is more token-conditional than the address).

### M5. Predicted-K decoder breaks at noise level cos 0.75

`pipeline_kv_medusa_06b_unbind.py` (v1) and `pipeline_kv_medusa_06b_unbind_v2.py`
(noise-augmented) — both collapse to 12–18% top-1 on predicted K from
the standalone KV-Medusa heads. The token-discriminative directions of
the K-manifold occupy a different subspace than the attention-relevant
directions that MSE-only training optimized for. The **decoder isn't
the bottleneck** (real-K ceiling stayed near 56%) — the head's K
predictions don't land in the decoder's basins.

### M6. Joint MSE+CE training (focused 1×1) finds the right subspace

`pipeline_kv_medusa_06b_joint_one.py` — single head, single decoder,
2000 steps, joint loss `MSE(K, V) + CE(decoder(K_pred) → token)`. Key
results:

| metric | value |
|--------|-------|
| Real-K decoder ceiling (post-train) | 0.529 |
| Predicted-K cos_k | 0.755 |
| **Predicted-K top-1** | **0.210** |
| **Predicted-K top-5** | **0.460** |

This is the first experiment in the session where predicted K
decoded to a token at any meaningful rate. Both cos_k and tok_acc
climbed simultaneously — the head found a region of K-space that is
both attention-correct and token-decodable.

### M7. Annealing pattern preserves anchor; second offset is harder

`pipeline_kv_medusa_06b_anneal_h2.py` — train head 2 + decoder
warm-started from M6. Head 1 (frozen anchor) stayed at 0.220 / 0.460.
Head 2 reached only **0.090 top-1, 0.240 top-5** in 1500 steps. Two
suspects: shared decoder pulls toward offset-1 distribution, and offset
2 carries less token information from h_t. Annealing methodology
preserves anchors but does not auto-extend to deeper offsets without
either per-offset decoders or joint multi-offset retraining.

## Why this matters

The "wormhole" thread (findings 13–22) measured the *structure* of the
K-cache. This finding is the first time the structure has been *read*
to recover token content. The HRR identification (K = unbinding key) is
no longer just an analogy — it's the description of a functional
operation we now have a pipeline for.

The current numbers (21% top-1, 46% top-5 on predicted K at offset 1)
are not yet decoder-quality. The session did not produce a usable
decoder-driven speedup. What it did produce:

- **A clean readout pipeline:** h_t → KV-Medusa head → predicted K →
  decoder → token. 1M parameters total, trained in ~10 min on Mac MPS.
- **A diagnosis of the gap:** the Q-path corruption when draft tokens
  are wrong is what blocks KV-Medusa from composing with token-Medusa
  drafters. Not a training problem; a mechanism problem.
- **Q > K as a token-info stream:** suggests a Q-Medusa head and
  combined K+V+Q decoder could push the ceiling well above 56–62%.

## What to test next

1. **Layer sweep** (in progress, `pipeline_kv_medusa_06b_layer_sweep.py`)
   — for each layer 0..27, train a probe predicting (K, V, Q) at
   layer 14 from h_at_L. Identifies where each head should attach.
2. **K+V+Q combined decoder** — decoder reads concatenated three
   streams. If the streams are complementary (independent views of
   the bound state), combined ceiling could be 70–80% top-1.
3. **Whitening on K input** — Plate's HRR uses unitary vectors
   precisely because they have isotropic unbinding. Σ⁻¹/² preprocessing
   should help disproportionately at deeper offsets.
4. **KV-rank-256 substrate on 0.6B** — Strix's 14B used this and got
   99.1% acceptance vs our 96%. Tests whether the smoother manifold
   carries the prediction quality.

## Reproduce

All scripts in `scripts/` with prefix `pipeline_kv_medusa_06b_*`. Results
in `results/pipeline_kv_medusa_06b_*.json`. Logs in `logs/`. The
joint-trained head + decoder are at `checkpoints/qwen_06b/kv_medusa_head_joint_one_1.pt`
and `checkpoints/qwen_06b/k_decoder_joint_one.pt`.

## Falsifiable predictions

1. **The 21% top-1 number reflects 0.6B's information content, not a
   training ceiling.** A KV-rank-256-compressed 0.6B substrate should
   give noticeably higher numbers (analogous to Strix's 14B 99.1% on
   rank-256 vs our 96% on vanilla).
2. **K+V+Q combined decoder will exceed Q-alone (0.615 top-1).** If
   it doesn't, the streams are highly redundant rather than
   complementary, contradicting the HRR view of K/Q/V as orthogonal
   roles.
3. **Layer sweep will show different optimal layers for K, V, Q.** If
   all three peak at the same layer, the address/probe/content
   decomposition is weaker than HRR predicts.

If predictions 1+2+3 fail, finding 24 needs revision toward "K-prediction
is structural but not extractable as a self-speculation decoder."
