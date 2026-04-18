# Finding 01 — Manifold dimension is a tokenizer-family property

## The claim

Trained LLMs sharing a tokenizer converge to the **same intrinsic
hidden-state manifold dimension** at the output layer, as measured by
the TwoNN estimator (Facco et al. 2017). Within the Qwen tokenizer
family — seven models spanning 0.6B → 32B parameters across dense and
MoE architectures, and a ternary-quantized variant — final-layer
TwoNN sits tightly in **9.07–10.89**.

The measurement is invariant within a tokenizer family to:
- Model size (0.6B → 32B: 53× parameter range).
- Architecture (dense, MoE, 1.58-bit ternary).
- Hidden dimension (1024 → 5120).
- Layer count (28 → 64).

Whether this dimension is **universal ACROSS tokenizer families is
not yet established.** Our cross-family sample is small (Phi-2 with
CodeGen tokenizer, BitNet-2B likely on a LLaMA-family tokenizer).
Phi-2 lands at 9.76 (peak 10.1), very close but not proven identical
to the Qwen range. More tokenizer diversity is needed before claiming
cross-family universality.

## Why it's a stop-and-think

The field treats `d_model` (hidden size) as the effective capacity of
a transformer LM. Models scale `d_model` with parameter count: 1024
at 0.6B, 5120 at 32B. A naive read says the "representation" fills
most of that space.

The measurement says otherwise: within a tokenizer family, the actual
intrinsic dimension is roughly constant at ~10 regardless of `d_model`.
The ambient space grows with scale, but the manifold does not.

If the dimension turns out to be **per-tokenizer-family** (the current
safest claim) rather than model-architecture-universal, that's still
striking: it says the manifold dimension is determined by the
language-prediction problem as mediated by the tokenizer, not by any
specific model that solves it. All Qwen-family models end up with the
same shape answer because they're solving the same boundary-condition
problem.

Several consequences follow either way:

1. **Per-token compute scales with manifold dim, not hidden size** (in
   any family where the measurement holds).
2. **Compression is not compression** — a rank-10 factorization is the
   NATURAL representation, not a lossy approximation.
3. **Most of `d_model` at 32B is "waste" for any individual token's
   state.** It exists to support the population of all tokens across
   contexts; any single token uses a 10-dim neighborhood.

## How it was measured

### Protocol (stage 1, `scripts/stage1_measure.py`)

1. Load the model in bf16.
2. Run a 10K-token calibration corpus through it with
   `output_hidden_states=True`.
3. For each layer, collect the hidden states across all positions as
   an `[N, d]` matrix.
4. Compute TwoNN on each layer's matrix (and compute PR + rank
   coverage for comparison).

### TwoNN, briefly

TwoNN estimates intrinsic dimension from the ratio
`μᵢ = r₂,ᵢ / r₁,ᵢ` where `r₁,ᵢ` and `r₂,ᵢ` are the distances from
point i to its first and second nearest neighbors. For data on a
d-dim manifold, μ follows a Pareto distribution with parameter d+1.
Estimator: `d_hat = 1 / mean(log μᵢ)`.

### Validation

Synthetic data with known intrinsic dim recovers it accurately:
- True 3D: TwoNN = 2.95
- True 5D: TwoNN = 5.22
- True 7D: TwoNN = 7.19
- True 10D: TwoNN = 9.49
- Random 2560D noise: TwoNN = 283 (i.e., reports high dim as expected)

The estimator works and the low values we see (~10) are not an
artifact of undersampling or a pathological distance distribution.

## The numbers

### Qwen-family (7 models, same tokenizer)

| model | params | hidden | layers | peak TwoNN | final TwoNN |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | 1024 | 28 | 11.1 | 9.09 |
| Qwen3-1.7B | 1.7B | 2048 | 28 | 12.2 | 8.98 |
| Qwen3-4B | 4B | 2560 | 36 | 12.7 | 9.52 |
| Qwen3-8B | 8B | 4096 | 36 | 13.1 | 9.38 |
| Qwen3-14B | 14B | 5120 | 40 | 13.3 | 9.38 |
| Qwen3-30B-A3B (MoE) | 30B (3B active) | 2048 | 48 | 13.0 | 9.07 |
| Qwen3-32B | 32B | 5120 | 64 | 14.8 | 10.89 |

Final TwoNN range across Qwen family: **9.07–10.89**. Spread under 2.
Confidence in within-tokenizer-family invariance: **high**.

### Cross-tokenizer samples (limited)

| model | tokenizer | params | hidden | peak TwoNN | final TwoNN |
|---|---|---|---|---|---|
| BitNet-b1.58 2B | ~LLaMA | 2B (ternary) | 2560 | 11.0 | 9.81 |
| Phi-2 | CodeGen | 2.7B | 2560 | 10.1 | 9.76 |

Two samples outside Qwen family; both land at final TwoNN in 9.76–9.81,
close to the Qwen range. **This is suggestive but not dispositive** —
we'd need ~5+ additional tokenizer families (GPT-2 BPE, Llama-3 128K,
Mistral 32K SentencePiece, T5 SentencePiece, ByT5 byte-level) before
extending the claim cross-family.

### Trend within the Qwen family

Peak TwoNN scales modestly with model size (11 → 15): larger models
explore a richer mid-stack manifold before collapsing. Final TwoNN
is invariant (9–11). The output-layer dimension is the within-family
constant.

## The layer profile (internal consistency check)

Within a single model, TwoNN follows a predictable pattern:
- Entry layers (0–3): low dim (~3–7), embedding compression.
- Mid-stack (10–25): peak (11–15 depending on model size).
- Final layers: collapse back to ~9–11.

Consistent with a spin-glass energy profile — the representation
expands to explore mid-stack (frustration peaks), then relaxes back
toward a well-defined output distribution.

## What it predicts

1. **Within-family**: any Qwen-family model we haven't measured should
   come in at final TwoNN 9–11.
2. **Cross-family test**: measuring Qwen vs GPT-2 vs Llama-3 vs Mistral
   vs T5 on the same protocol would settle whether the dimension is
   truly per-tokenizer or somewhat universal. Predict: either (a)
   slight differences that cluster by tokenizer family (per-tokenizer)
   or (b) all landing at ~10 (universal). Currently we can't
   distinguish.
3. **Smaller intrinsic dim** (in any family) should correspond to
   tokenizers with richer per-token semantic structure. Partial test:
   Phi-2 (smaller vocab, different tokenizer) has peak 10.1, lower
   than Qwen's 11–15 peaks. Consistent with "smaller vocab → each
   token carries more info → fewer dims needed" but not definitive
   on two samples.
4. **Distillation ceiling** within a family: compression can go to any
   rank ≥ the measured dim of that family. For Qwen, ~10.

## Limitations, honest version

1. Seven of nine measured models share the Qwen tokenizer. The
   "universal" language previously used overstated what the data
   supports; the safer claim is "invariant within the Qwen family,
   suggestive but not proven across tokenizer families."
2. The TwoNN estimator has known biases near the manifold boundary
   and with heavy-tailed distance distributions. We validated on
   synthetic but a rigorous finite-sample error bar is absent.
3. The measurement is sensitive to corpus choice. Different
   calibration corpora give slightly different numbers. We have not
   mapped this sensitivity formally.
4. TwoNN measures intrinsic dim, not linear rank. The two are
   different: a curved 10-dim manifold in 1024-dim space can require
   ~500 linear dims to cover (our r90 data confirms this). Both
   measurements are meaningful; this finding is about the intrinsic
   version.

## Reproduce

```bash
# On any machine with enough memory for the target model:
python scripts/stage1_measure.py --model-id Qwen/Qwen3-0.6B
# Results at results/stage1_manifold.json + .png
```

For models larger than MacBook memory:
```bash
python machines/z8g4/scripts/measure_manifold_large.py \
    --model meta-llama/Meta-Llama-3-70B \
    --out machines/z8g4/results/manifold_llama3_70b.json
```

## The open question that would close this finding

**Does a genuinely different-tokenizer LLM sit at the same ~10 dim?**

The cheapest, cleanest test:
```bash
# Run the same measurement on 5 tokenizer families, same-size models
for model in \
    openai-community/gpt2 \
    meta-llama/Llama-3.2-1B \
    mistralai/Mistral-7B-v0.1 \
    google/flan-t5-base \
    EleutherAI/pythia-1b; do
  python machines/z8g4/scripts/measure_manifold_large.py --model $model ...
done
```

If all land at 9–11: push the claim toward cross-family universal.
If they cluster by tokenizer family: confirm per-tokenizer. Either
answer is an important data point.

## Related

- [Finding 02](02_universal_rotation_curve.md) — the rotation curve
  between these 10D subspaces has the same SHAPE across tokenizer
  families (Pearson r > 0.97 between Qwen and Phi-2), which is a
  STRONGER cross-tokenizer claim than the dim number being identical.
- [Finding 04](04_head_pruning_redundancy.md) — number of active heads
  tracks the manifold dim independently.
- [Finding 05](05_manifold_floor.md) — the manifold dim sets a lower
  bound on compressed-model parameter count.
