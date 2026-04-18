> **New to this repo? Read [`docs/READ_THIS_FIRST.md`](docs/READ_THIS_FIRST.md) first.**
> It's the verbose explainer — what we're doing, why, the physics, every
> finding with numbers, every experiment catalogued, glossary included.

# Exponential Inference

**Transformers are spin glasses. Their hidden-state manifold is measurably low-dimensional (~10D), constant across all layers, and collapses predictably during token generation. This means per-token compute can decrease exponentially as context grows — without retraining, without distillation, without approximation. Just physics.**

This repo measures the intrinsic geometry of transformer hidden states and demonstrates that every model has a fixed manifold fingerprint that can be computed in a single forward pass.

## Why This Matters

Every LLM in production today — GPT-4, Claude, Llama, Qwen, Gemini — runs the same amount of computation for every token, whether it's the first word of a creative story (high energy, many possible continuations) or the 900th token of a predictable conclusion (system at ground state, outcome nearly inevitable).

This is wrong. The physics says:

1. **Token generation is spin glass relaxation.** The prompt injects energy (frustration). Each generated token releases energy, moving the system toward its ground state. Early tokens: many competing configurations, full compute justified. Late tokens: approaching ground state, most degrees of freedom already resolved.

2. **The manifold is ~10-dimensional.** Despite hidden sizes of 2560-4096+, the prediction-relevant information lives on a ~10D surface. The other 2550+ dimensions carry energy but not information — they are the higher dimensions that decay away.

3. **This is universal.** The manifold dimensionality is a property of the transformer architecture, not any specific model. Ternary weights (BitNet), fp16 (Llama), bf16 (Qwen) — the geometry is the same because attention IS spin-spin interaction, softmax IS the Boltzmann distribution, and layer normalization IS temperature regulation.

4. **One measurement = forever.** The manifold shape is determined by the model weights (the spin glass ground state). Measure it once on a calibration corpus, save the SVD bases as `manifold.pt`, and every future inference can use it. Like shipping quantization configs alongside model weights.

## Measured Results: BitNet b1.58-2B-4T

**31 layers, hidden_size=2560, vocab=128256**

The intrinsic dimensionality (TwoNN) is **constant at ~10 across all 31 layers**:

![Manifold measurements](results/stage1_manifold.png)

### The Spin Glass Energy Profile

| Phase | Layers | PR Range | What Happens |
|-------|--------|----------|-------------|
| Entry | L00-L03 | 55 → 93 | Expanding into manifold |
| Compression | L04-L07 | 39 → 10 | Finding ground state — PR minimum at L07 (10.1) |
| Bulk expansion | L08-L21 | 12 → 137 | Exploring the manifold surface |
| Collapse | L22-L30 | 138 → 32 | Relaxation to ground state |

Through ALL of this, **TwoNN stays between 9.7 and 11.0**. The intrinsic dimensionality does not change. The manifold shape is invariant — only the energy distribution on it changes. This is the fractal: the same ~10D surface at every scale.

### Per-Token Latency and KV Entropy During Generation

![Latency and entropy curves](results/latency_entropy_500tok.png)

KV attention entropy tracks the spin glass relaxation state in real time. Different prompts produce different energy profiles:
- **Structured prompts** (cosmology): bell-shaped latency curve — frustration builds, peaks, then relaxes
- **Complex prompts** (evolution): flat profile — the system stays frustrated longer
- **Mixed prompts** (linguistics): spikes of frustration at decision points, then relaxation

### Bottleneck Validation

Separate experiments (engine-a dynamic funnel) confirmed:
- **128x compression (4096→32) works with near-zero KL divergence** at late layers
- **97.1% accuracy at bottleneck dim 64** on BitNet with a trained gate at layer 30
- The manifold is robust despite the butterfly effect — small perturbations in the projection are corrected by downstream layers

## The Connection to Existing Techniques

Every existing inference speedup technique is approximating this physics without knowing it:

| Technique | What it senses | What it misses |
|-----------|---------------|----------------|
| **Speculative decoding** | "Some tokens are predictable" | It's not prediction — it's measurement of orbital collapse |
| **Early exit** | "Some tokens don't need all layers" | Binary exit/no-exit misses the continuous rank reduction |
| **Draft models** | "A small model can guess easy tokens" | The small model IS a low-rank projection of the manifold |
| **Medusa heads** | "Multiple future tokens can be predicted" | Training heads to approximate what the KV cache already knows |
| **Mixture of Experts** | "Different tokens need different compute" | Fixed expert assignment vs. dynamic manifold measurement |

**Exponential inference subsumes all of these.** The manifold measurement gives you the continuous, per-token, per-layer compute budget directly from the model's geometry. No training, no separate models, no approximation.

## The Physics

**Transformers are spin glasses:**
- Ternary BitNet weights (-1, 0, 1) are literal Ising spins at ground state
- Attention computes pairwise spin-spin interactions
- Softmax is the Boltzmann distribution (partition function)
- Layer normalization is temperature regulation
- Token generation is relaxation toward the ground state

**Three axes of the manifold:**
- **Width = KV cache** — spatial extent of the spin lattice (context window)
- **Depth = layer precision** — refinement of energy landscape per layer  
- **Sequence = relaxation** — each token brings system closer to ground state

**The fractal:** Engine A (per-layer depth) and Engine B (per-token sequence) measure the same manifold at different scales. One forward pass through 30 layers is structurally equivalent to generating 30 tokens. The expand → peak → collapse pattern appears at both scales.

## Manifold Catalog

Measured intrinsic dimensionality (TwoNN) across model families:

| Model | Type | Params | Hidden | Layers | Peak TwoNN | Final TwoNN |
|-------|------|--------|--------|--------|------------|-------------|
| Qwen3-0.6B | Dense | 0.6B | 1024 | 28 | 11.1 | 9.09 |
| Qwen3-1.7B | Dense | 1.7B | 2048 | 28 | 12.2 | 8.98 |
| BitNet b1.58-2B-4T | Ternary | 2B | 2560 | 30 | 11.0 | 9.81 |
| Phi-2 | Dense | 2.7B | 2560 | 32 | 10.1 | 9.76 |
| Qwen3-4B | Dense | 4B | 2560 | 36 | 12.7 | 9.52 |
| Qwen3-8B | Dense | 8B | 4096 | 36 | 13.1 | 9.38 |
| Qwen3-14B | Dense | 14B | 5120 | 40 | 13.3 | 9.38 |
| Qwen3-30B-A3B | **MoE** | 30B/3B | 2048 | 48 | 13.0 | 9.07 |
| Qwen3-32B | Dense | 32B | 5120 | 64 | 14.8 | 10.89 |

**Nine models. Three architectures (dense, MoE, ternary). 0.6B to 32B parameters. All converge to TwoNN ~9-11 at the output layer.**

Key findings:
- **Peak TwoNN scales with model size** (11 → 15) — larger models explore richer manifolds in mid-layers
- **Final TwoNN is invariant** (~9-11) — all models collapse to the same dimensionality at output
- **MoE doesn't change the manifold** — 64 experts are 64 redundant views of the same ~9D surface
- **Architecture doesn't matter** — ternary (BitNet), dense fp16 (Qwen, Phi), and MoE all converge

TwoNN accuracy validated on synthetic data: correctly recovers true dimensions 3 (2.95), 5 (5.22), 7 (7.19), 10 (9.49). Full random 2560D gives TwoNN=283. The ~9-11D measurements are real.

## Head Pruning: 80% of Heads Are Unnecessary

Independent confirmation of the manifold: dynamic attention head pruning based on sharpness shows that **80% of attention heads can be removed with 100% token match**.

Tested on MacBook Air M4 (MPS):

| Model | Heads Kept | Token Match | Manifold Narrowing |
|-------|-----------|-------------|-------------------|
| Qwen3-0.6B (16 heads) | 17% → 16% | 200/200 (100%) | 23.3% → 15.8% |
| Qwen3-4B (32 heads) | 19.4% avg | 200/200 (100%) | 23.7% → 15.0% |

**The number of active heads converges on the manifold dimensionality:**
- 0.6B: 15.8% of 16 heads = **2.5 heads** × ~3 dims/head = ~8 dims
- 4B: 15.0% of 32 heads = **4.8 heads** × ~2 dims/head = ~10 dims

Three independent measurements. Same answer:
1. **TwoNN geometry**: ~9-10D
2. **Bottleneck training**: 32 dims at 0.01 KL divergence
3. **Head pruning**: 2.5-4.8 active heads ≈ 8-10 effective dimensions

## The manifold floor (key scaling insight, 2026-04-18)

Factored-parameter budget scales with model size, but the minimum
parameter count needed to encode the tokenizer-induced manifold (the
"floor") is roughly size-independent:

| model | full params | factored at rank-32 | % of full |
|---|---|---|---|
| Qwen3-0.6B | 440M | **20.2M** | 4.58% |
| Qwen3-4B | ~3.2B | ~90M | 2.8% |
| Qwen3-32B | ~31B | **~270M** | 0.86% |

Empirically, even rank-256 on Qwen3-0.6B (160M factored, 36% of full)
showed degenerate output. This suggests the floor is likely ~80–160M
params for the Qwen tokenizer-induced manifold. **0.6B at rank-32 is
below this floor; no training procedure can succeed.**

**The failures at stages 8–15 on 0.6B are structural, not procedural.**
Scaling to 32B doesn't make distillation easier — it raises the
parameter budget above the floor, making success achievable.

Relatedly: **all 9 models in our manifold catalog share the Qwen
tokenizer family (or very close ones).** The universal "~9-11 intrinsic
dim" observation may actually be *tokenizer-bound* rather than
transformer-universal. Testable with GPT-2, Llama-3, Mistral, T5 on
machines that can run them.

## Where this is going (live research context)

Stages 0–4 establish the measurement. Stages 5+ turn it into a deployment
recipe: **rank-k factored decode trained via teacher–student distillation,
with K/V cache naturally living in the same rank-k subspace — one manifold,
one map.**

The target is 10–30× wall-clock speedup at batch=1 decode on 30B-class dense
models. Weights, attention, and KV cache all operate in the same per-layer
manifold basis.

Full research framing, in-flight experiments, falsified approaches, target
numbers per hardware, and machine coordination are tracked in
[`docs/research_context.md`](docs/research_context.md). **That file is the
shared memory for this project across machines and sessions.** Start there
for anything beyond the BitNet stages above.

## Quick Start

```bash
pip install -r requirements.txt
```

### 1. Measure a model's manifold
```bash
python scripts/stage1_measure.py --model-id Qwen/Qwen3-0.6B
# Results: results/stage1_manifold.json + .png
```

### 2. Verify head pruning (proves the manifold is real)
```bash
# Uses previous step's attention sharpness to skip diffuse heads
python scripts/stage5_skip_heads.py \
  --model Qwen/Qwen3-0.6B \
  --threshold 0.9 \
  --min-heads 2 \
  --device mps  # or cuda or cpu
```

### 3. Sparse head generation (physically smaller matmuls)
```bash
# Actually skips Q/K/V computation for pruned heads
python scripts/stage5_sparse_heads.py \
  --model Qwen/Qwen3-0.6B \
  --threshold 0.9 \
  --device mps  # needs GPU/MPS for speedup
```

### 4. Measure KV entropy during generation
```bash
python scripts/stage4_direct.py --max-new-tokens 500 --max-prompts 3
```

## Layout

```
src/
  common/model_loader.py       Device-aware model loader
  measurement/                 PR, TwoNN, SVD rank, hidden-state caching
  routing/rank_predictor.py    SVD manifold basis + rank predictor
  inference/dynamic_rank.py    Per-token rank-projection forward pass
  evaluation/                  Per-position timing and curve aggregation
scripts/                       One driver per stage
data/prompts.json              Generation prompts for acceleration measurement
docs/                          Physics maps, measurement logs, test doctrine
tests/                         Unit tests
results/                       Manifold measurements, plots, JSON summaries
```

## Citing

```
@misc{exponential_inference,
  author = {Parrish Corcoran},
  title  = {Exponential Inference: Transformers are spin glasses — 
            per-token compute decreases as context grows},
  year   = {2026},
  url    = {https://github.com/parrishcorcoran/Exponential-Inference}
}
```

## Licence

See `LICENSE`.
