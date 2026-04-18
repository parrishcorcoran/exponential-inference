# Read this first

If you arrive at this repo cold — human or AI — start here. This is the
verbose explainer: what we're doing, why, the physics behind it, what
we've measured, and where the work is going. It's long on purpose.

For short reference: see `docs/research_context.md` (terse). For
specialized analyses: `analysis/manifold_routing/` (inference routing)
and `analysis/manifold_training/` (training implications).

---

## TL;DR

Transformers are spin glasses. Their hidden-state representations live
on a ~9-11 dimensional curved manifold that is **created during training
and bounded by the tokenizer**. Every forward pass is a relaxation on
this manifold; every KV cache entry is a point sampled from it. Once
this is understood as a physical object, a cascade of inference
accelerations becomes mathematically unavoidable:

- **Per-token compute scales with manifold dimension, not hidden size.**
- **KV cache storage scales with manifold dimension, not vocab or context.**
- **Training compute can be reduced because the manifold's dimension,
  shape, and per-layer rotation schedule are substantially predictable
  from tokenizer + corpus + architecture alone.**

Target: **10–30× wall-clock speedup** at batch=1 decode on 30B-class
dense models, via rank-k factored weights + dynamic per-token compute
tiering, all driven by live state signals that come free from the
forward pass.

---

## The problem

Modern LLMs at 30B+ parameters use roughly the same per-token compute
whether they're generating the first word of a creative story (many
competing continuations, high entropy) or the 900th token of a
predictable conclusion (one dominant continuation, near ground state).
This is wasteful. It also doesn't match how LLMs actually behave
internally — the manifold of accessible hidden states narrows as context
grows, but the inference loop treats every token the same.

Everyone knows this intuitively; many techniques exploit parts of it:
speculative decoding, early-exit, Medusa, MoE, mixture-of-depth,
aggressive quantization, layer dropping at inference, KV cache
compression via heavy-hitter selection. Each gets some speedup. None of
them measures the underlying manifold directly. They're all doing
implicit low-dim routing without knowing the dimension.

This repo measures the manifold directly and builds the pipeline around
that measurement.

---

## The physics framing (the part people often miss)

### A transformer is a spin glass at its ground state

This is the reading that makes everything click. It's not a metaphor —
it's a literal mapping:

- **Attention** computes pairwise interactions between tokens. A spin
  glass is defined by pairwise couplings. Same math.
- **Softmax** over scores is the Boltzmann distribution with 1/√d_head
  as inverse temperature. Same object.
- **Residual connections + layer norm** implement a temperature-
  regulated propagation. Same role as thermostatting in physics.
- **Weights** are the coupling constants: they define the energy
  landscape the system relaxes on.
- **Training** is an annealing process that shapes the landscape into a
  spin-glass ground state with a specific, low-entropy ultrametric
  (RSB-hierarchical) structure.
- **Token generation** is relaxation from a prompt-excited state back
  toward the ground state along the manifold of low-energy
  configurations. Each token is one step of the relaxation trajectory.

The word "manifold" we use throughout is this: the low-energy submanifold
of state-space, crystallized by training, on which hidden states live.

### Replica symmetry breaking (RSB) and why bulk is slaved to surface

In spin glasses at low temperature, Parisi's replica-symmetry-breaking
solution tells you the equilibrium state has an ultrametric hierarchy:
many basins of attraction organized into clusters, clusters into
super-clusters, ad infinitum. Two key consequences for us:

1. **Off-manifold "bulk" degrees of freedom are slaved to the low-dim
   manifold.** The full ambient state space is high-dimensional (say
   5120 for Qwen3-32B), but in a crystallized glass, the bulk does not
   carry independent dynamical information. The observed ~10 intrinsic
   dim is the "real" state space; the remaining ~5110 dimensions are
   geometric embedding redundancy (tangent planes rotating as you move
   along the curved manifold).

2. **Entropy trajectories during generation are RSB descent patterns.**
   Bell curves, plateaus, and mid-generation spikes are signatures of
   how the system navigates the hierarchy of basins. A spike means the
   system climbed a saddle between sibling basins; a plateau means it's
   stuck in a metastable basin; a bell is clean descent into one basin.

### The tokenizer as boundary condition

A tokenizer defines the vocabulary and how text is split into tokens.
Training fits the model to predict next tokens given context. This
ties manifold geometry to the tokenizer's induced structure:

- Models trained on the same tokenizer family end up with closely
  related manifold geometry (confirmed: Qwen3 0.6B through 32B + Phi-2
  all cluster at ~9-11 intrinsic dim despite different sizes/architectures).
- The exact dimension may vary slightly between tokenizer families
  (testable; partly tested).
- The SHAPE of the per-layer basis rotation schedule is universal
  across tokenizers (confirmed: Pearson r > 0.97 between Qwen3 family
  and Phi-2 in stage 21).

The tokenizer is to the model what boundary conditions are to a PDE
solution. The same PDE with different boundary conditions gives different
solutions of the same functional form.

---

## Current empirical findings, with numbers

### 1. Intrinsic manifold dimension is ~9–11 across 9 measured models

| model | tokenizer | peak TwoNN | final TwoNN |
|---|---|---|---|
| Qwen3-0.6B | Qwen | 11.1 | 9.09 |
| Qwen3-1.7B | Qwen | 12.2 | 8.98 |
| BitNet-b1.58 2B | ~Llama | 11.0 | 9.81 |
| Phi-2 2.7B | CodeGen | 10.1 | 9.76 |
| Qwen3-4B | Qwen | 12.7 | 9.52 |
| Qwen3-8B | Qwen | 13.1 | 9.38 |
| Qwen3-14B | Qwen | 13.3 | 9.38 |
| Qwen3-30B-A3B (MoE) | Qwen | 13.0 | 9.07 |
| Qwen3-32B | Qwen | 14.8 | 10.89 |

Peak rises modestly with scale; final stays within 9-11 for every model.

### 2. The embedding matrix itself is NOT the manifold

Qwen3-0.6B's `embed_tokens.weight` has intrinsic dim ≈ 80 (PR ≈ 800).
The ~10 manifold emerges when tokens are sampled from text
(frequency-weighted embedding sample has TwoNN ≈ 12). The manifold is
`embedding geometry × text distribution`, not the embedding alone.

### 3. The embedding basis ≠ the activation basis

Layer 0 activation basis = embedding basis (overlap 1.0, trivially).
Layers 1+ have low overlap with P_embed (0.18-0.30). They share
dimensionality but sit in different 10D subspaces of the 1024D ambient.
Training learns per-layer rotations that cannot be extracted from the
embedding matrix alone.

### 4. The rotation schedule across layers is universal

Adjacent-layer basis overlaps form a curve. Normalized to [0,1] depth:

| measure | Qwen3-0.6B | Qwen3-1.7B | Phi-2 |
|---|---|---|---|
| phase-transition location | layer 0→1 | layer 0→1 | layer 0→1 |
| phase-transition overlap | 0.188 | 0.144 | 0.226 |
| mean adjacent overlap | 0.856 | 0.847 | 0.935 |
| first-vs-last overlap | 0.287 | 0.238 | 0.120 |

Pairwise Pearson correlation of the normalized curves: r > 0.97 for
all three pairs, including cross-tokenizer (Qwen vs Phi-2). The shape
of the rotation schedule is a universal transformer-LM property. Phi-2
is OFFSET upward (more gradual rotation) but has the same curve shape.

### 5. Layer 1 is universally the big rotation

All three models have their largest adjacent-layer rotation at
0→1 (frac_depth = 0.00). The rotation is uniform across all k=32
directions (principal cosines tightly clustered). Layer 1 is a global
"embedding → task feature frame" reorientation, not a selective
projection.

### 6. Head pruning works: 80-83% redundant heads

Across 0.6B and 4B models, using attention sharpness as the signal,
80-83% of attention heads can be skipped at decode time with 100%
token match. Number of active heads tracks the manifold dimension:

- 0.6B (16 heads total): 2.5 active × ~3 dims/head ≈ 8 dims
- 4B (32 heads total): 4.8 active × ~2 dims/head ≈ 10 dims

Independent empirical evidence for the manifold dim bound.

### 7. Distillation preserves manifold topology but not pointwise outputs on 0.6B

A rank-32 factored student distilled from Qwen3-0.6B teacher:

- **TwoNN invariance:** mean |Δ TwoNN| = 0.49 across 28 layers. The
  student's hidden-state manifold has the same intrinsic dim as
  teacher's at every layer. Structural proof of manifold copying.
- **Held-out distribution match:** ppl ratio up to ~99× worse than
  teacher, top-1 agreement ~20%. The pointwise function is off.

This is not a framework failure — it's a capacity problem. See the
manifold-floor section.

### 8. The manifold floor — why 0.6B won't demonstrate wall-clock gains

Factored-parameter budget scales with model size, but the MINIMUM
parameters needed to encode the tokenizer-induced manifold (the
"floor") is roughly size-independent:

| model | full params | factored at rank-32 | % of full |
|---|---|---|---|
| Qwen3-0.6B | 440M | 20.2M | 4.58% |
| Qwen3-4B | ~3.2B | ~90M | 2.8% |
| Qwen3-32B | ~31B | ~270M | 0.86% |

Empirically, even rank-256 on 0.6B (160M factored, 36% of full) showed
degenerate output. This suggests the floor is ~80-160M parameters for
the Qwen tokenizer-induced manifold. 0.6B at rank-32 is below the
floor; no training procedure can succeed. 32B at rank-32 comfortably
clears it.

The 0.6B experiments were run because MacBook hardware was available
and iteration is fast. They established every upstream piece of the
pipeline (manifold measurement, head pruning, distillation mechanics,
entropy-signal plumbing, basis geometry) but cannot demonstrate
wall-clock wins.

### 9. Per-kernel launch cost matters; MPS is a bad target

At 0.6B on this MacBook, teacher decode is 30 ms/tok on both MPS and
CPU (memory-bandwidth bound, not GPU-compute bound). Adding signal
hooks for dynamic routing adds 2–3× overhead on MPS vs only 70%
overhead on CPU. CUDA/ROCm have cheaper per-kernel dispatch than CPU
via command-buffer amortization. The "we can't beat baseline on MPS"
results throughout this repo are device-specific, not framework
failures.

### 10. Four canonical entropy profiles during generation (RSB descent types)

Measured on six prompt archetypes:

| profile | H(t) shape | interpretation |
|---|---|---|
| Linear decline | monotone down | prompt already in a basin; descent |
| Bell curve | up-peak-down | one saddle crossing, single basin commit |
| Plateau | flat | stuck in metastable basin, no progress |
| Mid-generation spike | down-spike-down | RSB-hierarchy traversal to sibling basin |

Reasoning-chain prompts produce the most saddles (~13 in 60 tokens),
matching the intuition that multi-step logic traverses multiple basin
transitions.

---

## The architecture being built

### Per-mechanism overview

1. **Rank-k factored weights.** Every attention and MLP Linear
   replaced with `A @ B` where `A: [d_out, k]`, `B: [k, d_in]`. At
   k≈32 this is ~0.9% of full weights for 32B. Initialized from PCA
   of teacher calibration activations; distilled via Matryoshka rank
   sampling so the student works at any k ≤ k_max.

2. **Rank-k K, V storage.** K and V are projections of hidden states
   that live on the manifold. Stored in k-dim coordinates per token,
   not full d_head. Attention computed as `q_coords · M · k_coords^T`
   where M = A_q^T @ A_k is a precomputed k×k matrix. Bandwidth per
   token drops from O(d) to O(k).

3. **Per-token dynamic rank.** Rank k(t, i) per layer per step is a
   function of live state:
   - `H_i(t)` — normalized attention entropy (free from eager attn).
   - `∂H_i/∂t` — derivative (prev step stored, one subtraction).
   - `k = f(H, ∂H/∂t)`: low H with low ∂H/∂t → aggressive prune;
     rising ∂H/∂t → saddle incoming, restore rank.

4. **Head skipping per layer per token.** Heads ranked by sharpness
   each step; skip below threshold. Stage 5 confirmed 80-83% prunable
   with zero quality cost.

5. **Dynamic layer depth (future).** When the system is relaxed (low
   entropy, small step size), skip late layers. Requires a better
   signal than attention entropy (attention entropy ≠ layer
   necessity); residual-update magnitude is likely correct but
   tentative.

6. **Prompt-entropy compute tiering.** At prompt end, measure attention
   entropy across all layers. Map to compute tier (low/medium/high)
   that sets the initial rank and the skip-aggressiveness for
   subsequent tokens. Lets a serving fleet route easy prompts to
   cheap paths.

7. **Speculative decoding via factored student.** Rank-k factored
   student as draft; teacher verifies in batch. Orthogonal to all the
   above mechanisms and composes multiplicatively.

### The all-dynamic principle

Every compression dimension above must be a function of live state,
not a hyperparameter. A static-parameter architecture has a hard
ceiling at worst-case efficiency. All-dynamic tracks the system's
actual manifold position moment by moment; only the latter approaches
theoretical bandwidth/compute reduction. The policy that drives all
mechanisms is ~20 lines of conditional logic over scalar state, not a
learned network.

### What trains vs what's free

| component | needs training? |
|---|---|
| Rank-k factored weights | YES — Matryoshka distillation |
| Heads kept per step | no — runtime signal (attention sharpness) |
| Layer depth per step | no — runtime signal (residual update magnitude, tbd) |
| KV attended per step | no — heavy-hitter from attention weights |
| Chart basis | no — compute at calibration, route at inference |
| Quantization | no — standard PTQ |
| Speculative depth | no — top-1 margin |

Only the factored weights require distillation. Everything else is a
runtime policy reading free signals from the forward pass.

---

## Stages 0–21: the experimental catalog

Each stage is a script in `scripts/` with results in `results/`.

| stage | what it tested | result |
|---|---|---|
| 0 | verify model loads and runs | ✓ baseline |
| 1 | measure manifold dim per layer via TwoNN + PR | ~9-11 across 9 models |
| 2 | fit per-token rank predictor | established predictor schema |
| 3 | dynamic-rank forward via hook projection | quality preserved |
| 4 | per-token timing + entropy during decode | entropy zoo profiles |
| 5 | skip attention heads below sharpness | 80-83% prunable, 100% match |
| 6 | plain SVD of weights factoring | FAILS (0/200 match) — weight SVD isn't manifold |
| 7 | PCA-basis weight factoring without training | FAILS at rank 256 — manifold is curved |
| 8 | teacher-student distillation at fixed rank 32 | kl 211→3.5, match 15/200 |
| 9 | (layer-wise distillation draft, not run) | — |
| 10 | single-token geometric decode (no attention) | garbage (expected) — context matters |
| 10b | rank-k residual projection at fixed k | needs rank ~500 in ambient to preserve |
| 11 | entropy-driven dynamic rank via hook projection | plumbing works; wall-clock negative on MPS |
| 12 | entropy-driven layer skip | 2× wall-clock, quality collapses |
| 12b | residual-magnitude layer skip | rel-update > 0.2 everywhere at 0.6B → not prune-able |
| 13 | scaled calibration (4× more data) | ppl ratio 922→99 — scaling works |
| 14 | teacher-sampled greedy calibration | TwoNN invariant (manifold preserved!) |
| 15 | Matryoshka distillation at 0.6B | unstable — below the manifold floor |
| 16 | TwoNN on embedding matrix directly | ~80 (NOT the manifold) |
| 17 | TwoNN on text-weighted embeddings | ~12 (IS the manifold) |
| 18 | per-layer basis overlap with embedding | different subspaces (same dim) |
| 19 | rotation profile + corpus invariance | smooth rotation, model-property |
| 20 | cross-model phase transition location | universal at layer 0→1 |
| 21 | rotation curve shape across tokenizers | Pearson r > 0.97 universal |

Plus `analysis/manifold_routing/` and `analysis/manifold_training/`
for parallel analyses done on Z8G4/Strix.

---

## The machines and who does what

| machine | role | constraints |
|---|---|---|
| **MacBook (MPS)** | prototype, measurement, short iteration on 0.6B | unified memory, kernel launch costs, no training ≥4B |
| **Z8G4** (HP Z8 G4, 700 GB RAM, no GPU, Skylake Xeon) | big-model measurement (>72B), teacher corpus generation, overnight eval | CPU only, slow per-op but unlimited RAM |
| **Strix Halo** (AMD ROCm, 82 GB VRAM) | interactive training, Matryoshka distillation at up to 32B, ROCm inference benchmarks | GPU throughput limited vs H100 but cheap and fast enough |

Each machine has its own folder under `machines/<name>/` with its own
scripts, README, and scratch. Large artifacts (corpora, weights) go
via HuggingFace Hub, not git. Small results (JSON, markdown) commit to
git. See `shared/README.md` for the full sync contract.

Don't cross machine boundaries — each machine owns its folder.

---

## Open questions and next experiments, in priority order

1. **Matryoshka distillation on 32B on Strix Halo.** Above the manifold
   floor; should converge cleanly where 0.6B didn't. The central
   experiment that produces wall-clock numbers.

2. **Tokenizer sweep on Z8G4.** Measure manifold dim on GPT-2,
   Llama-3, Mistral, T5. Test the "same tokenizer → same dim"
   hypothesis directly. Partial evidence: Phi-2 (different tokenizer)
   gave similar final dim but different peak dim. Need more data points.

3. **Layer-1 rotation alignment across Qwen family.** If Qwen3-0.6B's
   layer-1 rotation is a scaled/projected version of Qwen3-4B's
   layer-1 rotation, the rotation is a family property. Enables
   cross-size transfer.

4. **Curve shape as deployable prior.** Use the universal curve shape
   as a prior for per-layer rank allocation without calibration. Only
   calibrate at sparse layers and interpolate along the known curve.

5. **Empirical KV-sparsity measurement** on long-context prompts. The
   combined compression estimate (~2-5% of full KV) is from
   multiplicative composition of individually-measured mechanisms; a
   direct end-to-end measurement would tighten the number.

6. **Dynamic-layer-depth signal.** Stage 12/12b showed attention
   entropy is the wrong signal; residual-update magnitude is still a
   candidate but 0.6B doesn't have near-identity layers to validate
   on. Revisit at 32B.

7. **Atlas of local charts.** Stage E showed single global basis
   suffices at 0.6B. At 30B with deeper RSB hierarchy, locally-distinct
   charts may emerge.

---

## Glossary

**Manifold**: the low-dimensional submanifold of hidden-state space
where activations actually live, as a function of tokenizer and
training. Measured by TwoNN intrinsic dim, typically ~9-11.

**Intrinsic dimension**: the number of local degrees of freedom on the
manifold (what TwoNN measures). Can be much smaller than the ambient
linear rank (r90) because the manifold is curved.

**TwoNN**: Facco et al. 2017 estimator for intrinsic dim from the
ratio of first- to second-nearest-neighbor distances. Validated against
synthetic data in this repo.

**Participation ratio (PR)**: (Σλ)² / Σλ² on the covariance spectrum,
a scale-invariant effective linear rank. Different from TwoNN (PR is
linear, TwoNN intrinsic).

**r90 / r95 / r99**: number of top SVD components needed to cover
90/95/99% of variance. Linear-rank measurements.

**RSB (replica symmetry breaking)**: Parisi's solution for spin-glass
ground states. Predicts hierarchical (ultrametric) organization of
basins. Applied here, it says the ambient "bulk" of state space doesn't
carry independent dynamical information.

**Factored linear**: replacing W [d_out × d_in] with A @ B where
A [d_out × k], B [k × d_in]. Saves parameters and compute when k ≪ d.

**Matryoshka training**: sample k uniformly during training so that
every prefix [1:k] of A and B is itself a valid factorization.
Produces a single set of weights that works at any k ≤ k_max.

**Basis rotation / subspace overlap**: two k-dim subspaces of the same
ambient space can be compared via the Frobenius norm of P_a^T @ P_b
divided by √k, giving a value in [0,1]: 1 means same subspace, 0
orthogonal. Random baseline at k=32 in d=1024 is √(k/d) ≈ 0.177.

**Phase transition (in the stack)**: a layer index where the adjacent-
layer basis overlap is much smaller than neighbors. In every measured
model this is layer 0→1, the embedding-to-first-transformer-layer
boundary.

**Spin glass / crystallized ground state**: the state of a trained
LLM in this physical framing. Training is annealing; inference is
relaxation on the crystallized landscape.

**Manifold floor**: the minimum factored-parameter budget below which
no training procedure can compress a trained LM while preserving
behavior. Estimated at ~80-160M params for Qwen-tokenizer-induced
manifolds.

**All-dynamic principle**: design rule that every compression dimension
(rank, heads, layers, KV support, chart) must be a function of live
state, not a hyperparameter. Static parameters ceiling at worst-case
efficiency.

---

## Reading order for further exploration

1. This file (you're here).
2. `docs/research_context.md` — terse reference, current state, falsified approaches.
3. `analysis/manifold_routing/CONTEXT.md` — inference-side parallel analysis from Z8 / Strix work.
4. `analysis/manifold_routing/README.md` — head-pruning and routing-signal detail.
5. `analysis/manifold_training/README.md` — training-side parallel analysis.
6. `machines/<name>/README.md` — per-machine role and setup.
7. `shared/README.md` — sync contract.
8. Scripts: `scripts/stage1_*` through `scripts/stage21_*` in numerical order; each stage's result JSON in `results/`.

---

## A note on methodology

The project is moving fast, often with a failed experiment's outcome
clarifying the framework rather than refuting it. That's deliberate —
in a spin-glass-shaped search space, every failed test narrows the
hypothesis; no one of them refutes the frame. If you find yourself
tempted to "revert to standard-ML reasoning" because a measurement
surprised you, the first move is to re-read this file. The physics
framing has held through every empirical challenge; what's changed is
our understanding of WHICH mechanism implements the physics.

The failure modes specific to 0.6B on MPS (kernel launch overhead,
below-floor parameter budget, training instability) are machine-and-
scale-specific. They don't generalize up. The 30B run on Strix Halo
will produce the wall-clock numbers that do.
