# exponential-inference

> BitNet b1.58 2B is measurably at a spin-glass ground state. Its
> per-token hidden-state manifold collapses in a predictable,
> position-dependent way during generation — so per-token compute
> requirements *decrease* as context grows. This repo measures that
> collapse directly and exploits it at inference time via a per-token
> dynamic-rank forward pass. It is **not** a compression technique:
> the rank budget we use is extracted from the model's own geometry
> at inference time, per token, per layer.

Measured on `@@model_id` (@@n_layers layers, hidden size
@@hidden_size), running on `@@backend_name` (@@vram_gb GB).

![Per-token speedup vs generation position](results/acceleration_curve.png)
![Mean predicted rank vs generation position](results/rank_distribution.png)

## What the measurements say

### 1. The manifold is low-dimensional and changes shape across layers

We run a ~10K-token Wikipedia slice through BitNet and estimate the
intrinsic dimensionality at each of the @@n_layers decoder layers
using:

- **Participation ratio (PR)** — `(Σλᵢ)² / Σλᵢ²` on the covariance
  spectrum. Invariant to global scale; equals the rank on a flat
  spectrum.
- **TwoNN** (Facco et al. 2017) — non-parametric intrinsic dimension
  from the ratio of first- and second-nearest-neighbour distances.
- **r95** — number of SVD components needed to cover 95% of the
  activation energy.

@@layer_table

This is the "6 → 36 → 16" fingerprint the theory predicts: a shallow
compression, an expansion layer where mixed-context tokens spread out,
and a late-layer re-contraction around the output token distribution.

### 2. Early-layer manifold position predicts late-layer rank

For each token we take its position on the top-7 SVD basis at the
chosen source layer and fit a regressor to its per-layer effective
rank at layers 15/20/25/29. Both a linear model and a small MLP are
tried at source layers 5, 10, 15, 20; the first attempt clearing the
`R² ≥ 0.6` floor is accepted.

Result: @@predictor_line

### 3. Projecting each token to its predicted rank preserves the output

A `DynamicRankBitNet` wrapper registers forward pre-hooks on the
target decoder layers. At the entry of each target layer, every
token's hidden state is recentred, projected onto the top-`r` columns
of that layer's calibration SVD basis (where `r` is per token and per
layer, set by the predictor), and reconstructed.

- **Correctness gate** — at full rank, logits must match the
  unwrapped base model. Measured `max|Δlogits| = @@correctness_max_diff`
  (pass = @@correctness_passed).
- **Quality gate** — teacher-forced next-token accuracy is within
  tolerance of base (`@@base_accuracy`). Accepted safety multiplier on
  predicted ranks: `@@accepted_multiplier` (1.0 means the raw predictor
  is used).

### 4. Generation accelerates with position

Across the ten prompts in `data/prompts.json`, greedy-decoded at up to
2000 new tokens each, per-token speedup (base / dynamic) as a function
of generation position:

@@speedup_snapshots

Mean predicted rank (averaged over target layers), showing the
manifold tightening as context grows:

@@rank_snapshots

Sample generations (base and dynamic) live under
`results/generation_samples/` so output quality can be eyeballed
directly.

## Where this is going

Stages 0–4 above establish the measurement for BitNet-2B. The ongoing work
generalises the same measurement into a deployment recipe for arbitrary
trained LLMs at 30B-class scale: **rank-k factored decode trained via
teacher–student distillation, with K/V cache naturally living in the same
rank-k subspace (one manifold, one map).**

Full research context, in-flight experiments, falsified approaches, target
numbers, and machine coordination are tracked in
[`docs/research_context.md`](docs/research_context.md). That file is the
shared memory for this project across machines and sessions — start there
for anything beyond the BitNet stages.

## What this is not

- Not a compression technique. No extra training. No distillation.
  The rank budget is chosen at inference time from the model's own
  SVD structure on a calibration corpus.
- Not a speculative-decoding or draft-model scheme.
- Not about the ternary weights. The bf16 checkpoint exhibits the
  same geometric collapse; the bitnet.cpp kernels merely reproduce it
  faster. The final wall-clock re-measurement belongs in bitnet.cpp;
  this repo validates the mechanism in PyTorch.

## Honest limitations

- Measured on BitNet b1.58 2B specifically. The "ground state"
  framing predicts the same qualitative shape on any well-trained
  ternary-quantised model, but that is not yet tested here.
- Applying the same recipe to FP16 models that were not trained to a
  ternary ground state is proposed as future work and left un-done.
- Long-generation output quality is verified only by sample
  inspection plus held-out teacher-forced next-token accuracy.
  Head-to-head blind human evaluation has not been done.
- Wall-clock numbers on the PyTorch path carry overhead from the
  Python hooks. The integral speedup is reported as the geometric
  statement; the final wall-clock reproduction in bitnet.cpp is
  expected to be cleaner.

## Reproduce

Tested on a Strix Halo box with ROCm and ~82 GB unified VRAM. Any
ROCm- or CUDA-enabled PyTorch build should work.

```bash
pip install -r requirements.txt

# Stage 0: verify the base model loads and generates.
python scripts/stage0_verify.py

# Stage 1: cache per-layer hidden states, measure PR/TwoNN.
python scripts/stage1_measure.py

# Stage 2: fit the per-token rank predictor, walk source layer
# forward if R^2 < 0.6.
python scripts/stage2_fit_predictor.py

# Stage 3: correctness (full-rank == base) and quality (next-token
# accuracy) gates. If the raw predictor drops quality too much this
# loops the safety multiplier up to 4x.
python scripts/stage3_dynamic_forward.py

# Stage 4: measure the acceleration curve across 10 prompts x 2000
# tokens each. Emits results/acceleration_curve.png,
# results/rank_distribution.png, results/summary.json, and
# results/generation_samples/.
python scripts/stage4_acceleration.py

# Stage 5: render this README from the produced JSONs.
python scripts/render_readme.py
```

Tests:
```bash
python -m pytest tests/
```

## Layout

```
src/
  common/model_loader.py      ROCm-aware loader for BitNet-b1.58-2B-4T-bf16
  measurement/                PR, TwoNN, per-layer hidden-state caching
  routing/rank_predictor.py   SVD manifold basis + linear/MLP predictor
  inference/dynamic_rank.py   Per-token rank-projection forward pass
  evaluation/                 Per-position timing and curve aggregation
scripts/                      One driver per stage + render_readme.py
data/prompts.json             The ten generation prompts
tests/                        Unit tests for every stage
results/                      Produced artifacts; curves and JSON summaries
```

## Citing

If you find this useful, please cite as:

```
@misc{exponential_inference,
  author = {spinglassai},
  title  = {exponential-inference: BitNet 2B accelerates during generation},
  year   = {2026},
  url    = {https://github.com/parrishcorcoran/exponential-inference},
  note   = {Substack: https://spinglassai.substack.com}
}
```

## Licence

See `LICENSE`.
