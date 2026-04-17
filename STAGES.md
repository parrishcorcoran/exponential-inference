# Construction Plan

Each stage is a checkpoint. Do not proceed to stage N+1 before the user
approves the results of stage N.

- **Stage 0 — Scaffold & BitNet load check.** Directory layout, pinned
  dependencies, ROCm-aware model loader, a verification script that loads
  BitNet-b1.58 2B (bf16) and generates a short sequence.
- **Stage 1 — Manifold measurement.** TwoNN and participation-ratio
  estimators, cache hidden states on a 10K-token calibration corpus, report
  per-layer PR / TwoNN across all 30 layers. Expected shape for easy tokens:
  ~6 at L5, ~36 at L15, ~16 at L29. Hard tokens do not fully collapse.
- **Stage 2 — Per-token rank predictor.** From the cached states, fit
  (early-layer manifold position) -> (PR at layers 15/20/25/29). Try linear
  and small MLP. If R2 < 0.6 at layer 5, walk the prediction source layer
  forward until prediction quality is acceptable; report which layer ends up
  being used.
- **Stage 3 — Dynamic-rank forward pass.** Precompute per-layer SVD
  projection bases. Run early layers at full rank; after the prediction
  source layer, project each token into its predicted rank subspace for the
  remaining layers. Correctness gate: at full rank, outputs must match the
  base model bit-for-bit (modulo float noise). Quality gate: validation
  next-token accuracy within 1–2% of baseline.
- **Stage 4 — Acceleration curve.** Generate 2000 tokens from 10 prompts,
  baseline vs dynamic-rank, measure per-position wall-clock speedup and
  average rank. Save the curve, the rank distribution, and raw samples.
- **Stage 5 — Writeup.** Only after Stage 4 completes. README built around
  what the data says, not what we predicted.

## Execution model

The Claude Code container has no GPU. All real runs happen on the Strix
Halo (ROCm, 82 GB unified VRAM). The typical loop is:

1. Agent writes / modifies code in the container, commits to
   `claude/setup-exponential-inference-RRdcC`, pushes.
2. Human pulls on the Strix Halo, runs the stage entry point.
3. Human pastes results (or commits artifacts under `results/`) back for
   the next iteration.
