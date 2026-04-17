# Running on MacBook Air M4 (16GB)

BitNet b1.58-2B-4T fits comfortably in 16GB. Apple Silicon's Metal Performance Shaders (MPS) give real GPU acceleration.

## Setup

```bash
# Clone the repo
git clone https://github.com/parrishcorcoran/Exponential-Inference.git
cd Exponential-Inference

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate numpy scipy scikit-learn matplotlib tqdm datasets
```

## Run the Manifold Measurement

```bash
# Stage 0: verify the model loads (should take ~30s on M4)
python scripts/stage0_verify.py

# Stage 1: cache hidden states + measure manifold (~5 min on M4)
python scripts/stage1_measure.py
```

That's it. Results appear in:
- `results/stage1_manifold.json` — the manifold data (PR, TwoNN, SVD ranks per layer)
- `results/stage1_manifold.png` — visualization

## Run Generation with KV Entropy Measurement

```bash
# Measure per-token latency and KV attention entropy
python scripts/stage4_direct.py --max-new-tokens 500 --max-prompts 3
```

Results in `results/stage4_direct.json`.

## Run Rank-Reduced Generation (the actual experiment)

```bash
# Compare baseline vs rank-reduced generation
python scripts/stage4_rank_reduced.py \
  --max-new-tokens 200 \
  --max-prompts 2 \
  --max-rank 256 \
  --min-rank 8 \
  --target-layers 15 20 25 29
```

On M4 with MPS, the rank reduction should show real speedup because Apple Silicon GPU computes FLOPs proportional to rank (unlike CPU which is memory-bound).

## Expected Results

BitNet 2B on any hardware should show:
- **TwoNN ~10** across all 31 layers (the manifold dimensionality)
- **PR profile**: expand (10→143) → peak at L21 → collapse (143→32)
- **Per-token KV entropy** tracks the spin glass relaxation

## Notes

- The model is ~1.2GB — fits easily in 16GB
- MPS backend should be auto-detected by PyTorch
- If you get "MPS not available", make sure you have PyTorch >= 2.0 with Metal support
- Generation at ~15-30 tok/s expected on M4
