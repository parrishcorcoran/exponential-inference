# Strix Halo — Primary training & inference machine

AMD Strix Halo with ROCm, ~82 GB unified VRAM. Fast enough for
interactive training of rank-k factored students; memory large enough
to hold a 32B teacher alongside a factored student during distillation.

## Role in this project

Strix Halo is the **interactive / training** machine. Its niche:

1. **Rank-k factored student distillation.** Runs Matryoshka training
   of the factored student against a local teacher (up to 32B bf16
   fits with student and activations).
2. **Integrated all-dynamic runtime prototyping.** Builds the rank-k
   forward pass + dynamic policy (entropy-driven rank, saddle
   detection, head pruning). ROCm gives cheap per-kernel launches,
   unlike MPS where hooks eat the budget.
3. **Wall-clock benchmarking on ROCm.** Measures the actual inference
   speedup claims on hardware that can realize them.

## Setup

```bash
git clone https://github.com/parrishcorcoran/Exponential-Inference.git
cd Exponential-Inference
python3 -m venv .venv
source .venv/bin/activate

# ROCm PyTorch — adjust URL to your ROCm version.
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
pip install transformers accelerate numpy scipy scikit-learn datasets huggingface_hub

# Verify ROCm visible
python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

HF auth:
```bash
huggingface-cli login
```

## Data flow

**Owned by Strix Halo:**
- `machines/strix_halo/scratch/` — local cache of teacher weights,
  downloaded corpora, student checkpoints. **Gitignored.**
- `machines/strix_halo/results/` — small JSON, loss curves, evaluation
  summaries. **Committed to git.**

**NEVER commit to git from Strix:**
- Anything in `scratch/`.
- Teacher weights, full student checkpoints, teacher-output caches.

**Upload to HF Hub from Strix:**
- Trained factored student weights (`safetensors`) under
  `<username>/exponential-inference-student-<model>-r<rank>`.
- Model card markdown with recipe + eval numbers.

**Strix Halo ≠ Z8G4.** Only write to `machines/strix_halo/`. Consume
(download) from Z8G4's HF uploads; do not modify `machines/z8g4/`.

## Sync protocol

1. Pull from git: `git pull origin main`.
2. Pull latest HF corpora from Z8G4:
   ```
   huggingface-cli download <username>/corpus-<model> corpus.pt \
       --local-dir machines/strix_halo/scratch/corpora/
   ```
3. Train. Outputs go to `machines/strix_halo/results/*.json` and
   `machines/strix_halo/scratch/student_<run_id>/`.
4. Commit only small artifacts:
   ```
   git add machines/strix_halo/results/*.json
   git commit -m "strix: <experiment>"
   git push
   ```
5. Large artifacts (student weights, checkpoints):
   ```
   huggingface-cli upload <user>/exponential-inference-student-<tag> \
       machines/strix_halo/scratch/student_<run_id>/ . 
   ```
   Note the HF URL in a markdown under `machines/strix_halo/results/`.

## Scripts

- `scripts/train_matryoshka.py` — Matryoshka distillation of a factored
  student. Consumes a teacher corpus (from Z8G4 via HF) and a local
  teacher. Supports rank-sweep eval and the all-dynamic runtime.
- (forthcoming) `scripts/integrated_runtime.py` — rank-k forward pass
  with dynamic policy.

Each script has a CLI; run with `--help`.

## Tuning for ROCm

- **ATTN**: `attn_implementation="eager"` for training (we need
  `output_attentions=True` for entropy signal). For inference-only
  benchmarks you can switch to `sdpa` for speed.
- **Precision**: bf16 for forward, fp32 for factored A/B during
  training. ROCm handles this mix fine.
- **Optimizer**: AdamW with grad clip 0.5, LR 1e-4 to 3e-4. Use cosine
  schedule with ~10% warmup.
