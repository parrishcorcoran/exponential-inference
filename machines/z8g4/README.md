# Z8G4 — CPU-bound RAM monster

HP Z8 G4: dual Skylake-era Xeon, 700+ GB RAM, no GPU. Slow per-core but
can hold models that no single GPU can fit.

## Role in this project

Z8G4 is the **offline / oracle / big-model** machine. It is NOT for
interactive training. Its niche:

1. **Manifold measurement on models > 72B.** Load the biggest models
   possible (Llama-3-405B in 8-bit, Qwen-72B, DeepSeek-V3, etc.). Run
   Stage 1 style TwoNN + PR. Establishes scale universality of the
   ~9-11 intrinsic dim claim on models that don't fit anywhere else.

2. **Teacher-sampled corpus generation.** For any teacher that fits in
   RAM, generate a large on-policy calibration corpus (token IDs only
   — no hidden states, to keep artifacts transportable). Upload the
   token corpus to HuggingFace Hub as a dataset. Strix Halo consumes
   it during Matryoshka distillation.

3. **Overnight evaluation sweeps.** Once a student is trained on Strix
   and pushed to HF, Z8G4 can compute robust distribution metrics
   (ppl, KL, top-k) over millions of tokens of held-out text. Takes
   hours but runs unattended.

## Environment

```bash
# One-time setup
git clone https://github.com/parrishcorcoran/Exponential-Inference.git
cd Exponential-Inference
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate numpy scipy scikit-learn datasets huggingface_hub
```

For HF uploads:
```bash
huggingface-cli login
```

## Tuning for Skylake Xeon CPU-bound work

Z8G4 has two sockets. Decide per run whether to:
- **Pin to one socket** (avoid NUMA traffic, smaller but faster per op).
- **Span both** (more parallelism, but memory crosses sockets).

Small models (<10B): pin to one socket. `numactl -N 0 -m 0 python ...`.

Large models (>32B): let both sockets run, accept NUMA cost. Don't pin.

### Threadpool envs

```bash
# Start with physical cores on ONE socket.
export OMP_NUM_THREADS=$(lscpu -p=Core,Socket | grep -v '^#' | awk -F, '$2==0{print $1}' | sort -u | wc -l)
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# If spanning sockets, set to total physical cores.
```

Run a quick calibration pass and watch `htop` / `numastat`. Adjust until
memory bandwidth and CPU utilization both look saturated.

PyTorch-specific:
```bash
# Avoid over-subscription when using dataloader workers.
export TOKENIZERS_PARALLELISM=false
```

## Data flow

**Owned by Z8G4:**
- `machines/z8g4/scratch/` — local cache of hidden states, model weights,
  anything heavy. **Gitignored.** Stays on Z8G4.
- `machines/z8g4/results/` — small JSON result files. **Committed to
  git.** These are Z8G4's contributions to the shared record.

**NEVER commit to git from Z8G4:**
- Anything in `scratch/`.
- Raw model weights, KV caches, full activation snapshots.
- Corpora > 50 MB (use HF Hub for those).

**Upload to HF Hub (not git) from Z8G4:**
- Teacher-sampled token corpora (upload as HF dataset under
  `<username>/exponential-inference-corpus-<model>-<size>`).
- Measurement snapshots (hidden-state caches) if needed for downstream
  analysis on Strix.

**Strix Halo ≠ Z8G4.** Each machine has its own `machines/<name>/` folder
and never writes to the other's. Shared results end up in `shared/` only
after being summarized.

## Sync protocol

1. Pull from git at start of any session: `git pull origin main`.
2. Run experiments → results land in `machines/z8g4/results/*.json`.
3. Commit only small artifacts:
   ```
   git add machines/z8g4/results/*.json
   git commit -m "z8g4: <experiment>"
   git push
   ```
4. Large artifacts: `huggingface-cli upload <repo> <path>`. Note the HF
   URL in a markdown file under `machines/z8g4/results/`.
5. Strix Halo never touches `machines/z8g4/*`. Read-only.

## Scripts

- `scripts/measure_manifold_large.py` — Stage 1 (PR + TwoNN + r50/r90/r95/r99)
  on any HF model. Streaming forward pass, fp32 eigh on CPU. Handles
  models up to 500 GB on disk.
- `scripts/generate_teacher_corpus.py` — sample a teacher at T>0 over
  diverse seed prompts, save tokens only (no hidden states). Ready to
  push to HF Hub.
- `scripts/validate_student.py` — run a trained factored student against
  teacher on a large held-out text set. Compute ppl, KL, top-k
  distribution metrics. Output to `results/`.

Each script has a CLI; run with `--help` for details.
