# Z8G4 — CPU-bound RAM monster

HP Z8 G4: dual Skylake-era Xeon, **6 × 32 GB sticks per CPU socket** (so
roughly 192 GB per socket, check actual socket count for total — likely
384 GB or 768 GB depending on config), no GPU. Slow per-core but can
hold models that no single GPU can fit.

**Z8G4's killer role:** big-memory CPU host for tasks that don't need
GPU throughput. Current best fit: **cross-model manifold fingerprinting**
(the stage 60 task described below).

## Current priority (2026-04-21): cross-model manifold fingerprint

**Old priorities are superseded.** The previously assigned Qwen3-14B
corpus generation targeted the Holographic-Matryoshka-as-trained-
weight-factoring path, which has been FALSIFIED (0% vs original on
every tested model size; see `machines/strix_halo/results/validate_14b.log`).
Theory #6 (manifold-target training) is the current active track, and
the new priority below is its upstream diagnostic.

**The new priority: run `scripts/measure_manifold_fingerprint.py`
across models of varying size and tokenizer family, commit the
fingerprint JSONs back.**

The fingerprint combines:
- Per-layer bootstrap TwoNN dim (Finding 01 style)
- Per-layer-transition two-mode rotation spectrum (stage 58 style)
- Carry / flip / mid fraction per layer (stage 59 style)
- Rotation-operator eigenvalue distributions
- Two-mode concentration trend through the stack

This is what the main Mac session asked for after discovering the
two-mode structure. Z8G4's niche: run it on the models that don't
fit on Mac or Strix — Qwen3-32B, Qwen3-72B, Llama-3-70B, maybe
DeepSeek-V3. Each fingerprint is a JSON < 100 KB, committable. The
cross-architecture comparison is what lets us check whether the
two-mode structure is a universal transformer feature or a Qwen
quirk.

### Commands

Fingerprint individual models:

```bash
python machines/z8g4/scripts/measure_manifold_fingerprint.py \
    --model Qwen/Qwen3-32B \
    --out machines/z8g4/results/fingerprint_qwen3_32b.json

python machines/z8g4/scripts/measure_manifold_fingerprint.py \
    --model meta-llama/Llama-3.1-70B \
    --out machines/z8g4/results/fingerprint_llama3_70b.json

python machines/z8g4/scripts/measure_manifold_fingerprint.py \
    --model mistralai/Mistral-7B-v0.1 \
    --out machines/z8g4/results/fingerprint_mistral_7b.json
```

Use `numactl -N 0 -m 0` for models ≤ 32B; span sockets for larger.
Expected wall-clock per fingerprint: a few hours on CPU for 32B, a
day for 70B. Overnight / weekend scale.

### Then commit the results

```bash
git add machines/z8g4/results/fingerprint_*.json
git commit -m "z8g4: fingerprints for <model list>"
git push
```

The main Mac session picks up the JSONs and does cross-architecture
comparison. If the two-mode spectrum holds across tokenizer families
and scales, candidate Finding 14 gets formalized.

## Also still useful (secondary priority)

1. **High-N bootstrap TwoNN** on Qwen3 family. Current Finding 01
   measurements used N ≈ 300 hidden states per layer. Re-measure with
   N ≥ 3000 on at least 3 Qwen3 sizes. Tightens the noise caveat added
   to Finding 01.
2. **Multi-teacher ensemble embedding basis** for Theory #6. Average
   Qwen3-0.6B + 1.7B + 4B + 14B + 32B embedding-matrix PCA bases into
   a cleaner manifold measurement. Upload as a single HF artifact.
3. **Teacher-corpus generation** for Theory #6 scaled training — but
   wait for Strix to confirm the approach on smaller scale first.

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
