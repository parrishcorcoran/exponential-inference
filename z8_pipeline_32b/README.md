# Z8 — Qwen3-32B Layer-wise Compression Pipeline

This folder is a **self-contained handoff for Z8** to run the full
wormhole compression pipeline on Qwen3-32B using layer-wise cached
calibration.

Z8 has 700GB RAM, no GPU, dual Xeon (Cascade Lake, AVX-512). Layer-wise
calibration is the right approach because it doesn't need GPU compute
and the per-layer optimization is tractable on CPU.

Empirical speedup measured on Mac (smoke test): **442× per-step** for
layer-wise vs full-model FT, ~55-100× total wall-clock at typical
anneal stage counts.

---

## Project context (catch up to speed)

We've been measuring trained transformers and finding they have a
**universal wormhole topology** in their residual stream:

- Wide "mouths" at input and output (high-rank, ~50 dims active)
- Narrow "throat" in the middle (rank-1 in variance, magnitude grows 800×)
- Two-gate structure (entry wall L5, exit wall L19-L21 for 0.6B)
- Universal across model sizes (0.6B, 1.7B, 14B) AND precision regimes
  (fp16, BitNet ternary)

Findings 13-20 in `findings/` document this. `LEVERS.md` catalogs the
49 compression axes we've identified.

The methodology (`RUNBOOK.md`):
1. Measure shape (participation ratio per layer)
2. Apply trained-aware compression (anneal + finetune)
3. Find where finetuning can no longer recover quality

The recurring signal across 3 measurements (stages 119, 124b, 134):
**post-hoc projection ALWAYS fails on small models. Slow anneal +
finetune ALWAYS works at the same target.** Strix achieved rank-3
attention throat on 14B with QUALITY IMPROVEMENT (LASER effect).

What's new for this run: **layer-wise cached calibration** instead of
full-model finetuning. This is what GPTQ/AWQ use — capture each
Linear's input/output once, then optimize each Linear independently
against captured pairs. Much faster and CPU-tractable.

---

## What this pipeline does

### Phase 1: Shape measurement (~30-60 min on Z8 CPU)
- Per-layer participation ratio of residual stream
- Per-layer KV cache rank
- Confirms wormhole topology on 32B (we expect: deeper throat, wider
  mouths than 14B, since LASER slack scales with model size)

### Phase 2: Teacher capture (~30-60 min on Z8 CPU)
- Run 32B teacher forward ONCE on calibration corpus (~64 batches)
- For each Linear in attention (q_proj, k_proj, v_proj, o_proj):
  capture (input, output) tensor pairs
- Memory: ~50-100GB for captures (fits comfortably in 700GB)

### Phase 3a: Gentle weight rank anneal — q_proj, o_proj
- Per-layer rank reduction with sequential re-capture
- Step factor: ×0.95 (gentle 5% per stage)
- Target: 80% of full rank retained (~1.25× param reduction)
- Each Linear: 50-100 gradient steps on cached pairs (per stage)

### Phase 3b: Aggressive KV cache compression — k_proj, v_proj
- Push k_proj, v_proj rank toward 64-128 (from 5120 in 32B)
- Step factor: ×0.85 (15% per stage)
- Re-capture activations every 3 stages so compounding errors don't
  destroy downstream layers
- Goal: massive cache compression for downstream Medusa experiments
  (each compressed Linear ~30× smaller, enables 30-50 Medusa heads
  cheaply)

### Phase 4: Optional cleanup full-model FT (~few hours)
- Short end-to-end FT pass to clean up compounding from layer-wise
- Maybe 500-1000 steps full FT, much less than naive approach
- Skip if Z8 CPU is too slow for full forward; layer-wise quality
  should be acceptable on its own

### Phase 5: Save
- Full state dict
- Per-layer rank config
- Loader script for downstream use

---

## Expected wall time on Z8

Hardware: Z8 G4 with dual Xeon Cascade Lake, AVX-512, 700GB RAM, no GPU.

| Phase | Est. time |
|---|---|
| Phase 1: Shape | 30-60 min |
| Phase 2: Capture teacher | 30-60 min |
| Phase 3a: Weight anneal (8 stages × 56 layers × cached) | 4-8 hours |
| Phase 3b: KV anneal (12 stages × cached) | 4-8 hours |
| Phase 4: Optional cleanup FT (500 steps) | 4-12 hours |
| Save | <5 min |
| **Total** | **~12-30 hours** |

For aggressive runs (more anneal stages, more FT steps): scale linearly.

vs **naive full FT on 32B CPU**: 5-6 years (not feasible).

---

## What Z8 needs to do

```bash
# Pull latest from main
cd /path/to/Exponential-Inference
git pull

# Install deps if needed (transformers, datasets, torch)
.venv/bin/pip install -U torch transformers datasets

# Run the pipeline
.venv/bin/python z8_pipeline_32b/pipeline.py

# Default args run on Qwen3-32B with sensible budgets
# Override if needed:
#   --model Qwen/Qwen3-32B
#   --calib-batches 64
#   --weight-target-ratio 0.80
#   --kv-target-rank 64
```

Logs go to stdout. Checkpoints save to `z8_pipeline_32b/checkpoints/`.
Progress is incremental — if it crashes, the last completed stage is
on disk.

---

## What Mac will do while Z8 runs this

On Mac side (parallel work):
1. Build Medusa head training scripts that consume the compressed model
2. Build wide KV-Medusa heads (the big speedup unlock)
3. Write up findings 21+ documenting the layer-wise approach
4. Prep benchmark scripts for MMLU / HellaSwag / GSM8K against the
   compressed 32B once Z8 finishes

When Z8 finishes:
- Push compressed model to HuggingFace as `wormhole-qwen-32b`
- Run benchmarks, write up results
- Substack post + portfolio drop

---

## Verifying success

After the run, check `z8_pipeline_32b/results/pipeline_results.json`:

- `wormhole_shape.throat_pr` should be near 1 (rank-1 throat at 32B too)
- `weight_compression_ratio` should be ~1.2-1.5×
- `kv_compression_ratio` should be ~10-50× (we want it big for Medusa)
- `final_ppl` should be within ~10-30% of baseline on WikiText
  - 32B has more LASER slack than 0.6B, so might IMPROVE quality
- Per-layer ranks in `config.json` should follow wormhole shape (lower
  at throat, higher at gates)

---

## Downstream Medusa experiments (post-Z8)

After this pipeline runs:

```python
from z8_pipeline_32b.checkpoints.load import load_compressed_model

model, tokenizer = load_compressed_model("z8_pipeline_32b/checkpoints")

# model has FactoredLinear layers replacing q/k/v/o
# Cache footprint per token per layer: ~64 floats (vs 5120 baseline)
# Medusa head storage cost: ~64 floats × 28 layers × 16 bits = ~3.6KB per head
# vs uncompressed: ~28 × 5120 × 16 = ~290KB per head
# = 80× cheaper per head → can afford 50+ heads
```

Goal: train ~20-50 Medusa heads on the compressed throat → projected
5-10× decode throughput.

---

## Caveats

1. **Layer-wise compounds errors**. Sequential re-capture mitigates.
   Final cleanup FT (Phase 4) further mitigates if needed.
2. **CPU compute is slow**. Layer-wise tasks are tiny so it's fine,
   but full FT cleanup is expensive. Skip Phase 4 if it's too slow.
3. **Single-precision FP32 default on CPU**. PyTorch CPU bf16 isn't
   widely supported pre-Sapphire-Rapids. We use fp32.
4. **Model loading**: 32B fp16 weights download is ~64GB. Make sure
   HuggingFace cache has space.

---

## If something fails

- **OOM during capture**: reduce `--calib-batches` (default 64 → 32)
- **Capture too slow**: reduce `--seq-len` (default 256 → 128)
- **Quality bad after Phase 3a**: increase `--weight-target-ratio`
  (less aggressive, default 0.80 → 0.85)
- **Quality bad after Phase 3b**: increase `--kv-target-rank` (less
  aggressive, default 64 → 128)
- **Need to resume**: per-stage state is saved to disk; can resume by
  loading state from last completed stage

---

## Methodology references

- `findings/13_*.md` — Wormhole topology
- `findings/14_*.md` — Universal geometry, private mouth-2 decoder
- `findings/15_*.md` — Two-gate topology
- `findings/16_*.md` — KV cache as 360° field
- `findings/17_*.md` — Post-hoc projection floor (the recurring signal)
- `findings/18_*.md` — Compression topography
- `findings/19_*.md` — Certainty growth (H2O replacement)
- `findings/20_*.md` — Wormhole on BitNet (universal across precision)
- `LEVERS.md` — 49 compression axes catalog
- `RUNBOOK.md` — protocol for any model
- `SAVEPOINT_2026-04-24.md` — full project state at handoff

---

## Date

2026-04-24. Maintain and update with results.
