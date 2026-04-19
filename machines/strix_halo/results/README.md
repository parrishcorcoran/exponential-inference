# Strix Halo Results

Machine: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)
VRAM: 89 GB unified
ROCm: 7.13 nightly (PyTorch via rocm.nightlies.amd.com/v2/gfx1151/)

## Status
- GPU verified and working
- PyTorch ROCm operational
- **Qwen3-14B Holographic Matryoshka: RUN COMPLETE (2026-04-19 06:24 EDT).**
  100% token match at every tested k ∈ {32, 48, 64, 96, 128}.
  See `qwen3_14b_r32_128.json` and `run_14b.py`.

## The completed run (reference, not re-run)

**Qwen3-14B, k ∈ [32, 128], 2000 steps, ~35 min on Radeon 8060S, 33.9 GB peak.**

Result summary:

| k | compression | token match | notes |
|---|---|---|---|
| 32 | 160× | **100.0%** | smallest factored rank tested |
| 48 | 107× | 100.0% | |
| 64 | 80× | 100.0% | |
| 96 | 53× | 100.0% | |
| 128 | 40× | 100.0% | largest tested; headroom exists |

- 514M factored params (3.9% of full 14B model)
- Well above the manifold floor (~80–160M estimated in Finding 05)
- KL converged to 0.0 by step 500 of 2000
- 0.6B below-floor contrast: 0% match at any rank

**This is the Holographic Matryoshka empirical confirmation referenced
in Finding 10.**

## Technique recipe (kept for reproduction and for 32B follow-up)

**Technique name:** Holographic Matryoshka — nested rank-k factoring of
boundary weights that preserves bulk dim per Finding 10. Uses width
(Matryoshka rank-k, trained) and length (dynamic early-exit at
stabilization_depth, runtime). Bulk (MLP intermediate dim) stays full.

Per Finding 10 (holographic compressibility), the Matryoshka factoring is
boundary compression — it restricts bulk *rank*, not bulk dim. This is the
correct architecture. The remaining requirement is being above the manifold
floor (Finding 05 — 80-160M factored params).

Sizing for Qwen3-14B (d_model=5120, d_int=~13824, L=40):
- k = 64:  3 × 13824 × 64 × 40 ≈ **106M factored MLP params** — above 80M floor
- k = 128: ≈ **212M** — above 160M ceiling

Both k values should converge. Strix has 89 GB VRAM; Qwen3-14B in bf16 fits
with ~60 GB headroom for activations, student, and optimizer state.

## Commands to run

```bash
cd Exponential-Inference
source .venv/bin/activate

# 1) Generate teacher corpus on Z8G4 first (if not already on HF):
#    python machines/z8g4/scripts/generate_teacher_corpus.py \
#        --teacher Qwen/Qwen3-14B --out corpus.pt
#    huggingface-cli upload <user>/corpus-qwen3-14b corpus.pt

# 2) Pull corpus on Strix:
huggingface-cli download <user>/corpus-qwen3-14b corpus.pt \
    --local-dir machines/strix_halo/scratch/corpora/

# 3) Train:
python machines/strix_halo/scripts/train_matryoshka.py \
    --teacher Qwen/Qwen3-14B \
    --corpus machines/strix_halo/scratch/corpora/corpus.pt \
    --k-min 64 --k-max 128 \
    --steps 8000 \
    --lr 1e-4 \
    --out machines/strix_halo/results/matryoshka_qwen3_14b_r64_128.json \
    --save-student machines/strix_halo/scratch/student_14b_r64_128/
```

Expected wall-clock on Strix Halo (ROCm): 4–12 hours at 8000 steps with
14B teacher forward + student forward-backward per step. Monitor for
training divergence in the first 500 steps — if loss NaNs or climbs past
teacher, halve LR.

## Evaluation at completion

The script's final block evaluates at several k values in [k_min, k_max]
and records teacher_ppl, student_ppl, ppl_ratio, top1_agree, top5_agree.
Target: **top1_agree ≥ 0.80** at k=128 on the held-out eval. If met, this
is the first confirmed Matryoshka student above the manifold floor and
validates Finding 10's architecture.

Upload student weights to HF after a passing eval:
```bash
huggingface-cli upload <user>/exponential-inference-student-qwen3-14b-r64-128 \
    machines/strix_halo/scratch/student_14b_r64_128/ .
```
And commit the JSON eval to git:
```bash
git add machines/strix_halo/results/matryoshka_qwen3_14b_r64_128.json
git commit -m "strix: Qwen3-14B Matryoshka r64-128 first above-floor run"
```

## If it succeeds

The student can drive the integrated runtime (rank-k forward pass + dynamic
policy — rank, length early-exit, head pruning) and give actual wall-clock
inference speedup measurements on ROCm. This is what pushes the project
from "predicted 10-30x" to "measured 10-30x" on 14B.

## If it fails

Most likely cause: training instability (KL + hidden MSE loss ratio off, LR
too high for ROCm's mixed precision). Secondary: the manifold floor is
higher than Finding 05 estimated and 14B is borderline. Fallback: try k_min
= 128 only (fixed-rank, no Matryoshka sampling) to isolate whether
divergence is from rank sampling or from rank-k itself.

## Related
- [Finding 10](../../../findings/10_holographic_compressibility.md) —
  explains why factoring every MLP Linear is the right target.
- [Finding 05](../../../findings/05_manifold_floor.md) — sets the rank
  floor we need to clear.
- `scripts/train_matryoshka.py` — the training script itself.

