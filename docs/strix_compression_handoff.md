# Strix handoff: Compression + Medusa + early exit stack

## Task

Validate a stacked compression pipeline on Qwen3-14B as smoke test for
future 32B/70B deployment. The stack combines:

1. Early exit (per-layer LM probes)
2. Medusa speculative-decode heads (added one at a time)
3. Aware low-rank KV compression (MLA-style)
4. BitNet-style QAT ternary body weights
5. Embedding quantization
6. Shared-basis cross-layer factorization
7. Round-robin curriculum training

The novelty is the combination — no published work stacks all of these
with co-training on a pretrained model. Our GPU-friendly compression
choices specifically allow Medusa + early exit to coexist with aggressive
compression, which BitNet's CPU-only substrate could not.

## Priority order (phases)

Each phase validates before the next is launched. "Good enough, not
perfect" — 14B is smoke test, not final result.

### Phase 0 — Environment

```bash
cd /path/to/Exponential-Inference
git pull
# Kill any running stage 83 (IHT from earlier plan — today's findings
# show it's doomed without MLP blocks).
# Check disk: expect ~30GB for Qwen3-14B model cache.
```

### Phase 1 — Early exit (one script, one run)

```bash
python scripts/stage101_early_exit.py \
    --model Qwen/Qwen3-14B \
    --steps 2000 \
    --eval-every 200 \
    --lr 1e-4 \
    --batch-size 1 \
    --seq-len 256 \
    --save-probes checkpoints/stage101_probes.pt \
    --out results/stage101_early_exit.json
```

Expected: ~4-8 hours on Strix Halo.

Success criterion (loose): per-layer val CE should be descending
monotonically. By layer ~28 of 40, CE should be within 20% of the
final-layer CE. Any layer ≥25 producing correct top-1 for >60% of
val tokens is good enough to call early-exit working.

Report: layer-by-layer CE curve + which layer hits each confidence
threshold.

### Phase 2 — Medusa heads (one at a time)

Run 2a first (1 head). If it trains cleanly, add 2b (2 heads total, new
one trained while first frozen). Continue 2c, 2d, 2e.

```bash
# 2a — first Medusa head
python scripts/stage102_medusa.py \
    --model Qwen/Qwen3-14B \
    --num-heads 1 \
    --steps 2000 \
    --save-heads checkpoints/medusa_1.pt \
    --out results/stage102_medusa_1.json

# 2b — add second head, freeze first
python scripts/stage102_medusa.py \
    --model Qwen/Qwen3-14B \
    --num-heads 2 \
    --load-prev checkpoints/medusa_1.pt \
    --steps 2000 \
    --save-heads checkpoints/medusa_2.pt \
    --out results/stage102_medusa_2.json

# 2c, 2d, 2e — continue through head 5 similarly
```

Expected: ~4-6 hours per head = ~20-30 hours total for 5 heads.

Success criterion (loose): head k's val accuracy should be > 20% for
small k (head 1), decreasing to > 5% for head 5. Any pattern where
accuracy plateaus above random-chance (1/vocab ≈ 7e-6) is useful for
speculative decoding. Stop adding heads when val accuracy drops below
~5% — that head won't pay its cost.

Report: acceptance rate per head at teacher confidence thresholds.

### Phase 3 — Combined (early exit + Medusa)

Not a new training run — just integrated inference test. Run both
features together and confirm:
- No regression in base-model quality
- Combined inference speedup measurable

No separate script; inference test can be a notebook.

### Phase 4 — Compression round-robin (stage 103 — to be written)

Only after Phases 1-3 are validated. Stage 103 will be:

```bash
python scripts/stage103_roundrobin.py \
    --model Qwen/Qwen3-14B \
    --load-probes checkpoints/stage101_probes.pt \
    --load-medusa checkpoints/medusa_5.pt \
    --ranks 128,96,64,48,32,24,16 \
    --weight-bits 16,8,6,4,3,2,1.58 \
    --embed-bits 16,8,6,4 \
    --phases-per-step 150 \
    ...
```

This script doesn't exist yet — will draft after Phases 1-3 succeed.

## Memory budget on Strix Halo (~89GB VRAM)

**Phase 1 (early exit)**:
- Base model frozen in bf16: ~28GB
- Probes (40 × d×d affine): ~0.5GB
- AdamW states for probes only: ~1GB
- Activations (seq=256, batch=1, gradient checkpointing): ~5-10GB
- **Peak: ~40GB** — comfortable.

**Phase 2 (Medusa)**:
- Base model frozen: 28GB
- Medusa heads (small): <1GB
- AdamW states (active head): <1GB
- Activations: ~5-10GB
- **Peak: ~40GB** — comfortable.

**Phase 4 (compression round-robin)**: tighter; will detail when writing stage 103.

## Reporting back

Each phase produces a `results/stage*.json`. Commit and push:

```bash
git add results/stage101_early_exit.json
git add results/stage102_medusa_*.json
git add checkpoints/*.pt   # or store via HF Hub if too big
git commit -m "stage 101/102 strix results"
git push
```

Mac session will read results and decide whether to proceed to next phase.

## Troubleshooting

**OOM during Phase 1**: reduce `--batch-size` to something smaller
(already 1 — try gradient accumulation instead) or `--seq-len` to 128.

**Medusa head not training**: check `--lr` — 1e-4 might be too high for
layered MLP. Try 5e-5.

**Early exit probes all predict same thing**: the layer loss weights
might be unbalanced. Currently uniform `1/(L+1)`. Try `linspace(0.5, 1.0)`
to weight later layers more.

**Base model loads in fp16 but training crashes**: some ops may need
to upcast to fp32. The scripts already use `.to(torch.float32)` around
the trainable modules; if you see dtype mismatches, that's the fix area.

## Why this work is worth running

No prior paper has stacked BitNet-family ternary with MLA-style aware
KV compression with Medusa speculative decoding with early exit. The
blocker was that BitNet optimized for CPU inference (ternary bitwise
ops), which can't run Medusa's parallel drafting or early exit's
conditional computation efficiently. Our stack stays GPU-friendly
throughout, so Medusa and early exit can ride on top.

This 14B smoke test validates the pipeline. If it works, same pipeline
applies to Qwen2.5-72B or Qwen3-32B with minor adjustments.

Expected total Strix time: ~50-80 hours across all phases. Each phase
is pausable/resumable via git checkpoints.
