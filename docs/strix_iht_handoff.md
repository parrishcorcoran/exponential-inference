# Strix handoff: Iterative Holographic Transformer (IHT) prototype

## Task

Train two models from scratch on wikitext-2 and compare final perplexity:

1. **Baseline**: standard transformer at Qwen3-0.6B dimensions (d=1024, L=28, n_heads=16, SwiGLU MLP d_ffn=3072) — ~580M params
2. **IHT**: iterative holographic transformer at SAME dimensions (d=1024, L=28) but with no per-layer MLPs, replaced by rotation + context retrieval — ~215M params

**Success criterion**: IHT matches or beats baseline's final validation perplexity, despite having ~37% of the parameters.

If true, this validates the holographic-physics thesis: an architecture that directly implements holographic retrieval (HRR-style cumulative outer-product state + per-layer rotation + query-based retrieval) is more parameter-efficient than softmax attention + MLP for language modeling.

## Background

Full context in this repo's conversation history with the Mac session. Short version: we measured 15 transformers and found universal structural features (rotation curves, bimodal 0/π phase structure, 3-archetype token clustering). Stages 66-82 tested post-hoc applications; stage 81 tried a light phase-control prototype (null result). Stage 83 is the genuine architectural test: replace standard attention + MLP with a holographic-math equivalent and see if it works.

## What's in `scripts/stage83_iterative_holographic.py`

- `StandardTransformer`: baseline at d=1024, L=28. Standard MHA + SwiGLU MLP + RoPE + RMSNorm + weight tying.
- `IterativeHolographic`: IHT.
  - Stage 1 — causal hologram state via cumulative outer products: `S_t = S_{t-1} + k_t ⊗ v_t`
  - Stage 2 — 28 iteration layers, each: `h → R_l · h + W_Q_l · h + α · (S_{t-1} · q)` with residual
  - No softmax, no MLP. Cheaper per layer than baseline.
- Same training loop, same data, same optimizer. Clean comparison.

## Command to run on Strix

```bash
cd /path/to/Exponential-Inference
git pull

python scripts/stage83_iterative_holographic.py \
    --d-model 1024 \
    --n-layers 28 \
    --n-heads 16 \
    --d-ffn 3072 \
    --seq-len 128 \
    --batch-size 4 \
    --steps 5000 \
    --lr 3e-4 \
    --eval-every 500 \
    --out results/stage83_strix.json
```

(Device auto-detects to CUDA/ROCm.)

### Expected runtime on Strix

- Baseline (580M params, standard transformer): **~2-4 hours** (~1-2 s/step on modern GPU)
- IHT (215M params, but d×d context state operations are still expensive due to O(T·d²) retrieval): **~3-6 hours** — the outer-product state `S` is memory-intensive

Total: **~6-10 hours** for both models. Overnight-friendly.

## Memory budget on Strix

For d=1024, L=28, seq=128, batch=4:

**Baseline**:
- Model: 580M × 2 bytes (bf16) = 1.2 GB
- AdamW states (fp32): 4.6 GB
- Activations: ~2-3 GB
- **Peak: ~8-10 GB**

**IHT**:
- Model: 215M × 2 bytes = 430 MB
- AdamW states (fp32): 1.7 GB
- **Hologram state S**: `batch × seq × d × d × 4 bytes` = `4 × 128 × 1024 × 1024 × 4` = **2.1 GB** (!)
- Activations: ~1-2 GB
- **Peak: ~6-8 GB**

If OOM on IHT, reduce batch to 2 or seq to 64. Adjust steps up to maintain tokens-per-epoch.

## Expected outputs

- `results/stage83_strix.json` — full training history + final comparison
- Printed summary at end with verdict: "IHT MATCHES baseline" or "IHT worse by X%"

## Interpretation framework

### If IHT val_ppl within 5% of baseline val_ppl

**STRONG WIN**. The holographic architecture achieves comparable language modeling at ~37% params. Scale next.

### If IHT worse by 5-20%

**PARTIAL**. The architecture direction is right but needs refinement. Candidates:
- Add a small per-layer MLP (hybrid) for non-linearity
- Better initialization for R_l (from the measured universal rotation curve in Finding 02)
- More layers (60-80) at same parameter budget
- Different retrieval mechanism (e.g., multiple retrieval heads)

### If IHT much worse (>50%)

**REJECTED as-is**. Linear retrieval may be too weak at this scale. Next architecture to try:
- Hybrid: IHT base + periodic softmax attention layers (every 4th layer)
- Keep the hologram idea but use multiple parallel states (multi-head retention)

## Known architectural limitations to test

1. **Linear retrieval has O(d/log d) capacity limit** per Plate's HRR analysis. For seq=128 tokens this should be fine. If we scale to longer sequences, this may break.

2. **Rotation matrix `R_l` is full `d × d`** — could overfit. Initialization is near-identity; could constrain to orthogonal via Givens rotations if training diverges.

3. **`α` scale factor** initialized at 0.1. If training is unstable, reduce to 0.01 or add warmup.

## Troubleshooting

**OOM on IHT**: reduce `--batch-size` to 2 or `--seq-len` to 64.

**IHT diverges** (loss → inf): set `--lr 1e-4` (smaller) and investigate rotation matrix magnitudes.

**IHT learns nothing** (loss flat): check that `S_prev` is being formed correctly, that rotation matrix has non-trivial gradient, that alpha is nonzero.

## After the run

Commit `results/stage83_strix.json` + a brief markdown findings note. Push back to repo. Mac session will interpret results + decide next steps.

## Why this is worth running

- Architecture tests are cheap (~10 hours on one GPU for a decisive answer)
- Outcome is clean binary: match baseline or not
- If it works, we have a path to 3× parameter-efficient transformers that scale
- If it doesn't, we know this specific implementation of holographic math isn't enough and need to iterate

The holographic thesis has been measurement-validated across 15 models (fingerprint catalog). This is the first architectural test of whether the math that SHOULD work according to the physics actually trains end-to-end as a language model.

Good luck.
