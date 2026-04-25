# Pipeline Smoke Test on Qwen3-0.6B

End-to-end demonstration of the full wormhole compression pipeline on
Qwen3-0.6B. Designed as a smoke test to surface roadblocks before
running the real pipeline on Strix at 1.7B / 14B scale.

## What this does

1. **Phase 1: Shape measurement** — wormhole topology, KV cache geometry
2. **Phase 3a: Gentle weight rank anneal** — small ~5% reductions on
   q/k/v/o/gate/up/down projections, finetune between steps. Comfortable
   tolerance, not pushing limits.
3. **Phase 3b: Aggressive KV cache compression** — large reductions on
   k_proj, v_proj rank with finetune. Goal is significant compression
   so downstream Medusa-head experiments have small per-head storage cost.
4. **Save**: full state dict + config + loader for downstream Medusa work.

## Why aggressive KV but gentle weights?

Wide KV-Medusa enables ~10-50× decode throughput by adding many parallel
prediction heads. Each head's storage cost scales with KV cache size.
So:
- Small weight compression preserves model quality (model is the brain)
- Big KV compression makes Medusa heads cheap to add (cache is the playground)

Per-head storage at uncompressed KV: ~1.5MB for 0.6B (28 layers × 1024 dims × 16 bits × 2 for K+V)
Per-head storage at our target KV (rank 256): ~360KB
Per-head with Q4 stacked: ~90KB

That's 16-100× cheaper per head, enabling experiments with 20-50 heads.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | The full pipeline script |
| `checkpoints/model_state.pt` | Compressed model state dict |
| `checkpoints/config.json` | Per-layer ranks, base model, ranks |
| `checkpoints/load.py` | Helper to reload the compressed model |
| `results/smoke_results.json` | Phase-by-phase metrics |

## Run

```bash
cd pipeline_smoke_06b
.venv/bin/python pipeline.py
```

Expected runtime on Mac MPS: ~30-60 minutes for 0.6B.

## Verifying it worked

After the run, check `results/smoke_results.json`:
- `wormhole_shape.throat_pr` should be near 1 (rank-1 throat confirmed)
- `weight_rank.final_compression_ratio` should be ~1.05-1.2× (gentle)
- `kv_compression.final_compression_ratio` should be ~3-8× (aggressive)
- `final_ppl` should be within ~30% of baseline

## Downstream Medusa experiments

```python
from checkpoints.load import load_compressed_model
model, tokenizer = load_compressed_model("pipeline_smoke_06b/checkpoints")

# model has FactoredLinear layers replacing k_proj and v_proj
# throat layer index in config["throat_layer"]
# compressed cache fits ~50 Medusa heads at the storage cost of 1 uncompressed
```

## Why 0.6B for the smoke test

- Fits in 13GB Mac RAM
- Runs in 30-60 min vs days at 1.7B
- Finds the same pipeline bugs (the issues are in the code, not the model)
- Same script will run on Strix for 1.7B/14B with no changes
