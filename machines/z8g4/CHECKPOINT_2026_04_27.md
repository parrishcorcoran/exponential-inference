---
name: Z8G4 Checkpoint — April 27, 2026
description: Session handoff. KV-Medusa 32B results in hand. Next step: build verification loop for actual wall clock decode speedup.
---

# Z8G4 CHECKPOINT — April 27, 2026

## WHAT HAPPENED THIS SESSION

### KV-Medusa on Qwen3-32B — DONE
Trained 10 KV-Medusa heads on the uncompressed 32B base model. Each head predicts future K and V cache entries from the hidden state at the middle layer (L32).

**Results (strict findings, measured):**

| Head | Offset | cos_k | cos_v | Accept (>0.7) |
|------|--------|-------|-------|---------------|
| 1 | t+1 | 0.900 | 0.381 | 100% |
| 2 | t+2 | 0.890 | 0.313 | 100% |
| 3 | t+3 | 0.893 | 0.289 | 100% |
| 4 | t+4 | 0.889 | 0.292 | 100% |
| 5 | t+5 | 0.887 | 0.290 | 100% |
| 6 | t+6 | 0.892 | 0.267 | 100% |
| 7 | t+7 | 0.891 | 0.272 | 100% |
| 8 | t+8 | 0.891 | 0.277 | 100% |
| 9 | t+9 | 0.893 | 0.287 | 100% |
| 10 | t+10 | 0.892 | 0.286 | 100% |

- K prediction is flat at ~0.89 across all offsets (no degradation)
- V prediction is weaker (~0.27-0.38) but V is content-specific and harder
- 100 training steps per head, ~31M params each (MLP: 5120→2560→1024)
- Total training time: ~4.4 hours on CPU
- Trained on OpenWebText, validated on held-out samples
- Results: `results/kv_medusa_32b.json`
- Script: `z8_pipeline_32b/kv_medusa_cpu.py`
- Head weights: `z8_pipeline_32b/kv_medusa_results/`

**NOT measured yet:** actual wall clock decode speedup. Cosine similarity ≠ correct tokens. Need verification loop.

### Inference Speed Benchmark on Qwen3-14B — DONE
Measured real wall clock forward pass and generation speed with factored (SVD) models at various ranks.

| Rank | Params | Forward | Fwd Speedup | Gen/tok | Gen Speedup |
|------|--------|---------|-------------|---------|-------------|
| Full | 14.8B | 5,871ms | 1.0x | 478ms | 1.0x |
| 1024 | 5.7B | 2,918ms | 2.0x | 284ms | 1.7x |
| 512 | 3.6B | 2,202ms | 2.7x | 165ms | 2.9x |
| 256 | 2.6B | 1,144ms | 5.1x | 166ms | 2.9x |
| 128 | 2.1B | 963ms | 6.1x | 116ms | 4.1x |

- Script: `z8_pipeline_32b/bench_inference_speed.py`
- Generation plateaus at ~165ms/tok between rank 512 and 256 (overhead-bound)
- Rank 128 breaks through to 116ms/tok
- These are RAW SVD without fine-tuning — quality is garbage. Speed is real.

### Whitening A/B Test (Partial) — DONE
Compared raw SVD vs activation-whitened SVD on Qwen3-14B. Only completed rank 128 and 256 before being killed for other experiments.

| Rank | Raw PPL | Whitened PPL | Improvement |
|------|---------|-------------|-------------|
| 128 | 44.6M | 224K | 199x better |
| 256 | 4.9M | 262K | 18.6x better |

- Whitening: collect activation covariance (X^T X), Cholesky, whiten W before SVD, undo in factored form
- 32 calibration samples, 280 covariance matrices, took 645s
- Script: `z8_pipeline_32b/test_whitening.py`
- **Rank 512 and 1024 NOT tested yet** — these are the critical ones

### SVD Spectrum Analysis on Qwen3-14B — DONE
Analyzed all 280 projections to find natural rank at 99% energy retention.

- MLP projections need 92-96% of full rank for 99% energy (barely compressible by energy alone)
- K/V need 87-93% of max rank
- Q/O need 65-75%
- Median rank for 99% energy: 3691 out of 5120
- Even at 99% energy, PPL jumps to 2.69x teacher
- Conclusion: energy retention ≠ quality. Fine-tuning is essential.
- Results: `z8_pipeline_32b/rectangle_14b.log`

### 32B Thermostat Attempts — NOT RUNNING
Multiple attempts to run the thermostat compression on 32B. Issues encountered:
1. IPEX optimize corrupts weight format before SVD — removed IPEX
2. Rank 1024 starting point (20% of full rank) gives PPL 14M — too aggressive
3. Script is written and ready at `z8_pipeline_32b/thermostat_32b.py`
4. Needs either: (a) much higher starting rank, or (b) whitened SVD to make rank 1024 survivable
5. The whitening test (rank 512/1024 results) would determine the right approach

### Strix Findings (pulled from repo)
Strix (MacBook Halo) ran several experiments on 4B:
- KV-Medusa: 10.9 tokens/step, 5.17x speedup, 99%+ acceptance
- Whitened rectangle on 4B: best PPL 9.0 at rank 392 (BETTER than teacher 11.0 — LASER effect)
- Plain rectangle: best PPL 10.0 at rank 568 (also better than teacher)
- KV-256 compressed base achieves PPL 9.7 (better than teacher 11.0)
- Scripts: `scripts/pipeline_kv_medusa.py`, `scripts/pipeline_uniform_rect_4b_whitened.py`, etc.

## WHAT TO DO NEXT (in priority order)

### 1. Build Medusa Verification Loop (IMMEDIATE — hours not days)
We have 10 trained heads on 32B. Build the actual speculative decode loop:
- Draft 10 tokens using heads (predict K/V → run attention with predicted cache)
- ONE batched verification forward pass through real model
- Accept tokens where predicted K matches actual K
- Measure ACTUAL tokens/second vs standard autoregressive decode
- This gives the first real wall clock number

### 2. Finish Whitening Test (hours)
Complete rank 512 and 1024 comparison on 14B. If whitened rank 1024 gives <2-5x teacher PPL, that's our starting point for the 32B thermostat. Script exists: `z8_pipeline_32b/test_whitening.py`

### 3. Thermostat with Whitened SVD on 32B (days)
Once we know the right starting rank from the whitening test:
- Add whitened SVD to `z8_pipeline_32b/thermostat_32b.py` (replace `factorize_linear` with `factorize_whitened`)
- Start at a rank where quality is recoverable
- 448 independent thermostats, 1% drops, per-projection cutoffs
- Coherency tests at each checkpoint (8 prompts, top-1/top-5 token match vs teacher)
- OpenWebText training data

### 4. Medusa on Compressed Model (hours after thermostat)
Retrain 10 KV-Medusa heads on the compressed model. Should train even faster since compressed model's KV structure is simpler (lower rank = more predictable).

### 5. End-to-End Benchmark (minutes)
Full pipeline wall clock: compressed model + Medusa verification loop, tokens/second.

## HARDWARE

```
HP Z8 G4 Workstation
  CPU: 2× Intel Xeon Gold 5218 (Cascade Lake)
  RAM: 384GB DDR4-2666 (12× 32GB, all channels populated on both sockets)
  Storage: NVMe
  GPU: None
  OS: Ubuntu 24.04.4 LTS, kernel 6.17.0
```

User also has:
- 9 additional 32GB DDR4-2666 sticks (can go to 672GB)
- 6× 32GB DDR4-2400 sticks (DO NOT MIX — will drag all channels to 2400)
- 16× 16GB DDR4-2400 sticks (DO NOT MIX)
- HP Z840 ($600, dual Xeon E5-2600 v3/v4, no AVX-512) — considering selling for MacBook M5 money

### Required Environment Variables
```bash
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export TOKENIZERS_PARALLELISM=false
```

### Python Environment
```bash
/home/supercomputerz8/MedusaBitNet/.venv/bin/python
# Has: torch 2.8.0+cpu, IPEX, transformers, datasets
```

### Performance Notes
- DO NOT use ipex.optimize() before SVD — it repacks weights and SVD reads garbage
- CPU governor must be "performance" (check: cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
- torch.set_num_threads(32) — do NOT use 64 (hyperthreading kills performance)
- Clear RAM between big runs: sudo swapoff -a && sudo swapon -a && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
- 32B forward pass: ~1.5-3s (depends on seq_len and whether model is fresh in cache)

## MODELS CACHED ON DISK
```
Qwen/Qwen3-0.6B:    1.5GB
Qwen/Qwen3-1.7B:    4.1GB
Qwen/Qwen3-4B:      8.1GB
Qwen/Qwen3-8B:     16.4GB
Qwen/Qwen3-14B:    29.6GB
Qwen/Qwen3-32B:    65.5GB
Qwen/Qwen3-30B-A3B: 61.1GB
```

## KEY FILES FROM THIS SESSION
```
z8_pipeline_32b/kv_medusa_cpu.py          — KV-Medusa training script (CPU)
z8_pipeline_32b/kv_medusa_results/        — Trained head weights + results JSON
z8_pipeline_32b/bench_inference_speed.py  — Wall clock inference benchmark
z8_pipeline_32b/test_whitening.py         — Whitened vs raw SVD A/B test
z8_pipeline_32b/thermostat_32b.py         — Multi-axis thermostat (ready, not run successfully)
z8_pipeline_32b/rectangle_14b.py          — Rectangle packing prototype
results/kv_medusa_32b.json               — Published KV-Medusa results
```

## USER NOTES
- Name: Parrish Corcoran
- This is a portfolio piece for ML job applications
- Strict about publishing only measured results, not projections
- "I don't want you to get excited until we actually have wall clock"
- DM on Slack at milestones (user ID: U0ASKMS30UR)
- Checks in every couple hours to orchestrate
