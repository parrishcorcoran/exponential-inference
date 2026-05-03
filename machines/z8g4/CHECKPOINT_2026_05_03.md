---
name: Z8G4 Checkpoint — May 3, 2026
description: HRR attention conversion experiments. Soft blend works at 2% but walls. Fade-out approach worse. CUDA ready for V100s. Key findings documented.
---

# Z8G4 CHECKPOINT — May 3, 2026

## HRR ATTENTION CONVERSION — EXPERIMENTAL RESULTS

### What We Proved
1. **Attention IS holographic correlation** — FFT circular correlation produces cos=0.993 identical scores to Q@K^T matmul
2. **Superposed HRR is faster at long sequences** — 6x at 1024 tokens, 4x at 2048 (32B dimensions)
3. **Output normalization is critical** — without normalizing HRR output to match standard output scale, even 1.8% blend produces PPL 82M. With normalization: PPL 39.93 (2.5% above teacher)
4. **Soft blend wall at ~2.2% HRR** — quality degrades past 2% HRR blend on 0.6B with 200 ft steps
5. **Fade-out approach worse than soft blend** — diverges at just 0.1% HRR
6. **Overfitting is a problem** — 200 steps on repeated data causes quality degradation over rounds

### Core Operation Benchmark (32B dimensions: 40Q/8KV heads, 128d)
| SeqLen | Standard | HRR Superposed | Speedup |
|--------|----------|----------------|---------|
| 256 | 2.5ms | 2.4ms | 1.04x |
| 512 | 26.8ms | 9.6ms | 2.78x |
| 1024 | 97.1ms | 16.1ms | 6.03x |
| 2048 | 363.5ms | 87.9ms | 4.14x |

### Why the Wall Exists
1. **HRR superposition is lossy** — 256 items in 128-dim vector, SNR ~6%
2. **Softmax does sharp selective retrieval** — HRR can't replicate putting 99% weight on one token
3. **Error correction can't learn the mapping** in 200 steps on limited data
4. **The cumsum trick is causal but accumulates noise** — position 256 has 256 items superposed

### What Might Break Through
1. **Much more training data** — streaming OWT instead of repeated batches
2. **V100 GPUs** — 10x faster iteration, can do thousands of rounds
3. **Per-layer conversion** — convert one layer at a time, not all simultaneously
4. **LOLCATS-style MSE matching** — train HRR to match standard output per-layer first, then swap
5. **Sliding window HRR** — only superpose last 16 tokens instead of all 256
6. **Manifold-aware HRR** — project K to ~10D manifold before superposition
7. **Hybrid: 80% HRR + 20% softmax** — route hard tokens to standard, easy tokens to HRR

### Related Work Discovered
- **Hrrformer (ICML 2023, EleutherAI)** — HRR attention from scratch, 280x faster training, near SOTA
- **LOLCATS (ICLR 2025, Hazy Research)** — softmax→linear conversion in 40M tokens via MSE matching
- **HALO** — Qwen3→RNN-hybrid conversion in 2.3B tokens
- **Residual Linear Attention** — error correction for linear attention
- **GLA, DeltaNet, RWKV-7** — various linear attention + correction architectures
- None of these ship in production — the quality gap remains

### Connection to Tony Plate / Vaswani
- K in KV cache = unbinding key (Tony Plate HRR 1995)
- Q@K^T = circular correlation (holographic retrieval)
- Softmax = cleanup memory (noise removal)
- V = bound content
- Attention IS holographic, just implemented with explicit storage instead of superposition
- The O(n²) comes from storing N separate K,V pairs instead of superposing them

## OTHER EXPERIMENTS THIS SESSION

### PID Axis Testing on 4B — COMPLETE
| Axis | Free Zone | Best PPL | Status |
|------|-----------|----------|--------|
| K rank | 75% (4x free) | 20.78 | LEVER |
| V rank | 55% (~2x free) | 19.83 | LEVER |
| MLP rank | ~0% | 26.45 | NOT lever |
| Norm squash | ~0% | 26.45 | NOT lever |
| Embed rank | 0% (catastrophic) | 26.45 | NOT lever |

### nGPT Geometry Across Scale — COMPLETE
| Model | Mean Norm | CV | Spherical? |
|-------|-----------|-----|-----------|
| 0.6B | 0.969 | 0.323 | Most nGPT |
| 32B | 1.784 | 0.356 | Least |
No convergence toward nGPT with scale.

### 14B Magnitude Anneal — KILLED (was at round 53)
- Magnitude 0.602, mean norm 1.097, PPL 18.06
- Free zone: 18% (rounds 1-11)
- Was heading to 0.2 for Phase 2 but killed by mistake
- Need to restart (with checkpointing this time)

### Inference Speed Benchmark (14B) — COMPLETE
| Rank | Gen/tok | Speedup |
|------|---------|---------|
| Full | 478ms | 1.0x |
| 512 | 165ms | 2.9x |
| 128 | 116ms | 4.1x |

## HARDWARE

### CUDA Installed — Ready for V100s
- Driver: 535.288.01
- CUDA: 12.0
- PyTorch: 2.5.1+cu121
- V100s not physically installed yet (printing shrouds)

### System
```
HP Z8 G4, 2x Xeon Gold 5218, 384GB DDR4-2666
Python: /home/supercomputerz8/MedusaBitNet/.venv/bin/python
Cached corpus: data/owt_tokens_50M.pt (53.2M tokens)
```

## KEY FILES
```
z8_pipeline_32b/hrr_soft_blend.py         — Soft blend approach (best so far)
z8_pipeline_32b/hrr_fade_standard.py      — Fade-out approach (worse)
z8_pipeline_32b/hrr_routed.py             — Binary router approach (too aggressive)
z8_pipeline_32b/hrr_superposed_32b.py     — Core operation benchmark
z8_pipeline_32b/hrr_attention_poc.py      — FFT vs matmul equivalence test
z8_pipeline_32b/pid_compress.py           — PID compression framework
z8_pipeline_32b/pid_magnitude_ngpt.py     — Magnitude anneal to nGPT
z8_pipeline_32b/pid_tau_ngpt.py           — Tau interpolation to nGPT (not run)
z8_pipeline_32b/pid_results/              — All results JSONs
docs/CHECKLIST_32B.md                     — Master compression checklist
docs/ORTHOGONAL_AXES.md                   — Confirmed axes
data/owt_tokens_50M.pt                    — Cached corpus
```

## WHAT TO DO NEXT

1. **Install V100s** — 10x faster iteration makes HRR experiments viable
2. **Try per-layer HRR conversion** — one layer at a time, not all simultaneously
3. **Try LOLCATS-style MSE matching first** — train HRR to match standard output, THEN swap
4. **Restart 14B magnitude anneal** with checkpointing
5. **Continue PID axis testing** — Q rank, O rank, magnitude on 4B
6. **Build KV-Medusa verification loop** — actual wall clock decode speedup

## LESSONS LEARNED
- HRR output scale is wildly different from standard attention — MUST normalize
- Binary routing creates cliffs — use soft blending
- PID gains need to be tuned per experiment — too conservative wastes days
- Router with constant init (all same score) creates dead zones — need variance
- 200 ft steps on repeated data causes overfitting — need streaming or larger corpus
- NEVER kill a long-running process without checkpointing first
