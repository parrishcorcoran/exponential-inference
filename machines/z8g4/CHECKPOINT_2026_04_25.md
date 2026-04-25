---
name: Z8G4 Full Checkpoint — April 25, 2026
description: Complete state dump for session handoff. Performance breakthrough: 133x speedup found. All optimization findings, running experiments, and next steps.
---

# Z8G4 CHECKPOINT — April 25, 2026

## THE BREAKTHROUGH: 133x Speedup Found

The Z8 G4 was running at **0.75% of its actual capability.** A 32B model forward pass went from 210s to **1.57s** with pure software fixes:

| Fix | Speedup | How |
|-----|---------|-----|
| CPU governor: powersave → performance | ~2x | `echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Threads: 64 → 32 (disable HT) | 16x | `OMP_NUM_THREADS=32`, `torch.set_num_threads(32)` |
| IPEX optimize | 3.9x | `model = ipex.optimize(model, dtype=torch.float32, inplace=True)` |
| KMP/OMP settings | ~2x | See env vars below |
| **Combined** | **133x** | |

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

### System Settings (already applied, persist across reboot with sysctl)
```bash
# CPU governor (needs re-applying after reboot)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    sudo sh -c "echo performance > $cpu"
done

# Memory
sudo sysctl vm.swappiness=1
sudo sysctl vm.dirty_ratio=40
sudo sysctl vm.dirty_background_ratio=10
sudo sysctl vm.max_map_count=1048576

# THP
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
sudo sh -c "echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag"
```

## HARDWARE PROFILE

```
HP Z8 G4 Workstation
  CPU: 2× Intel Xeon Gold 5218 (Cascade Lake, 2019)
    - 16 cores / 32 threads per socket
    - 2.30 GHz base, 3.90 GHz turbo
    - AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL, AVX-512_VNNI
    - NO avx512_bf16 (that's Cooper Lake+)
    - NO AMX (that's Sapphire Rapids+)
  RAM: 376 GB DDR4-2666 (12 channels, 6 per socket)
  NUMA: 2 nodes, 192GB each, distance 10/21
  L1: 32KB/core, L2: 1MB/core, L3: 22MB/socket
  Storage: NVMe
  GPU: None (integrated display only)
  OS: Ubuntu 24.04.4 LTS, kernel 6.17.0
```

### Measured Performance (with optimizations)
- **Peak FLOPS**: 908 GFLOPS (fp32, 1024×1024 matmul)
- **Theoretical peak**: 2.3 TFLOPS (fp32 AVX-512)
- **Utilization**: 37% on single matmul, 22% on training loop
- **INT8 VNNI**: 1.49 TOPS (1.88x over fp32)
- **Memory bandwidth**: 58-72 GB/s measured (200 GB/s theoretical)
- **NUMA penalty**: 1.15x (minimal — not the bottleneck)
- **32B forward pass**: 1.57s
- **0.6B forward pass**: ~0.18s

### Bottleneck Analysis
- **NOT compute-bound** (37% utilization)
- **NOT memory-bound** (30% bandwidth utilization)
- **OVERHEAD-bound**: Python loop + autograd + tensor metadata = 80% of step time
- The MSE loss computation was 80% of training step time (56 separate tensor allocations)
- Solution: pre-allocated buffers, skip loss tracking, sub-batch gradients

## CACHED TRAINING SPEEDS

| Version | Description | ms/step | Steps/sec |
|---------|-------------|---------|-----------|
| Original | Full model fwd+bwd, 64 threads | 10,000 | 0.1 |
| Thread fix | Full model, 32 threads | 600 | 1.7 |
| Cached V1 | Cached I/O, autograd | 330 | 3 |
| Cached + compile | torch.compile | 444→374 | 2.7 |
| Manual grad | No autograd, pre-alloc buffers | 356 | 2.8 |
| V5 sub-batch | B=128, rotate 14/56 projections | **8.8** | **114** |
| V5 + INT8 | VNNI quantized forward | ~4.7 | ~213 |

## MANIFOLD FINGERPRINTS COMPLETED

15 models across 7 tokenizer families. All show TwoNN 8-12:

| Model | Family | TwoNN | Rotation | Carry |
|-------|--------|-------|----------|-------|
| TinyLlama-1.1B | Llama | 7.99 | 1.530 | 0.168 |
| Qwen3-0.6B | Qwen | 8.75 | 1.527 | 0.174 |
| Bloom-7B | BigScience | 8.75 | 1.542 | 0.224 |
| Qwen3-4B | Qwen | 8.07 | 1.492 | 0.194 |
| Phi-2 | Microsoft | 8.26 | 1.431 | 0.279 |
| Qwen3-1.7B | Qwen | 9.52 | 1.548 | 0.166 |
| Mistral-7B | Mistral | 9.13 | 1.499 | 0.212 |
| Yi-1.5-34B | Yi | 9.23 | 1.486 | 0.255 |
| Qwen3-14B | Qwen | 9.25 | 1.527 | 0.191 |
| Qwen3-8B | Qwen | 9.84 | 1.537 | 0.185 |
| Qwen3-32B | Qwen | 10.27 | 1.495 | 0.218 |
| GPT-NeoX-20B | EleutherAI | 10.78 | 1.445 | 0.253 |
| Qwen3-30B-A3B | Qwen MoE | 11.36 | 1.546 | 0.209 |
| Qwen2.5-72B | Qwen | 11.58 | 1.453 | 0.271 |
| Mixtral-8x7B | Mistral MoE | 11.65 | 1.496 | 0.207 |

## THERMOSTAT RANK ANNEALING RESULTS

### 0.6B Non-cached (5250 steps, ~14 hours)
- Start: rank 64, PPL ratio 44M×
- End: rank 64, PPL ratio **1.555×** (nearly hit 1.5× threshold)
- Never triggered a rank drop — reached near-teacher quality at rank 64

### 0.6B Cached (10,000 steps, ~55 minutes)
- Rank 64 → 41 (23 rank drops, all by patience exhaustion)
- Loss: 4.78 (rank 64) → 5.44 (rank 41)
- Smooth loss increase — no cliff found
- Each rank drop recovers quickly then plateaus

### 32B Non-cached thermostat (200 steps before killed)
- Teacher PPL: 10.93
- Step 200: PPL 83.8, ratio 7.7× (recovering fast)
- Each step was ~10 min (now would be ~5s with optimizations)

## HOLOGRAPHIC TRANSFORMER RESULTS

Custom architecture: parallel multi-view attention (N views replace L layers).

| Model | Params | Data | Best PPL |
|-------|--------|------|---------|
| Holographic V1 | 10.6M | 500k | 709 |
| Standard V1 | 18.9M | 500k | 823 |
| Holographic V2 | 10.6M | 2M | 458 |
| Standard V2 | 18.9M | 2M | 497 |
| Holographic V3 | 10.6M | 20M | 283 |
| Standard V3 | 18.9M | 20M | 326 |

Holographic wins at every data size with 44% fewer params. But can't generate coherent text at this scale — PPL comparison only.

## MANIFOLD-AWARE TRAINING FINDINGS

### Verified Winners
1. **Manifold init**: 2.6× faster early convergence (PCA basis from calibration)
2. **FFN LoRA targets**: 8.6% better eval than standard qv targets
3. **Rank 4-10 optimal**: rank 32 hurts (too many params for small data)

### Verified Failures
1. Layer freezing by PR: overfits (11% worse eval)
2. Gradient projection: no benefit (gradients already on manifold)
3. Per-layer LR scaling: hurts convergence
4. Rotation propagation: overfits
5. Holographic/phase init: same as flat PCA (curvature needs nonlinear)

### Holographic Reconstruction Test
Layer 5 → Layer 29 at rank 10: cosine similarity **0.7522**
(81.6% of tokens > 0.7, 99.8% > 0.5)
Strongly supports holographic framing.

## 32B PIPELINE STATUS

`z8_pipeline_32b/pipeline.py` — fixed `dtype` → `torch_dtype` bug.

Phase 1 completed on one run:
- Baseline PPL: 7.71
- Throat: PR=1.01 at L7, pump=1435×
- Wormhole topology confirmed at 32B

Phase 2+ needs relaunch with all optimizations.

### Pipeline Timing Estimates (with 133x speedup)
- Phase 1 (shape): ~2 seconds
- Phase 2 (capture): ~2 minutes
- Phase 3a (weight anneal, 8 stages): ~1 hour
- Phase 3b (KV anneal, 15 stages): ~2 hours
- Full pipeline: ~3-4 hours
- 50K fine-tuning steps: ~3 days

## LIBXSMM

Built from source at `/home/supercomputerz8/libxsmm/`
- Full AVX-512 support compiled
- ctypes overhead negates gains (need C extension for real benefit)
- Sweet spot: matrices ≤ 64×64

## KEY FILES

```
machines/z8g4/scripts/
  numa_unified.py              — NUMA benchmark + model sharding
  thermostat_rank_anneal.py    — Cached thermostat rank annealing
  measure_manifold_fingerprint.py — Cross-model fingerprinting
  holographic_model.py         — Holographic transformer architecture
  train_holographic.py         — Training + baseline comparison

machines/z8g4/results/
  fingerprint_*.json           — 15 model fingerprints
  thermostat_*.json            — Annealing results
  holographic_*.json           — Architecture comparison

z8_pipeline_32b/
  pipeline.py                  — 32B compression pipeline (fixed)
  README.md                    — Pipeline spec

analysis/
  manifold_routing/            — Inference optimization findings
  manifold_training/           — Training optimization findings
```

## MODELS CACHED ON DISK

```
Qwen/Qwen3-0.6B:    1.5GB
Qwen/Qwen3-1.7B:    4.1GB
Qwen/Qwen3-4B:      8.1GB
Qwen/Qwen3-8B:     16.4GB
Qwen/Qwen3-14B:    29.6GB
Qwen/Qwen3-32B:    65.5GB
Qwen/Qwen3-30B-A3B: 61.1GB
Qwen/Qwen2.5-72B:  (cleared to save disk)
microsoft/phi-2:     5.6GB
microsoft/bitnet:    1.2GB + 4.8GB (bf16)
Mixtral-8x7B:       (cleared)
Yi-1.5-34B:         (cleared)
```

Disk: 935GB total, ~300GB free.

## WHAT TO DO NEXT

1. **Relaunch 32B pipeline** with all optimizations (estimated 3-4 hours)
2. **INT8 VNNI integration** into pipeline for additional 1.88x on forward
3. **Full 50K fine-tuning** on 32B (3 days with optimizations)
4. **Scale to 72B** (fits in 376GB, forward pass ~3s with optimizations)
5. **Benchmark against Strix Halo** results for validation

## CRITICAL: AFTER NVME INSTALL

After new NVMe cards are installed:
1. Re-apply CPU governor: `for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do sudo sh -c "echo performance > $cpu"; done`
2. Re-apply sysctl settings (see above)
3. Set env vars in `.bashrc` or session
4. Verify: `python -c "import torch; torch.set_num_threads(32); print(torch.get_num_threads())"` should print 32
5. LIBXSMM is at `/home/supercomputerz8/libxsmm/` — already compiled
