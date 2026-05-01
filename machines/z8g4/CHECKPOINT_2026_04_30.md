---
name: Z8G4 Checkpoint — April 30, 2026
description: Session handoff. 14B magnitude anneal to nGPT running (round 32 of ~230). V100s incoming. CUDA ready. PID framework built and tested.
---

# Z8G4 CHECKPOINT — April 30, 2026

## CURRENTLY RUNNING

### 14B Two-Phase Magnitude Anneal to nGPT — PID 211870
- **Phase 1 (ACTIVE):** Freeze weights, shrink magnitude, train only norms
- **Current state (round 32):** magnitude 0.654, PPL 18.07, CV 0.319, mean norm 1.191
- **Target:** magnitude 0.2, then save checkpoint, then switch to Phase 2
- **Phase 2:** Unfreeze all weights, continue shrinking to unit norm
- **ETA:** ~19 hours to reach 0.2 magnitude at current PID rate (~0.6%/round)
- **Log:** `z8_pipeline_32b/pid_magnitude_ngpt_14b.log`
- **Script:** `z8_pipeline_32b/pid_magnitude_ngpt.py`
- **Checkpoint saves:** Model saved to `z8_pipeline_32b/pid_results/phase1_14b_model/` when magnitude hits 0.2 or PID hits floor

**Key observations from this run:**
- Free zone: rounds 1-11 (magnitude 1.0→0.82), PPL IMPROVED from 17.07 to 16.37 (LASER effect)
- Best PPL: 16.37 at round 5 (magnitude 0.922)
- Free zone ended at round 12 (magnitude 0.801)
- PID rides the 5% quality line perfectly — PPL locked at 18.06-18.08 since round 18
- CV stays exactly 0.319 throughout — distribution shape unchanged, only scale moves
- Norms absorb magnitude reduction remarkably well
- Previous run killed at round 53 (magnitude 0.602, mean norm 1.097) — DON'T DO THAT AGAIN
- This run reproduces previous run exactly (same PPL at same rounds)

**When this run finishes Phase 1 (hits 0.2):**
1. Model checkpoint saved automatically to disk
2. Phase 2 starts automatically — unfreezes all weights
3. Phase 2 uses 300 fine-tune steps/round, lr=2e-5
4. Phase 2 PID has lower gains (kp=0.3, ki=0.01, kd=0.1) for stability
5. Goal: push mean norm to 1.0 and CV below 0.15 = nGPT conversion

**When Phase 2 completes (nGPT achieved or wall hit):**
1. Commit and push results
2. Save model as `phase1_14b_model` (or `ngpt_14b` if conversion succeeds)
3. Update CHECKLIST_32B.md with magnitude axis results

## WHAT WE LEARNED THIS SESSION

### PID Compression Framework — BUILT AND TESTED
Script: `z8_pipeline_32b/pid_compress.py`
- PID controller maintains quality at exactly target% above teacher
- Tested 7 axes on Qwen3-4B with 5% target:

| Axis | Free Zone | Best PPL | Status |
|------|-----------|----------|--------|
| K rank | **75%** (4x free) | 20.78 (21% better) | LEVER |
| V rank | **55%** (~2x free) | 19.83 (25% better) | LEVER |
| MLP rank | ~0% (breaks at 5%) | 26.45 | NOT a lever |
| Norm squash | ~0% (breaks at 5%) | 26.45 | NOT a lever |
| Embed rank | 0% (catastrophic) | 26.45 | NOT a lever |
| Magnitude | 18% free, ongoing | 16.37 (14B) | LEVER (running) |
| Q rank | not tested yet | — | TODO |
| O rank | not tested yet | — | TODO |

### nGPT Geometry Measurement — COMPLETE
Script: `z8_pipeline_32b/measure_ngpt_all.py`
Results: `z8_pipeline_32b/pid_results/ngpt_geometry_scale.json`

| Model | Mean Norm | CV | Dist→1.0 |
|-------|-----------|-----|----------|
| 0.6B | 0.969 | 0.323 | 0.031 |
| 1.7B | 1.663 | 0.288 | 0.663 |
| 4B | 1.286 | 0.308 | 0.286 |
| 8B | 1.796 | 0.298 | 0.796 |
| 14B | 1.821 | 0.319 | 0.821 |
| 32B | 1.784 | 0.356 | 0.784 |

**Key finding:** NO convergence toward nGPT with scale. Bigger models get LESS spherical. 0.6B is closest to unit norm. Pretraining does NOT naturally produce hypersphere geometry.

### KV-Medusa on 32B — COMPLETE (from previous session)
Results: `results/kv_medusa_32b.json`
- 10 heads, cos_k 0.887-0.900, 100% acceptance
- Zero degradation from offset 1 to 10
- Need verification loop for actual wall clock (NOT BUILT YET)

### Inference Speed Benchmark on 14B — COMPLETE (from previous session)
| Rank | Forward | Gen/tok | Speedup |
|------|---------|---------|---------|
| Full | 5,871ms | 478ms | 1.0x |
| 1024 | 2,918ms | 284ms | 1.7x |
| 512 | 2,202ms | 165ms | 2.9x |
| 256 | 1,144ms | 166ms | 2.9x |
| 128 | 963ms | 116ms | 4.1x |

### Whitening A/B Test — PARTIAL (from previous session)
| Rank | Raw PPL | Whitened PPL | Improvement |
|------|---------|-------------|-------------|
| 128 | 44.6M | 224K | 199x |
| 256 | 4.9M | 262K | 18.6x |
| 512 | NOT TESTED | — | — |
| 1024 | NOT TESTED | — | — |

## HARDWARE CHANGES

### V100 GPUs — INCOMING
- 2× NVIDIA V100 16GB — user has them, printing shrouds for installation
- PCIe cards, will be GPU 1 and GPU 2 (Quadro K2200 stays as GPU 0 for display)
- V100 = compute 7.0, Tensor Cores, 15 TFLOPS fp32, 125 TFLOPS fp16

### CUDA — INSTALLED AND READY
- Driver: 535.288.01 (already had it for Quadro K2200)
- CUDA toolkit: 12.0
- PyTorch: 2.5.1+cu121 (replaced torch+cpu in venv)
- Verified working: `torch.cuda.is_available() = True`
- When V100s are physically installed, they'll appear automatically
- No additional software setup needed

### RTX PRO 6000 Blackwell 96GB
- User considering purchase ($2,099 on eBay)
- 96GB VRAM = 32B fits on one card
- Zero-feedback seller — cautioned about risk
- Cost equivalent: ~840 hours of H100 rental

### Hardware Still Available
- 9× 32GB DDR4-2666 sticks (could go to 672GB total)
- 6× 32GB DDR4-2400 sticks (DO NOT MIX — drags all channels to 2400)
- 16× 16GB DDR4-2400 sticks (DO NOT MIX)
- HP Z840 — user considering selling for MacBook M5 money

## WHAT TO DO NEXT (priority order)

### 1. Wait for 14B magnitude anneal to finish Phase 1 + Phase 2
- Phase 1 is running, ~19 hours to reach 0.2 magnitude
- Phase 2 will start automatically after checkpoint save
- DO NOT KILL THIS PROCESS (PID 211870)

### 2. When V100s are installed
- Run `nvidia-smi` to verify detection
- Test: `python -c "import torch; print(torch.cuda.device_count())"`
- Move 4B experiments to GPU (one per card, 2x parallel throughput)
- 14B experiments: pipeline parallel across both cards

### 3. Build KV-Medusa verification loop (for actual wall clock)
- Have 10 trained heads from previous session
- Need the speculative decode loop: draft → verify → accept
- This gives the first real decode speedup number

### 4. Finish whitening test (rank 512/1024)
- Critical for determining 32B thermostat starting point

### 5. Continue PID axis testing
- Q rank, O rank still untested
- MLP width, Q heads on larger models
- Stack confirmed axes for additivity test

### 6. Run magnitude anneal on 4B and 32B
- 4B: quick validation (~2 hours on CPU, minutes on V100)
- 32B: CPU only (too big for V100 16GB)

## CRITICAL NOTES FOR NEXT SESSION

1. **DO NOT KILL THE RUNNING PROCESS (PID 211870)** — it's the 14B magnitude anneal
2. Check process: `ps -p 211870 -o pid,rss,pcpu,etime`
3. Check progress: `tail -5 z8_pipeline_32b/pid_magnitude_ngpt_14b.log`
4. The cached corpus is at `data/owt_tokens_50M.pt` (53.2M tokens) — use for all future experiments
5. PyTorch is now CUDA-enabled (2.5.1+cu121) — CPU code still works normally
6. The PID framework (`z8_pipeline_32b/pid_compress.py`) is the main tool for testing axes

## KEY FILES

```
z8_pipeline_32b/pid_compress.py              — PID compression framework (all axes)
z8_pipeline_32b/pid_magnitude_ngpt.py        — Two-phase magnitude anneal to nGPT
z8_pipeline_32b/pid_magnitude_phase2.py      — Phase 2 standalone (DON'T USE — wrong approach)
z8_pipeline_32b/measure_ngpt_all.py          — nGPT geometry across model scales
z8_pipeline_32b/pid_results/                 — All PID results (JSON)
z8_pipeline_32b/kv_medusa_cpu.py             — KV-Medusa training
z8_pipeline_32b/bench_inference_speed.py     — Wall clock inference benchmark
z8_pipeline_32b/test_whitening.py            — Whitened vs raw SVD
data/owt_tokens_50M.pt                       — Cached OpenWebText corpus (53.2M tokens)
docs/CHECKLIST_32B.md                        — Master compression checklist
docs/ORTHOGONAL_AXES.md                      — Confirmed orthogonal axes (from Strix)
results/kv_medusa_32b.json                   — KV-Medusa results on 32B
```

## ENVIRONMENT

```bash
# Python
/home/supercomputerz8/MedusaBitNet/.venv/bin/python
# PyTorch 2.5.1+cu121, CUDA 12.1, transformers, datasets

# CPU optimization
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export TOKENIZERS_PARALLELISM=false

# Clear RAM between big runs
sudo swapoff -a && sudo swapon -a && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

## USER NOTES
- Name: Parrish Corcoran
- Portfolio piece for ML job applications
- Strict about publishing only measured results, not projections
- Master plan: convert models to nGPT form, build library of converted models
- PID-controlled compression is the methodology
- Slack DM at milestones (user ID: U0ASKMS30UR)
- Lesson learned: NEVER kill a long-running process without saving checkpoint first
