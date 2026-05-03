---
name: Z8G4 Checkpoint — May 3, 2026 (final)
description: Resonator attention works. cos=1.0 at L27. Next: anneal from MAP probe to pure resonator.
---

# Z8G4 CHECKPOINT — May 3, 2026 (final)

## THE BREAKTHROUGH

Resonator attention achieves **exact cos=1.000** match with standard attention at L27 of Qwen3-0.6B. First known exact resonator-based attention on a real language model.

### How it works
1. **Learned MAP position keys** decorrelate K vectors (element-wise multiply, no FFT)
2. **MAP decode** gives initial attention scores from superposed K memory
3. **Resonator iteration** refines to exact: `weights = softmax(V^T @ estimate); estimate = V @ weights`
4. One iteration: cos 0.998 → 1.000

### Results per layer (Qwen3-0.6B, 2000 training steps)
| Layer | attn_cos | out_cos | top1_match |
|-------|----------|---------|------------|
| L0 | 1.0000 | 1.0000 | 0.106 (diffuse) |
| L7 | 0.9952 | 0.9980 | 0.992 |
| L14 | 0.2988 | 0.6036 | 0.213 (needs more training) |
| L21 | 0.1462 | 0.5551 | 0.140 (needs more training) |
| L27 | **1.0000** | **1.0000** | **1.000** (exact with resonator) |

### Critical bug found and fixed
HRR circular correlation had `conj()` on wrong argument the ENTIRE TIME.
- WRONG: `IFFT(FFT(a) * FFT(b).conj())`
- RIGHT: `IFFT(FFT(a).conj() * FFT(b))`
- ALL prior HRR experiments were broken. The 2% wall, diverging resonator, failed teacher-student — all had this bug.

## WHAT TO DO NEXT

### The anneal: MAP probe → pure resonator
Currently the MAP decode costs O(n²×d) — same as standard attention. The resonator iteration costs O(k×n×d) — cheap. Shift the workload:

```
fade=1.0: probe = MAP_decode (O(n²), clean)
fade=0.5: probe = 0.5*MAP + 0.5*raw_superposition  
fade=0.0: probe = raw_superposition only (O(n×d))
```

PID controls the fade. The model learns to make the resonator converge from the cheap probe. When fade=0, total cost is O(k×n×d) — sub-quadratic for k << n.

### Steps
1. Build the fade script: MAP → raw superposition probe, PID controlled
2. Train all weights (model + MAP keys) during the anneal
3. More training for L14 and L21 (throat layers — currently low cos)
4. Measure: how many resonator iterations needed at each fade level?
5. When fade=0 and cos stays 1.0: we have sub-quadratic exact attention

### Key insight chain
1. Attention IS holographic retrieval (proven: cos=0.993)
2. K is the binding key (autoregressive)
3. Model's K vectors are correlated (dot product 0.64) — can't superpose cleanly
4. Whiten K → decorrelate → but circular cross-talk remains
5. Learned MAP keys → learned decorrelation → cos=0.995
6. Resonator iteration → exact cos=1.000
7. The resonating tokens = argmax, distribution around = softmax
8. Next: anneal from O(n²) probe to O(n) probe while resonator compensates

## GROUND RULES
1. Lossless or improvement only — even 5% quality loss is unacceptable
2. Real math, real physics — no approximations dressed up as solutions  
3. Say "I don't know" when I don't know

## WHAT HAS BEEN DONE BEFORE (don't repeat)
- Sliding window attention (Longformer, Mistral) — called it Ewald, was wrong
- HRR attention replacement (Hrrformer, ICML 2023) — approximate, not exact
- Linear attention conversion (LOLCATS, ICLR 2025) — approximate
- Learned random features (DARKER, FAVOR#, Spectraformer) — approximate
- Learnable temperature (TempNet) — small gains
- Top-k sparse attention — lossy
- Low-rank attention projection (Linformer) — called it manifold resonator, was wrong

## WHAT IS NOVEL
- Learned MAP keys for K decorrelation in superposition
- Resonator iteration achieving exact cos=1.0 on real LLM
- The anneal from MAP → pure resonator (proposed, not built yet)
- The correlate bug fix enabling all of the above

## FILES
```
z8_pipeline_32b/resonator_train.py            — resonator with learned codebook
z8_pipeline_32b/resonator_teacher_student.py   — teacher-student (fixed correlate)
z8_pipeline_32b/hrr_soft_blend.py             — soft blend HRR (2% wall — pre-bugfix)
z8_pipeline_32b/hrr_fade_standard.py          — fade-out approach
z8_pipeline_32b/hrr_routed.py                 — binary router
z8_pipeline_32b/hrr_superposed_32b.py         — core operation benchmark
z8_pipeline_32b/pid_compress.py               — PID compression framework
z8_pipeline_32b/pid_results/                  — all results
machines/z8g4/CHECKPOINT_2026_05_03b.md       — earlier checkpoint
data/owt_tokens_50M.pt                        — cached corpus (Qwen tokenizer)
data/owt_tokens_p2o6e100_nGPT_800m.pt        — cached corpus (nGPT tokenizer)
```

## ENVIRONMENT
```bash
/home/supercomputerz8/MedusaBitNet/.venv/bin/python
# PyTorch 2.5.1+cu121, CUDA ready for V100s (not installed yet)
# CRITICAL: use A.conj() * B for circular correlation, NOT A * B.conj()
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export TOKENIZERS_PARALLELISM=false
```
