---
name: Z8G4 Checkpoint — May 3, 2026 (evening)
description: CRITICAL BUG FOUND in HRR correlate. MAP binding with learned position keys next. Resonator path viable.
---

# Z8G4 CHECKPOINT — May 3, 2026 (evening)

## CRITICAL DISCOVERY: correlate() was BACKWARDS

The circular correlation function had conj() on the WRONG argument the entire time.

WRONG: `IFFT(FFT(a) * FFT(b).conj())`
RIGHT: `IFFT(FFT(a).conj() * FFT(b))`

Impact: EVERY HRR experiment before this was broken. The 2% wall, diverging resonator, failed teacher-student — all had this bug.

After fix: single-item recovery went from cos=-0.05 to cos=0.78 (random) and cos=1.00 (spectrum-normalized).

## CURRENT STATE: MAP Binding with Learned Position Keys

We're building a resonator attention replacement using MAP (element-wise multiply) instead of HRR (circular convolution):

**What works:**
- Single-item recovery: cos=1.0 with spectrum normalization
- MAP binding with ±1 keys: zero expected cross-talk
- Q·K score recovery from superposition: cos=0.20 without training

**What's next (about to build):**
- Train the position keys to decorrelate K vectors
- Teacher-student: MSE(MAP_scores, real Q·K scores)
- Learnable position keys, `seq × head_dim` params per layer
- No FFT needed — just element-wise multiply

**The full pipeline (if this works):**
1. Learned position keys decorrelate K vectors
2. MAP superposition stores all K in one vector: M_K = sum(pos_j * K_j)
3. Probe: decode score_j = Q · (pos_j * M_K) ≈ Q · K_j
4. This gives attention scores in O(n*d) not O(n²*d)
5. Softmax + retrieve V normally

## KEY INSIGHT FROM THIS SESSION

The user's chain of reasoning:
1. Attention IS holographic retrieval (proven: cos=0.993)
2. K is the binding key, used for both bind and unbind (autoregressive)
3. Most K vectors are orthogonal to query — noise should cancel
4. Problem: model's K vectors are correlated (dot product 0.64), not orthogonal
5. Solution: whiten to decorrelate, or learn position keys that decorrelate
6. The resonating tokens = argmax, the distribution around it = softmax
7. Train the model to learn its own codebook

## GROUND RULES (established this session)
1. Lossless or improvement only — even 5% quality loss is unacceptable
2. Real math, real physics — no approximations dressed up as solutions
3. Say "I don't know" when I don't know

## FILES
```
z8_pipeline_32b/resonator_train.py           — resonator with learned codebook (v1)
z8_pipeline_32b/resonator_teacher_student.py  — teacher-student with HRR (had bug)
z8_pipeline_32b/manifold_resonator_long.py    — low-rank projection (not novel)
z8_pipeline_32b/manifold_attention.py         — PCA-based attention routing
z8_pipeline_32b/weighted_layer_attention.py   — per-layer attention budget
z8_pipeline_32b/hrr_soft_blend.py             — soft blend (2% wall — had bug)
z8_pipeline_32b/ewald_attention.py            — sliding window (not real Ewald)
z8_pipeline_32b/pid_compress.py               — PID axis testing framework
z8_pipeline_32b/pid_results/                  — all results
```

## WHAT NOT TO DO
- Don't implement known techniques and call them novel
- Don't kill long-running processes without checkpointing
- Don't use `A * B.conj()` for circular correlation — use `A.conj() * B`
