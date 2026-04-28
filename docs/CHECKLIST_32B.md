# 32B Compression Checklist

All confirmed orthogonal axes applied to Qwen3-32B.
Check off as each is measured with wall clock on 32B.

## Weight Axes (model size reduction)

- [ ] **1. K projection rank (SVD)** — Free zone: 4x (ppl improves). Wall: rank 128. Measured on 14B. Need 32B measurement + fine-tune.
- [ ] **2. V projection rank (SVD)** — Free zone: 1.6x. Wall: rank 642. Measured on 14B. Need 32B.
- [ ] **3. Weight bits (per-channel quant)** — Free zone: Q6. Wall: Q3. Measured on 14B. Need 32B INT8 VNNI.
- [ ] **4. Embed bits (quantization)** — Free zone: Q6. Wall: Q3. Measured on 14B. Need 32B.
- [ ] **5. MLP width (row pruning)** — Free zone: 10%. Wall: 25%. Measured on 14B. Need 32B.
- [ ] **6. Q head pruning** — Free zone: 10/40 heads. Wall: 15/40. Measured on 14B. Need 32B (40 Q heads, 8 KV heads).
- [ ] **7. Low-rank Q projection (SVD)** — Free zone: rank 256. Measured on 14B. Need 32B.
- [ ] **8. Magnitude (scale weights down)** — Free zone: 13%. Wall: 20%. Measured on 4B. Need 32B.

## Cache Axes (inference memory reduction)

- [ ] **9. K cache Q4** — +0.4 ppl with fine-tune. Wall: Q2. Measured on 14B. Need 32B.
- [ ] **10. V cache Q4** — +2.3 ppl with fine-tune. Wall: Q2. Measured on 14B. Need 32B.

## Decode Axes (tokens per step)

- [x] **11. KV-Medusa (predict future KV)** — cos_k 0.887-0.900 at offsets 1-10. 100% acceptance at 0.7. Measured on 32B. **NEEDS: verification loop for wall clock.**
- [ ] **12. Regular Medusa (predict next token)** — 37% at t+1 on 4B. Need 32B.

## Methodology

- [ ] **13. Whitened SVD** — 199x better than raw at rank 128 on 14B. Need rank 512/1024 results. Need 32B integration.
- [ ] **14. Thermostat anneal (all axes)** — Script ready. Need working starting point (whitened SVD). Per-layer independent cutoffs.
- [ ] **15. Rectangle packing** — Prototype on 14B showed wormhole shape. Need post-thermostat repack.

## Potential Axes (not yet tested on any model)

- [ ] **16. Cross-layer weight tying** — Tie weights of adjacent cheap layers + fine-tune
- [ ] **17. Layer dropping** — Some layers improve when skipped (L13, L15 on 14B). Drop + fine-tune.
- [ ] **18. Activation quantization** — Quantize hidden states between layers
- [ ] **19. Attention head sharing across layers** — Share KV heads between adjacent cheap layers
- [ ] **20. MLP rank (SVD on gate/up/down)** — MLP weights may be low-rank. SVD + fine-tune.
- [ ] **21. Embedding pruning (vocabulary trim)** — Remove unused vocab rows
- [ ] **22. RoPE frequency compression** — Reduce position encoding dimensions
- [ ] **23. Norm fusion** — Merge adjacent RMSNorms
- [ ] **24. KV head merging (reduce GQA groups)** — Merge similar KV heads + fine-tune
- [ ] **25. Dynamic precision (per-token bit width)** — Certainty-driven quantization
- [ ] **26. Looped layers (repeat one block)** — Replace cheap layer sequence with loop
- [ ] **27. Distillation** — Train smaller model from compressed teacher
- [ ] **28. Unit-norm anneal (nGPT conversion)** — Equalize per-row magnitude. Early signal positive on 0.6B.

## Confirmed NOT levers

- ~~Head angle rotation~~ — Gauge symmetry, rotation-invariant
- ~~Q/O rank reduction~~ — Freezes at full rank, Q and O need full rank
- ~~Shallow draft (first N layers)~~ — 0% accuracy at 10 layers, wormhole needs all layers
- ~~Cross-model KV routing~~ — cos 0.007, different learned representations

## Combined Compression Target

Free zones only (all confirmed axes stacked):
```
K rank 4x * V rank 1.6x * Weight Q6 2.67x * Embed Q6 2.67x *
MLP 90% 1.1x * K Q4 4x * V Q4 4x * Q heads 1.3x *
Q rank 1.2x * Magnitude 1.15x = ~350x theoretical

Realistic (coupling): 30-50x compression at near-baseline quality
+ KV-Medusa: 10+ tokens/step decode acceleration
```

## Measured Wall Clock (14B, CPU fp32, 32 threads)

| Config | Forward | Gen/tok | Speedup |
|--------|---------|---------|---------|
| Full 14.8B | 5,871ms | 478ms | 1.0x |
| Rank 1024 (5.7B) | 2,918ms | 284ms | 1.7x |
| Rank 512 (3.6B) | 2,202ms | 165ms | 2.9x |
| Rank 256 (2.6B) | 1,144ms | 166ms | 2.9x |
| Rank 128 (2.1B) | 963ms | 116ms | 4.1x |

## What's Next (priority order)

1. Build KV-Medusa verification loop → first real decode wall clock on 32B
2. Finish whitening test (rank 512/1024) → find viable starting rank
3. Start thermostat with whitened SVD on 32B → compress with quality
4. Measure each axis independently on 32B → fill out this checklist
5. Stack axes → combined compression measurement
6. Rectangle repack → GPU-friendly deployment shape
