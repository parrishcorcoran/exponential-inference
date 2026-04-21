# Active checklist — open experiments anyone can pick up

Items in priority order within each category. Any Claude session on any
machine (Mac, Strix, Z8) can grab an open item and run it. Move an item
to "in progress" by commiting a stub file in `scripts/` with the next
stage number; move to "done" by committing results + a short writeup.

---

## Holographic / spectral (origin: point #1)

The Stage 58 two-mode rotation finding (angle 0 and π dominant) plus
stage 59 carry-channel analysis are being deepened in parallel by the
main session.

- [ ] **[CLAIMED: Z8G4]** **Cross-model manifold fingerprint**. Run
  `machines/z8g4/scripts/measure_manifold_fingerprint.py` on Qwen3-32B,
  Qwen3-72B, Llama-3-70B, and one non-Qwen-family model. Compare the
  two-mode spectrum, carry-fraction trajectory, and rotation curve
  across scales and tokenizer families. Each fingerprint is <100KB
  JSON; commit all back.
- [ ] **Validate two-mode spectrum on Qwen3-14B** (Strix). Does the
  bimodal distribution persist at larger scale? <1 hour on ROCm.
- [ ] **Per-position frequency decomposition**. FFT of the rotation
  curve (Finding 02) per model. What frequencies carry the signal?
- [ ] **Cross-layer mutual information** at varying layer gaps. Does
  information content decay with layer distance, or persist via the
  carry channel?
- [ ] **Phase-alignment-based dim estimator**. A replacement for TwoNN
  that uses phase relationships (derived from the two-mode spectrum)
  rather than point distances.

## Detailed manifold map (origin: point #2)

- [ ] **Per-token local tangent basis**. Not global PCA but the tangent
  plane at each token's position. Needs KNN + local PCA per point.
- [ ] **Cross-layer Christoffel-style connection coefficients**.
  Quantify how the manifold curves between layers.
- [ ] **Corpus density map on the manifold**. Where do tokens cluster?
  Hottest regions via KDE on projected coordinates.
- [ ] **Bootstrap TwoNN at N ≥ 2000**. Rerun Finding 01's measurements
  with larger hidden-state pools to get ±0.2 dim precision instead
  of ±0.6. Use `scripts/stage55_twonn_variance.py` as reference.

## Minimum architecture (origin: point #3)

- [ ] **Clean-slate d_int sweep** (Strix). Train from scratch at d_int
  ∈ {512, 1024, 2048, 3072} with fixed d_model, L. Find the bulk
  floor for this architecture. Keep d_model = 1024, L = 28.
- [ ] **Clean-slate d_model sweep** (Strix). Fix d_int, L; sweep
  d_model ∈ {64, 128, 256, 512, 1024}. Find the boundary floor.
- [ ] **Clean-slate L sweep** (Strix). Fix d_model, d_int; sweep
  L ∈ {8, 14, 20, 28, 40}. Find the layer-count floor.
- [ ] **Joint minimum finder**. Given the three floors, what's the
  smallest (d_model, d_int, L) that still produces coherent text?

## Train-vs-route split (origin: point #4)

- [ ] **Offline calibration file generator**. Script that produces a
  `manifold.pt` artifact per tokenizer family containing:
  per-layer PCA basis, rotation operators, projection matrices.
  Read at model boot, used at inference.
- [ ] **Single-matmul per-token position lookup**. During inference,
  compute the token's manifold coords in one matmul against the
  precomputed basis. Benchmark wall-clock.
- [ ] **Lookup-table router**. Key = quantized manifold coords;
  value = (width, length) routing decision. Near-zero runtime cost.
- [ ] **Full-pipeline latency benchmark**. Router + sparse forward pass
  + KV compression + head pruning composed. Measure vs baseline.

## Theory #6 scaling (origin: stage 54 series)

Primary open test track; the centerpiece claim.

- [ ] **Multi-teacher ensemble manifold** (Z8). Load Qwen3 {0.6B, 1.7B,
  4B, 14B, 32B} embedding matrices; average their PCA bases. Train a
  small student against the ensemble manifold. Measure per-bucket
  accuracy (stage 56 buckets). Ceiling-break check: does the student
  beat any single teacher on that teacher's own high-entropy bucket?
- [ ] **Scaled training on Strix**. 100M-param student, 10M wikitext
  tokens, 20k steps, checkpointed perplexity. Target: top-10%
  confidence agreement > 0.7 (useful speculative-decoding threshold).
- [ ] **Divergence signature tracking**. Plot `(student_ppl −
  teacher_ppl)` every 500 steps during long training. Watch for
  the crossover point (negative = student exceeds teacher on corpus).
- [ ] **Resolution-bucket A/B test**. After training, evaluate
  student vs teacher on each entropy bucket separately (stage 56).
  Report per-bucket student accuracy and per-bucket student-teacher
  disagreement that favors the student.

## Strix's geodesic architecture validation (origin: 2026-04-20 discovery)

- [ ] **Commit a `geodesic_validation.log`** showing teacher vs student
  generation samples on 5-10 prompts. Claims of "2.4× faster, 43×
  smaller, matches 0.6B teacher" exist only in commit body. Without
  side-by-side samples, the claim can't be distinguished from prior
  overclaims (14B "100% match" that turned out to be self-
  consistency).
- [ ] **Per-bucket geodesic eval**. Same resolution buckets as stage
  56 applied to the geodesic's outputs vs Qwen3-0.6B teacher.

## Rules of engagement

- If you start an item, commit a stub first to claim it
- If you complete an item, commit the result + a short markdown with
  the key numbers and interpretation
- If an item turns out to be ill-posed or already answered, remove it
  from this list with a note
- If you discover something new mid-task, add a new item here before
  the old one is closed

Maintained at the head of the main branch. Pull before starting work.
