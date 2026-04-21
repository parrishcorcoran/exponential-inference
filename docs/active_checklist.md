# Active checklist — open experiments anyone can pick up

Items in priority order within each category. Any Claude session on any
machine (Mac, Strix, Z8) can grab an open item and run it. Move an item
to "in progress" by commiting a stub file in `scripts/` with the next
stage number; move to "done" by committing results + a short writeup.

---

## The 5-tier strategy (added 2026-04-21)

The project is now organized around 5 parallel tiers:

- **Tier 1 — Dynamic manifold routing on EXISTING models** (product,
  near-term). No retraining; drop-in speedup for any open-weight LLM.
- **Tier 2 — Geodesic ODE architecture** (ARCHIVED — pure geodesic
  doesn't match the empirical two-mode physics, Strix attempts
  collapsed). Scripts remain for reference but no new investment.
- **Tier 3 — Two-channel holographic architecture** (research, high-
  upside). Purpose-built for measured two-mode rotation structure.
- **Tier 4 — Shared manifold-routing runtime** (infrastructure,
  unifies Tiers 1 and 3). Build the routing logic once; reuse
  everywhere.
- **Tier 5 — Training primitives from manifold + holographic**
  (compound speedup on Tier 3 training). Makes the smart-small-model
  training tractable on consumer hardware.

Items below are tagged with their tier.

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

## Tier 1 — Dynamic manifold routing on existing models (new 2026-04-21)

The near-term product. Drop-in speedup for any open-weight LLM. No
retraining. Builds the shared manifold-routing runtime (Tier 4) as a
side effect.

- [ ] **`runtime/wrap_with_manifold.py`**: library API
  `wrap_with_manifold(model, manifold_path) → wrapped_model`. Wrapped
  model's `generate()` uses dynamic routing.
- [ ] **Dynamic head pruning at runtime**. Per-token select active
  heads based on attention sharpness (Finding 04). Keep ≥20% of
  heads (stage 51's floor).
- [ ] **Dynamic length per token**. Per-token early-exit at
  stabilization_depth (Finding 09). Skip remaining layers when state
  has locked.
- [ ] **Dynamic KV compression**. Per-layer rank-k KV bottleneck at
  inference (stage 38). Rank 128 known safe.
- [ ] **Router from manifold coordinates**. Read current manifold
  position via precomputed PCA basis. Look up (active_heads, exit_layer,
  kv_rank) from a trainable policy OR a fixed table.
- [ ] **Wall-clock benchmark on Qwen3-14B**. End-to-end speedup vs
  baseline generate(). Target: 2-5× at 95%+ quality preservation.
- [ ] **Package as `exponential-inference-runtime` pip library**.
  For broad adoption once the benchmark shows real speedup.

## Tier 3 — Two-channel holographic architecture (new 2026-04-21)

Purpose-built for the measured two-mode rotation structure. Stage 60
bake-off in progress tests whether it even works at small scale.

- [ ] **[IN PROGRESS]** **Stage 60 bake-off**: standard transformer vs
  two-channel holographic at 20M params on wikitext-2. Running.
- [ ] **Hrrformer integration (if Tier 3 proceeds)**: use HRR binding
  for per-channel attention (O(T log T) complexity). Fork from
  github.com/NeuromorphicComputationResearchProgram/Hrrformer.
- [ ] **Strix-scale holographic training**. If stage 60 shows
  architectural parity or advantage, scale to 100M params on Strix.
- [ ] **Benchmark vs conventional 7B on MMLU/HumanEval**. Gate 2 —
  the claim doesn't land without benchmark parity.

## Tier 5 — Training primitives (new 2026-04-21)

Compound training speedups applicable primarily to Tier 3. Stacked,
these should make the smart-small-model training tractable on
consumer hardware (10× total speedup target).

- [ ] **Manifold-aware initialization**. Seed student weights from
  per-layer PCA bases in manifold.pt so the student starts near the
  correct structure. Skip the random→organized phase (first 100-500
  steps). Estimated 25-50% training time saved.
- [ ] **Manifold-regularized loss**. Add penalty term for hidden
  states drifting OFF the measured manifold. Prevents learning
  off-manifold noise. 10-30% faster convergence.
- [ ] **Natural gradient on manifold**. Replace Euclidean gradient
  descent with gradient computed w.r.t. the manifold's local metric.
  2-5× convergence speedup on curved optimization landscapes.
- [ ] **Curriculum via manifold resolution**. Train at rank-10 target
  first, expand to rank-64, then rank-128. Coarse structure fast,
  refinement later.
- [ ] **Rotation-curve-aware per-layer learning rate**. Early layers
  (big rotations) get low LR, late layers (small rotations) get
  higher LR. Following Finding 02's universal curve.
- [ ] **Two-channel staged curriculum**: carry → flip → mix. Train
  carry channel to convergence (easy objective) first. Then add
  flip. Then learn mixing. Avoids joint-optimization difficulty.
- [ ] **Carry-freeze, flip-train**. After initial convergence,
  freeze carry channel, only train flip + MLPs. Halves trainable
  params in late-stage training.
- [ ] **Trainable rotation operators, not weights**. Parameterize
  layer transitions as Givens/quaternion rotations (~1k params per
  layer) instead of full weight matrices (~1M per layer). Radical
  reduction in trainable space. Expressiveness untested.
- [ ] **Cross-tokenizer-family carry-channel pretraining**. Pretrain
  a carry-channel encoder on one family, transfer to others.
  Amortize expensive "find manifold" work.
- [ ] **Training-data curriculum from resolution buckets**. Use
  stage 56's entropy buckets. Easy tokens first (early
  convergence), hard tokens later.
- [ ] **Ternary/quantized manifold training**. Quantize student
  weights to ternary along manifold axes during training. 8-16×
  memory reduction during training. Inspired by BitNet.
- [ ] **Early-exit backprop**. Use stabilization_depth signal during
  training — backprop only through layers that haven't stabilized
  for this specific example. Faster training on easy examples.

## Tier 6+ — speculative extensions (new 2026-04-21)

Not yet active. Flag for future consideration.

- [ ] **Tier 6: Post-training alignment.** Instruction tuning,
  RLHF, safety — all expressed through manifold coordinates
  instead of token-level supervision.
- [ ] **Tier 7: Speculative co-decoding.** Two students of different
  widths jointly decoding, verified via manifold agreement. Another
  2-4× serving-time speedup.
- [ ] **Tier 8: Self-supervised manifold learning.** Discover the
  manifold from tokenizer + corpus alone, no teacher model needed.
  Most radical version of the teacher-free approach.

## Strix's geodesic architecture (Tier 2, ARCHIVED 2026-04-21)

The pure-geodesic path is archived — it doesn't match the measured
two-mode physics, and Strix's attempts (32B collapsed to repetition,
others unvalidated) haven't shown quality. Scripts remain in
`machines/strix_halo/scripts/train_geodesic*.py` for reference.

If Tier 3 holographic fails for specific reasons, we might revisit
geodesic as a simpler fallback. Otherwise, no new investment.

## Rules of engagement

- If you start an item, commit a stub first to claim it
- If you complete an item, commit the result + a short markdown with
  the key numbers and interpretation
- If an item turns out to be ill-posed or already answered, remove it
  from this list with a note
- If you discover something new mid-task, add a new item here before
  the old one is closed

Maintained at the head of the main branch. Pull before starting work.
