# Research context

Shared memory across machines and sessions. The framing, the experiments run so
far, what's been falsified, what's in flight.

Stage1–4 (BitNet dynamic-rank) is described in the README. Stage5+ builds on
those measurements and pivots to a general recipe for rank-bounded decode on
arbitrary trained LLMs.

---

## The claim

A trained LLM's residual stream lives on a low-dimensional manifold. We've
measured intrinsic dimension (TwoNN) across nine models
(`Qwen3-{0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B, 32B}`, `BitNet-b1.58 2B`, `phi-2`)
and it converges to ~9–11 mid-stack regardless of hidden size (1024 → 5120).
Participation ratio collapses to ≈1 on some layers in the smaller models
(`results/Qwen_Qwen3-0.6B_manifold.json`, layers 3–4).

**If the manifold dimension is real and exploitable, per-token decode compute
scales with the manifold, not with the hidden size.** That's the 10–30×
wall-clock claim on 30B-class dense models at batch=1 decode.

---

## The integrated view: one manifold, one map

The forward pass and generation are two trajectories on the same manifold:

- **Forward pass:** h₀ → h₁ → … → h_L (within one token, across layers)
- **Generation:** h_t → h_{t+1} → … (across tokens, at any fixed layer)

Layers apply distinct operators; tokens iterate the full stack. The *dynamics*
differ, but the *state space* is shared. One per-layer manifold basis describes
both.

That means weights, K-cache, V-cache, and attention all operate in the **same**
rank-k subspace by construction — the KV cache is not a separate compression
target, it is part of the forward-pass map. Build the student correctly and
K_t, V_t are already rank-k; no post-hoc compression needed.

Concretely, at each layer `i`:

```
basis P_i  [d_in, k]         # top-k directions of input activations (calibration)
A_q = W_q @ P_i              # factored Q projection
A_k = W_k @ P_i              # factored K projection (K_t arrives in rank-k coords)
A_v = W_v @ P_i
# K_cache, V_cache stored in rank-k coordinates from birth
# attention: Q_new · K_cache is a rank-k dot product
# O(T · k) instead of O(T · d), and the `k` is the manifold dimension
```

No separate KV compression routine. No separate KV calibration. One map.

---

## Why plain rank reduction fails, and distillation succeeds

Intrinsic dimension ≠ linear rank. TwoNN = 10 says the manifold has 10 local
degrees of freedom; `r90 ≈ 470` (from stage1) says the *linear* rank needed to
cover 90% of variance is much higher. A 10-dim manifold non-linearly embedded
in 500 linear dimensions needs ~500 linear basis vectors to reconstruct.

Stage 6 (plain `SVD(W)`) and Stage 7 (basis-PCA factoring without training)
both confirmed this empirically:

| approach | rank | weights kept | token match (200) |
|---|---|---|---|
| Stage 6 — plain `SVD(W)` | 256 | 36.7% | 0/200 (gibberish) |
| Stage 7 — basis-PCA init | 256 | 36.7% | 6/200 (repetition) |

Linear factoring is capped by the **linear** dim, not the intrinsic one.

Distillation is the bridge. A rank-k factored student trained to match a
frozen teacher's logits and hidden states learns the *nonlinear* projection
that keeps the computation on the manifold. Stage 8 (`stage8_distill_factored.py`)
drops KL 211 → 3.5 (60×) in 1500 steps at k=32 on Qwen3-0.6B.

---

## How this is different from existing methods

Medusa / EAGLE / speculative decoding / early-exit / mixture-of-depth / MoE
all implicitly measure some slice of the manifold (typically 2–6 dimensions)
and route on it. MoE-64 is structurally equivalent to bottleneck-64 per
token; each approach gets ~2–4× on top of a modern baseline. **None of them
measure the intrinsic manifold dimension nor use it as the routing target.**

We measure the full manifold up front, then build weights and caches that
live on it.

---

## Pipeline

### Phase 1 — Calibration (once per model)

1. Run teacher on ~10k–100k tokens of diverse text.
2. Collect per-layer input covariances for every target Linear.
3. Eigendecompose (full eigh for d ≤ 2048; randomized SVD for larger — **note:
   randomized path in current `stage8_distill_factored.py` produced bad
   bases; reverted to full eigh, TODO re-implement for 32B scale**).
4. Top-k eigenvectors form `P_i` for each layer.

### Phase 2 — Teacher output cache (once per model, ~10 min on H100)

Forward teacher once over calibration corpus. Save hidden states and logits
to disk. From here on, training never runs 32B inference again.

### Phase 3 — Distillation (2–6 hours on A100 / H100 / Strix Halo)

Student is teacher with every `q_proj / k_proj / v_proj / o_proj / gate_proj /
up_proj / down_proj` replaced by a `BasisFactoredLinear` initialized from
`P_i`. Freeze everything else. Loss:

- KL(teacher || student) on final logits, temperature 1.
- Relative hidden-state MSE per layer (normalize by teacher RMS — unweighted
  sum MSE *diverges*, learned the hard way).

Training ~270M factored params on top of frozen teacher weights. Small job by
modern standards; fits easily on A100 40GB.

### Phase 4 — Deployment

Cast factored weights to bf16. KV cache is naturally rank-k. Attention at
long context is `O(T·k)` instead of `O(T·d)`.

---

## Target numbers

At batch=1 decode on 30B-class dense models. First two columns are baseline
and rank-32 factored weights only; the fourth column is the combined effect
at long context (integrated rank-k forward + rank-k KV).

| hardware | baseline ms/tok | rank-32 factored | 32k-ctx integrated |
|---|---|---|---|
| H100 SXM | ~19 | ~1.5–2 | ~0.5–0.8 |
| A100 80GB | ~32 | ~2–3 | ~1.0–1.5 |
| M3 Ultra | ~80 | ~3–5 | ~2.0–3.0 |
| Strix Halo (256 GB/s) | OOM full | ~8–12 (target) | ~3–5 (target) |

These assume:
- Weight factoring gives speedup ≈ bandwidth-ratio at batch=1 decode (well-
  established for memory-bandwidth-bound workloads).
- No fused-kernel work required on H100/A100. On MPS/ROCm, kernel-launch
  overhead could cap the speedup at ~5× unless we write a fused factored-
  matmul kernel.

---

## Training cost

- Phase 1 + 2 (one-time per model): ~$0.50–1 on cloud H100, or free on Strix
  Halo.
- Phase 3 (single full distillation run): **$3–5** on cloud A100/H100, or ~4–6
  hours on Strix Halo.
- Full R&D cycle (10 trial runs, rank ablation, step-count ablation): **~$30–
  50 cloud, or free on Strix Halo overnight**.

Cheap because the expensive step (32B forward) is a one-time precompute. The
distillation loop only touches the small factored student — effectively a
270M-param training job, not a 32B one.

---

## Machines

- **MacBook (MPS)** — fast iteration. Smoke tests on Qwen3-0.6B/4B. Short
  distillation loops (minutes). Any numerical path changes get tested here
  before committing to the long runs.
- **Strix Halo (ROCm, 82 GB VRAM)** — primary compute. Teacher output caches
  live here. Full distillation at 4B / 32B scale. KV-integrated decode
  benchmarking.

Per-script naming: `stageN_<what>.py` under `scripts/`, JSON artifacts under
`results/stageN_*.json`.

---

## Open experiments, in priority order

1. **Finish Qwen3-0.6B rank-32 distillation** at 15k+ steps, diagnose whether
   match tracks KL monotonically. If greedy-match lags KL, add teacher-argmax
   cross-entropy term (direct top-1 supervision, not just distribution shape).
2. **Expand calibration corpus to ~10k+ tokens** (currently 733 toks/layer in
   stage8 — too few for high-fidelity teacher reproduction).
3. **Qwen3-4B distillation on Strix Halo.** Same recipe, bigger model,
   longer runs. Measure real wall-clock on ROCm.
4. **KV as part of the map.** Rewrite `BasisFactoredLinear` so K and V live in
   rank-k coordinates throughout the cache. Attention becomes
   `O(T·k + d·k)` by construction. Test at T ∈ {512, 4k, 32k}.
5. **Manifold-guided speculative decoding.** Token trajectory on the manifold
   is smooth (Stage 5 showed heads narrow with context). Extrapolate N steps
   in manifold coordinates, project back, verify cheaply. Draft = geometric
   extrapolation on the same model.
6. **Qwen3-32B.** Precompute teacher outputs on Strix Halo, distill on A100 or
   Strix Halo.

---

## Falsified / do-not-re-run

- **Plain `SVD(W)` at any rank ≤256** — `results/stage6_factored_Qwen_Qwen3-0.6B.json`.
  0/200 match. Global weight SVD does not find the manifold.
- **Basis-PCA factoring without training** — `results/stage7_basis_factored_*.json`.
  Marginal at rank 256 (6/200), collapses at lower rank. Init is fine, but
  the nonlinear compression needs *training*.
- **End-to-end distillation with unweighted-sum hidden MSE across layers** —
  gradients diverge (loss → 350k). Must use relative MSE per layer, or lower
  the LR to 3e-4 with tight grad clipping.
- **Randomized SVD for d > 2048 covariance matrices** — current implementation
  in `stage8_distill_factored.py` was producing NaN bases. Reverted to full
  eigh. Needed for 32B down_proj (d=27648); re-implement with better power-
  iteration and check QR step.

---

## Stuff we believe but haven't measured yet

- That rank-k K/V coordinates from a distilled rank-k student preserve
  attention fidelity at long context. (Follow-up: Exp 4.)
- That manifold smoothness across tokens supports multi-step speculative
  extrapolation without a separate draft model. (Follow-up: Exp 5.)
- That the 10–30× ratio actually materializes on H100/A100 with custom fused
  kernels. (Follow-up: bench after Phase 3 on Strix Halo lands.)

---

## Style notes for anyone (Claude or human) reading this

- Do not re-derive the claim from scratch. The manifold is measured across 9
  models; that's the starting point.
- Do not retry anything in the "Falsified" list without reading why.
- When the user says "we want 10×" they mean at batch=1 decode on 30B-class,
  not on 0.6B. Speedup at small scale is capped because embedding + lm_head
  dominate.
- Don't argue the physics from priors that don't account for the manifold
  measurement. The priors that predict this is impossible all assume
  data-agnostic low-rank compression. We are not doing that.
