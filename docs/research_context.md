# Research context

Shared memory across machines and sessions. The framing, the experiments run so
far, what's been falsified, what's in flight.

Stage1–4 (BitNet dynamic-rank) is described in the README. Stage5+ builds on
those measurements and pivots to a general recipe for rank-bounded decode on
arbitrary trained LLMs.

---

## The claim (physics framing, not ML framing)

A trained LLM is a spin glass at its crystallized ground state. The residual
stream lives on a thin curved low-dimensional region of state space (the
"boundary layer"): **measurable, fixed by training, ~9–11 dim**. Prompts
inject energy (frustrate the system). Generation is relaxation trajectory
along the boundary layer back toward ground state.

We've measured intrinsic dimension (TwoNN) across nine models
(`Qwen3-{0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B, 32B}`, `BitNet-b1.58 2B`, `phi-2`)
and it converges to ~9–11 mid-stack regardless of hidden size (1024 → 5120).
Participation ratio collapses to ≈1 on some layers in the smaller models
(`results/Qwen_Qwen3-0.6B_manifold.json`, layers 3–4).

**Per-token decode compute scales with the boundary-layer dimension, not the
hidden size.** That's the 10–30× wall-clock claim on 30B-class dense models
at batch=1 decode.

### RSB resolves bulk vs. surface

In a spin glass at ground state (= trained LLM), Parisi's replica symmetry
breaking solution says the bulk off-manifold degrees of freedom are
**ultrametrically slaved to the low-dim manifold**. They do not carry
independent dynamical information. The `r90 ≈ 470` linear rank measured at
mid-layers is **geometric embedding redundancy of the curved 9-dim manifold
in 5120-dim space** (tangent planes rotating along the curve), **not**
independent bulk information that needs to be tracked.

Consequence: rank-k (k ≈ 9) is sufficient for KV storage, attention compute,
and layer-to-layer transport. The bulk is dynamically redundant once the
system is crystallized. **Do not store K/V at "bulk" rank (~500); store them
at manifold rank (~9).** Anyone who reaches for "but we need the bulk for
dynamics" is applying non-glass intuition to a glass system.

---

## Kernels should be simple — the current code is a compromise

The ultimate form of a forward pass on the boundary layer is **not matmul**,
it's routing:

- Residual stream stored as `(T, L, k)` tensor. k ≈ 9. Never materialized to d.
- Per-layer transport: `c' = M_i · c` where M_i is k×k (or a tiny k→k learned
  nonlinearity). ~81 ops per token per layer at k=9.
- Attention: `q_coords · M_{qk,i} · k_coords^T` where M_{qk,i} = A_q^T A_k is a
  k×k matrix. Softmax, weighted sum of V_coords. ~O(T·k) per head per layer,
  with k=9.
- Only at the end: project from k-dim manifold coords back to d, apply vocab
  projection.

Current `BasisFactoredLinear` (`scripts/stage8_distill_factored.py`) produces
full d_out output because it has to plug into HuggingFace attention. This is
an engineering compromise — the double-matmul `A·(B·x)` has kernel-launch
overhead and doesn't expose k-dim routing. **It is not the end state.** The
real kernel is ~200 lines of rank-k routing ops, cache-resident, looking
more like graph traversal than tensor computation.

Stage 10 builds that stripped-down form. Whether it needs training at all is
testable — if the per-layer transport is near-linear in rank-k coords, we can
extract M_i by least-squares regression on calibration trajectories and skip
distillation entirely. That is the direct geometric test of whether the
physics framing is clean or needs neural-network machinery.

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

## The all-dynamic principle (fundamental, not negotiable)

If compression dimensions are stacked maximally and cheaply, **every
dimension must be dynamic**. Not "some dynamic, some static." Every one.
Any static dimension locks compute to worst-case for the hardest input
the system will ever see, wasting cycles on every easier input.

| dimension | static (wrong) | live-state function of |
|---|---|---|
| rank k per layer | fixed | H_i(t), ∂H_i/∂t |
| heads kept | top-N fixed | per-head sharpness each step |
| layer depth | all layers always | residual-update magnitude, early-exit H_final |
| KV attended | whole cache | attention weight distribution (heavy hitters) |
| chart basis | one per layer | local manifold position (nearest cluster) |
| quantization | fp16 | tokens-per-basin (early high, late low) |
| speculative depth | fixed K | top-1 margin |

Cost of the signals: **free** (all come from tensors the forward pass
already materializes — attention weights, logits, hidden states). No new
compute. Just read the state, decide.

The "policy" is ~20 lines of conditional logic over scalar state, one
rule per dimension. Not a learned network. It's the routing gate for
every mechanism simultaneously, reading the same cheap live state.

Static-parameter architecture has a ceiling at worst-case efficiency.
All-dynamic architecture tracks the system's actual manifold position
moment-by-moment. Only the latter can approach theoretical
bandwidth/compute reduction. The former is a toy; the latter is the
system.

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

## The manifold floor — why 0.6B can't demonstrate the method

The factored-params budget scales with model size, but the "floor"
(minimum parameters needed to encode the tokenizer-induced manifold)
is roughly size-independent. So:

| model | full params | factored at k=32 | % of full |
|---|---|---|---|
| Qwen3-0.6B | 440M | **20.2M** | 4.58% |
| Qwen3-4B | ~3.2B | ~90M | 2.8% |
| Qwen3-32B | ~31B | **~270M** | 0.86% |

**0.6B at rank-32 is only 20M factored parameters.** Empirically, even
rank-256 on 0.6B (160M factored, ~36% of full) showed degenerate output
in stage 10b. This suggests the absolute floor is likely 80-160M+
params for the Qwen tokenizer-induced manifold.

**Implication: the 0.6B platform is structurally unsuitable for this
method.** Failures at stages 8–15 aren't procedural (bad training, wrong
hyperparams, MPS overhead) — they're structural. You cannot compress a
function below its floor, regardless of training procedure.

**Scaling doesn't make distillation easier; it raises the budget above
the floor.** At 32B rank-32 gives 270M factored params — comfortably
above the empirical floor. Training should converge cleanly, NOT because
32B is simpler, but because 32B at rank-32 is ABOVE the floor whereas
0.6B at rank-32 was BELOW it.

**Correlate: our 9-model catalog all shares (or closely shares) the
same tokenizer family.** The "universal ~9-11 dim" finding may actually
be "universal within this tokenizer family." Untested; would predict
that models with genuinely different tokenizers (GPT-2, Llama-3, Mistral,
T5) could sit at different manifold dim. A day of Z8G4 work to
discriminate.

**Practical consequence for the Strix Halo training plan:**
- Don't try to reproduce the 0.6B results and "scale them up." The 0.6B
  results are below-floor; scaling literally cannot reproduce bad
  numbers at higher scale.
- **Current target: Qwen3-14B Holographic Matryoshka at k ∈ [64, 128]**
  (first above-floor run; fits well within Strix Halo's 89 GB VRAM).
  32B remains feasible after 14B lands. See
  `machines/strix_halo/results/README.md` for the recipe.
- Finding 10 (holographic compressibility) names the technique
  **Holographic Matryoshka** and confirms the existing training script
  is architecturally correct: factoring every Linear is boundary
  compression (restricts rank, preserves d_int). Bulk dim stays full —
  exactly what the physics requires.
- If 14B at k=64 still struggles, try k=128 (fixed, no Matryoshka
  sampling) to isolate rank-sampling instability from below-floor failure.

## State at 0.6B / MPS checkpoint (2026-04-18)

Pieces established on 0.6B (at least observational/structural):

| # | piece | status | key result |
|---|---|---|---|
| 1 | manifold measured, 9 models | ✓ | TwoNN ≈ 9-11 universal |
| 2 | head skipping stage 5 | ✓ | 83% heads prunable, 100% match |
| 3 | distillation preserves manifold (TwoNN invariance) | ✓ stage 14 | mean \|ΔTwoNN\| = 0.49 |
| 4 | calibration scaling | ✓ stage 8→13 | 922× → 99× ppl ratio at 4× data |
| 5 | entropy signal free from eager attention | ✓ stage 11 | plumbing works |
| 6 | saddle detection ∂H/∂t | ✓ stage F | 4 profiles match RSB zoo |
| 7 | integrated rank-k attention math | ✓ stage D (math) | ~100× storage reduction projected |
| 8 | atlas observation | ✓ stage E | single basis sufficient at 0.6B |
| 9 | CPU vs MPS kernel overhead | ✓ | hook cost 2× cheaper on CPU |
| 10 | entropy-driven dynamic rank | ✗ (MPS hook overhead) | projection-and-back adds more overhead than savings |
| 11 | residual-magnitude layer skip | ✗ | no near-identity layers at 0.6B |
| 12 | Matryoshka nested-rank training | ✗ (unstable on MPS) | diverged; needs more care |

**Items 10/11/12 fail for reasons rooted in 0.6B scale + MPS kernel overhead, not in the physics. The missing pieces are engineering, not framework.**

The integrated all-dynamic runtime is specified in the design, each mechanism is individually validated, but the composite training (Matryoshka) needs Strix Halo's memory + stable training dynamics to land cleanly.

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

## How to read experimental outcomes in this project

Every experiment that doesn't reproduce the teacher's exact output is
**narrowing the hypothesis space**, not falsifying the framing. The
framing has survived all of stage 5–10b; each result tightened a
constraint.

| stage | what "failed" measured | how to read it |
|---|---|---|
| 5: head pruning (works) | 83% heads skippable with 100% match | manifold narrows; baseline confirmed |
| 6: plain SVD(W) | 0/200 match | weight-SVD isn't aligned with activation manifold |
| 7: basis-PCA factoring | 6/200 at rank 256 | manifold is curved; single global chart insufficient |
| 8: distillation (works) | kl 211 -> 0.4 (530x), match 15/200 | nonlinear chart-transition can be learned |
| 10: single-token geometric | garbage | attention coupling is part of the dynamics |
| 10b: fixed-rank projection | ~500 linear rank needed | r90 ≈ 470 measured; ambient rank confirmed |

**What these collectively tell us the deployment recipe must be:**

1. Atlas of local charts per layer, not a single global basis (from 7, 10b).
2. Cross-token coupling preserved — full attention + KV cache (from 10).
3. Per-token dynamic rank tied to current entropy (from 10b + entropy trace
   shapes).
4. Dynamic layer depth — skip late layers when relaxed (from 5 + entropy
   shape).
5. Optional distillation as a finishing adapter to smooth chart transitions
   (from 8).

None of these conclusions are "standard ML tricks"; they are constraints
read directly from measurement. When reading or running a new experiment,
read the result as narrowing, not as refutation.

## The entropy-profile zoo

Measured KV-attention entropy during generation shows four canonical
shapes, each a different mode of descent through the boundary layer's
RSB-hierarchical energy landscape:

- **Bell curve** — single-basin relaxation. Peak frustration = moment of
  commitment, then monotone descent.
- **Linear decline** — prompt already near basin. No real frustration;
  system rolls downhill.
- **Plateau** — system trapped in a metastable basin. Locally relaxed,
  globally not. No further acceleration available on this token.
- **Mid-generation spike (complex prompts)** — saddle crossing between
  sibling basins at the same RSB level. System explored one basin, found
  it wrong, climbed out. Spike height = saddle height; duration = time
  to identify the new basin.

Two signals to watch at inference:
- **H(t)**: current entropy. Rank tracks this.
- **∂H/∂t > 0**: saddle crossing in progress; restore rank until it
  drops again.

You can't classify prompts in advance; adapt live.

The spike phenomenon itself is strong evidence for the physics framing.
Pure statistical learners produce monotone uncertainty decrease. Spin
glasses are the only class where mid-trajectory spikes are expected,
because only RSB hierarchy traversal produces them.

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
- **Do not reflexively reach for standard-ML tricks** (distillation,
  teacher-student, speculative decoding, draft models). They each recover a
  fraction of what the geometric framing gives you directly, at higher
  engineering cost. Distillation may still be needed to absorb nonlinearity
  on the boundary layer, but it is not the primary mechanism; the geometric
  map is.
- **RSB means bulk is slaved to surface.** When tempted to say "but we need
  bulk information for dynamics" — don't. The physics says the bulk is
  ultrametrically redundant with the manifold in a crystallized system.
  r90 ≈ 470 is embedding-curvature, not independent DOF.
- **Kernels should be tiny.** 81-op matmuls, not million-op matmuls.
  Anything doing d_model-size matmuls in the hot path is missing the point.
