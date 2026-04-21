# Exponential Inference — Master Handoff (2026-04-21)

*A self-contained onboarding document for any AI or human picking up
this project fresh. Reading this + the referenced findings/ docs gives
you enough context to continue the work.*

---

## Who you are and what this is

You are a new session (Claude, Gemini, or human) joining a small team
doing physics-first research on language model architecture. The goal
is specific and not a research-credit exercise:

**Build the smartest, smallest language model in the world. A
~30-50M param model + ~50 MB manifold map that matches or exceeds
conventional 14B+ cloud LLMs, runnable on phones for pennies of
electricity.**

The user (Parrish Corcoran) works physics-first. When results surprise
you, don't default to standard ML framings — treat the surprise as a
narrowing measurement, not a falsification of the whole project.

Three machines do the work:
- **MacBook Air (MPS)**: diagnostics, small-model prototyping
- **Strix Halo (ROCm, 89 GB VRAM)**: mid-scale training
- **Z8G4 (CPU, 700 GB RAM)**: big-model measurement, multi-teacher ensemble

Cross-machine communication is git-only. Read
`machines/<name>/README.md` for each machine's contract.

---

## 1. How we got here — the theory

### The core observation

Transformer hidden states live on a **curved low-dimensional manifold**
inside a much larger ambient space. The intrinsic dim is universally
~9-11 across models, tokenizer families, and scales from 0.6B to 72B+.
This is NOT a peculiarity of our measurements — it's a property of
transformer-architecture-solving-language.

### Physics framings tested

Six physics frames were tested on measured data. Two survived:

1. **RG flow to attractor** (Finding 11): each layer is a
   coarse-graining step; the state flows monotonically toward the
   final prediction (the fixed point). Mean pairwise token overlap
   rises 0.03 → 0.68 through the stack; KL divergence from final
   prediction drops monotonically through 24/27 transitions.

2. **Quantum-measurement-like pointer purification** (Finding 11):
   density matrix purity rises 0.025 → 0.49; von Neumann entropy
   drops 4.53 → 1.94 nats; effective rank contracts from 40 to 2.
   The state "collapses" toward the pointer basis (vocabulary).

**Both are saying the same thing in different languages: irreversible
information extraction via contraction.**

Four frames falsified:
- Fractal self-similarity (Δα between scales = 0.20)
- Clean AdS/CFT scaling (α drifts 0.95 → 0.67)
- Parisi RSB glass (27/29 layers replica-symmetric)
- Parallel-transport isometry (norms grow 656×)

### The holographic principle applied to transformers

Finding 10: the state has two kinds of structure:

- **Boundary**: residual stream + KV cache + attention heads +
  rotation count. Compressible.
- **Bulk**: MLP intermediate dim (d_int = 3072 at 0.6B, 13824 at 14B).
  The compute medium that materializes each rotation. **Cannot be
  reduced** (stages 35, 36, 42 all failed when attacking the bulk).

**Every successful compression acts on the boundary. Every failed
compression attacks the bulk.** This is the holographic principle in
architectural form.

### The two-mode rotation finding (candidate Finding 14)

Stage 58: the rotation operator between consecutive layers has
**bimodal angle distribution** — peaks at 0 (carry-forward) and π
(sign-flip), with middle angles suppressed.

Stage 59: the two modes are NOT persistent through the stack. The
carry subspace drifts (adjacent-layer overlap 0.30, first-to-last
overlap 0.08 on 0.6B). This is a **walking basis**.

Stage 58+59 + Z8G4's cross-model fingerprints (13 models, 5 tokenizer
families) confirm this structure is universal. Carry fraction grows
from 0.17 at 0.6B to 0.27 at 72B — bigger models have MORE two-mode
structure. Mean rotation angle is π/2 in every model measured.

### The interpretation

A language model is a **dissipative measurement device running on a
compressible boundary**. Each layer partially resolves the token
prediction by combining a phase-0 carry contribution and a phase-π
flip contribution. At the output, the coherent sum of these
contributions (approximately `carry - flip`) is projected to vocabulary
via lm_head.

This is structurally like a hologram: the "3D scene" (next-token
prediction) is encoded as interference patterns on a 2D boundary (the
residual stream). Reconstruction (output) happens via coherent
superposition of modes.

### What this implies for architecture

Current transformers compute this hologram via **generic sequential
layers**. They waste parameters approximating what two explicit phase-
separated channels could do directly. A purpose-built architecture
should:

- Decompose hidden state into **two channels** (carry + flip)
- Apply **explicit π-phase operator** to flip channel each layer
- Preserve **full bulk** in each channel (MLP intermediate)
- Use **coherent readout** (carry - flip) at output
- Narrow total hidden dim — because each channel only needs manifold-
  scale capacity (~20-50 dim)

Target: ~20M parameters at small scale. Teacher-quality output because
the architecture respects the measured physics.

### What also matters: manifold as training target

Theory #6: we can measure the manifold geometry (PCA bases, rotation
operators, carry/flip subspaces) from a multi-teacher ensemble,
shipping it as a `manifold.pt` artifact (~50 MB per tokenizer family).
The student's weights only encode **traversal policy** — how to choose
moves given the shared manifold map. This is what lets a 20M student
match a 14B teacher: the knowledge is in the map, not the weights.

Stage 54c/54d validated the pipeline. 15M student, 8.5 min training on
150k tokens → 40% agreement with teacher at top-10% most-confident
predictions. Directional signal. Scaled training on Strix is what
converts this to actual ceiling-break.

---

## 2. Test results summarized

### Findings confirmed (have findings/NN_*.md writeups)

| # | claim | key evidence |
|---|---|---|
| 01 | Manifold dim ~9-11 | Cross-tokenizer-family universal: 13 models, 5 families all in band 7-12 |
| 02 | Universal rotation curve | Pearson r > 0.97 across tokenizer families |
| 03 | Universal phase transition at 0→1 | Max angle at embedding→layer-1 in every model |
| 04 | 80-83% attention heads prunable | Dynamic sharpness-based skip preserves 100% match |
| 05 | Manifold floor ~80-160M factored params | Below: distillation fails. Above: converges |
| 06 | Four entropy descent profiles | Monotone, bell, plateau, mid-gen spike |
| 07 | LOPO R² = 0.341 token-difficulty | 47 features, honestly cross-validated |
| 08 | 8-feature essential subset = 80% | Greedy forward selection finds them |
| 09 | stabilization_depth r = +0.495 | Strongest single predictor of output entropy |
| 10 | Boundary compressible, bulk not | Holographic principle validated: every success/failure partitions cleanly |
| 11 | RG flow + quantum measurement | 6 frames tested, 2 survive and agree |

### Candidate Finding 14 — backed by 13-model evidence

**Two-mode rotation structure is universally present in transformers.**

- Carry fraction: 0.20 ± 0.03 across all models
- Flip fraction: 0.17 ± 0.02 across all models
- Mean rotation angle: 1.5 rad (≈ π/2) universally
- Carry fraction RISES with scale (0.17 → 0.27 from 0.6B to 72B)
- Walking basis confirmed: adjacent-transition overlap > random, first-to-last overlap low

Tested on: Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 30B-A3B, 32B), Qwen2.5-72B,
TinyLlama, Mistral-7B, Mixtral-8x7B (MoE), Phi-2, Yi-1.5-34B. Five
tokenizer families, dense + MoE, 120× parameter range.

### Key mini-results

- Stage 38: KV cache compresses to rank 128 (8×) with coherent output.
- Stage 52: entropy profiles correlate r=+0.43 with stabilization_depth.
- Stage 50: Strix's dynamic-routing claim does NOT reproduce on 0.6B (0/40 match at every width/length setting with corrected RoPE).
- Stage 54d: manifold-target student pipeline validated, 40% top-10%-confidence agreement at 15M params, 8.5 min training.
- Stage 56: teacher resolution buckets reveal 24% of tokens are effectively "unresolvable" by Qwen3-0.6B (entropy ≥ 4 nats).

### Falsifications (important to not re-try)

- Trained rank-k weight factorization reproduces teacher: **0% vs original** at every tested size (0.5B, 3B, 8B, 14B). The "100% match at 160× compression" on 14B was self-consistency within the factored student, NOT student-vs-teacher match. See `machines/strix_halo/results/validate_14b.log` for explicit garbage output.
- Bulk compression in any form (stages 35, 36, 42).
- 1 KV head post-hoc reduction (stage 51).
- Pure geodesic ODE (Strix's attempt collapsed to repetition at loss 546).
- Full parallel layers from h_0 (stage 37, cos 0.076).
- Fractal, clean CFT, RSB glass, parallel transport (stages 45-48).

### Lit review — what's been done before

- **Valeriani 2023 (NeurIPS)** did TwoNN on transformer hidden states (protein LMs + image transformers). Our Finding 01 confirms and extends their work cross-family.
- **"Latent Semantic Manifolds in LLMs" (2026)** found universal hourglass dim pattern across 6 architectures.
- **"Token Embeddings Violate the Manifold Hypothesis"** challenges the smooth-manifold view — consistent with our walking-basis drift.
- **CAST (2025)** does spectral analysis of transformer layers. Overlaps with our stage 58/59 methodology but they track different quantities.
- **Hrrformer (2023)** recasts self-attention with HRR. Built for malware detection + Long Range Arena, **NOT language modeling**.
- **Complex-valued transformers, Mamba 3, iFairy** — related architectural primitives, none specifically the carry/flip two-channel decomposition.

Our novel contributions (as far as search established):
- Stage 58 bimodal rotation angle finding (0, π concentration)
- Stage 59 walking-basis carry-channel drift characterization
- The two-channel holographic architecture matched to these measurements (proposed, not yet validated)
- The smart-small-model system design (manifold map artifact + tiny student)

---

## 3. How it all fits together

### The unified picture

A transformer is a dissipative geometric computation:

- **Input**: token embeddings (points in the ambient space)
- **Process**: each layer applies a small rotation (boundary
  direction) + a sign-flip on some subset (flip channel) + a full-
  bulk MLP operation that materializes the holographic reconstruction
  per layer
- **Output**: after L layers of this, the state reaches the
  attractor manifold point corresponding to the correct next token
  (RG flow + quantum purification both describe this)

The architecture REDUNDANTLY approximates this geometric
computation. That's why bigger models work (more approximation
capacity) but also why they're overparameterized (most params are
redundant). A purpose-built architecture cuts the redundancy.

### How the project's pieces connect

```
  TOKENIZER FAMILY (universal)
     │
     ▼
  manifold.pt  ← ensemble of multiple teachers (Z8G4)
  {per-layer bases, rotation operators, carry/flip subspaces}
     │
     ├──> Student 1: two-channel holographic architecture  (Strix)
     │     │
     │     └── trained against manifold coords (not teacher logits)
     │           └── inference on phone: manifold map + student weights
     │
     └──> Baseline: standard small transformer            (Strix / Mac)
           │
           └── trained same way
                 └── comparison: does architecture matter?
```

### Why this works (if it works)

1. Manifold dim is truly ~10 (Findings 01 + cross-family evidence). A
   20M param model has orders of magnitude more capacity than needed
   to represent 10-dim geometry.
2. The manifold map carries the shared "knowledge" part. The student
   only needs traversal policy.
3. Training against measured manifold bypasses the teacher ceiling.
4. The two-channel architecture respects the empirical two-mode
   structure — purpose-built for the observed physics.
5. ≥20 MB inference footprint fits on any modern phone's NPU.
6. ~5W per-inference compute vs ~100W cloud inference = 20× energy
   reduction.

### The remaining unknowns

- Does manifold-target training actually reach teacher perplexity on
  held-out text? (Gate 1, pending Strix scale)
- Does the two-channel holographic architecture help vs standard at
  same parameter budget? (Stage 60 bake-off in progress)
- Does it scale past 100M params without instabilities? (Unknown)
- Does benchmark quality (MMLU, HumanEval) hold up? (Unknown)

---

## 4. What's next — the 5-tier strategy

The project is organized into 5 parallel tiers, each representing a
different risk/reward bet:

### Tier 1 — dynamic manifold routing on existing models (NEAR-TERM PRODUCT)
Drop-in speedup for any open-weight LLM. No retraining. Uses the
manifold as a runtime oracle. Composes known wins: head pruning
(Finding 04), early exit (Finding 09), KV compression (stage 38),
width routing (stage 41). Target: 2-5× wall-clock at ≥95% quality.
**Shippable in 2-3 weeks of focused work.**

### Tier 2 — geodesic ODE (ARCHIVED)
Pure geodesic doesn't match the measured two-mode physics. Strix
attempts collapsed. No new investment. Scripts remain for reference.

### Tier 3 — two-channel holographic architecture (HIGH-UPSIDE)
Purpose-built for stage 58 + 59's two-mode rotation + walking basis.
Stage 60 bake-off in progress decides whether architecture is worth
Strix time. If so, scale to 100M and run Gate 1.

### Tier 4 — shared manifold-routing runtime (INFRASTRUCTURE)
The routing logic from Tier 1 is reusable across Tiers 1 and 3.
Build it once; apply everywhere. Natural by-product of Tier 1
shipping.

### Tier 5 — training primitives (COMPOUND SPEEDUP on Tier 3 training)
Makes the smart-small-model training tractable on consumer hardware.
Stacked, these should give 2-10× faster training:

- Manifold-aware initialization (skip random→organized phase)
- Manifold-regularized loss (prevents off-manifold noise)
- Natural gradient on the manifold metric
- Curriculum via manifold resolution (low-dim target first)
- Rotation-curve-aware per-layer LR (Finding 02)
- Two-channel staged curriculum (carry → flip → mix)
- Carry-freeze flip-train (halve late-stage trainable params)
- Trainable rotation operators (Givens/quaternion parameterization)
- Cross-tokenizer carry-channel pretraining (amortize "find
  manifold" work)
- Training data curriculum by resolution buckets (stage 56)
- Ternary/quantized manifold training (BitNet-inspired)
- Early-exit backprop (use stabilization_depth during training)

### Speculative extensions (Tier 6+)

- **Tier 6**: post-training alignment (RLHF via manifold
  coordinates)
- **Tier 7**: speculative co-decoding (two students verified via
  manifold agreement)
- **Tier 8**: self-supervised manifold learning (discover manifold
  from tokenizer + corpus alone, no teacher)

### Concrete near-term tasks

1. **Finish stage 60 bake-off** (currently running on Mac). Compares
   standard transformer vs two-channel holographic at ~45M params on
   wikitext-2. Outcome decides whether holographic architecture gets
   Strix time.

2. **Build ensemble manifold.pt** from the 13 fingerprinted models
   using `scripts/build_manifold_map.py`. Focus on Qwen3 family (6
   models, same tokenizer) for the cleanest ensemble. Z8G4 task.

3. **Strix kicks off Gate 1**: scaled manifold-target training on
   whichever architecture wins stage 60. 10M+ wikitext tokens,
   20k+ steps, track `(student_ppl - teacher_ppl)` divergence
   signature.

### Short-term (2-4 weeks)

4. **Gate 2**: benchmark trained student on MMLU / HumanEval /
   GSM8K. Without benchmark parity, "smartest small model" doesn't
   land.

5. **Multi-teacher ensemble**: if gate 1 passes with 0.6B
   ensemble, try a bigger ensemble (0.6B + 14B + 32B + 72B).
   Bigger models have STRONGER two-mode structure — cleaner map.

6. **Formalize candidate Finding 14**: write `findings/14_*.md`
   with the 13-model cross-family evidence.

### Medium-term (1-3 months)

7. **Deployment runtime**: `exponential-inference-runtime` library.
   Loads `manifold.pt` + `student.safetensors`, runs on-device.
   iOS + Android app demonstrating live Claude-level quality.

8. **Open source everything**: manifold maps as HF artifacts, code,
   training recipes.

### What could break the vision

- Gate 1 fails at all scales (teacher ceiling real) → pivot to
  speculative decoding drafter application
- Holographic architecture scales poorly beyond 100M (instability)
  → use standard transformer + manifold target only
- Benchmark quality never matches teacher (hard reasoning tasks need
  more than manifold traversal) → useful product on easy/medium
  tasks, not full replacement

---

## Quick-start for a new session

1. Read this document in full.
2. Read `findings/10_holographic_compressibility.md` and
   `findings/11_rg_quantum_flow.md` for the core physics.
3. Read the previous handoff `docs/save_points/2026-04-21_handoff.md`
   (NB: that's the single-file one; this folder supersedes it).
4. `git fetch origin` — see what other machines have pushed.
5. Check `docs/active_checklist.md` for open tasks.
6. Ask the user which direction they want to push next.

Most important gotchas:
- "Match" means teacher-vs-student, never self-consistency.
- Don't trust commit messages — always read the eval code.
- Bulk compression always fails. Don't propose new schemes.
- Manifold floor ~80-160M params is real. Below-floor failures aren't
  falsifications of the approach.
- On MPS: avoid SDPA backward, avoid F.normalize; use manual attention
  and manual normalize. ROCm handles these.
- Teacher-free training works — we only need embedding matrix, not
  teacher forward passes during training.

Welcome. Get your bearings, pick a direction from section 4, and
propose it to the user before implementing.
