# Exponential-Inference — Roadmap

**Thesis: magnitude is a band-aid for structural issues, not architecture.**

Per-row magnitude variance in pretrained Qwen3 weights was added during pretraining as compensation for underlying structural issues — attention routing imbalances, residual scale mismatches, layer-wise contribution ratios. The body settled into using row magnitudes to compensate rather than fix the underlying structure (Adam took the path of least resistance). The structural problems are still there, hidden under the magnitude variance. A slow anneal of α (per-row magnitude) toward uniform during continued training forces the body to expose and fix those underlying structural problems instead of patching them.

If true, the result is a *cleaner* model than base — possibly better — with all weight rows on the unit hypersphere, no per-row α anywhere. This is a new compression and architecture frontier beyond nGPT (per-row α) and BitNet (per-tensor α).

---

## Phase A0 — Lossless nGPT-form conversion ✓ DONE (2026-05-04)

Split each targeted Linear into `W̃` (unit-norm rows) + `α` (per-row magnitude). Function preserved by construction. +0.0014 nats bf16 split noise.

- Artifact: `model_package/Qwen3-0.6B-nGPT-form/`
- Validation: 97.80% top-1 agreement, val_ce delta +0.001366 nats, 3/4 coherency match
- Scripts: `scripts/ngpt_lossless_convert.py`, `scripts/ngpt_load.py`

---

## Phase A1 — Polish to perfect nGPT (in progress)

Fine-tune A0 with W̃ unit-norm projection enforced after each step. Closes the bf16 gap, becomes true Loshchilov-nGPT trained on Qwen3 weights.

**Smoke-test result (9 steps, 80K tokens):** gap went from `+0.001366` to `-0.006320` nats (better than base). W̃ rows held at mean 0.999999. 3/4 coherency match.

Full run currently running — 100M token budget, early-stop on 2 evals at/below base.

### A1 sub-deliverables

- **A1 (this run):** `model_package/Qwen3-0.6B-nGPT-perfect/` — the polished artifact
- **A1b — Speedup proof:** train A1 vs base Qwen3 on identical tokens; show nGPT trains faster (Loshchilov's claim, validated post-hoc). ~6h paired experiment.
- **A1c — nGPT equivalence:** verify A1 has the structural properties Loshchilov's paper documents (W̃ unit-norm, bounded singular values, hypersphere geometry, well-conditioned gradients). Defends the "this is true nGPT, not nGPT-shaped" claim.
- **A1d — Benchmark essentially lossless:** lm-eval-harness on base, A0, A1 (arc_easy, arc_challenge, hellaswag, piqa, winogrande, lambada, wikitext PPL). Show A1 ≥ base on all tasks within bf16 noise.
- **A1e — Scale to all Qwen3/Qwen3.5 sizes:** Qwen3-0.6B (this run) → 1.7B → 4B → 8B → 14B → 32B (if Z8 G4 VRAM allows). ~3-4h per size.
- **A1f — Push to Hugging Face Hub:** `huggingface.co/parrishcorcoran/Qwen3-{size}-nGPT` for each size. Model card with architecture description, conversion methodology, benchmark table, loading instructions. MIT license. NO Claude attribution.

This is the public-facing literature contribution: "first post-hoc nGPT conversion of a major open model family at multiple scales, validated lossless and demonstrating Loshchilov's training-speedup claims."

---

## Phase A2 — Slow magnitude anneal (the actual research goal)

After A1 lands, anneal α from per-row base norms toward uniform 1.0 over a long training run. Slow anneal forces the body to expose and fix the underlying structural issues that magnitude was compensating for.

**Recipe:**
- Initialize from A1 (clean perfect-nGPT artifact)
- Forward: `y = (x @ W̃ᵀ) · α_t` where `α_t[i] = (1-τ)·α_base[i] + τ·1.0` and τ ramps 0 → 1 across training
- W̃ rows projected to unit-norm after each step (constraint maintained)
- α gets gradient updates only through (1-τ) channel — at τ=1 effectively removed
- **No bake.** Continuous anneal.
- Loss: KL distillation from frozen base + CE + hidden state MSE
- Corpus: diverse round-robin (OWT + wikitext + C4)
- Token budget: 5–10B initially. Slowness requires tokens.

**Critical measurement gates (non-negotiable):**
- wikitext PPL every 30 min — kill if > 1.5× base
- arc-easy 100-sample spot-check every 30 min — kill if drop > 5pts
- val_ce on diverse corpus every 10 min
- Coherency 4-prompt check every 30 min — kill if any becomes degenerate

**Acceptance:**
- val_ce within 0.05 nats of base (or better)
- wikitext PPL within 1.5× base (or better)
- arc/hellaswag/piqa within 3pts of base
- All α uniform at 1.0 (or removed entirely)
- All weight rows ‖W̃[i]‖ = 1.0 ± 1e-6
- Coherency: 4/4 prompts match base

If achieved, this is the *novel* result — a transformer with no per-row magnitude anywhere, structurally rebuilt to function without it.

**Empirical evidence the thesis is right:** April 2026 reached α=1 with severely broken recipe (3h, OWT-only, no teacher, no measurement gates). Model still produced grammatical English — degraded but coherent. With proper recipe (slow + diverse + teacher + gates), the body should rebuild cleanly.

---

## Phase A3 — 1-bit quantize W̃ (downstream, after A2)

Once A2 produces a model with unit-norm rows and α removed, quantize W̃ rows to sign(W̃) ∈ {-1, +1}. No scale parameter needed (rows are unit-norm by construction). True 1-bit nGPT.

Compare against Bonsai (per-128-group 1.125-bit) and BitNet b1.58.

---

## Sequencing

1. **A0** ✓ done
2. **A1** (in progress) → produces perfect-nGPT artifact at 0.6B
3. **A1b/A1c/A1d** in parallel after A1 finishes — speedup proof, nGPT equivalence, benchmarks
4. **A1e** scale to other Qwen3 sizes (1.7B, 4B, 8B, 14B, 32B)
5. **A1f** push all to Hugging Face Hub
6. **A2** the actual research goal — slow magnitude anneal on top of A1
7. **A3** 1-bit on top of A2 — the IP / Bonsai-competitor artifact

A0–A1f = the public literature deliverable. A2–A3 = the moonshot research that builds on it.
