# Exponential-Inference — Roadmap

**Thesis: magnitude is noise.**

Pretraining naturally drives weight rows toward uniform magnitude (Qwen3-0.6B has mean row norm = 1.0001, but residual CV = 0.25–0.6 per layer). Adam + RMSNorm don't push the residual variance to zero because nothing in the loss landscape requires it — RMSNorm absorbs magnitude downstream, so weights drift in their non-uniformity without consequence.

That residual is noise. The signal is the row *direction* on the hypersphere. The goal of this project is to remove the noise — produce a Qwen3 where every weight row has unit magnitude, with no per-row α anywhere — and demonstrate that the resulting model retains quality (because the body never *needed* the magnitude variance, just learned to live with it).

If true, this is a new compression frontier beyond nGPT (per-row α) and BitNet (per-tensor α). No model on the planet currently has it.

---

## Sequence (in priority order)

### A0 — Lossless nGPT-form conversion (foundation, foundation only)

**Purpose:** extract W̃ (unit-norm directions) from base Qwen3-0.6B as the *initialization* for A2. The α we save is **noise to be discarded**, not signal to preserve.

Convert each targeted Linear (`q/k/v/o_proj`, `gate/up/down_proj`):

- `W̃[i, :]` = `W[i, :] / ‖W[i, :]‖` (unit-norm rows — signal)
- `α[i]` = `‖W[i, :]‖` (per-row magnitudes — noise we're recording but throwing out)

**Acceptance:**
- max abs logit diff vs base < 5e-3
- top-1 token agreement ≥ 99%
- val_ce ≤ base + 0.005 nats
- W̃ tensor saved separately so A2 can load it directly

A0 ships as a verified intermediate, but it is **not the artifact** — A2 is.

---

### A2 — Pure-direction Linear (the actual goal)

Drop α entirely. Linear becomes:

- `y = x @ W̃ᵀ` — no α, no per-row magnitude factor
- W̃ rows constrained to unit-norm during training (renormalize after each optimizer step)
- All magnitude logic flows through RMSNorm and residuals — which is what they're for

**Recipe (the discipline matters more than the architecture):**
- Initialize W̃ from A0 (unit-norm directions extracted from base)
- Loss: KL distillation from frozen base Qwen3 + CE + hidden state MSE
- Corpus: diverse round-robin (OWT + wikitext-103 + C4) — never single-source
- Optimizer: standard, with W̃ unit-norm projection after each step
- **Critical measurement gate:** wikitext PPL + arc-easy spot-check every 30 min; kill on any explosion. April's failure was metric tunnel-vision (OWT val_ce looked great while wikitext exploded 26 → 670). We do not repeat that.
- Tokens: 1B initially; scale to 5–10B if metrics hold

**Acceptance:**
- val_ce within 0.05 nats of base
- wikitext PPL within 1.5× base
- arc-easy / hellaswag / piqa within 3pts of base
- All weight rows have ‖W̃[i]‖ = 1.0 ± 1e-6
- Coherency checks at every save: 4/4 prompts produce non-degenerate completions

If achieved, this is the publishable result and the IP-defensible artifact.

---

### A3 — 1-bit quantize W̃ (downstream, only if A2 succeeds)

Once A2 produces a clean unit-norm-only model, quantize W̃ rows to sign(W̃) ∈ {-1, +1}. No α anywhere (we already removed it in A2). This is genuinely 1-bit nGPT with no scale parameter — the bit-rate floor.

Compare against Bonsai (per-128-group 1.125-bit) and BitNet b1.58 on identical benchmarks at 0.6B scale.

---

### A1 — nGPT-style fine-tune (fallback, only if A2 fails)

If A2 explodes (wikitext PPL or downstream benchmarks won't recover), fall back to nGPT proper: keep α as a learnable per-row parameter, train W̃ + α with unit-norm projection on W̃. This is the "we couldn't strip magnitude entirely but we got the nGPT structure" outcome. Still publishable, just not the headline result.

A1 is the safety net, not the destination.

---

## Sequencing rationale

1. **A0 first** — necessary to extract W̃ as initialization for A2. Couple of hours.
2. **A2 directly** — the actual goal. If magnitude is noise, this should converge with proper measurement gating (wikitext + benchmarks every 30 min, not just val_ce).
3. **A3 if A2 succeeds** — the 1-bit story is much cleaner on top of A2's pure-direction model.
4. **A1 if A2 fails** — fallback to nGPT proper. Floor outcome, not the moonshot.

**Why this is different from previous attempts:** v8/v10/v10_linear were architectural compromises (per-tensor α + bake) that fought the body. April had the right pointing but broken measurement. A2 has the right pointing AND proper measurement gates. The thesis "magnitude is noise" predicts A2 should *not* require heroic recovery — the body never needed the magnitude variance to function.
