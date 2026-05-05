# Exponential-Inference — Roadmap

**Thesis: magnitude is a band-aid for structural issues, not architecture.**

Per-row magnitude variance in pretrained Qwen3-0.6B weights was added during pretraining as compensation for underlying structural issues — attention routing imbalances, residual scale mismatches, layer-wise contribution ratios. Adam + the loss landscape settled on "give some rows larger magnitudes" as the path of least resistance to fix these structural problems. The underlying issues are still there, hidden under the magnitude variance.

If we slowly anneal α (per-row magnitude) toward uniform during continued training, the body is forced to **expose and fix the underlying structural problems** that magnitude was compensating for. The slow anneal is critical — yanking the band-aid fast (April / v8 / v10) collapses the model because the body never gets time to rebuild structure. Done correctly, the body rebuilds structurally and the magnitude band-aid becomes unnecessary. By τ=1, α is gone AND the body has real structural fixes, not patches.

If true, the result is a *cleaner* model than base — possibly better — with all weight rows on the unit hypersphere, no per-row α anywhere. This is a new compression and architecture frontier beyond nGPT (per-row α) and BitNet (per-tensor α).

---

## Sequence (in priority order)

### A0 — Lossless nGPT-form conversion (initialization step)

Split each targeted Linear (`q/k/v/o_proj`, `gate/up/down_proj`):

- `W̃[i, :]` = `W[i, :] / ‖W[i, :]‖` (unit-norm rows — the directions we keep)
- `α[i]` = `‖W[i, :]‖` (per-row magnitude — initially preserved so function = base)

A0 sets up the parameterization where W̃ and α are separately addressable, so the slow anneal of α can target only α without disturbing direction.

**Acceptance:**
- max abs logit diff vs base < 5e-3
- top-1 token agreement ≥ 99%
- val_ce ≤ base + 0.005 nats
- W̃ and α saved as distinct tensors

A0 is a couple hours' work. Foundation, not deliverable.

---

### A2 — Slow magnitude anneal (the actual goal)

**Anneal α toward 1.0 over a long training run.** The slowness is the entire point — it gives the body time to rebuild structure that magnitude was compensating for.

**Recipe:**
- Initialize from A0 (W̃ + α split, α = base row norms, function = base)
- Forward: `y = (x @ W̃ᵀ) · α_t` where `α_t[i] = (1-τ)·α_base[i] + τ·1.0` and τ ramps 0 → 1 across training
- W̃ rows projected to unit-norm after each optimizer step
- α gets gradient updates only through the (1-τ) channel — at τ=1, α has no path to drift; effectively removed
- **No bake.** The anneal is continuous. Never snap τ.
- Loss: KL distillation from frozen base Qwen3 + CE + hidden state MSE
- Corpus: diverse round-robin (OWT + wikitext-103 + C4) — never single-source
- Token budget: 5–10B initially. Slowness requires tokens.
- LR: standard, possibly warmup-decay; α and W̃ may want separate LRs

**Critical measurement gates (non-negotiable):**
- wikitext PPL every 30 min — kill if > 1.5× base
- arc-easy 100-sample spot-check every 30 min — kill if drop > 5pts
- val_ce on diverse corpus (not just OWT) every 10 min
- Coherency 4-prompt check every 30 min — kill if any prompt becomes degenerate

These gates exist because April's failure was metric tunnel-vision: OWT val_ce looked great while wikitext exploded 26 → 670. We do not repeat that failure.

**Acceptance:**
- val_ce within 0.05 nats of base (or better)
- wikitext PPL within 1.5× base (or better)
- arc-easy / hellaswag / piqa within 3pts of base
- All weight rows ‖W̃[i]‖ = 1.0 ± 1e-6
- α uniform at 1.0 (or removed entirely from forward)
- Coherency: 4/4 prompts produce non-degenerate completions
- **Bonus structural-rebuild signal:** attention entropy, residual scales, layer-wise contribution metrics shift visibly from base — that's evidence the body actually rebuilt rather than just absorbed

This is the publishable result and the IP-defensible artifact.

---

### A3 — 1-bit quantize W̃ (downstream, after A2 succeeds)

Once A2 produces a model with unit-norm rows and no α, quantize W̃ rows to sign(W̃) ∈ {-1, +1}. Since rows are unit-norm by construction, sign quantization is clean — no scale parameter needed at all. This is genuinely 1-bit nGPT with no scale: the bit-rate floor.

Compare against Bonsai (per-128-group 1.125-bit) and BitNet b1.58 on identical benchmarks at 0.6B scale.

---

### A1 — nGPT-style fine-tune (fallback if A2 fails)

If A2 explodes despite slow anneal + measurement gates, fall back to nGPT proper: keep α as a learnable per-row parameter, train W̃ + α with unit-norm projection on W̃. This produces a clean nGPT-form artifact (not the band-aid removal, just the structural rewrite). Still publishable as "first post-hoc nGPT conversion of Qwen3," just not the headline result.

A1 is the safety net, not the destination.

---

## Sequencing rationale

1. **A0** — necessary to set up the W̃ + α split. Couple of hours.
2. **A2** — the goal. Slow band-aid removal. 5–10B token run with hard measurement gates. The slowness and the measurement discipline are what previous attempts lacked.
3. **A3** — clean 1-bit story is much easier on top of A2's pure-direction model than from base.
4. **A1** — fallback if A2 fails despite proper recipe. Floor outcome.

## Why this is different from previous attempts

- **April:** right pointing, broken measurement (OWT-only val_ce, missed wikitext explosion)
- **v8 / v10 / v10_linear:** bake snaps master mid-anneal, interrupting structural rebuild; per-tensor α is architectural compromise
- **A2:** slow continuous anneal + frozen teacher anchor + diverse corpus + hard measurement gates + no bake. Lets the body rebuild structure under continuous pressure with safety checks.

The thesis predicts A2 should *not* require heroic recovery if done properly — the body has been holding structural issues together with magnitude band-aids and would benefit from being forced to fix them.
