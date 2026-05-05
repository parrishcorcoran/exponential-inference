# Exponential-Inference — Roadmap

Goal: post-hoc nGPT-form Qwen3 + 1-bit downstream as the IP/research artifact.

The plan below is sequenced. Each step builds on the previous — A0 is the foundation, everything else assumes A0 has been finalized.

---

## A0 — Lossless nGPT-form conversion (foundation)

**Status:** validation works; needs HF packaging.

Convert each targeted Linear in Qwen3-0.6B from `W` to a split parameterization:

- `W̃[i, :]` = `W[i, :] / ‖W[i, :]‖` (unit-norm rows)
- `α[i]` = `‖W[i, :]‖` (per-row scalar magnitude)
- forward: `y = (x @ W̃ᵀ) · α + bias`

Function is preserved by construction: `α[i] · W̃[i, :] = W[i, :]` exactly in fp32, +0.0014 nats noise in bf16.

**Targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

**Deliverable:** HF-loadable artifact at `model_package/Qwen3-0.6B-nGPT-form/`:

- `config.json` with `auto_map` pointing to `modeling_qwen3_ngpt.py`
- `modeling_qwen3_ngpt.py` defining `NGPTLinear` and the patched Qwen3 architecture
- `model.safetensors` with split parameters (`*.weight` = W̃, `*.alpha` = α)
- `tokenizer.json` etc.
- Loaded via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`

**Acceptance:**

- max abs logit diff vs base < 5e-3 on a 2K-token batch
- top-1 token agreement ≥ 99%
- val_ce ≤ base + 0.005 nats on OWT validation tail
- wikitext PPL within 0.5 of base
- Coherency: 4/4 prompts match base completion

Once A0 is banked, this is the "first post-hoc nGPT conversion of Qwen3-0.6B" artifact, lossless to bf16 noise.

---

## A1 — nGPT-style fine-tune (training-benefits demo)

Load A0. Fine-tune both `W̃` and `α` with the **unit-norm projection of W̃ rows after each optimizer step** — the constraint that makes it actually nGPT during training, not just relabeled parameters.

**Recipe:**

- Each training step: forward → backward → optimizer step → `W̃[i] ← W̃[i] / ‖W̃[i]‖` per row
- Loss: KL distillation from frozen base Qwen3 teacher + CE + hidden state MSE
- Corpus: diverse round-robin (OWT + wikitext-103 + C4)
- LR: separate rates for W̃ and α (α typically higher)
- Tokens: ~500M (~6 hours on local Z8 G4)

**Comparison:** continue standard fine-tune of base Qwen3 on identical tokens for the same time budget.

**Acceptance:**

- A0 + nGPT fine-tune reaches lower val_ce than base + standard fine-tune on identical tokens
- Wikitext PPL improves vs base
- Arc-easy / hellaswag / piqa within 1pt of base
- Closes the +0.0014 nats bf16 split noise

This becomes the publishable "post-hoc nGPT-form fine-tunes faster than base fine-tune" result.

---

## A3 — 1-bit quantize W̃ (the IP story)

On top of A0/A1, quantize `W̃` to sign-only:

- `W̃_q[i, j] = sign(W̃[i, j]) ∈ {-1, +1}`
- α stays fp16 (per-row magnitude scale)
- forward: `y = (x @ W̃_qᵀ) · α + bias` — but `W̃_q` is now 1-bit-storable

Because W̃ rows are unit-norm by construction (from A0), sign quantization is clean — no shared per-tensor or per-group scale needed beyond α.

**Recipe:**

- Initialize from A1 (clean nGPT-form weights)
- Quantization-aware fine-tune: forward uses `sign(W̃)`, gradients use straight-through estimator
- KL distillation from A1 (full-precision nGPT teacher) + base Qwen3 (text-quality anchor)
- Diverse corpus, ~1B tokens

**Comparison:** Bonsai (per-128-group 1.125-bit) and BitNet b1.58 on identical benchmarks at 0.6B scale.

**Acceptance:**

- arc-easy / hellaswag / piqa within 2pts of A1
- Wikitext PPL ≤ 1.5× A1
- Quality at 1-bit comparable or better than Bonsai/BitNet b1.58 at similar effective bit-rate

This is the genuine PrismML / NVIDIA / Apple-relevant artifact — direct competitor to Bonsai with cleaner geometric structure.

---

## A2 — Pure-direction Linear (moonshot)

Run only after A0/A1/A3 are banked. This is the swing-for-the-fences experiment.

**Drop α entirely.** Linear becomes:

- `y = x @ W̃ᵀ` (no α, no per-row magnitude knob)
- W̃ rows constrained to unit-norm during training
- Body must learn to compute without per-row magnitude expressiveness — all magnitude logic flows through RMSNorm and residuals

**Why this matters if it works:** No model on the planet has this. nGPT keeps α. BitNet keeps per-tensor α. Bonsai keeps per-group α. A2 is genuinely pure-rotation Linear with no magnitude factor anywhere.

**Recipe:**

- Init from A0's W̃ (throw α away, keep only the unit-norm directions)
- Train only W̃ with unit-norm projection per step
- Frozen base Qwen3 teacher KL + hidden state MSE + diverse corpus
- **Critical:** monitor wikitext PPL + arc-easy spot-check every 30 min; kill on explosion (April's tunnel-vision failure mode)
- Initial budget 6–12 hours; scale to 24–48h only if metrics hold

**Acceptance (high bar):**

- Val_ce within 0.05 nats of A0
- Wikitext PPL within 2× A0
- All downstream benchmarks within 3pts of A0

If this holds, the result is publishable as a new compression frontier beyond nGPT and BitNet. If it explodes (likely), kill early and extract whatever signal we can about which layers/rows resist uniformity.

---

## Sequencing rationale

1. **A0 first** because every downstream task needs the W̃ + α split. It's the parameterization, not the change.
2. **A1 next** because it produces a clean nGPT-form artifact (closes bf16 noise) AND the training-benefits demo, which is the publishable result that supports A0's value.
3. **A3 in parallel/after A1** because it's the IP-defensible artifact (1-bit). PrismML's whole company is 1-bit; this is the room they care about.
4. **A2 last** because it's high-risk research. A0+A1+A3 give the floor (publishable artifact + acquihire story). A2 is the moonshot on top.

A0+A1+A3 = floor outcome. A2 succeeded = exceptional outcome.
