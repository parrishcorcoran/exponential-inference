# Qwen Halo — Full Pipeline Checklist

Target: Qwen3-14B-AWQ (Q4) → maximally compressed + accelerated model.
Every step uses thermostat-controlled annealing with fine-tuning.
Each step saves a checkpoint. Professional eval (not wikitext).

## Evaluation setup

- **Training data**: OpenWebText or RedPajama (not wikitext — too small)
- **Eval data**: held-out OpenWebText + MMLU subset + HellaSwag
- **Metrics**: val_ppl, MMLU accuracy, generation coherence
- **Fine-tune budget per step**: 500 steps, lr=5e-5, norms + target axis
- **Thermostat rule**: reduce axis by 5-10%, FT, eval. Accept if quality holds (< 1.5x threshold). Back off if it doesn't. Freeze and move to next axis.

## Phase 1 — Model shape (wormhole compression)

### Step 1.1: Load Q4 base + baseline eval
- [ ] Load Qwen3-14B-AWQ
- [ ] Eval baseline ppl on OpenWebText holdout
- [ ] Eval MMLU 5-shot (sample)
- [ ] Generate 10 diverse prompts, save outputs
- [ ] Save as `checkpoints/pipeline/step0_baseline/`

### Step 1.2: Wormhole shape measurement
- [ ] Measure r99 per layer (confirm wormhole on Q4 model)
- [ ] Identify throat, passage, mouth boundaries
- [ ] Compare with bf16 wormhole (stage 117) — should match

### Step 1.3: KV rank anneal (thermostat)
- [ ] Anneal KV projections: 768 → 512 → 384 → 256
- [ ] Thermostat: stop when ppl > 1.2x baseline
- [ ] Fine-tune norms + KV 500 steps between each
- [ ] Save `checkpoints/pipeline/step1_kv_rank/`

### Step 1.4: Per-layer K sensitivity + schedule
- [ ] Measure per-layer K damage at target rank
- [ ] Identify wall layers (protect) vs cavity layers (crush)
- [ ] Apply per-layer K schedule (protect walls, crush cavities)
- [ ] Fine-tune 500 steps
- [ ] Save `checkpoints/pipeline/step2_k_perlayer/`

### Step 1.5: Throat attention rank anneal
- [ ] Anneal throat (identified from 1.2) attention ranks: 256 → 128 → 64 → 32 → 16 → 8 → 4
- [ ] Thermostat controlled, FT between each
- [ ] Save `checkpoints/pipeline/step3_throat_rank/`

### Step 1.6: MLP width anneal (bathtub-shaped)
- [ ] Throat MLP: 100% → 95% → 90% → 85% → 80% → 75% → 70%
- [ ] Passage MLP: 100% → 95% → 90% → 85%
- [ ] Mouth MLP: keep 100%
- [ ] Thermostat + FT at each step
- [ ] Save `checkpoints/pipeline/step4_mlp_width/`

### Step 1.7: Weight quantization verification
- [ ] Model is already Q4 (AWQ) — verify quality held through above steps
- [ ] If degraded, apply per-layer precision schedule (Q6 edges, Q4 middle)
- [ ] Save `checkpoints/pipeline/step5_weight_quant/`

### Step 1.8: Embed quantization
- [ ] Anneal embed: Q6 → Q5 → Q4
- [ ] Thermostat + FT
- [ ] Save `checkpoints/pipeline/step6_embed/`

## Phase 2 — Cache optimization

### Step 2.1: V rank uniform anneal
- [ ] Anneal V projections uniformly: 256 → 192 → 160 → 128
- [ ] Thermostat + FT
- [ ] Save `checkpoints/pipeline/step7_v_rank/`

### Step 2.2: K cache quantization (Q4)
- [ ] Apply per-channel Q4 to k_proj weights
- [ ] FT 500 steps
- [ ] Verify quality
- [ ] Save `checkpoints/pipeline/step8_k_quant/`

### Step 2.3: V cache quantization (Q4)
- [ ] Apply per-channel Q4 to v_proj weights
- [ ] FT 500 steps
- [ ] Save `checkpoints/pipeline/step9_v_quant/`

### Step 2.4: Dynamic eviction implementation
- [ ] Build proper KV cache eviction (modify cache during generation, not attention mask)
- [ ] Implement entropy-based scoring: per-token confidence from logits
- [ ] Test keep rates: 90% → 80% → 70% → 60% → 50%
- [ ] Compare with H2O baseline
- [ ] Target: 3x eviction at < 2 ppl cost
- [ ] Save `checkpoints/pipeline/step10_eviction/`

## Phase 3 — Heads (decode acceleration)

### Step 3.1: Early exit probes
- [ ] Train per-layer LM probes (every 5th layer)
- [ ] 2000 steps on training corpus
- [ ] Measure per-layer val CE
- [ ] Save `checkpoints/pipeline/step11_early_exit/`

### Step 3.2: Medusa heads (speculative decode)
- [ ] Train head 1 (predict t+1): 2000 steps
- [ ] Train head 2 (predict t+2): 2000 steps
- [ ] Train head 3 (predict t+3): 2000 steps
- [ ] Measure acceptance rates
- [ ] Save `checkpoints/pipeline/step12_medusa/`

### Step 3.3: KV-Medusa heads (predict future KV)
- [ ] Design: predict next-token K,V from current hidden state
- [ ] Train on compressed model's KV representations
- [ ] Measure cache prediction accuracy
- [ ] Save `checkpoints/pipeline/step13_kv_medusa/`

### Step 3.4: Wide Medusa (20-50 heads)
- [ ] Enabled by cache compression: each head's KV cost is 16-20x cheaper
- [ ] Train heads 4-20 sequentially
- [ ] Measure diminishing returns curve
- [ ] Find optimal head count
- [ ] Save `checkpoints/pipeline/step14_wide_medusa/`

## Phase 4 — Integration + benchmark

### Step 4.1: Full stack integration
- [ ] Load final checkpoint with all optimizations
- [ ] Verify quality: ppl, MMLU, generation samples
- [ ] Wall-clock benchmark: tok/s with all features enabled
- [ ] Memory benchmark: peak VRAM usage

### Step 4.2: Comparison table
- [ ] vs original Qwen3-14B (bf16)
- [ ] vs Qwen3-14B-AWQ (Q4 baseline)
- [ ] vs published compression results (SVDLLM, SliceGPT, etc.)
- [ ] Publish numbers: compression ratio, quality, speed

### Step 4.3: Package + release
- [ ] Save final model to HuggingFace
- [ ] Write model card with compression details
- [ ] Push to Exponential-Inference repo
- [ ] Write Substack post (via meta-prompt)

## Progress tracking

| Step | Status | PPL | Notes |
|------|--------|-----|-------|
| 0. Baseline | | | |
| 1.3 KV rank | | | |
| 1.4 K per-layer | | | |
| 1.5 Throat rank | | | |
| 1.6 MLP width | | | |
| 1.7 Weight quant | | | |
| 1.8 Embed quant | | | |
| 2.1 V rank | | | |
| 2.2 K cache Q4 | | | |
| 2.3 V cache Q4 | | | |
| 2.4 Eviction | | | |
| 3.1 Early exit | | | |
| 3.2 Medusa | | | |
| 3.3 KV-Medusa | | | |
| 3.4 Wide Medusa | | | |
| 4.1 Integration | | | |
