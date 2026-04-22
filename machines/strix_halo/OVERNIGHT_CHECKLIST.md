# Overnight Checklist — 2026-04-22

## Priority 1: Scale to 14B
- [ ] Download 14B SAE (running)
- [ ] Run Holographic Transformer on 14B
- [ ] Compare routing profile: 0.6B vs 14B
- [ ] Measure: does 14B have more defined tokens? Different head ranking?

## Priority 2: Enforce routing (actual speedup)
- [ ] Build custom forward that ACTUALLY skips layers/heads based on SAE routing
- [ ] Measure real wall-clock speedup (not theoretical)
- [ ] Compare text quality: routed vs full compute

## Priority 3: Calibrate routing thresholds
- [ ] Run on diverse text (WikiText, code, creative writing)
- [ ] Find optimal defined/branching feature thresholds
- [ ] Measure token match at different routing aggressiveness levels

## Priority 4: Multi-layer SAE
- [ ] Compare SAE at layer 0 vs layer 14 vs layer 27 for routing quality
- [ ] Which layer's SAE gives best routing predictions?

## Priority 5: Package for HuggingFace
- [ ] Update model_package/ with SAE router
- [ ] ExponentialForCausalLM with holographic routing built in
- [ ] trust_remote_code=True compatible

## Priority 6: Push everything
- [ ] Commit all results
- [ ] Push to GitHub
- [ ] Update save point on Desktop
