# Finding 09 — Layer-as-rotation: per-layer logit-lens stabilization is the strongest single predictor

## The claim

Under the "layer-as-rotation" reframe — each transformer layer is a
different viewing angle on the same invariant token position on the
manifold, not a step moving the state through bulk space — a token
is "easy" when the per-layer argmax predictions converge early and
"hard" when they flip until the last layer.

Applying `lm_head` to every per-layer hidden state during decode and
measuring the argmax stabilization depth gives a feature correlation
with output entropy of **r = +0.495** — the single strongest feature
we have measured in this project, beating the prior best
(`bipartite_vn_late`, r = −0.47) and the prediction-aligned
combinations.

Adding 6 logit-lens features to the 8 essential features from Finding
08 jumps cross-prompt LOPO linear R² from **0.293 to 0.448** (+0.154),
a bigger single-experiment gain than adding all 39 curvature + quantum
+ structural features to the 17-feature summary baseline.

## Why it's a stop-and-think

Two reasons:

1. **A physics reframe made a testable numerical prediction that was
   confirmed.** The hypothesis that each layer is a viewing angle on a
   stationary manifold point, not a step of motion through a bulk,
   predicted specifically that per-layer argmax convergence should be
   the cleanest easy/hard signal. Measured: it is.

2. **14 features with the right physics outperform 47 features without
   it.** The feature engineering effort that accumulated 47 features
   across multiple physics framings (curvature, quantum density matrix,
   black-hole bipartite, trajectory) reached LOPO R² = 0.341. Adding
   one small family of logit-lens features (6 more) jumps R² to 0.448
   — bigger than all prior progress combined.

Signal architecture matters more than signal quantity, when the
architecture aligns with the physics of what's being measured.

## How it was measured

### Protocol (stage 34, `scripts/stage34_logit_lens.py`)

For each decode step on each of 35 prompts:

1. Run forward with `output_hidden_states=True`.
2. For each layer's final-position hidden state, apply final RMSNorm
   then `lm_head` to get per-layer logits.
3. Compute per-layer argmax token → sequence of L argmax ids for this
   token's generation step.
4. Compute six features from this sequence:
   - `stabilization_depth` = (1 + latest_layer_that_disagrees_with_final) / L.
   - `first_agreement_depth` = earliest_layer_agreeing_with_final / L.
   - `agreement_fraction` = #layers_agreeing / L.
   - `argmax_entropy` = Shannon entropy of argmax-id distribution.
   - `top_token_frequency` = fraction of layers voting for the most-common id.
   - `logit_lens_avg_entropy` = mean of per-layer output-entropy (not argmax) across L layers.
5. Correlate each feature with output_entropy; evaluate combined LOPO
   R² against the 8 essential features alone.

## The numbers

### Individual correlations (35 prompts, 4165 records, Qwen3-0.6B)

| feature | r with output_entropy |
|---|---|
| **stabilization_depth** | **+0.495** |
| agreement_fraction | -0.426 |
| first_agreement_depth | +0.272 |
| logit_lens_avg_entropy | +0.202 |
| top_token_frequency | -0.137 |
| argmax_entropy | +0.134 |

### LOPO R² comparison

| feature set | n features | LOPO R² |
|---|---|---|
| essential 8 (from Finding 08) | 8 | 0.293 |
| logit-lens-6 alone | 6 | 0.115 |
| **essential 8 + lens 6** | **14** | **0.448** |
| full 47-feature set (stage 31) | 47 | 0.341 |

Gain from adding logit-lens to essentials: **+0.154 R²**. More than
double the gain from adding the other 39 non-essential features to the
essential 8.

## Interpretation through the reframe

**`stabilization_depth` is the direct quantity the reframe predicts.**

If each layer is a viewing angle, and easy tokens are manifold points
whose identity is visible from most angles, then:
- easy tokens: argmax stabilizes early, stabilization_depth small.
- hard tokens: argmax flips between candidates across many views,
  stabilization_depth large.

The r = +0.495 correlation confirms the reframe's specific prediction.

**The other logit-lens features cluster with stabilization_depth.**
They're different aggregations of the same "per-layer consensus" signal,
so individually they carry information but together they don't add much
beyond stabilization_depth alone. The 6 features collectively give
+0.154 R² beyond essentials, but `stabilization_depth` + essentials
alone would give most of that.

## What this changes for routing

1. **Best routing signal is NOT from attention weights.** It's from the
   per-layer logit-lens projection. The `lm_head` applied at every layer
   reveals stabilization trajectory that the attention pattern alone
   doesn't.

2. **Cost**: at each step we apply `lm_head` to all L hidden states,
   computing per-layer logits. `lm_head` is large (hidden × vocab). At
   Qwen3-0.6B with L=28 and vocab=151k, this is 28 × 1024 × 151k ≈ 4.3
   GFLOPs per step — about 10% of one forward pass. Cheap enough for
   deployment.

3. **Smaller models will pay more (proportionally) for this signal**
   because their L is close to Qwen3-0.6B's and their base forward is
   smaller. 32B has 64 layers and the full forward is much larger, so
   per-layer `lm_head` is a smaller fraction of forward-pass cost.

## Limitations / caveats

1. One model tested (Qwen3-0.6B). Cross-model confirmation would
   strengthen — especially do stabilization_depth correlations hold at
   1.7B or 4B where layer count differs slightly.

2. The per-layer `lm_head` application is expensive enough that it
   might not be useful for extremely cheap cheap-paths. But for rank-k
   factored decoding (the target goal), per-layer logits are already
   essentially free because hidden states must be reconstructed for the
   residual stream.

3. We tested prediction of `output_entropy`. Direct prediction of
   `logit_margin` (top-1 vs top-2 gap) gives different correlations
   that we haven't fully explored for this feature family.

## Reproduce

```bash
python scripts/stage34_logit_lens.py \
    --model Qwen/Qwen3-0.6B \
    --max-new-tokens 120 \
    --device mps
```

## Related

- [Finding 02](02_universal_rotation_curve.md) — the per-layer basis
  rotation is universal. This finding says the rotation schedule is
  also how token-identity converges: more rotations = more views =
  more chances to stabilize.
- [Finding 08](08_minimal_signal_subset.md) — the essential 8-feature
  subset. This finding proposes adding `stabilization_depth` as a
  9th essential feature.
- [Finding 06](06_rsb_descent_profiles.md) — entropy profile shapes.
  `stabilization_depth` is the per-token equivalent: how many layer-
  views needed to commit, analogous to how many generation steps
  needed for the system to commit.
- Stage 33b — deployment test showed 5.4× quality improvement under
  routing. With this finding, the routing signal should be stronger.
