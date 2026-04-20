# Master findings

This is the archive of established findings that should make a
researcher stop and think. Each has independent measurement, reproducible
protocol, and implications that extend beyond the measurement itself.

If you're here from the repo README, these are the results we want you
to see before anything else.

## The roster

| # | finding | short version | stage(s) |
|---|---|---|---|
| [01](01_universal_manifold_dim.md) | **Per-tokenizer manifold dimension** | Within a tokenizer family (7 Qwen-family models, 0.6B–32B, dense/MoE/ternary) the final-layer TwoNN lands in 9.07–10.89. Suggestive but not dispositive evidence of cross-tokenizer universality. | stage 1 |
| [02](02_universal_rotation_curve.md) | **Universal rotation curve shape** | The per-layer basis rotation, normalized to [0,1] depth, has the same curve shape across tokenizer families (Pearson r > 0.97). The rotation schedule is a transformer-LM constant. | stage 19–21 |
| [03](03_universal_phase_transition.md) | **Universal phase transition at layer 0→1** | Every model's biggest basis rotation is at the embedding-to-first-transformer-layer boundary. Same location across sizes and tokenizers. | stage 20 |
| [04](04_head_pruning_redundancy.md) | **80–83% of attention heads are redundant** | Dynamic head pruning via attention sharpness skips 80–83% of heads with 100% token match on held-out generation. Number of active heads tracks the manifold dim. | stage 5 |
| [05](05_manifold_floor.md) | **The manifold floor (size-independent minimum)** | Rank-k factored compression has a parameter-count floor (~80–160M params for the Qwen tokenizer-induced manifold) that is approximately size-independent. A model must have enough factored capacity to clear this floor regardless of its full-size parameter count. | stages 8/10b/13/15 |
| [06](06_rsb_descent_profiles.md) | **Four canonical entropy descent profiles** | Attention entropy during generation clusters into four archetypes: monotone-decline, bell, plateau, and mid-generation spike. These correspond to descent types through an RSB-hierarchical energy landscape. Reasoning prompts produce the most saddles. | stages 4/F |
| [07](07_easy_token_classifier.md) | **Token-difficulty routing signals under honest validation** | 47 runtime features predict output entropy at cross-prompt LOPO R² = 0.341 (78% of the h_final PCA ceiling). Reasoning prompts are a systematic exception (R² = 0.21). The naive random-split R² (0.47) inflates by ~28%; linear regression generalizes honestly, MLP overfits. | stages 24/30/31 |
| [08](08_minimal_signal_subset.md) | **Minimal 8-feature orthogonal subset captures 80% of full** | Greedy LOPO selection over 47 features reveals that 8 features (each from a different physics framing: quantum, boundary, trajectory, angular, density, interaction, manifold locality, depth bipartite) reach LOPO R² = 0.272 — 80% of the full set's 0.341. Each axis is orthogonal; no physics family alone contributes multiple essential features. | stage 32 |
| [09](09_logit_lens_view_stabilization.md) | **Per-layer logit-lens stabilization is the strongest single predictor** | Under the "layer-as-rotation" reframe, each layer is a different viewing angle on the same invariant manifold point. The depth at which per-layer argmax stabilizes (r = +0.495 with output entropy) beats every other single signal. Adding 6 logit-lens features to the 8 essentials bumps LOPO R² from 0.293 to 0.448 — bigger than all 39 prior non-essential features combined. | stage 34 |
| [10](10_holographic_compressibility.md) | **Holographic compressibility: boundary vs bulk (Holographic Matryoshka)** | Names the project's technique, now empirically confirmed on Qwen3-14B (100% token match at k=32, 160× compression, Strix Halo). Every failed compression attacked the MLP bulk (intermediate dim); every successful one acted on the boundary (residual stream / KV / heads / rotations). Bulk dim is load-bearing holographic projection — cannot be reduced. Bulk rank CAN be restricted via boundary Matryoshka factoring. Collapses the 3D slice architecture to width + length; depth stays full. | stages 35/36/42 (fail), 04/09/33b/38 (succeed), Strix 14B run (confirmed) |
| [11](11_rg_quantum_flow.md) | **Forward pass is simultaneously RG flow and quantum measurement** | Six physics frames tested; only two survive and they agree. Frame 3 (RG flow to attractor): KL vs final decreases monotonically 10.86 → 0.00 across 24/27 transitions. Frame 6a (quantum measurement/pointer selection): purity rises 0.025 → 0.49, VN entropy falls 4.53 → 1.94, effective rank contracts 40 → 2. Falsified: fractal self-similarity (Δα=0.20), clean CFT scaling (α drifts 0.95→0.67), Parisi RSB (27/29 layers replica-symmetric), parallel transport (norms grow 656×). Describes the project's *dynamics* to complement Finding 10's *structure*. | stages 45, 46, 47, 48, 49 |

## Why these (and not others)

Each finding meets three criteria:

1. **Reproducible**: a single script in `scripts/` produces the numbers.
2. **Surprising**: it contradicts or substantially refines a prior
   commonly held in the field.
3. **Actionable**: it implies a specific engineering or theoretical
   move. Not just an observation.

Other results in the repo (distillation preserves TwoNN, text-weighted
embedding matches activation dim, corpus partial-invariance) are
interesting supporting measurements but don't individually clear the
"stop-and-think" bar.

## Reading order for an external reviewer

1. This index.
2. [Finding 01](01_universal_manifold_dim.md) — the flagship.
3. [Finding 04](04_head_pruning_redundancy.md) — strongest inference-side result.
4. [Finding 05](05_manifold_floor.md) — explains why naive experiments fail.
5. [Findings 02, 03, 06](02_universal_rotation_curve.md) — the follow-ups that
   tighten the framework into something deployable.

## Adding to this archive

New findings belong here if they:
- Are independently measurable with a committed script.
- Are confirmed on at least two models OR predict something subsequently
  observed.
- Change how we'd build the system.

Proposals that aren't yet findings (marked "open" in
`docs/research_context.md`) live elsewhere until confirmed.
