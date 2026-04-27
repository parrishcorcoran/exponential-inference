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
| [10](10_holographic_compressibility.md) | **Holographic compressibility: boundary vs bulk (Holographic Matryoshka)** | Names the project's technique. Every failed compression attacked the MLP bulk (intermediate dim); every successful one acted on the boundary (residual stream / KV / heads / rotations). Bulk dim is load-bearing holographic projection — cannot be reduced. Collapses the 3D slice to width + length; depth stays full. **Corrected scope (2026-04-20):** the technique is an *inference-time dynamic routing architecture* on the unfactored original model, not a trained weight compression. Trained rank-k factorization was tested at 0.5B, 3B, 8B, 14B — 0% match vs original at every size. Dynamic width (head masking) + length (early exit) + batch parallelism on the unfactored Qwen3-14B produces coherent text, 1.6×–3.0× wall-clock, 99× batch throughput. | stages 35/36/42 (fail), 04/09/33b/38 (succeed), Strix 14B dynamic routing (confirmed architecture) |
| [11](11_rg_quantum_flow.md) | **Forward pass is simultaneously RG flow and quantum measurement** | Six physics frames tested; only two survive and they agree. Frame 3 (RG flow to attractor): KL vs final decreases monotonically 10.86 → 0.00 across 24/27 transitions. Frame 6a (quantum measurement/pointer selection): purity rises 0.025 → 0.49, VN entropy falls 4.53 → 1.94, effective rank contracts 40 → 2. Falsified: fractal self-similarity (Δα=0.20), clean CFT scaling (α drifts 0.95→0.67), Parisi RSB (27/29 layers replica-symmetric), parallel transport (norms grow 656×). Describes the project's *dynamics* to complement Finding 10's *structure*. | stages 45, 46, 47, 48, 49 |
| [12](12_bitnet_scaling_mechanism.md) | **BitNet ternary scales via width × superposition** | Ternary weights at small d_model (0.6B, d=1024) lose information; at d≥4096 the wider superposition recovers fp-equivalent representational capacity. Ternary needs ~4× width to match fp-resolution. | stages 110-112 |
| [13](13_wormhole_residual_stream.md) | **Residual stream rank collapses in middle layers** *(originally framed as "wormhole"; superseded by Finding 22's per-layer topography view)* | At 0.6B and 14B, mid-layer participation ratio collapses to near 1 with high-rank "mouths" at edges. The collapse is real per-layer; the global "wormhole" framing is not universal across architectures (see Finding 22). | stages 113-117 |
| [14](14_universal_geometry_private_decoder.md) | **Throat coordinates are universal; mouth-2 decoder is private** | Mid-layer (low-rank) representations are tokenizer-invariant across model sizes (R²=0.93 at throat). The exit-side mouth (last 1/3 of layers) is model-private (R²=0.08). The compressed channel is universal; the unpacker is local. | stages 121-123 |
| [15](15_two_gate_wormhole.md) | **Entry/exit gates frame the throat region** *(formerly "two-gate wormhole")* | The low-rank middle is bounded by an entry gate (~L5 on 0.6B, rank ~140) and an exit gate (~L21, rank ~700 — hardest layer). The throat is not a uniform tunnel but a corridor between two compression-resistant walls. | stage 127 |
| [16](16_kv_cache_field_geometry.md) | **KV cache is angularly uniform, scale-free, non-conservative** | The K-cache geometry is uniform across layers in 14B (different from 0.6B), with high Gini in attention scores. K-cache behaves as a non-conservative scalar field. | stages 128-134 |
| [17](17_post_hoc_projection_floor.md) | **Post-hoc projection has a floor; trained-aware compression breaks through** | Naive low-rank projections of a trained model hit a quality floor. The fix is making the model aware of compression during training (or fine-tuning into the rank-restricted subspace). | stage 135 |
| [18](18_compression_topography.md) | **KV-cache compression has 5+ independent axes** | K rank, V rank, bits, clustering, attention Gini — each has different per-layer shape and they're orthogonal. Stack multiplicatively; projected 100–300× cache compression. | stages 136-139 |
| [19](19_certainty_growth.md) | **Output entropy decreases through generation; per-position compression schedule** | Average across 5 sequences: output entropy 4.00 → 2.71 nats (−32%), top-1 confidence 0.350 → 0.443 (+27%), attention Gini 0.674 → 0.922 (+37%). The model gets more committed late; that commitment is a principled per-position compression budget signal that replaces H2O's heuristic. | stage 139 |
| [20](20_bitnet_wormhole_universality.md) | **K-axis topography appears on BitNet b1.58-2B** *(formerly "BitNet wormhole universality")* | The K-axis low-rank middle, sharper than fp16, appears on ternary models too. The structure is magnitude-driven, not precision-dependent. | stage 142 |
| [21](21_moe_wormhole_sharper.md) | **K-axis topography on Granite MoE: 17-layer cavity** *(formerly "MoE wormhole sharper")* | The K-axis low-rank region is sharper on MoE models — a 17-layer cavity. MoE expert routing produces a sharper middle compression than dense. | stage 142b |
| [22](22_topology_revision_no_universal_wormhole.md) | **Topology revision: no universal wormhole** | Per-layer cavity anneal on 14B reveals scattered cavities, not a clean throat. The "global wormhole" framing was overstated; the actual structure is per-axis, per-layer topography. Supersedes findings 13/15/20/21's wormhole language. | stage 143 |
| [23](23_dual_layer_sweep_pairs_dont_help.md) | **Dual-layer KVQ probes don't beat single-layer; K/V peak at L14, Q at L15** | Closed-form ridge regression over 378 layer pairs in Qwen3-0.6B for predicting K, V, Q at target L14, offset +1. Single-layer matches every dual-layer pair within noise (Δ +0.002 for V, +0.000 for K, -0.044 for Q). The residual stream is additive (h[L+1] = h[L] + Δ) so concat doesn't add information. K/V peak at L14 (their compute layer); Q peaks at L15 (next-position propagation). | Z8 G4 closed-form sweep |
| [24](24_k_as_unbinding_key.md) | **K is the literal HRR unbinding key; tokens decode off the K-manifold** | At Qwen3-0.6B layer 14, real cached K alone decodes to the correct token at 56.0% top-1 via a frozen LM head; real Q decodes at 61.5% top-1, 78.5% top-5. K-Medusa heads predict K with cos 0.74-0.81 uniform across 10 offsets (replicates Strix's 14B finding). Joint head+decoder MSE+CE training closes the predicted-K → token gap (21% / 46% top-1/top-5 at offset 1) — the methodology that turns the structural finding (#23) into something that reads tokens off the manifold. | stages 144-159 |

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
