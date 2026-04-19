# Finding 10 — Holographic compressibility: boundary vs bulk

This finding names the project's core technique: **Holographic Matryoshka**.
It is the training-and-inference architecture that follows from this
principle — nested rank-k factoring of boundary weights (Matryoshka),
preserving bulk dim (holographic), plus per-token dynamic early-exit on
length.

## The claim

The transformer has two kinds of state, and they compress differently:

- **Boundary** = the manifold projection. Residual stream (hidden dim),
  K/V cache, embedding, lm_head. These live on the ~10-dim manifold and
  are compressible by rank-k factorization.
- **Bulk** = the MLP intermediate activation space (d_int = 3072 for
  Qwen3-0.6B, 9728 for Qwen3-4B). This is the compute medium that
  expands the manifold coordinates into a wide operational space
  where the nonlinearity (SiLU × up) operates, then projects back.

**The bulk dimension count cannot be reduced.** It is the holographic
projection required to materialize each rotation step. Reducing it,
pruning positions within it, or factoring its output to a narrower
intermediate breaks generation in the same way.

The bulk's *operational rank* (how many independent directions the
bulk expresses on a given token) CAN be restricted via boundary rank-k
factoring — that's what Matryoshka distillation does. The bulk's
*dimension count* cannot.

## Why it's a stop-and-think

**Every failed compression attempt in this project has been a bulk
compression. Every successful one has been a boundary compression.**
This single principle predicts the outcome of each attempt in advance.

We did not see the pattern until we had the holographic framing. Once
you see it, the three compression axes of the 3D slice architecture
collapse to two (width and length), and the architecture simplifies:
keep bulk dim full at every layer, compress only boundary rank and
rotation count.

The intuition is the same as the Bekenstein–Hawking formulation of
black-hole thermodynamics: the amount of information in a region is
bounded by the area of its boundary, not the volume of its bulk. The
bulk's role is to materialize the reconstruction that the boundary
encodes; the bulk's *dimension* is part of what that reconstruction
requires.

## Evidence — every attempt cleanly partitions

### Failed compressions (all attacked the bulk)

| stage | what we tried | bulk-attack mechanism | result |
|---|---|---|---|
| 35 | rank-k MLP factoring on untrained 0.6B | factored gate/up/down through PCA — restricts bulk rank below the manifold floor | 1-9/100 match across ranks 32-512 |
| 36 | rotation-native narrow MLP | reduced intermediate dim from 3072 to k | 0/100 match across all k |
| 42 | oracle top-k intermediate pruning | dropped bulk positions by |int_act| | 2×-compression → 4/80 match; worse at 20× |

Stage 36 is the sharpest falsification: it rebuilt the MLP with
intermediate dim = k instead of 3072 (compressing d_int itself), and
produced garbage at every tested rank up to 512. Stage 42 showed the
same under the harshest assumption (oracle per-token pruning of the
bulk) — keeping only the top-50% of bulk positions by |int_act| still
broke output.

### Successful compressions (all acted on the boundary)

| finding/stage | what we compressed | boundary-aligned mechanism | result |
|---|---|---|---|
| 04 | 80-83% of attention heads skipped | heads are rotation-specialists bolted onto the residual stream | 100% token match at 80% skip |
| 09 / 33b | early-exit at stabilization_depth | skipping future rotations, not bulk | 5.4× quality preservation under routing |
| 38 | KV cache rank-128 (8× compression) | K/V are boundary projections of hidden state | coherent output, 8× memory win |
| Matryoshka (planned) | rank-k factored weights above the manifold floor | rank-restricts bulk via boundary factoring, does not reduce d_int | untested at scale, infrastructure ready on Strix |

**The boundary/bulk partition predicts every prior result.** It also
resolves the apparent failure of Matryoshka at 0.6B (stage 15): that
failure was the manifold floor, not a holographic violation. At
sufficient rank and above the floor, Matryoshka does not reduce bulk
dim — it only restricts operational rank, which is compatible with the
principle.

## The sharper distinction between bulk dim and bulk rank

This matters to avoid re-conflating.

- **Bulk dim** = number of intermediate positions (d_int = 3072).
  **Cannot be reduced.** Pruning positions (stage 42), narrowing the
  intermediate (stage 36), or approximating the MLP by a low-rank
  tensor decomposition that changes d_int all break the hologram.
- **Bulk rank** = the effective rank of the intermediate activation
  across tokens. If the MLP's inputs live on a rank-k subspace of the
  d_model-wide residual stream, the bulk activations lie in a rank-k
  restricted *subspace of the full d_int space* — but they use all
  d_int positions. This is what Matryoshka-factored MLPs do, and it
  works above the manifold floor.

Physically: d_int positions are non-negotiable, but you don't need all
d_int directions to be independent. The bulk's job is to span the
output, which a rank-k signal through all d_int positions can do
(assisted by the SiLU non-linearity, which partially re-expands rank).

## What this changes for the architecture

**The 3D slice collapses to 2D + 1 dynamic feedback signal:**

| axis | what it is | compressible? | mechanism |
|---|---|---|---|
| width | residual stream rank | yes — via Matryoshka boundary factoring | trained nested rank-k |
| length | number of rotations used | yes — per-token early exit | stabilization_depth from Finding 09 |
| **depth** | **MLP intermediate dim count** | **no — bulk is load-bearing** | **full d_int at every layer** |

"Depth" as we've been using it in the 3D slice framework is not a free
axis. It must stay at teacher's d_int.

This simplifies the training target and aligns with existing
Matryoshka machinery on Strix Halo. The script at
`machines/strix_halo/scripts/train_matryoshka.py` factors every Linear
including gate/up/down — this is holographically correct **because the
factoring preserves d_int**; it only restricts operational rank. The
student's intermediate activations still occupy all d_int positions;
they just lie on a rank-k subspace within that space.

## Related compressions, in this frame

- **Head pruning** (Finding 04) is boundary — each attention head
  writes into the residual stream, so skipping a head skips a rotation
  contribution on the boundary. Bulk (d_int) is untouched. Predicts
  high compressibility.
- **KV compression** (stage 38) is boundary — K and V are projections
  of the hidden state onto attention-key/value subspaces. They live
  on the boundary. Predicts ~8× compression, which matches.
- **Early-exit / length compression** (Finding 09 / stage 33b) skips
  future rotations entirely — the bulk doesn't run at all for skipped
  layers. This is a length-axis operation, not a bulk operation. Works.

## What this rules out

Several directions that looked promising before Finding 10 now predict
failure:

- Depth-sparse MoE where experts span narrow sub-pieces of d_int
  per layer (short-fat slices in our 3D framework). The expert itself
  reduces bulk dim at that layer — bulk compression. Predicted to fail.
- Intermediate-dim Matryoshka (nested *bulk* rank with d_int varying
  per step). Fails for the same reason.
- Oracle-guided bulk pruning at runtime, even with a perfect predictor
  of which positions matter per token. Stage 42 tested the oracle and
  confirmed failure.

## Limitations / caveats

1. Tested on 0.6B only. The principle should hold at 4B+ (manifold
   floor argument), but Matryoshka training there has not yet been
   run. Strix Halo infrastructure is ready.
2. The "holographic" framing is a physics metaphor that fits the data
   well, but the exact correspondence (what is the "black hole" for a
   transformer?) is not formalized.
3. The sharp line between boundary and bulk assumes the MLP
   intermediate is uniquely the bulk. Attention's QK^T intermediate
   *might* also be bulk, but we haven't tested compressing it directly
   (Finding 04 showed head skipping works, which is a different kind
   of compression).

## Reproduce

The failures and successes that support this finding are already
reproducible via scripts 35, 36, 38, 42 and the earlier Finding 04/09
stages. No new script is needed — this is a re-reading of the
combined evidence.

For the forward-looking test, the Matryoshka training that acts on
boundary rank (preserving d_int) is in
`machines/strix_halo/scripts/train_matryoshka.py`.

## Related

- [Finding 01](01_universal_manifold_dim.md) — the ~10-dim manifold
  is the boundary whose compressibility this finding explains.
- [Finding 05](05_manifold_floor.md) — Matryoshka failures at 0.6B
  are now explained as below-floor, not holographically ruled out.
- [Finding 04](04_head_pruning_redundancy.md) — head-prunability
  follows from heads being boundary-aligned.
- [Finding 09](09_logit_lens_view_stabilization.md) — length-axis
  compression (stabilization-based early exit) is the second live
  compression axis alongside width.
