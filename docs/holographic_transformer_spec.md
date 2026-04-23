# Holographic Transformer — Design Spec (Future Work)

**Status**: design only. Not yet built or validated. Written as a
target once the compression matrix has enough data to make principled
lever choices.

## Motivation

Stages 86-91 attempted a "holographic oracle" — replace trained MLPs
with closed-form whitened ridge retrieval, or with nonparametric
Hopfield, or with drift-aware variants. All failed in the same way:
per-layer angular direction preserved (h_cos +0.60), but prediction
collapsed to 1-2%. We interpreted this as "the bilinear gate is load-
bearing," but the deeper read is: we were replacing *some* of what
the trained weights encode, and missing the rest.

The compression matrix (stages 107, 108, Strix Qwen Halo) exposes
what we were missing:

| lever | what it encodes | measured cost |
|---|---|---|
| Weight precision (bits) | retrieval precision per binding | Q8/Q6 universally cheap; Q4 cliff without fine-tune |
| Per-tensor α (learnable) | amplitude trajectory per layer | BitNet-proven: compensates Q4/ternary when trained |
| KV rank | effective hologram capacity | 14B: 10× free slack. 0.6B: zero slack. SCALE-dependent. |
| Embed bits | bind-side precision on token IDs | universally cheap, slight bonus on 14B |
| SwiGLU rank (SVD) | MLP expressivity | free up to d_model on 0.6B; untested at 14B |
| d_ffn (naive) | MLP width if no rotation | expensive at all sizes (first-k rows lose info) |
| Layer gate | which layers are active | untested but measurable (dead zone is 69-73% of layers) |
| Per-head α / # heads | parallel retrieval channels | **prediction: opposite lever to KV rank** (untested) |
| Head dim | precision per channel | coupled with # heads (shared budget) |

Each lever is a measurable compensation channel. The HRR transformer
failures were from trying to replace structure without including these
compensators.

## Proposed architecture

Holographic transformer with every compensation lever **explicit and
trainable**:

```
for each layer l:
    h_norm = RMSNorm(h)

    # HRR attention block
    # key/value projection with learnable low-rank (MLA-style)
    k = W_K_l(h_norm)                       # [B, T, d_kv * n_kv_heads]
    v = W_V_l(h_norm)
    q = W_Q_l(h_norm)

    # split into heads
    k = split_heads(k, n_heads_l)           # per-layer head count
    v = split_heads(v, n_heads_l)
    q = split_heads(q, n_heads_l)

    # build or retrieve from hologram state
    S = cumulative_outer_product(k, v)      # [B, T, d_q, d_v]

    # holographic retrieval
    y = S @ q.unsqueeze(-1)                 # [B, T, d_v]

    # AMPLITUDE LEVER: per-tensor α trained end-to-end
    y = alpha_attn_l * y

    # SHARPENER: soft Hopfield cleanup or silu gate
    y = cleanup_l(y)

    # merge heads
    y = merge_heads(y)
    h = h + y

    # HRR MLP block (or traditional SwiGLU with low-rank)
    h_mlp = MLP_low_rank_l(RMSNorm(h))
    h_mlp = alpha_mlp_l * h_mlp             # amplitude lever
    h = h + h_mlp

# Quantization-aware throughout:
# all weight matrices store fp32 master + forward uses QAT STE at bits_l
# lever (adaptive per-layer bit count)
```

Key features:

1. **Every compression axis explicit**: `bits_l`, `rank_l`, `n_heads_l`,
   `alpha_attn_l`, `alpha_mlp_l` are all learnable or assignable per
   layer.

2. **No wasted depth**: dead-zone layers (identified by manifold
   measurements) use aggressive compression; active layers stay high
   rank. The "bathtub" shape of both 0.6B and 14B manifolds suggests
   this is universal.

3. **Compensation baked in**: α scalars trained end-to-end are the
   BitNet-proven mechanism that translates precision loss into
   amplitude compensation.

4. **Measurement-initialized**: rank schedule per layer comes from
   the manifold measurement (stage 107 companions for each size). No
   uniform rank.

## Predictions (opposite levers, untested)

From HRR physics + our measurements, candidate compensating pairs:

| compress | boost | mechanism |
|---|---|---|
| KV rank ↓ | # heads ↑ | spread signal across more retrieval channels |
| weight precision ↓ | per-tensor α ↑ | BitNet's proven amplitude compensation |
| layer count ↓ | rank per layer ↑ | each remaining layer does more |
| embed precision ↓ | vocab-size ↓ | fewer tokens, each needing less precision |

Each pair is testable: fix axis A at a harsh setting, vary axis B, see
if quality recovers. The user's example — "**+128 heads + rank 16**
should balance" — is exactly this hypothesis in the KV/heads pair.

## Why this is different from what we tried in 86-91

| stage 86-91 | this design |
|---|---|
| Replace MLPs with oracle | Train jointly with oracle structure |
| Fixed unbinding (linear/Hopfield) | Trainable amplitude α per layer |
| Uniform rank attempt | Per-layer rank from manifold |
| No precision axis | QAT bits per layer |
| No explicit head count lever | n_heads per layer configurable |

Stage 86-91 asked: "can a pure holographic retrieval replace the
trained MLPs?" Answer: no, because the MLP weights carry more than
pure retrieval. This design asks: "what if the holographic retrieval
has every compensation the MLP has?" — and then every compensation is
a parameter, not a missing piece.

## Out-of-scope (what NOT to build yet)

- HRR cross-sequence attention (fancy variant; start with causal single-stream)
- Resonator networks (extra cleanup; start with silu)
- Complex-valued rotations (theoretically cleaner; defer)
- Hierarchical HRR binding (for syntactic trees; later)

## Prerequisites before building

1. **Complete marginal cost matrix** on 0.6B, 4B, 14B minimum.
2. **Opposite-lever validation**: at least one pair confirmed
   (heads ↔ rank, α ↔ bits, layers ↔ rank).
3. **Manifold measurement** on target size (we have 0.6B, 4B, 14B,
   32B, 30B-A3B, 1.7B, 8B, phi-2).
4. **Scaling law fit** on compression ratio vs model size so we can
   predict where this architecture should pay off.

## When to build

Once prerequisites 1-3 are complete (estimated 2-4 weeks of lever
exploration), build a stage 110+ that implements this spec at 0.6B
scale as proof-of-concept, then scale to 4B if promising.

## Expected payoff if it works

- Param-efficient: each compensation is ~d or d² params, not d × d_ffn
- GPU-friendly: dense matmul throughout, tensor cores usable
- Scale-independent compression: since every lever is explicit, gains
  compound instead of hitting size-dependent slack limits
- Physics-grounded: each design choice is measurement-derived, not
  benchmark-chased

Honest caveat: if opposite-lever pairs DON'T work (axes are all shared
budget, never compensating), this design falls back to standard QAT
transformer. Still useful, but the holographic framing doesn't buy
additional compression beyond what QAT alone gives.
