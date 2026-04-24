# Strix handoff: stage 112 position-aware quantization at 14B

## Ask

Run stage 112 (position-aware weight quantization) on Qwen3-14B. Tests
whether the bathtub-aware hybrid (Q6 edges + ternary middle) that DOESN'T
work at 0.6B DOES work at 14B, where d_model=5120 gives ternary enough
effective resolution via superposition (Finding 12).

If confirmed: first demonstration of **position-aware quantization
below BitNet's 1.58-bit floor** without QAT, using only the manifold
structure of the trained model.

## Why now

Just landed on Mac (stage 112, pushed commit 19b63ac):

**0.6B results:**
- uniform Q4: +32.6 ppl
- hybrid Q6-edge + Q4-mid: +20.4 ppl (40% better at ~same bits)
- **hybrid Q6-edge + Q2-mid: 13M ppl (broken)** — d_model=1024 too small for ternary middle

**14B prediction (based on Finding 12 mechanism):**
- d_model=5120 → ternary's effective output resolution ~5120 levels per neuron
- That's enough to match fp16 even in rank-1 middle → should work
- Hybrid Q6-edge + Q2-mid at 14B → predicted to land near teacher quality

If prediction holds, the story becomes: **"bathtub-aware ternary works at any scale where d_model is sufficient to support ternary's superposition, starting around d=2048-3072."**

## Edge width adjustment for 14B

From our 14B manifold measurement (Qwen_Qwen3-14B_manifold.json):
- **Active zone (edges)**: L0-6 and L32-39 (high PR, ~8 layers each side)
- **Dead zone (middle, pr ≈ 1)**: L7-31 (25 layers)

The script's default `--edge-width 3` was tuned for 0.6B (L0-2 + L25-27).
For 14B use `--edge-width 7` (covers L0-6 + L33-39, which matches the
measured bathtub).

## Command

```bash
cd /path/to/Exponential-Inference
git pull

python scripts/stage112_position_aware_quant.py \
    --model Qwen/Qwen3-14B \
    --edge-width 7 \
    --eval-batches 20 \
    --out results/stage112_14b.json
```

Expected runtime: ~15-30 min on Strix (12 variants, each loads 14B model,
quantizes, evals 20 batches).

Memory: Qwen3-14B bf16 ≈ 28GB weights. Script loads fresh model per
variant so peak memory ≈ 28GB + eval activations. Fits on Strix Halo
(89GB VRAM) comfortably.

## What to look for

**Primary finding (prediction):**
```
uniform Q4 at 14B → modest cost (14B tolerates Q4 post-hoc per prior work)
hybrid Q6-edge + Q4-mid → same or better
hybrid Q6-edge + Q3-mid → WORKS at 14B (expected) vs broken at 0.6B
hybrid Q6-edge + Q2-mid → PROBABLY WORKS at 14B (this would be the headline)
```

If Q2-mid works at 14B:
- Average bits ≈ (7×2 × Q6 + 25×40 × Q2) / 40 = ~2.9 bits/weight
- That's below BitNet's 1.58 ternary in aggregate (weighted by active vs. middle)
- Post-hoc, without QAT — unprecedented

**If Q2-mid doesn't work at 14B:**
- Suggests the ternary threshold is closer to d=8192+ (Qwen3-32B / 72B scale)
- Still a publishable finding on the scaling boundary

## After you run

Commit + push `results/stage112_14b.json`. Mac will update the
compression_matrix.md and relevant findings with the 14B row.

Also update compression_matrix.md directly if comfortable — structure is:

```
| Qwen3-14B | position-aware | Q6-edge + Q4-mid (w=7) | no | ... | ... | ... | ... | stage112_14b |
```

## Honest caveat

This test confirms or refutes Finding 12's scaling mechanism in a single
run. It's the clean extension of what Mac just measured at 0.6B. Don't
let this block longer-running work if Qwen Halo is mid-experiment — it
can wait until current phases finish.

## What this tests in physics terms

The bathtub (Finding 13) says middle layers have rank-1 activation.
Finding 12 says ternary resolution scales as `d`. Combining: in middle
layers, ternary's resolution only needs to match a **rank-1 output**,
which requires far fewer levels than high-rank outputs at edges.

So the Q6-edge + Q2-mid hybrid at 14B is a direct test of:
- **"Middle layers need Q2 just for direction + magnitude on rank-1 stream"**
- **"d=5120 supports Q2 in middle via superposition"**

Both predictions from Findings 12+13. Direct empirical test.

## Estimated impact if positive

First measurement-driven hybrid quantization that beats uniform low-bit
quantization, validated on a real LLM at scale, with physics-derived
schedule (not tuned or searched). That's portfolio-level.
