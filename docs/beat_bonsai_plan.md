# Beat-Bonsai Plan — Realistic Path to <1.58 bits Lossless

Bonsai's published number: 89% downstream retention at ~1.25 bits/weight (binary + per-128-group scale + bias). This document lays out the concrete path from where we are today to crossing that bar — and beyond, toward true lossless.

## Where we are right now

```
✅ Lossless ops library (rotation, permutation, magnitude factoring, per-group rotation)
✅ Validation harness (val CE, K1 residual, intra-row CV, Δ=0 verify)
✅ Hardware: Z8 + 2×V100 (32GB) + 780GB RAM
✅ Coherency-test discipline (capability damage detection)

❌ Structural additions untested (the high-leverage move)
❌ K=1 projection never run with full preconditioning chain
❌ QAT distillation loop not built (only ablation training so far)
❌ Capability evals not wired up (need MMLU, GSM8K to compare to Bonsai's 89%)
❌ No 7B replication, no 14B at all
```

Stage 209 narrowed the answer: per-Linear and per-group lossless ops sit at the Gaussian floor (CV ≈ 0.754, K1 residual ≈ 0.600). The library-only path is closed at g=128. All remaining probability mass is in **structural additions** and **group-size sweep**.

## The 6-stage path

### Stage A — Structural additions probe (Week 1, 0.6B)

The single most-informative experiment we can run.

- Zero-init per-output bias on every targeted Linear (~14K DOF on 0.6B)
- Zero-init per-group bias per row — Bonsai's exact lever (~600K DOF)
- Verify Δ=0
- Measure: does K1 residual drop when adapters absorb K=1 reconstruction error?

**Decision gate:** If structural additions don't help K1 residual, the recipe is fundamentally broken and we pivot. If they do, we proceed.

### Stage B — Group-size sweep (Week 1-2, 0.6B)

Find the bit-budget sweet spot.

- g ∈ {32, 64, 128, 256}
- Measure K1 residual at each, with full preconditioning
- Pick g that crosses 99% retention
- Most likely answer: g=64 (1.5 bits) if g=128 stalls

### Stage C — Full preconditioning + K=1 projection (Week 2-3, 0.6B)

Stack everything that works:

- SmoothQuant migration → Hadamard rotation → per-group rotation → magnitude factoring → permutation → structural adapters
- Then actually project to K=1 binary
- Measure CE drift WITHOUT training (pure projection)

**Decision gate:** if pre-training CE drift is <0.1 nats, training will converge. If >0.5 nats, we need more structure.

### Stage D — QAT distillation loop (Week 3-6, 0.6B)

Bonsai's actual training recipe.

- KL divergence to FP teacher (Qwen3-0.6B-base)
- ~500M tokens (Z8 can do this in days at 0.6B)
- Best-state checkpointing on val CE + coherency probe
- Body weights binary; structural adapters fp16

**Decision gate:** if 0.6B retention >90% on val CE, we beat Bonsai's 89%. Wire up MMLU/GSM8K to confirm.

### Stage E — Scale validation at 7B (Week 7-10)

~10 days/anneal on Z8. ONE run.

- Same recipe, scaled token budget proportionally
- **Decision gate:** if 7B retention ≥ 0.6B retention, recipe scales. Continue. If 7B drops materially, debug before 14B.

### Stage F — 14B campaign (Week 11-30)

Where we beat Bonsai publicly.

- 3 weeks per anneal, plan for 3-4 anneals = recipe iteration room
- Full multi-metric eval: MMLU, GSM8K, HumanEval, ARC, HellaSwag
- The number that matters: **average downstream retention**, not perplexity alone
- Bonsai's 89% is a downstream-benchmark number — match the comparison

## Compute budget on Z8

| Stage | Cost / run | Runs | Total GPU-hrs |
|-------|-----------|------|---------------|
| A     | ~2 hr     | 5    | 10            |
| B     | ~3 hr     | 4    | 12            |
| C     | ~1 hr     | 1    | 1             |
| D     | ~3 days   | 5    | 360           |
| E     | ~10 days  | 1    | 480           |
| F     | ~21 days  | 4    | 4032          |
|       |           |      | **~4900 GPU-hrs** |

That's ~56% utilization of the year on 2×V100. Comfortable, with headroom for ablations.

## Probability tree

```
P(Stage A shows K1 drop)              ≈ 55%
P(Stage C K=1 projection < 0.5 nats)  ≈ 60% | A worked
P(Stage D QAT converges to >89%)      ≈ 75% | C looked good
P(7B replicates)                       ≈ 80% | 0.6B works
P(14B beats Bonsai on downstream)     ≈ 85% | 7B works

Joint: 0.55 × 0.60 × 0.75 × 0.80 × 0.85 ≈ 17%

Plus partial-credit paths (1.5 bits at g=64 instead):
        + ~30% (alternative path through Stage B)
                                              ≈ ~50%
```

**~50% probability of beating Bonsai's 89% retention at <1.58 bits within 30 weeks**, conditional on executing through Stage F.

## The single most-important move right now

**Stage A.** Run it this week. Tells us in <1 day whether the structural-additions hypothesis is alive. Everything downstream depends on it.

## Missing infrastructure beyond stages

Capability eval harness:

- Wire up `lm-eval-harness` against MMLU, GSM8K, HumanEval, ARC
- Run on FP baseline first → establish what Qwen3-0.6B base scores
- This becomes the metric we report. Bonsai's 89% = average downstream, not perplexity.

## Hardware-conditioned size sweet spot

Memory is not the constraint on Z8 (780GB RAM enables ZeRO-3 offload up to ~150B fp16). Compute is.

| Size | Time/anneal (Z8) | P(<1.58 lossless, 1 yr) |
|------|------------------|-------------------------|
| 0.6B | hours            | 24%                     |
| 7B   | ~10 days         | 45%                     |
| **14B** | **~21 days**  | **~60%** ← sweet spot   |
| 32B  | ~50-60 days      | ~52%                    |
| 70B  | ~140 days        | ~30%                    |

**14B is the sweet spot for Z8 + 2×V100 hardware.** Above 14B, compute crushes faster than scale buys theoretical possibility.

## Commercial framing

Beating Bonsai by 1-3pp is a paper, not a product. The probability tree for $1M+ outcomes:

```
P(beat Bonsai)                            ≈ 50%
P(true lossless <1.58 | beat Bonsai)      ≈ 70%
P(true lossless <1.25 | lossless <1.58)   ≈ 55%

P($1M+ outcome | lossless <1.58)          ≈ 60% (talent market dominates)
P($1M+ outcome | lossless <1.25)          ≈ 85% (acquihire / IP license viable)
```

What actually unlocks the $$:

1. Reproducibility across architectures (Llama, Mistral, Qwen) — Apple wants Llama
2. Real hardware speedup — a Triton/CUDA kernel for 1.25-bit inference
3. Quality on downstream tasks (MMLU, GSM8K, HumanEval), not just perplexity
4. Open-source release timing — secured offer/funding before public release

The single highest-leverage thing alongside research: **a Triton kernel for 1.25-bit inference**. Hyperscalers buy deployable IP, not papers.
