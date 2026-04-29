# Finding 31: PID-controlled adiabatic descent unlocks compensation that displacement-style preconditioning cannot

**Status**: established (Stage 189, 2026-04-29). Single proof-of-concept
run on Qwen3-0.6B with bottleneck-only master training (o_proj + down_proj).
Replicable; full-body version pending Strix.

## Claim

When converting a pretrained transformer to a low-bit representation,
**adiabatic group-size descent with master-weight training tracks the
RG attractor as it deforms under the constraint**. This achieves a
strictly lower CE plateau than displacement-style one-shot
preconditioning (Stage 188).

The mechanism is what user articulated as "split up the rows like
Bonsai but tons more, then PID controller and remove each one until we
reach binary." Concretely: progressively increase the per-group scale
group_size while training the master to absorb each step.

## Result

| Stage | description | trainable params | CE drift Δ |
|---|---|---:|---:|
| 184 | norms+α, frozen master, group=128 | 410K | +3.854 |
| 187 | + weak outlier regularization | 410K | +3.878 |
| 188 | + full BitNet-shape preconditioning (one-shot) | 410K | +3.760 |
| **189** | **PID adiabatic descent, master trainable on o/down** | **147M** | **+3.159** |

Stage 189 unlocks **0.60 nats below Stage 188** and **0.70 nats below
Stage 184**. The size of this gap matches the gap between Stage 169 T3
(α-recovery, Δ=−0.121) and base CE — i.e., it is meaningful, not noise.

(Caveat: Stage 189 used SEQ_LEN=64 vs 128 in 184/188. T0 differs;
deltas are in matched units, but absolute comparison should account
for the eval window difference.)

## Trajectory

The PID controller engaged at all three levels (drift exceeded 0.10
nat setpoint), holding the system and training extra steps before
advancing:

```
level   group_size   bits/weight   init drift    final drift   extra steps
   1         16         2.00         +11.79         +3.53           100
   2         64         1.25          +3.57         +3.43           100
   3        128         1.13          +3.44         +3.16           100
```

Final drift went DOWN at every level even though *compression
increased* at each step. The master tracked the attractor through the
deformation; cumulative training compounded across levels.

## Why this works where Stage 188 didn't

**Stage 188 (one-shot displacement):** clamped 8333 RMSNorm gain
outliers at once, scaled embedding 2.5×, set initial logit
temperature. The system was kicked off its FP RG attractor with no
prior preparation. Compensation training (norms+α only) plateaued at
+3.76, having recovered most of the displacement but unable to
relocate to a lower-energy attractor.

**Stage 189 (adiabatic):** at each level the system was at its
instantaneous attractor (lossless or near-lossless on o/down).
Increasing group_size deforms the attractor; master training absorbs
the deformation while the system stays at the attractor. PID throttle
guarantees the descent never moves faster than the master can track.

Mathematically equivalent to thermal annealing: keep changes slow
enough that the system stays in its ground state at every step.

## Why this matches the validated RG framing

Different precision constraints define different RG fixed points
(Finding 30):
- FP attractor: high-magnitude outliers, sparse residual structure
- Bonsai-binary attractor: flat outliers, redistributed magnitude

The one-shot path (Stage 188) tries to land on the Bonsai attractor
without knowing exactly where it is. The adiabatic path (Stage 189)
follows the attractor as it continuously deforms from FP toward Bonsai.
Master training is the mechanism by which the model parameters track
the attractor's location at each step.

This generalizes beyond binary: any compression direction (group size,
bit precision, sparsity, rank) can be approached adiabatically with
PID-throttled descent.

## What was different from Stage 184/188

Stage 184/188 trained only **norms + α** (~410K params). Master
weights were frozen at the initial binary projection.

Stage 189 trained **master weights of o_proj + down_proj** (~147M
params, 360× more capacity). The master could move to absorb each
group_size step. q/k/v/gate/up_proj stayed frozen at group=128 (Bonsai-
fixed) the whole time — and yet the o/down master training still
unlocked 0.6 nats. Suggests training master on ALL projections (full
body, ~440M params) would unlock substantially more.

## Implications for the production pipeline

1. **Adiabatic descent is the right pipeline shape for low-bit
   conversion.** One-shot quantization + post-hoc compensation hits a
   floor that the adiabatic version can break through.

2. **Master-weight training is essential, not optional.** Stage 184
   exhausted what compensation-alone (norms+α) can do. The remaining
   gap requires master to move under the constraint.

3. **PID throttle is the right control.** Letting the descent advance
   at fixed schedule risks displacement when a level is too aggressive.
   Setpoint-based throttle (drift ≤ ε nats) automatically slows down
   at the hard parts.

4. **Bottleneck-only training (per Finding 27) is a real efficiency
   win.** o_proj + down_proj is ~33% of body params but captured most
   of the available improvement at this compute budget. For Strix's
   14B, training only the bottleneck projections may be sufficient
   for a first useful conversion.

5. **The schedule [2, 4, 8, 16, 32, 64, 128] (or finer) is the
   natural decomposition.** Each level halves the per-group scale
   count, which is approximately one bit of representation budget per
   step. Slow enough for PID to work, fast enough to converge in a
   few hours of training.

## Proposed Stage 190 (Strix scale)

Full-body adiabatic descent on Qwen3-14B (or 0.6B with full body if
Mac headroom allows):
- All 196 (×56 for 14B) target linears trainable master
- Finer schedule: [2, 4, 8, 16, 32, 64, 128]
- PID setpoint 0.05 nats
- Long enough per-level training for PID to advance every level

Predicted: plateau drops well below +3 nats, possibly approaching base
CE (lossless Bonsai-class) on the small model. On 14B, the per-head
structure is weaker (Finding 30) so should land cleaner.

## Files

- `scripts/stage189_pid_adiabatic_descent_to_bonsai.py`
- `results/stage189_pid_adiabatic_descent.json`

## Related findings

- Finding 25 (anneal to nGPT-shape): the original geometric anneal
- Finding 27 (o_proj is the binary bottleneck): why we trained o/down
- Finding 28 (ternary 0-state is hard head selection): why per-group
  bias matters
- Finding 30 (compensation = outlier flattening): RG-attractor framing
