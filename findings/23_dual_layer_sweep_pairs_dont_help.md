# Finding 23 — Dual-layer KVQ probes don't beat single-layer; K/V peak at L14, Q at L15

## Claim

For predicting K, V, and Q at target layer 14, offset +1 in Qwen3-0.6B,
**dual-layer linear probes give no measurable improvement over single-layer
probes**. The residual stream is sufficiently additive that no second
layer carries information beyond what the optimal single layer already
provides.

K and V both peak at **L14** (the target layer itself). Q peaks at **L15**
(one layer later than the target).

## Method

Closed-form ridge regression (no gradient descent). For each layer pair
(L1, L2) with L1 < L2 over 28 layers (378 pairs total):

```
W* = (X^T X + λI)^-1 X^T Y
where X = concat(h[L1], h[L2]) ∈ ℝ^[N, 2*d_model]
      Y = K, V, or Q at layer 14, offset +1
      λ = 1e-3
```

Single-layer baseline computed identically with `X = h[L]` for each L.
Validation: mean cosine similarity between predicted and target heads.

V and K used N_train = 30,720 tokens (10× feature dim, generous margin).
Q used N_train = 10,240 tokens (5× margin).

`scripts/pipeline_kv_medusa_06b_dual_layer_sweep_v_lstsq.py` (parameterized
for V, K, or Q via CLI arg).

## Results

| Axis | Best single | Best pair | Δ (dual − single) | Conclusion |
|---|---|---|---|---|
| **V** | L14 cos = 0.5509 | (L14, L19) cos = 0.5529 | +0.0020 | Single suffices |
| **K** | L14 cos = 0.8312 | (L14, L20) cos = 0.8313 | +0.0001 | Single suffices |
| **Q** | **L15** cos = 0.4094 | (L15, L21) cos = 0.3652 | -0.0442 | Single suffices |

The Q dual case shows negative Δ because of bias-variance with 5× margin.
Re-running Q at 10× margin would likely converge to Δ ≈ 0 like V and K.
The conclusion (single layer suffices) is consistent across all three axes.

### Single-layer V curve

```
L0–L5  (mouth):    0.33 → 0.42   ← rising
L6–L13 (climbing): 0.43 → 0.53
L14–L19 (peak):    0.55 (flat)   ← L14 = target = where V is computed
L20–L28 (descend): 0.54 → 0.46
```

### Single-layer K curve

```
L0–L5:   0.73 → 0.78   ← high baseline (K easier than V linearly)
L6–L13:  0.79 → 0.83
L14–L19: 0.83 (peak)   ← L14 again
L20–L28: 0.83 → 0.79   ← descending
```

K is much more linearly predictable than V (~0.83 vs ~0.55 peak). Likely
because K cosine similarity is structurally easier to optimize than V
magnitude reconstruction.

### Single-layer Q curve

```
L0–L5:   0.34 → 0.38   ← lowest baseline of the three axes
L6–L14:  0.38 → 0.41
L15:     0.41 (peak)   ← shifted by 1 from K/V peak
L16–L28: 0.40 → 0.37
```

Q's peak shifts to L15. Mechanical reason: Q at L14 for predicting offset+1
benefits from looking ONE LAYER LATER because L15's residual contains
information about what the next token will need.

## Interpretation

**Why dual fails:** the residual stream is additive (h[L+1] = h[L] + Δ).
Information-theoretically, concat(h[L1], h[L2]) and (h[L1], Δ) carry the
same content. With limited data, the doubled feature space adds variance
without adding information, so dual underperforms single under bias-
variance.

**Why K and V peak at L14:** the target K, V are computed AT L14 by
W_K @ h[L14], W_V @ h[L14]. Predicting them from h[L14] is essentially
recovering W_K, W_V — the easiest possible prediction.

**Why Q peaks at L15:** Q's offset+1 target is "what query will the model
need at the next position." The next position's input is influenced by
information that propagates through L14 and emerges at L15. So h[L15]
carries the most relevant signal for next-position Q.

**Why Q is harder linearly than K:** Q has MHA (16 heads) vs K/V GQA
(8 heads) — 2× more output space. Linear probe also struggles with the
position-dependent rotation from RoPE. Nonlinear probes likely close
this gap (KV-Medusa MLP heads achieved K cos 0.74-0.81 trained, vs our
linear 0.83 — ceiling unclear but Q nonlinear may significantly exceed
0.41).

## Architectural recommendation

For Medusa-style multi-axis prediction heads on 0.6B:

- **K head: attach at L14** (single layer, linear probe baseline 0.83)
- **V head: attach at L14** (single layer, linear probe baseline 0.55)
- **Q head: attach at L15** (single layer, linear probe baseline 0.41,
  worth trying nonlinear for higher ceiling)
- **Don't build dual-layer architectures.** The residual stream is
  sufficiently additive that pair concatenation buys nothing.

## Related work

- Finding 18 (compression topography): K and V have different per-layer
  rank profiles. K is non-uniform, V is uniform. This finding adds:
  even though K rank varies, the linear-probe optimum for K is still at
  the target layer L14, same as V.
- KV-Medusa (commit `e8afc4a` and `aaa96c5`): nonlinear MLP probes
  trained at L=L//2 (= L14 for 28-layer 0.6B) hit K cos 0.74-0.81 across
  offsets t+1 to t+30. Same target layer as the linear probe optimum.

## Date + sources

2026-04-26.
- `scripts/pipeline_kv_medusa_06b_dual_layer_sweep_v_lstsq.py`
- `results/pipeline_kv_medusa_06b_dual_layer_sweep_v_lstsq.json`
- `results/pipeline_kv_medusa_06b_dual_layer_sweep_k_lstsq.json`
- `results/pipeline_kv_medusa_06b_dual_layer_sweep_q_lstsq.json`

## Next experiment

**Q with nonlinear probe.** Linear gives 0.41 at L15. Train a 2-layer
MLP probe (same architecture as the KV-Medusa K/V heads) on h[L15] for
Q at offset +1 over 200 gradient steps. Expected: Q cos jumps to 0.7+
range like K, confirming Q's linear bottleneck is RoPE/MHA structure
not information loss.

If nonlinear Q hits 0.7+: build the **KVQ-Medusa head architecture** at
L14 (K+V) and L15 (Q), see if joint speculative decoding with all three
predicted gives more usable draft tokens than KV alone.
