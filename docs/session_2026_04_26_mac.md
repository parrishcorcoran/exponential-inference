# Mac Session — 2026-04-26

Comprehensive log covering today's work on the Mac (MPS, Qwen3-0.6B vanilla).
Stages 144–159, plus Finding 24. Companion to Finding 23 (Z8 G4 closed-form layer sweep).

## Theme

**Reading the address book.** Prior work measured the structure of the
K-cache (rank profile, manifold dimension, per-layer compressibility).
This session asked the next question: given that K predicts uniformly
across a 10-token horizon, **can we use the predicted K to recover the
token at that position?** That's HRR's unbinding question, and the answer
is starting to be yes — given the right training methodology.

## Stages run

| stage | what | status |
|-------|------|--------|
| 144 | KV-Medusa replication on vanilla Qwen3-0.6B (10 heads × 300 steps) | done — cos_k 0.71–0.77 uniform |
| 145 | KV substitution token-acceptance test (oracle drafts) | done — 96.0% top-1 |
| 146 | Wallclock benchmark (baseline tok/s, verify pass time) | done — 5.55× projected |
| 147 | Standard token-Medusa heads on 0.6B (5 heads × 300 steps) | done — 32%/5%/2.5%/2.5%/2% |
| 148 | Combined token-Medusa + KV-Medusa acceptance test | done — KV adds 0% |
| 149 | Tree-Medusa top-K probe (K = 1..64) | done — chained ceiling 3.16 |
| 150 | Conditional KV-Medusa training (head conditional on token) | done — slight cos lift, no acceptance gain |
| 151 | Combined cond-KV vs uncond-KV vs no-sub | done — all three identical at token level |
| 152 | Eagle drafter training (1500 steps, 1-layer transformer) | done — undertrained, val tok_acc 0.148 |
| 153 | Eagle autoregressive draft + verify test | done — chained 1.40 |
| 154 | K-decoder v1 (real-K → token via frozen LM head) | done — ceiling 0.560 |
| 155 | K-decoder v2 (noise-augmented) | done — degraded to 0.180 |
| 156 | Joint focused 1×1 (head + decoder, MSE+CE) | done — **predicted-K top-1 = 0.21, top-5 = 0.46** |
| 157 | Anneal head 2 with frozen head-1 anchor | done — anchor preserved, h2 hit 0.09/0.24 |
| 158 | Q-only decoder ceiling test | done — **real-Q top-1 = 0.615, top-5 = 0.785** |
| 159 | Layer sweep (28 layers × 200 steps) for K/V/Q optimal attachment | running |

## Key results

### KV-Medusa replicates at 0.6B scale

Strix's 14B novel result (cos_k uniform across 10 offsets, no decay)
holds at 0.6B vanilla. Confirms the K-uniformity is a structural
property of the K-manifold, not a 14B substrate artifact.

| | 0.6B (this session) | 14B (Strix) |
|---|---|---|
| K-cosine range | 0.71–0.77 | 0.75–0.81 |
| Token acceptance (oracle drafts) | 96.0% | 99.1% |
| Tokens/decode step (chained) | 8.76 | 10.9 |
| Projected speedup vs baseline | 5.55× | 5.17× |

### The Q-bottleneck

The combined token-Medusa + KV-Medusa pipeline shows zero KV gain.
Diagnosis: when token-Medusa drafts a wrong token, the input embedding
at that position propagates through layers 0–13 and produces a wrong
**Q at layer 14**. Q is the unbinding probe; a wrong probe retrieves
the wrong content regardless of how clean the K, V cache is. KV-Medusa
fixes the address book and content store, but cannot fix the probe.
Eagle (or any token-quality-improving drafter) is what addresses the
Q-path.

### K is the unbinding key — and we can read it

`pipeline_kv_medusa_06b_unbind.py` trains a 1M-parameter linear
projection from cached K → frozen LM head and measures token recovery.

**Real K → token: 0.560 top-1.** K alone carries over half the token
signature. This is the structural validation of the HRR identification
of K as the unbinding key (Plate 1995; Hopfield Networks Is All You
Need, Ramsauer 2020).

**Real Q → token: 0.615 top-1, 0.785 top-5.** Q is even more
informative. Consistent with HRR: the probe is more token-conditional
than the address.

### The methodology lesson

The first three K-decoder attempts (v1 train-on-clean, v2
noise-augmented, parallel multi-offset joint) all failed to recover
useful predicted-K → token accuracy. The breakthrough was **focused
1×1 joint training**: one head, one decoder, one offset, joint MSE+CE
loss. Result: predicted-K top-1 = 0.21, top-5 = 0.46.

The lesson, in line with this project's annealing methodology
elsewhere: don't train heads in parallel; anneal one offset at a time,
let head and decoder co-adapt, then move forward. The address book is
already there; the readout has to be co-adapted.

## Frame revision

The "wormhole" terminology used in earlier sessions is replaced
throughout this session's docs by:

- **Address book / address space** — the K-manifold's role in attention.
- **Probe / unbinding key** — what Q does, what K is.
- **Throat / bottleneck layer** — when referring to the layer where
  representation rank is minimized (formerly "wormhole throat").
- **Manifold topography** — the per-layer cavities-and-walls landscape
  measured by finding 22.

Finding 22 already supersedes the global "wormhole" claim. This
session continues that revision.

## Artifacts

### Scripts (new this session)

```
scripts/pipeline_kv_medusa_06b.py                  # KV-Medusa replication
scripts/pipeline_kv_medusa_06b_token_test.py       # KV substitution acceptance
scripts/pipeline_kv_medusa_06b_speed.py            # wallclock benchmark
scripts/pipeline_medusa_heads_06b.py               # standard token-Medusa
scripts/pipeline_kv_medusa_06b_combined.py         # combined test
scripts/pipeline_kv_medusa_06b_tree.py             # tree-Medusa probe
scripts/pipeline_kv_medusa_06b_conditional.py      # conditional KV-Medusa
scripts/pipeline_kv_medusa_06b_combined_cond.py    # combined cond-KV test
scripts/pipeline_eagle_drafter_06b.py              # Eagle drafter training
scripts/pipeline_eagle_test_06b.py                 # Eagle autoregressive
scripts/pipeline_kv_medusa_06b_unbind.py           # K-decoder v1
scripts/pipeline_kv_medusa_06b_unbind_v2.py        # K-decoder v2
scripts/pipeline_kv_medusa_06b_joint.py            # multi-head joint
scripts/pipeline_kv_medusa_06b_joint_one.py        # focused 1×1 (best so far)
scripts/pipeline_kv_medusa_06b_anneal_h2.py        # anneal forward
scripts/pipeline_kv_medusa_06b_q_only.py           # Q ceiling
scripts/pipeline_kv_medusa_06b_layer_sweep.py      # layer sweep (running)
scripts/pipeline_kv_medusa_06b_kvq.py              # K+V+Q decoder (queued)
scripts/pipeline_kv_medusa_06b_kv_only.py          # K+V decoder (queued)
```

### Results

```
results/pipeline_kv_medusa_06b.json
results/pipeline_kv_medusa_06b_token_test.json
results/pipeline_kv_medusa_06b_speed.json
results/pipeline_medusa_heads_06b.json
results/pipeline_kv_medusa_06b_combined.json
results/pipeline_kv_medusa_06b_tree.json
results/pipeline_kv_medusa_06b_conditional.json
results/pipeline_kv_medusa_06b_combined_cond.json
results/pipeline_eagle_drafter_06b.json
results/pipeline_eagle_test_06b.json
results/pipeline_kv_medusa_06b_unbind.json
results/pipeline_kv_medusa_06b_unbind_v2.json
results/pipeline_kv_medusa_06b_joint_one.json
results/pipeline_kv_medusa_06b_anneal_h2.json
results/pipeline_kv_medusa_06b_q_only.json
results/pipeline_kv_medusa_06b_layer_sweep.json     # running
```

### Checkpoints

```
checkpoints/qwen_06b/kv_medusa_head_{1..10}.pt      # original KV-Medusa heads
checkpoints/qwen_06b/medusa_head_{1..5}.pt          # standard token-Medusa
checkpoints/qwen_06b/kv_medusa_cond_head_{1..5}.pt  # conditional KV-Medusa
checkpoints/qwen_06b/eagle_drafter.pt               # Eagle drafter
checkpoints/qwen_06b/k_decoder.pt                   # v1 K-decoder (real-K trained)
checkpoints/qwen_06b/k_decoder_v2.pt                # v2 noise-augmented
checkpoints/qwen_06b/kv_medusa_head_joint_one_1.pt  # **focused 1×1 head** (best)
checkpoints/qwen_06b/k_decoder_joint_one.pt         # **focused 1×1 decoder** (best)
checkpoints/qwen_06b/kv_medusa_head_joint_one_2.pt  # anneal h2
```

## Open frontier (after layer sweep finishes)

1. **K+V+Q combined decoder** — independence test for the three streams.
   If complementary, ceiling could hit 70–80% top-1.
2. **Whitening** — Σ⁻¹/² preprocessing on K input. HRR's unitary-vector
   ideal made empirical.
3. **KV-rank-256 substrate** on 0.6B — does Strix's 99.1% acceptance
   carry over once the manifold is smoothed?
4. **Continue annealing** through offsets 3..10 with per-offset
   decoders (the 2×1 anneal showed the shared decoder is the bottleneck).
5. **Q-Medusa head** — predict Q at future positions from h_t. Train it
   with the same MSE+CE joint pattern as the K-head.

## Cross-machine context

Strix has CUDA + larger batches available. Two natural Strix experiments
to queue up after this session:

- **Properly trained Eagle** (50K+ steps, batch=8). The 1500-step Mac
  attempt landed at 14.8% val tok_acc (undertrained).
- **KV-Medusa on Strix's 14B KV-256 substrate** with the same
  K-decoder pipeline. The 99.1% vs 96% gap may be substrate-driven.

Z8 has MoE-class capacity. Could test K-decoder on Qwen3-MoE 30B-A3B
once the methodology stabilizes — would the K-as-unbinding-key claim
hold across the MoE expert routing layer?

## Session takeaway

The map is real (K-manifold is structured), the lookup is starting to
work (joint MSE+CE training, 21% / 46% predicted-K → token), and the
diagnosis is clean (Q-path corruption blocks naive composition with
weak token drafters). Moving from "structural finding" to "usable
decoder" depends on getting K+V+Q combined plus the right substrate
plus the right drafter for Q. None of those is obviously out of reach.
