"""Find the MINIMUM resources each token needs.

Per token: what is the LEAST compute that still gives the correct answer?
- Minimum KV heads (angles): ablate heads until prediction changes
- Minimum layers (depth): ablate from end until prediction changes
- Minimum KV positions: which context positions are essential

Not averages. Not correlations. EXACT minimum per token.
"""
import torch
import torch.nn.functional as F
import numpy as np

device = "cuda"

print("=" * 70)
print("MINIMUM RESOURCES PER TOKEN")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
H = model.config.hidden_size
HEAD_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS
GQA = N_HEADS // N_KV
lm_head = model.lm_head.weight
fnorm = model.model.norm

text = "The theory of general relativity describes gravity as the curvature of spacetime caused by mass"
ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
T = ids.shape[1]

print(f"Model: L={N_LAYERS} H={N_HEADS} KV={N_KV}")
print(f"Sequence: {T} tokens")

# Get ground truth: full model hidden states + predictions
with torch.no_grad():
    out = model.model(ids, output_hidden_states=True)
    hs = out.hidden_states
    final_logits = F.linear(fnorm(hs[-1]), lm_head)[0]
    final_preds = final_logits.argmax(-1)  # [T]

print(f"\nMeasuring minimum resources per token...\n")
print(f"{'Pos':>4} {'Token':>12} {'→':>2} {'Next':>12} {'Min L':>6} {'Min KV°':>8} {'Min ctx':>8}")
print("-" * 58)

all_min_layers = []
all_min_kv_degrees = []

for t in range(T - 1):
    target = final_preds[t].item()
    target_str = tokenizer.decode([target])
    tok_str = tokenizer.decode(ids[0, t:t+1])

    # ── Minimum layers: exit from end until prediction changes ──
    min_layers = N_LAYERS
    for exit_l in range(N_LAYERS, 0, -1):
        h = fnorm(hs[exit_l][:, t:t+1, :])
        pred = F.linear(h, lm_head)[0, 0].argmax().item()
        if pred != target:
            min_layers = exit_l + 1
            break
        min_layers = exit_l

    # ── Minimum KV heads (angles): measure per-head contribution ──
    # At the resolution layer: which heads' attention actually matters?
    # Use the hidden state difference when masking each KV head group
    res_layer = min(min_layers, N_LAYERS) - 1
    if res_layer < 0:
        res_layer = 0

    layer = model.model.layers[res_layer]
    attn = layer.self_attn
    h_in = hs[res_layer]

    # Compute full Q, K, V for this layer
    h_norm = layer.input_layernorm(h_in)
    q = attn.q_proj(h_norm).view(1, T, N_HEADS, HEAD_DIM)
    k = attn.k_proj(h_norm).view(1, T, N_KV, HEAD_DIM)

    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q_t = q[0, t]  # [N_HEADS, HEAD_DIM]
    k_all = k[0, :t+1]  # [t+1, N_KV, HEAD_DIM]

    # Per-KV-head: attention energy (how much this head contributes)
    kv_importance = []
    for kv_h in range(N_KV):
        # Q heads that use this KV head
        q_heads_for_kv = list(range(kv_h * GQA, (kv_h + 1) * GQA))
        energy = 0
        for qh in q_heads_for_kv:
            # Attention score: q[qh] @ k[kv_h].T
            scores = (q_t[qh] * k_all[:, kv_h]).sum(-1)  # [t+1]
            energy += scores.float().abs().max().item()
        kv_importance.append(energy)

    # Sort KV heads by importance, find minimum needed
    kv_ranked = sorted(range(N_KV), key=lambda x: kv_importance[x], reverse=True)
    # How many KV heads needed? (approximation: heads covering >90% of total importance)
    total_imp = sum(kv_importance)
    cum = 0
    min_kv = 0
    for kv_h in kv_ranked:
        cum += kv_importance[kv_h]
        min_kv += 1
        if cum >= 0.9 * total_imp:
            break

    # Convert KV heads to degrees: each head = 360/N_KV degrees
    degrees_per_head = 360.0 / N_KV
    min_degrees = min_kv * degrees_per_head

    # ── Minimum context positions ──
    # Which positions in the context have the most attention energy?
    ctx_importance = np.zeros(t + 1)
    for qh in range(N_HEADS):
        kv_h = qh // GQA
        scores = (q_t[qh] * k_all[:, kv_h]).sum(-1).float().cpu().numpy()
        ctx_importance += np.abs(scores)

    # How many context positions cover 90% of attention?
    ctx_ranked = np.argsort(-ctx_importance)
    ctx_total = ctx_importance.sum()
    cum_ctx = 0
    min_ctx = 0
    for pos in ctx_ranked:
        cum_ctx += ctx_importance[pos]
        min_ctx += 1
        if cum_ctx >= 0.9 * ctx_total:
            break

    all_min_layers.append(min_layers)
    all_min_kv_degrees.append(min_degrees)

    print(f"  {t:>3} {tok_str:>12} → {target_str:>12} L{min_layers:>3}   {min_degrees:>5.0f}°   {min_ctx:>3}/{t+1} pos")

# Summary
ml = np.array(all_min_layers)
md = np.array(all_min_kv_degrees)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Min layers: mean={ml.mean():.1f} min={ml.min()} max={ml.max()}")
print(f"  Min KV degrees: mean={md.mean():.0f}° min={md.min():.0f}° max={md.max():.0f}°")
print(f"  Layer savings: {(1 - ml.mean()/N_LAYERS)*100:.0f}%")
print(f"  KV savings: {(1 - md.mean()/360)*100:.0f}%")

print(f"\n  Distribution:")
for l in sorted(set(all_min_layers)):
    n = sum(1 for x in all_min_layers if x == l)
    pct = n / len(all_min_layers) * 100
    bar = "█" * int(pct)
    print(f"    L{l:>2}: {n:>3} tokens ({pct:>4.0f}%) {bar}")

print(f"\n  KV degree distribution:")
for d in sorted(set(all_min_kv_degrees)):
    n = sum(1 for x in all_min_kv_degrees if x == d)
    pct = n / len(all_min_kv_degrees) * 100
    print(f"    {d:>3}°: {n:>3} tokens ({pct:>4.0f}%)")

print(f"\nDone.", flush=True)
