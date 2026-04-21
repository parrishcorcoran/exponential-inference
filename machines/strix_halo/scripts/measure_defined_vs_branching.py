"""Measure: can geometric properties of the hidden state distinguish
DEFINED positions (one continuation) from BRANCHING positions (multiple)?

No trained classifier. Pure measurement.

For each token position in diverse text:
  1. Run full forward, get the output distribution
  2. Label: defined (top-1 dominates) vs branching (multiple high-prob tokens)
  3. Measure geometric features of the hidden state at that position
  4. Check separation — do geometric features distinguish the two?

If they do: the manifold tells you the tree structure directly.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time

device = "cuda"

print("=" * 70)
print("MANIFOLD MEASUREMENT: defined vs branching positions")
print("Pure geometry. No trained classifier.")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
HIDDEN = model.config.hidden_size
VOCAB = model.config.vocab_size

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Diverse text — different domains, different difficulty
# ═══════════════════════════════════════════════════════
texts = [
    "The cat sat on the mat and looked out the window at the birds flying by.",
    "Quantum entanglement allows particles to be correlated regardless of the distance separating them.",
    "To make pasta, boil water in a large pot, add salt, then cook the noodles for eight minutes.",
    "The Riemann zeta function has non-trivial zeros only on the critical line with real part one-half.",
    "She walked into the room and immediately noticed something was different about the arrangement of furniture.",
    "Gradient descent iteratively adjusts parameters in the direction that minimizes the loss function.",
    "The dog barked loudly at the mailman who came every day at the same time without fail.",
    "Superconductivity emerges when Cooper pairs of electrons form a macroscopic quantum ground state below Tc.",
    "He picked up the phone and dialed the number, waiting nervously for someone to answer on the other end.",
    "The Fourier transform decomposes a function of time into constituent frequencies with complex amplitudes.",
    "Once upon a time in a land far away there lived a king who ruled with wisdom and kindness.",
    "Attention mechanisms compute weighted sums over value vectors where weights are derived from query-key compatibility.",
    "The weather today is sunny with a high of seventy-five degrees and a slight breeze from the west.",
    "Topological quantum error correction encodes logical qubits in the homology of a surface code lattice.",
    "I went to the store to buy milk, eggs, bread, and some fresh vegetables for dinner tonight.",
]

# ═══════════════════════════════════════════════════════
# Collect measurements
# ═══════════════════════════════════════════════════════
print(f"\nCollecting measurements from {len(texts)} passages...", flush=True)

all_records = []
lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

for text_idx, text in enumerate(texts):
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # (L+1) × [1, T, H]

        embedding = hidden_states[0][0]       # [T, H] — manifold position
        final_hidden = hidden_states[-1][0]    # [T, H]

        # Output distribution at each position
        final_logits = F.linear(final_norm(hidden_states[-1]), lm_head_weight)[0]  # [T, V]
        probs = F.softmax(final_logits.float(), dim=-1)  # [T, V]

        # ── Label: defined vs branching ──
        # Use the actual distribution shape
        top_probs, top_ids = probs.topk(10, dim=-1)  # [T, 10]
        top1_prob = top_probs[:, 0]                    # [T]
        top2_prob = top_probs[:, 1]                    # [T]
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # [T]
        margin = top1_prob - top2_prob                  # [T]

        # ── Geometric features from hidden states (MEASUREMENTS) ──

        # 1. Embedding norm — energy at manifold position
        emb_norm = embedding.norm(dim=-1)  # [T]

        # 2. Hidden state norm at final layer
        final_norm_val = final_hidden.norm(dim=-1)  # [T]

        # 3. Per-layer hidden state norms — trajectory energy
        layer_norms = torch.stack([hidden_states[l][0].norm(dim=-1) for l in range(len(hidden_states))])  # [L+1, T]

        # 4. Norm velocity: how fast is the state changing between layers
        norm_diffs = (layer_norms[1:] - layer_norms[:-1]).abs()  # [L, T]
        norm_velocity_mean = norm_diffs.mean(dim=0)  # [T]
        norm_velocity_std = norm_diffs.std(dim=0)    # [T]

        # 5. Layer-to-layer cosine similarity (update direction consistency)
        cos_sims = []
        for l in range(1, len(hidden_states)):
            update = hidden_states[l][0] - hidden_states[l-1][0]  # [T, H]
            if l > 1:
                prev_update = hidden_states[l-1][0] - hidden_states[l-2][0]
                cos = F.cosine_similarity(update, prev_update, dim=-1)  # [T]
                cos_sims.append(cos)
        cos_sims = torch.stack(cos_sims)  # [L-2, T]
        update_alignment_mean = cos_sims.mean(dim=0)  # [T]
        update_alignment_std = cos_sims.std(dim=0)

        # 6. Early vs late layer alignment (Finding 08: layer_halves_align)
        mid = len(hidden_states) // 2
        early_update = hidden_states[mid][0] - hidden_states[0][0]    # [T, H]
        late_update = hidden_states[-1][0] - hidden_states[mid][0]    # [T, H]
        halves_align = F.cosine_similarity(early_update, late_update, dim=-1)  # [T]

        # 7. Stabilization depth (Finding 09) — direct measurement
        per_layer_argmax = []
        for l in range(1, len(hidden_states)):
            h = hidden_states[l]
            h_normed = final_norm(h)
            logits_l = F.linear(h_normed, lm_head_weight)
            per_layer_argmax.append(logits_l[0].argmax(-1))  # [T]
        per_layer_argmax = torch.stack(per_layer_argmax)  # [L, T]
        final_argmax = per_layer_argmax[-1]  # [T]
        agrees = (per_layer_argmax == final_argmax.unsqueeze(0))  # [L, T]
        L = agrees.shape[0]
        disagrees = ~agrees
        layer_indices = torch.arange(L, device=device).unsqueeze(1).float()
        last_disagree = (disagrees.float() * layer_indices).max(dim=0).values
        stab_depth = (last_disagree + 1) / L  # [T]

        # 8. Local KNN distance in embedding space (manifold density)
        # For each token's embedding, distance to nearest neighbors in the same sequence
        emb_dists = torch.cdist(embedding.unsqueeze(0).float(), embedding.unsqueeze(0).float())[0]  # [T, T]
        emb_dists.fill_diagonal_(float('inf'))
        knn1_dist = emb_dists.min(dim=-1).values  # [T]
        knn3_dist = emb_dists.topk(3, largest=False, dim=-1).values.mean(dim=-1)  # [T]

        # 9. Centeredness — distance from mean embedding
        emb_mean = embedding.mean(dim=0, keepdim=True)  # [1, H]
        centeredness = F.cosine_similarity(embedding, emb_mean.expand_as(embedding), dim=-1)  # [T]

        # 10. Update kurtosis — shape of per-layer updates
        all_updates = torch.stack([hidden_states[l][0] - hidden_states[l-1][0] for l in range(1, len(hidden_states))])  # [L, T, H]
        update_norms = all_updates.norm(dim=-1)  # [L, T]
        update_mean = update_norms.mean(dim=0, keepdim=True)
        update_std = update_norms.std(dim=0, keepdim=True) + 1e-8
        update_z = (update_norms - update_mean) / update_std
        update_kurtosis = (update_z ** 4).mean(dim=0) - 3.0  # [T] excess kurtosis

    # Store records
    tokens = [tokenizer.decode(ids[0, i:i+1]) for i in range(T)]
    for t in range(T - 1):  # skip last (no next token to predict)
        all_records.append({
            "text_idx": text_idx,
            "pos": t,
            "token": tokens[t],
            "next_token": tokens[t+1] if t+1 < T else "",
            # Labels (from output distribution)
            "top1_prob": top1_prob[t].item(),
            "top2_prob": top2_prob[t].item(),
            "entropy": entropy[t].item(),
            "margin": margin[t].item(),
            # Geometric measurements
            "emb_norm": emb_norm[t].item(),
            "final_norm": final_norm_val[t].item(),
            "norm_velocity_mean": norm_velocity_mean[t].item(),
            "norm_velocity_std": norm_velocity_std[t].item(),
            "update_alignment_mean": update_alignment_mean[t].item(),
            "update_alignment_std": update_alignment_std[t].item(),
            "halves_align": halves_align[t].item(),
            "stab_depth": stab_depth[t].item(),
            "knn1_dist": knn1_dist[t].item(),
            "knn3_dist": knn3_dist[t].item(),
            "centeredness": centeredness[t].item(),
            "update_kurtosis": update_kurtosis[t].item(),
        })

    if (text_idx + 1) % 5 == 0:
        print(f"  {text_idx+1}/{len(texts)} passages done", flush=True)

print(f"\nTotal records: {len(all_records)}", flush=True)

# ═══════════════════════════════════════════════════════
# Analysis: separation without any classifier
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ANALYSIS: geometric separation of defined vs branching")
print(f"{'='*60}")

# Convert to arrays
features = [
    "emb_norm", "final_norm", "norm_velocity_mean", "norm_velocity_std",
    "update_alignment_mean", "update_alignment_std", "halves_align",
    "stab_depth", "knn1_dist", "knn3_dist", "centeredness", "update_kurtosis"
]

top1_probs = np.array([r["top1_prob"] for r in all_records])
entropies = np.array([r["entropy"] for r in all_records])
margins = np.array([r["margin"] for r in all_records])

# Define: position is "defined" if top1 > 0.8, "branching" if top1 < 0.4
# This is NOT a threshold for the system — it's a labeling scheme for analysis
defined_mask = top1_probs > 0.8
branching_mask = top1_probs < 0.4
ambiguous_mask = ~defined_mask & ~branching_mask

n_defined = defined_mask.sum()
n_branching = branching_mask.sum()
n_ambiguous = ambiguous_mask.sum()
print(f"\nPositions: {n_defined} defined (p>0.8), {n_branching} branching (p<0.4), {n_ambiguous} ambiguous")

# For each geometric feature: mean ± std for defined vs branching
print(f"\n{'Feature':>25} {'Defined':>12} {'Branching':>12} {'Sep':>8} {'Corr w/ p1':>10}")
print("-" * 70)

for feat in features:
    vals = np.array([r[feat] for r in all_records])

    if n_defined > 0 and n_branching > 0:
        d_mean = vals[defined_mask].mean()
        d_std = vals[defined_mask].std()
        b_mean = vals[branching_mask].mean()
        b_std = vals[branching_mask].std()

        # Cohen's d: effect size of separation
        pooled_std = np.sqrt((d_std**2 + b_std**2) / 2) + 1e-10
        cohens_d = abs(d_mean - b_mean) / pooled_std

        # Pearson correlation with top1_prob (continuous measure)
        if vals.std() > 1e-10:
            corr = np.corrcoef(vals, top1_probs)[0, 1]
        else:
            corr = 0.0

        print(f"{feat:>25} {d_mean:>6.3f}±{d_std:.3f} {b_mean:>6.3f}±{b_std:.3f} "
              f"{cohens_d:>7.2f}d {corr:>+9.3f}")

# Correlation matrix between geometric features and output distribution
print(f"\n{'='*60}")
print("CORRELATIONS: geometry → output distribution")
print(f"{'='*60}")

print(f"\n{'Feature':>25} {'r(top1)':>8} {'r(entropy)':>10} {'r(margin)':>10}")
print("-" * 60)

for feat in features:
    vals = np.array([r[feat] for r in all_records])
    if vals.std() > 1e-10:
        r_top1 = np.corrcoef(vals, top1_probs)[0, 1]
        r_ent = np.corrcoef(vals, entropies)[0, 1]
        r_margin = np.corrcoef(vals, margins)[0, 1]
    else:
        r_top1 = r_ent = r_margin = 0.0
    print(f"{feat:>25} {r_top1:>+8.3f} {r_ent:>+10.3f} {r_margin:>+10.3f}")

# Show some examples
print(f"\n{'='*60}")
print("EXAMPLES: most defined and most branching positions")
print(f"{'='*60}")

sorted_by_top1 = sorted(all_records, key=lambda r: r["top1_prob"], reverse=True)

print("\nMost DEFINED (highest top1_prob):")
for r in sorted_by_top1[:8]:
    print(f"  '{r['token']:>12}' → '{r['next_token']:<12}' p1={r['top1_prob']:.3f} "
          f"stab={r['stab_depth']:.3f} halves={r['halves_align']:.3f} "
          f"vel={r['norm_velocity_mean']:.1f}")

print("\nMost BRANCHING (lowest top1_prob):")
for r in sorted_by_top1[-8:]:
    print(f"  '{r['token']:>12}' → '{r['next_token']:<12}' p1={r['top1_prob']:.3f} "
          f"stab={r['stab_depth']:.3f} halves={r['halves_align']:.3f} "
          f"vel={r['norm_velocity_mean']:.1f}")

# Save
with open("machines/strix_halo/results/defined_vs_branching.json", "w") as f:
    json.dump({
        "n_records": len(all_records),
        "n_defined": int(n_defined),
        "n_branching": int(n_branching),
        "records_sample": all_records[:50],
    }, f, indent=2)
print(f"\nSaved results.", flush=True)
