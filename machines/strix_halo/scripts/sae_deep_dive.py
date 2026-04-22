"""SAE Deep Dive: parse every piece of routing information from SAE output.

For each token, the SAE gives us 16384 features with activations.
Parse this into discrete routing decisions:
- Which layer resolves this token (correlate SAE features with stab_depth)
- How many heads needed (correlate with head importance)
- Per-head activation patterns
- Feature clusters that correspond to routing decisions
"""
import torch
import torch.nn.functional as F
import numpy as np
import json

device = "cuda"

print("=" * 70)
print("SAE DEEP DIVE: extracting routing information")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size          # 1024
N_LAYERS = model.config.num_hidden_layers  # 28
N_HEADS = model.config.num_attention_heads  # 16
N_KV = model.config.num_key_value_heads     # 8
HEAD_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS  # 128

lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

# Load SAE (use layer 14 = 50% depth — best for routing decisions)
SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"
sae_ckpt = torch.load(f"{SAE_PATH}/ae_50.pt", map_location=device, weights_only=False)
sae_enc_w = sae_ckpt["encoder.weight"].float().to(device)  # [16384, 1024]
sae_enc_b = sae_ckpt["encoder.bias"].float().to(device)
sae_dec_w = sae_ckpt["decoder.weight"].float().to(device)  # [1024, 16384]
SAE_K = sae_ckpt["k"].item()
SAE_DIM = sae_enc_w.shape[0]

print(f"Model: H={H}, L={N_LAYERS}, heads={N_HEADS}, kv={N_KV}")
print(f"SAE: {SAE_DIM} features, k={SAE_K}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

def sae_encode(h):
    """Full SAE encode → activations for all features."""
    pre = h.float() @ sae_enc_w.T + sae_enc_b
    return F.relu(pre)

# ═══════════════════════════════════════════════════════
# Collect detailed per-token data
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "Climate change driven by human activities threatens ecosystems worldwide through rising temperatures.",
    "Evolution by natural selection explains how populations of organisms change over many generations.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "The development of antibiotics in the twentieth century transformed medicine and saved millions of lives.",
]

print(f"\nCollecting detailed measurements...", flush=True)

all_data = []

for text in texts:
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states

        # Teacher measurements
        final_logits = F.linear(final_norm(hidden_states[-1]), lm_head_weight)[0]
        probs = F.softmax(final_logits.float(), dim=-1)
        top1_prob = probs.max(dim=-1).values
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)

        # Stabilization depth
        per_layer_argmax = []
        for l in range(1, len(hidden_states)):
            h_normed = final_norm(hidden_states[l])
            logits_l = F.linear(h_normed, lm_head_weight)
            per_layer_argmax.append(logits_l[0].argmax(-1))
        per_layer_argmax = torch.stack(per_layer_argmax)
        final_argmax = per_layer_argmax[-1]
        disagrees = (per_layer_argmax != final_argmax.unsqueeze(0))
        L = disagrees.shape[0]
        layer_idx_t = torch.arange(L, device=device).unsqueeze(1).float()
        last_disagree = (disagrees.float() * layer_idx_t).max(dim=0).values
        stab_depth = (last_disagree + 1) / L

        # Per-head attention contribution at layer 14 (SAE layer)
        layer14 = model.model.layers[14]
        h14 = hidden_states[14][0]  # [T, H]
        h14_norm = layer14.input_layernorm(hidden_states[14])

        # Full Q, K, V at this layer
        q = layer14.self_attn.q_proj(h14_norm).view(1, T, N_HEADS, HEAD_DIM)
        # Per-head Q norms (proxy for head activation)
        per_head_q_norm = q[0].float().norm(dim=-1)  # [T, N_HEADS]

        # SAE features at layer 14
        h14_flat = hidden_states[14][0]  # [T, H]

        for t in range(T):
            sae_acts = sae_encode(h14_flat[t])  # [16384]
            active_mask = sae_acts > 0
            n_active = active_mask.sum().item()

            # Top-k features
            topk = sae_acts.topk(min(SAE_K, 20))

            # Per-head Q norms for this token
            head_norms = per_head_q_norm[t].cpu().numpy()  # [N_HEADS]

            # Correlation: which SAE features align with which heads?
            # The SAE decoder columns map features back to hidden space.
            # Each head occupies a slice of hidden space.
            # Feature i's decoder vector → project onto each head's subspace.
            top_feat_indices = topk.indices.cpu().numpy()
            top_feat_values = topk.values.cpu().numpy()

            # For top features: measure which head subspace they project into
            feat_head_alignment = np.zeros(N_HEADS)
            for fi in range(min(10, len(top_feat_indices))):
                feat_idx = top_feat_indices[fi]
                feat_val = top_feat_values[fi]
                # Decoder column for this feature → [H]
                dec_vec = sae_dec_w[:, feat_idx].cpu().numpy()
                # Project onto each head's subspace
                # Q weight: [Q_DIM, H] → reshape to [N_HEADS, HEAD_DIM, H]
                q_w = layer14.self_attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
                for head in range(N_HEADS):
                    head_proj = q_w[head].float().cpu().numpy()  # [HEAD_DIM, H]
                    alignment = np.abs(head_proj @ dec_vec).sum()
                    feat_head_alignment[head] += alignment * feat_val

            tok_str = tokenizer.decode(ids[0, t:t+1])

            all_data.append({
                "token": tok_str,
                "top1_prob": top1_prob[t].item(),
                "entropy": entropy[t].item(),
                "stab_depth": stab_depth[t].item(),
                "sae_n_active": n_active,
                "sae_top_val": top_feat_values[0] if len(top_feat_values) > 0 else 0,
                "sae_mean_val": float(sae_acts[active_mask].mean()) if n_active > 0 else 0,
                "sae_max_feature": int(top_feat_indices[0]) if len(top_feat_indices) > 0 else -1,
                "head_norms": head_norms.tolist(),
                "feat_head_alignment": feat_head_alignment.tolist(),
                "top_features": top_feat_indices[:10].tolist(),
                "top_values": top_feat_values[:10].tolist(),
            })

print(f"Collected {len(all_data)} token records")

# ═══════════════════════════════════════════════════════
# Analysis: build the routing table
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SAE → ROUTING TABLE")
print(f"{'='*70}")

# 1. SAE features → stabilization depth
print(f"\n1. SAE FEATURES → RESOLUTION LAYER")
sae_n = np.array([d["sae_n_active"] for d in all_data])
sae_top = np.array([d["sae_top_val"] for d in all_data])
sae_mean = np.array([d["sae_mean_val"] for d in all_data])
stab = np.array([d["stab_depth"] for d in all_data])
top1 = np.array([d["top1_prob"] for d in all_data])
ent = np.array([d["entropy"] for d in all_data])

corrs = {
    "sae_n_active": np.corrcoef(sae_n, stab)[0, 1],
    "sae_top_val": np.corrcoef(sae_top, stab)[0, 1],
    "sae_mean_val": np.corrcoef(sae_mean, stab)[0, 1],
}
print(f"  Correlation with stabilization_depth:")
for k, v in corrs.items():
    print(f"    {k}: r = {v:+.3f}")

# 2. Per-head analysis
print(f"\n2. SAE → WHICH HEADS MATTER")
head_norms_all = np.array([d["head_norms"] for d in all_data])  # [N, N_HEADS]
feat_align_all = np.array([d["feat_head_alignment"] for d in all_data])  # [N, N_HEADS]

print(f"\n  Per-head mean Q norm:")
for h in range(N_HEADS):
    mean_norm = head_norms_all[:, h].mean()
    std_norm = head_norms_all[:, h].std()
    # Correlation of this head's norm with top1_prob
    r_top1 = np.corrcoef(head_norms_all[:, h], top1)[0, 1]
    r_stab = np.corrcoef(head_norms_all[:, h], stab)[0, 1]
    print(f"    Head {h:>2}: norm={mean_norm:.1f}±{std_norm:.1f}  r(top1)={r_top1:+.3f}  r(stab)={r_stab:+.3f}")

print(f"\n  SAE feature → head alignment (which heads each feature activates):")
for h in range(N_HEADS):
    mean_align = feat_align_all[:, h].mean()
    r_top1 = np.corrcoef(feat_align_all[:, h], top1)[0, 1] if feat_align_all[:, h].std() > 0 else 0
    print(f"    Head {h:>2}: mean_align={mean_align:.1f}  r(top1)={r_top1:+.3f}")

# 3. Defined vs branching: what SAE features differ?
print(f"\n3. DEFINED vs BRANCHING: which SAE features distinguish them?")
defined_mask = top1 > 0.8
branch_mask = top1 < 0.3

if defined_mask.sum() > 5 and branch_mask.sum() > 5:
    # Most common top features in defined vs branching
    def_features = []
    branch_features = []
    for i, d in enumerate(all_data):
        if defined_mask[i]:
            def_features.extend(d["top_features"][:5])
        elif branch_mask[i]:
            branch_features.extend(d["top_features"][:5])

    from collections import Counter
    def_common = Counter(def_features).most_common(10)
    branch_common = Counter(branch_features).most_common(10)

    print(f"\n  Top SAE features for DEFINED tokens (top1 > 0.8):")
    for feat, count in def_common:
        print(f"    Feature {feat}: appears {count} times")

    print(f"\n  Top SAE features for BRANCHING tokens (top1 < 0.3):")
    for feat, count in branch_common:
        print(f"    Feature {feat}: appears {count} times")

    # Do they overlap?
    def_set = set(f for f, _ in def_common)
    branch_set = set(f for f, _ in branch_common)
    overlap = def_set & branch_set
    unique_def = def_set - branch_set
    unique_branch = branch_set - def_set
    print(f"\n  Overlap: {len(overlap)} features shared")
    print(f"  Unique to defined: {len(unique_def)} features: {unique_def}")
    print(f"  Unique to branching: {len(unique_branch)} features: {unique_branch}")

# 4. Summary table
print(f"\n{'='*70}")
print("ROUTING TABLE SUMMARY")
print(f"{'='*70}")
print(f"\n  {'SAE Measurement':>25} {'→ Routes to':>20} {'Correlation':>12}")
print(f"  {'-'*60}")
print(f"  {'Mean activation':>25} {'Layer depth':>20} {corrs['sae_mean_val']:>+12.3f}")
print(f"  {'N active features':>25} {'Layer depth':>20} {corrs['sae_n_active']:>+12.3f}")
print(f"  {'Top activation':>25} {'Layer depth':>20} {corrs['sae_top_val']:>+12.3f}")
print(f"  {'Mean activation':>25} {'Entropy':>20} {np.corrcoef(sae_mean, ent)[0,1]:>+12.3f}")
print(f"  {'Mean activation':>25} {'Top1 prob':>20} {np.corrcoef(sae_mean, top1)[0,1]:>+12.3f}")

# Per-head routing
best_head_r = -1
best_head = -1
for h in range(N_HEADS):
    r = abs(np.corrcoef(head_norms_all[:, h], top1)[0, 1])
    if r > best_head_r:
        best_head_r = r
        best_head = h
print(f"  {'Head Q norm (best)':>25} {'Top1 prob':>20} {best_head_r:>+12.3f} (head {best_head})")

print(f"\nDone.", flush=True)
