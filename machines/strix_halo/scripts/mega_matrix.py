"""MEGA MATRIX: every measurement → find what precisely predicts
length, head count, and KV head count.

Measurements collected per token:
- SAE: all 16384 feature activations, n_active, sparsity, top vals
- Per-head: Q norm, K norm, attention entropy, attention max, output norm
- Per-layer: hidden norm, layer delta, logit lens prediction, stabilization
- Global: output entropy, top1/top2/top5 prob, margin, hidden velocity
- Cross-layer: norm trajectory, cosine between layers, rotation magnitude

Targets:
1. LENGTH: which layer this token resolves at (stabilization depth)
2. HEAD COUNT: how many heads are needed (measured by leave-one-out impact)
3. KV COUNT: how many KV angles needed (measured by KV rank contribution)
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

device = "cuda"

print("=" * 70)
print("MEGA MATRIX — every measurement vs (length, heads, kv)")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS
GQA = N_HEADS // N_KV
lm_head_w = model.lm_head.weight
final_norm = model.model.norm

# Load SAE (layer 50%)
SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"
sae_ckpt = torch.load(f"{SAE_PATH}/ae_50.pt", map_location=device, weights_only=False)
sae_enc_w = sae_ckpt["encoder.weight"].float().to(device)
sae_enc_b = sae_ckpt["encoder.bias"].float().to(device)
SAE_DIM = sae_enc_w.shape[0]

print(f"Model: H={H} L={N_LAYERS} heads={N_HEADS} kv={N_KV}")
print(f"SAE: {SAE_DIM} features")

texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "Climate change driven by human activities threatens ecosystems worldwide through rising temperatures.",
    "Evolution by natural selection explains how populations of organisms change over many generations.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "The development of antibiotics in the twentieth century transformed medicine and saved millions of lives.",
    "The French Revolution transformed society by uprooting centuries of tradition and absolute monarchy.",
    "Coffee originated in Ethiopia and became one of the most widely consumed beverages across the globe.",
]

print(f"\nCollecting EVERYTHING on {len(texts)} passages...", flush=True)

# Storage
features = defaultdict(list)  # feature_name → [values per token]

for text_idx, text in enumerate(texts):
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        hs = out.hidden_states  # (L+1) × [1, T, H]

        # ══════════════════════════════════════════
        # TARGET 1: LENGTH (stabilization depth)
        # ══════════════════════════════════════════
        per_layer_argmax = []
        per_layer_top1_prob = []
        per_layer_entropy = []
        for l in range(1, len(hs)):
            h_normed = final_norm(hs[l])
            logits_l = F.linear(h_normed, lm_head_w)[0]
            probs_l = F.softmax(logits_l.float(), dim=-1)
            per_layer_argmax.append(logits_l.argmax(-1))
            per_layer_top1_prob.append(probs_l.max(-1).values)
            per_layer_entropy.append(-(probs_l * (probs_l + 1e-10).log()).sum(-1))

        pla = torch.stack(per_layer_argmax)
        final_argmax = pla[-1]
        disagrees = (pla != final_argmax.unsqueeze(0))
        L = disagrees.shape[0]
        layer_idx_t = torch.arange(L, device=device).unsqueeze(1).float()
        last_disagree = (disagrees.float() * layer_idx_t).max(dim=0).values
        stab_depth = ((last_disagree + 1) / L).cpu().numpy()

        # Final output
        final_logits = F.linear(final_norm(hs[-1]), lm_head_w)[0]
        final_probs = F.softmax(final_logits.float(), dim=-1)
        top1_prob = final_probs.max(-1).values.cpu().numpy()
        top2_vals = final_probs.topk(2, dim=-1).values
        top5_vals = final_probs.topk(5, dim=-1).values
        output_entropy = -(final_probs * (final_probs + 1e-10).log()).sum(-1).cpu().numpy()
        margin = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
        top5_mass = top5_vals.sum(-1).cpu().numpy()

        # ══════════════════════════════════════════
        # TARGET 2: HEAD COUNT (per-head importance)
        # Measure at multiple layers
        # ══════════════════════════════════════════
        head_data = {l_idx: {} for l_idx in [0, 7, 14, 21, 27]}

        for l_idx in head_data.keys():
            if l_idx >= N_LAYERS:
                continue
            layer = model.model.layers[l_idx]
            attn = layer.self_attn
            h_l = hs[l_idx]
            h_norm = layer.input_layernorm(h_l)

            # Q, K projections
            q = attn.q_proj(h_norm).view(1, T, N_HEADS, HEAD_DIM)
            k = attn.k_proj(h_norm).view(1, T, N_KV, HEAD_DIM)

            # Per-head Q norm
            q_norms = q[0].float().norm(dim=-1)  # [T, N_HEADS]
            # Per-KV-head K norm
            k_norms = k[0].float().norm(dim=-1)  # [T, N_KV]

            head_data[l_idx] = {
                "q_norms": q_norms.cpu().numpy(),  # [T, 16]
                "k_norms": k_norms.cpu().numpy(),   # [T, 8]
            }

        # ══════════════════════════════════════════
        # SAE features
        # ══════════════════════════════════════════
        h_sae = hs[14][0]  # layer 14 (50%)
        sae_pre = h_sae.float() @ sae_enc_w.T + sae_enc_b
        sae_acts = F.relu(sae_pre)  # [T, 16384]

        sae_n_active = (sae_acts > 0).sum(-1).cpu().numpy()
        sae_mean = []
        sae_max = []
        sae_top5_sum = []
        for t in range(T):
            active = sae_acts[t][sae_acts[t] > 0]
            sae_mean.append(active.mean().item() if len(active) > 0 else 0)
            sae_max.append(sae_acts[t].max().item())
            sae_top5_sum.append(sae_acts[t].topk(5).values.sum().item())
        sae_mean = np.array(sae_mean)
        sae_max = np.array(sae_max)
        sae_top5_sum = np.array(sae_top5_sum)

        # SAE sparsity ratio
        sae_sparsity = sae_n_active / SAE_DIM

        # ══════════════════════════════════════════
        # Hidden state trajectory features
        # ══════════════════════════════════════════
        # Norms at each layer
        layer_norms = np.array([hs[l][0].float().norm(dim=-1).cpu().numpy() for l in range(len(hs))])
        # [L+1, T]

        # Velocity between consecutive layers
        layer_deltas = np.array([
            (hs[l+1][0] - hs[l][0]).float().norm(dim=-1).cpu().numpy()
            for l in range(len(hs)-1)
        ])  # [L, T]

        # Cosine between consecutive layers
        layer_cosines = []
        for l in range(1, len(hs)-1):
            cos = F.cosine_similarity(hs[l][0].float(), hs[l+1][0].float(), dim=-1)
            layer_cosines.append(cos.cpu().numpy())
        layer_cosines = np.array(layer_cosines)  # [L-1, T]

        # ══════════════════════════════════════════
        # Per-token: store ALL features
        # ══════════════════════════════════════════
        for t in range(T):
            # Targets
            features["TARGET_stab_depth"].append(stab_depth[t])
            features["TARGET_top1_prob"].append(top1_prob[t])
            features["TARGET_entropy"].append(output_entropy[t])

            # Output distribution
            features["output_margin"].append(margin[t])
            features["output_top5_mass"].append(top5_mass[t])

            # SAE
            features["sae_n_active"].append(sae_n_active[t])
            features["sae_mean_act"].append(sae_mean[t])
            features["sae_max_act"].append(sae_max[t])
            features["sae_top5_sum"].append(sae_top5_sum[t])
            features["sae_sparsity"].append(sae_sparsity[t])

            # Hidden norms at key layers
            for l in [0, 7, 14, 21, 27, 28]:
                if l < len(hs):
                    features[f"norm_L{l}"].append(float(layer_norms[l, t]))

            # Layer deltas at key points
            for l in [0, 1, 7, 14, 21, 26]:
                if l < len(layer_deltas):
                    features[f"delta_L{l}"].append(float(layer_deltas[l, t]))

            # Layer cosines
            for l in [0, 1, 7, 14, 21, 25]:
                if l < len(layer_cosines):
                    features[f"cosine_L{l}"].append(float(layer_cosines[l, t]))

            # Norm trajectory stats
            norms_t = layer_norms[:, t]
            features["norm_mean"].append(float(norms_t.mean()))
            features["norm_std"].append(float(norms_t.std()))
            features["norm_max"].append(float(norms_t.max()))
            features["norm_growth"].append(float(norms_t[-1] / (norms_t[0] + 1e-10)))

            # Delta trajectory stats
            deltas_t = layer_deltas[:, t]
            features["delta_mean"].append(float(deltas_t.mean()))
            features["delta_std"].append(float(deltas_t.std()))
            features["delta_max"].append(float(deltas_t.max()))
            features["delta_early_vs_late"].append(
                float(deltas_t[:7].mean() / (deltas_t[21:].mean() + 1e-10)))

            # Per-layer logit lens entropy
            for l_check in [0, 7, 14, 21, 27]:
                if l_check < len(per_layer_entropy):
                    features[f"lens_entropy_L{l_check}"].append(
                        float(per_layer_entropy[l_check][t].cpu()))
                    features[f"lens_top1_L{l_check}"].append(
                        float(per_layer_top1_prob[l_check][t].cpu()))

            # Per-Q-head norms (layer 14 = middle)
            if 14 in head_data and "q_norms" in head_data[14]:
                for h in range(N_HEADS):
                    features[f"qnorm_h{h}"].append(float(head_data[14]["q_norms"][t, h]))

            # Per-KV-head norms (layer 14)
            if 14 in head_data and "k_norms" in head_data[14]:
                for kv in range(N_KV):
                    features[f"knorm_kv{kv}"].append(float(head_data[14]["k_norms"][t, kv]))

            # Velocity (hidden state change between consecutive tokens)
            if t > 0:
                vel = (hs[14][0, t] - hs[14][0, t-1]).float().norm().item()
            else:
                vel = 0
            features["velocity"].append(vel)

    print(f"  {text_idx+1}/{len(texts)} done", flush=True)

# ══════════════════════════════════════════════════════
# CORRELATION MATRIX
# ══════════════════════════════════════════════════════
N = len(features["TARGET_stab_depth"])
feat_names = sorted([k for k in features.keys() if not k.startswith("TARGET")])
target_names = ["TARGET_stab_depth", "TARGET_top1_prob", "TARGET_entropy"]

print(f"\n{'='*70}")
print(f"MEGA CORRELATION MATRIX ({N} tokens, {len(feat_names)} features)")
print(f"{'='*70}")

# Compute correlations
results = []
for fn in feat_names:
    vals = np.array(features[fn])
    if vals.std() < 1e-10:
        continue
    row = {"feature": fn}
    for tn in target_names:
        tvals = np.array(features[tn])
        r = np.corrcoef(vals, tvals)[0, 1]
        row[tn] = r
    results.append(row)

# Sort by absolute correlation with stab_depth (LENGTH)
print(f"\n{'FEATURE':>25} {'r(LENGTH)':>10} {'r(TOP1)':>10} {'r(ENTROPY)':>10}")
print("-" * 58)

# Top correlates for LENGTH
results_by_length = sorted(results, key=lambda x: abs(x.get("TARGET_stab_depth", 0)), reverse=True)
print(f"\nTOP CORRELATES FOR LENGTH (stabilization depth):")
for r in results_by_length[:20]:
    print(f"  {r['feature']:>25} {r['TARGET_stab_depth']:>+10.3f} {r['TARGET_top1_prob']:>+10.3f} {r['TARGET_entropy']:>+10.3f}")

# Top correlates for TOP1 (proxy for head count needed)
results_by_top1 = sorted(results, key=lambda x: abs(x.get("TARGET_top1_prob", 0)), reverse=True)
print(f"\nTOP CORRELATES FOR HEAD COUNT (top1_prob):")
for r in results_by_top1[:20]:
    print(f"  {r['feature']:>25} {r['TARGET_stab_depth']:>+10.3f} {r['TARGET_top1_prob']:>+10.3f} {r['TARGET_entropy']:>+10.3f}")

# KV-specific: which KV head norms correlate
print(f"\nKV HEAD NORMS → TARGETS:")
for kv in range(N_KV):
    fn = f"knorm_kv{kv}"
    if fn in [r["feature"] for r in results]:
        r = [x for x in results if x["feature"] == fn][0]
        print(f"  KV{kv}: r(length)={r['TARGET_stab_depth']:>+.3f}  r(top1)={r['TARGET_top1_prob']:>+.3f}  r(entropy)={r['TARGET_entropy']:>+.3f}")

# Q HEAD NORMS → TARGETS
print(f"\nQ HEAD NORMS → TARGETS:")
for h in range(N_HEADS):
    fn = f"qnorm_h{h}"
    if fn in [r["feature"] for r in results]:
        r = [x for x in results if x["feature"] == fn][0]
        print(f"  Q{h:>2}: r(length)={r['TARGET_stab_depth']:>+.3f}  r(top1)={r['TARGET_top1_prob']:>+.3f}  r(entropy)={r['TARGET_entropy']:>+.3f}")

print(f"\nDone. {len(results)} features measured.", flush=True)
