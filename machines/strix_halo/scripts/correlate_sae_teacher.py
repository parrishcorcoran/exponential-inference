"""Correlate SAE features with teacher behavior and predictor measurements.

Three sources of information about each token:
1. SAE encoder output (sparse features) — the manifold read
2. Teacher behavior (stabilization depth, output entropy, top1 prob)
3. Our old predictor measurements (hidden_norm, velocity, knn_dist)

If all three agree: the SAE reads the manifold, our predictors approximated it,
and the teacher's behavior reflects manifold position.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import json

device = "cuda"

print("=" * 70)
print("CORRELATE: SAE ↔ Teacher ↔ Predictors")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

# Load SAE for multiple layers
SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"

saes = {}
for layer_pct in [0, 10, 20, 30, 40, 50]:
    ckpt = torch.load(f"{SAE_PATH}/ae_{layer_pct}.pt", map_location=device, weights_only=False)
    # SAE layer index: percentage of total layers
    layer_idx = int(layer_pct / 100 * N_LAYERS)
    saes[layer_idx] = {
        "encoder_w": ckpt["encoder.weight"].float().to(device),  # [16384, 1024]
        "encoder_b": ckpt["encoder.bias"].float().to(device),    # [16384]
        "k": ckpt["k"].item() if ckpt["k"].numel() == 1 else 64,
    }
    print(f"  SAE layer {layer_idx} ({layer_pct}%): {ckpt['encoder.weight'].shape[0]} features, k={saes[layer_idx]['k']}")

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def sae_encode(hidden_state, sae):
    """Encode hidden state with SAE → sparse features."""
    # h @ W^T + b → ReLU → top-k
    pre_act = hidden_state.float() @ sae["encoder_w"].T + sae["encoder_b"]
    acts = F.relu(pre_act)
    # Top-k sparsity
    k = min(sae["k"], acts.shape[-1])
    topk = acts.topk(k, dim=-1)
    return topk.values, topk.indices, acts


# ═══════════════════════════════════════════════════════
# Collect all three measurements on diverse text
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The development of antibiotics in the twentieth century transformed medicine and saved millions of lives.",
    "Climate change driven by human activities threatens ecosystems worldwide through rising temperatures.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "Evolution by natural selection explains how populations of organisms change over many generations.",
]

print(f"\nCollecting measurements on {len(texts)} passages...", flush=True)

all_records = []

for text in texts:
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states

        # Teacher behavior: output distribution
        final_logits = F.linear(final_norm(hidden_states[-1]), lm_head_weight)[0]
        probs = F.softmax(final_logits.float(), dim=-1)
        top1_prob = probs.max(dim=-1).values    # [T]
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # [T]

        # Teacher behavior: stabilization depth
        per_layer_argmax = []
        for l in range(1, len(hidden_states)):
            h = hidden_states[l]
            h_normed = final_norm(h)
            logits_l = F.linear(h_normed, lm_head_weight)
            per_layer_argmax.append(logits_l[0].argmax(-1))
        per_layer_argmax = torch.stack(per_layer_argmax)
        final_argmax = per_layer_argmax[-1]
        disagrees = (per_layer_argmax != final_argmax.unsqueeze(0))
        L = disagrees.shape[0]
        layer_idx_t = torch.arange(L, device=device).unsqueeze(1).float()
        last_disagree = (disagrees.float() * layer_idx_t).max(dim=0).values
        stab_depth = (last_disagree + 1) / L

        # SAE features at each available layer
        for sae_layer, sae in saes.items():
            if sae_layer >= len(hidden_states):
                continue
            h = hidden_states[sae_layer][0]  # [T, H]

            for t in range(T):
                topk_vals, topk_idx, full_acts = sae_encode(h[t], sae)

                # SAE measurements
                n_active = (full_acts > 0).sum().item()
                top_val = topk_vals[0].item() if len(topk_vals) > 0 else 0
                mean_val = topk_vals.mean().item() if len(topk_vals) > 0 else 0
                sparsity = n_active / full_acts.shape[0]

                # Predictor measurements (from hidden state)
                h_vec = h[t].float()
                hidden_norm = h_vec.norm().item()
                velocity = 0.0
                if t > 0:
                    velocity = (h[t] - h[t-1]).float().norm().item()

                all_records.append({
                    "sae_layer": sae_layer,
                    # SAE
                    "sae_n_active": n_active,
                    "sae_top_val": top_val,
                    "sae_mean_val": mean_val,
                    "sae_sparsity": sparsity,
                    # Teacher
                    "top1_prob": top1_prob[t].item(),
                    "entropy": entropy[t].item(),
                    "stab_depth": stab_depth[t].item(),
                    # Predictors
                    "hidden_norm": hidden_norm,
                    "velocity": velocity,
                })

print(f"Total records: {len(all_records)}")

# ═══════════════════════════════════════════════════════
# Correlate
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CORRELATIONS")
print(f"{'='*60}")

# Group by SAE layer
for sae_layer in sorted(saes.keys()):
    records = [r for r in all_records if r["sae_layer"] == sae_layer]
    if len(records) < 20:
        continue

    print(f"\n  SAE Layer {sae_layer} ({len(records)} records):")

    sae_features = ["sae_n_active", "sae_top_val", "sae_mean_val", "sae_sparsity"]
    teacher_features = ["top1_prob", "entropy", "stab_depth"]
    predictor_features = ["hidden_norm", "velocity"]

    all_features = sae_features + teacher_features + predictor_features

    # Correlation matrix
    print(f"\n  {'':>18}", end="")
    for tf in teacher_features:
        print(f" {tf:>10}", end="")
    for pf in predictor_features:
        print(f" {pf:>12}", end="")
    print()
    print("  " + "-" * 75)

    for sf in sae_features:
        vals_sf = np.array([r[sf] for r in records])
        print(f"  {sf:>18}", end="")

        for tf in teacher_features:
            vals_tf = np.array([r[tf] for r in records])
            if vals_sf.std() > 1e-10 and vals_tf.std() > 1e-10:
                r = np.corrcoef(vals_sf, vals_tf)[0, 1]
                print(f" {r:>+10.3f}", end="")
            else:
                print(f" {'N/A':>10}", end="")

        for pf in predictor_features:
            vals_pf = np.array([r[pf] for r in records])
            if vals_sf.std() > 1e-10 and vals_pf.std() > 1e-10:
                r = np.corrcoef(vals_sf, vals_pf)[0, 1]
                print(f" {r:>+12.3f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

# Also: predictor ↔ teacher (baseline comparison)
print(f"\n  PREDICTOR ↔ TEACHER (baseline):")
records_all = [r for r in all_records if r["sae_layer"] == list(saes.keys())[-1]]
for pf in predictor_features:
    vals_pf = np.array([r[pf] for r in records_all])
    print(f"  {pf:>18}", end="")
    for tf in teacher_features:
        vals_tf = np.array([r[tf] for r in records_all])
        if vals_pf.std() > 1e-10 and vals_tf.std() > 1e-10:
            r = np.corrcoef(vals_pf, vals_tf)[0, 1]
            print(f" {r:>+10.3f}", end="")
        else:
            print(f" {'N/A':>10}", end="")
    print()

print(f"\nDone.", flush=True)
