"""Extrapolate on the manifold, not in ambient space.

Previous test failed because:
1. Extrapolating in 5120D ambient space — noise from 5110 irrelevant dims
2. Using all layers — early layers curve fast, late layers are flat

Fix:
1. Project hidden states to intrinsic manifold (~10-14D via PCA)
2. Extrapolate in manifold coordinates
3. Use late-layer hidden states where curvature is minimal
4. Account for the curvature by using the rotation curve shape
"""
import torch
import torch.nn.functional as F
import numpy as np
import json

device = "cuda"

print("=" * 70)
print("MANIFOLD EXTRAPOLATION — project to intrinsic dims first")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

HIDDEN = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

prompts = [
    "The theory of general relativity describes gravity as",
    "The capital of France is Paris, which is known for",
    "To solve a quadratic equation, you can use the",
    "Water freezes at zero degrees Celsius and boils at",
    "The Fibonacci sequence starts with zero and one, then each",
    "Neural networks learn by adjusting weights through a process called",
    "The speed of light in a vacuum is approximately",
    "Once upon a time in a kingdom far away there lived a",
]

N_GEN = 20

# ═══════════════════════════════════════════════════════
# Step 1: Collect hidden states for PCA basis
# Need many hidden states to find the manifold axes
# ═══════════════════════════════════════════════════════
print("\nCollecting hidden states for manifold basis...", flush=True)

# Collect from LATE layers (35-39) where curvature is minimal
collect_layers = [35, 37, 39]
all_hidden_for_pca = []

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        for layer_idx in collect_layers:
            # All token positions at this layer
            h = out.hidden_states[layer_idx][0]  # [T, H]
            all_hidden_for_pca.append(h.cpu().float())

all_hidden_for_pca = torch.cat(all_hidden_for_pca, dim=0)  # [N, H]
print(f"PCA basis: {all_hidden_for_pca.shape[0]} vectors from layers {collect_layers}", flush=True)

# Compute PCA
mean = all_hidden_for_pca.mean(dim=0, keepdim=True)
centered = all_hidden_for_pca - mean
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

# How many components capture the manifold?
var_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
print(f"\nVariance explained by top-K components:")
for k in [5, 10, 14, 20, 30, 50, 100]:
    if k < len(var_explained):
        print(f"  K={k:>3}: {var_explained[k-1]*100:.1f}%")

# Test multiple K values for extrapolation
K_VALUES = [10, 14, 20, 50, 100, 500]

# Projection matrices for each K
projections = {}
for K in K_VALUES:
    # Vt[:K] are the top-K right singular vectors [K, H]
    projections[K] = Vt[:K].to(device)  # project to K dims

mean_gpu = mean.to(device)

# ═══════════════════════════════════════════════════════
# Step 2: Generate and extrapolate on manifold
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MANIFOLD EXTRAPOLATION RESULTS")
print(f"{'='*60}")

# Use layer 39 (flattest part of rotation curve)
EXTRAP_LAYER = 39

for prompt in prompts[:6]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt}'")

    with torch.no_grad():
        # Generate N tokens, collect hidden states at late layer
        out = model(ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values

        step_hidden = [out.hidden_states[EXTRAP_LAYER][0, -1].clone()]
        step_final = [out.hidden_states[-1][0, -1].clone()]
        step_tokens = [out.logits[0, -1].argmax().item()]

        for step in range(N_GEN - 1):
            tok_input = torch.tensor([[step_tokens[-1]]], device=device)
            out = model(tok_input, past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = out.past_key_values
            step_hidden.append(out.hidden_states[EXTRAP_LAYER][0, -1].clone())
            step_final.append(out.hidden_states[-1][0, -1].clone())
            step_tokens.append(out.logits[0, -1].argmax().item())

    gen_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
    print(f"  Generated: '{gen_text[:70]}'")

    step_hidden = torch.stack(step_hidden)  # [N, H]
    step_final = torch.stack(step_final)

    # Get token probabilities for defined/branching labeling
    with torch.no_grad():
        all_logits = model.lm_head(model.model.norm(step_final.unsqueeze(0)))[0]
        all_probs = F.softmax(all_logits.float(), dim=-1)
        top1_probs = all_probs.max(dim=-1).values  # [N]

    # Test extrapolation at each K
    for K in K_VALUES:
        proj = projections[K]  # [K, H]

        # Project hidden states to manifold
        centered_h = step_hidden - mean_gpu
        manifold_coords = centered_h @ proj.T  # [N, K]

        # Linear extrapolation in manifold space
        matches = 0
        total = 0
        match_at_defined = 0
        total_defined = 0
        cos_sum = 0.0

        for t in range(2, len(manifold_coords)):
            # Extrapolate on manifold
            delta = manifold_coords[t-1] - manifold_coords[t-2]
            m_pred = manifold_coords[t-1] + delta

            # Project back to ambient space
            h_pred = (m_pred @ proj) + mean_gpu.squeeze(0)

            # Get token prediction
            h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0).to(torch.bfloat16))
            logits = model.lm_head(h_normed)[0, 0]
            pred_tok = logits.argmax().item()
            actual_tok = step_tokens[t]

            # Cosine in manifold space
            cos = F.cosine_similarity(
                m_pred.unsqueeze(0).float(),
                manifold_coords[t].unsqueeze(0).float()
            ).item()
            cos_sum += cos

            if pred_tok == actual_tok:
                matches += 1
            total += 1

            # Track defined positions separately
            if t < len(top1_probs) and top1_probs[t-1].item() > 0.8:
                total_defined += 1
                if pred_tok == actual_tok:
                    match_at_defined += 1

        avg_cos = cos_sum / max(total, 1)
        pct = matches / max(total, 1) * 100
        def_pct = match_at_defined / max(total_defined, 1) * 100

        if K in [10, 14, 50, 500]:
            print(f"  K={K:>3}: {matches}/{total} ({pct:.0f}%) matches, "
                  f"defined: {match_at_defined}/{total_defined} ({def_pct:.0f}%), "
                  f"avg manifold cos={avg_cos:.3f}")

    # Detailed per-token for K=14 (manifold dim)
    K = 14
    proj = projections[K]
    centered_h = step_hidden - mean_gpu
    manifold_coords = centered_h @ proj.T

    print(f"\n  Per-token detail (K={K}, layer {EXTRAP_LAYER}):")
    print(f"  {'t':>4} {'Token':>12} {'Predicted':>12} {'Match':>6} {'m_cos':>7} {'p1':>6} {'Type':>8}")

    for t in range(2, min(len(manifold_coords), 15)):
        delta = manifold_coords[t-1] - manifold_coords[t-2]
        m_pred = manifold_coords[t-1] + delta
        h_pred = (m_pred @ proj) + mean_gpu.squeeze(0)

        with torch.no_grad():
            h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0).to(torch.bfloat16))
            logits = model.lm_head(h_normed)[0, 0]
            pred_tok = logits.argmax().item()

        actual_tok = step_tokens[t]
        cos = F.cosine_similarity(
            m_pred.unsqueeze(0).float(),
            manifold_coords[t].unsqueeze(0).float()
        ).item()
        p1 = top1_probs[t-1].item() if t-1 < len(top1_probs) else 0
        tok_type = "DEFINED" if p1 > 0.8 else "branch" if p1 < 0.3 else "mid"
        match = "YES" if pred_tok == actual_tok else ""

        print(f"  {t:>4} {tokenizer.decode([actual_tok]):>12} "
              f"{tokenizer.decode([pred_tok]):>12} {match:>6} "
              f"{cos:>7.3f} {p1:>6.3f} {tok_type:>8}")

# ═══════════════════════════════════════════════════════
# Summary: manifold cosine vs ambient cosine
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMPARISON: manifold vs ambient extrapolation quality")
print(f"{'='*60}")

# One prompt, detailed comparison
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    out = model(ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    sh = [out.hidden_states[EXTRAP_LAYER][0, -1].clone()]
    st = [out.logits[0, -1].argmax().item()]
    for step in range(14):
        tok_input = torch.tensor([[st[-1]]], device=device)
        out = model(tok_input, past_key_values=past, use_cache=True,
                   output_hidden_states=True)
        past = out.past_key_values
        sh.append(out.hidden_states[EXTRAP_LAYER][0, -1].clone())
        st.append(out.logits[0, -1].argmax().item())

sh = torch.stack(sh)

print(f"\n{'K':>5} {'Avg m_cos':>10} {'Avg a_cos':>10} {'Matches':>8}")
for K in K_VALUES:
    proj = projections[K]
    mc = (sh - mean_gpu) @ proj.T

    m_cos_sum = 0
    a_cos_sum = 0
    matches = 0
    total = 0

    for t in range(2, len(mc)):
        # Manifold extrapolation
        delta = mc[t-1] - mc[t-2]
        m_pred = mc[t-1] + delta
        h_pred = (m_pred @ proj) + mean_gpu.squeeze(0)

        m_cos = F.cosine_similarity(m_pred.unsqueeze(0).float(), mc[t].unsqueeze(0).float()).item()

        # Ambient extrapolation (for comparison)
        a_delta = sh[t-1] - sh[t-2]
        a_pred = sh[t-1] + a_delta
        a_cos = F.cosine_similarity(a_pred.unsqueeze(0).float(), sh[t].unsqueeze(0).float()).item()

        with torch.no_grad():
            h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0).to(torch.bfloat16))
            pred_tok = model.lm_head(h_normed)[0, 0].argmax().item()
        if pred_tok == st[t]:
            matches += 1

        m_cos_sum += m_cos
        a_cos_sum += a_cos
        total += 1

    print(f"  {K:>3}   {m_cos_sum/total:>9.4f}  {a_cos_sum/total:>9.4f}  {matches}/{total}")

print(f"\nDone.", flush=True)

with open("machines/strix_halo/results/manifold_extrapolation.json", "w") as f:
    json.dump({"extrap_layer": EXTRAP_LAYER, "k_values": K_VALUES}, f, indent=2)
print("Saved results.", flush=True)
