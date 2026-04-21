"""Manifold read: project to intrinsic coordinates, measure local geometry.

Step 1: Build the manifold atlas (PCA basis + calibration points)
        from diverse text through the full model.

Step 2: For each generated token, project hidden state to manifold
        coordinates, compute local density and curvature.

Step 3: Test if density/curvature correlate with token definedness
        (top1 probability) BETTER than any signal we've tried.

If this works: density → head count, curvature → layer count.
Direct manifold measurement → routing.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time

device = "cuda"

print("=" * 70)
print("MANIFOLD READ — intrinsic coordinates + local geometry")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
H = model.config.hidden_size

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Step 1: Build manifold atlas from diverse text
# ═══════════════════════════════════════════════════════
print(f"\nBuilding manifold atlas...", flush=True)

atlas_texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations.",
    "Dolphins are highly intelligent marine mammals known for their social behavior and communication systems.",
    "The process of fermentation has been used for thousands of years to produce bread wine and cheese.",
    "Modern architecture emphasizes clean lines open spaces and the integration of buildings with surroundings.",
    "The immune system protects the body against infection through a complex network of cells and proteins.",
    "Jazz music originated in the African American communities of New Orleans in the late nineteenth century.",
    "Volcanic eruptions can have devastating effects on local ecosystems but also create fertile soil.",
    "The printing press invented by Gutenberg around 1440 revolutionized the spread of knowledge throughout Europe.",
    "Coral reefs are among the most diverse ecosystems on the planet supporting thousands of marine species.",
    "The development of antibiotics in the twentieth century transformed medicine and saved millions of lives.",
    "Ancient Egyptian civilization flourished along the Nile River for over three thousand years.",
    "Artificial neural networks were inspired by the structure and function of biological brains.",
    "The theory of plate tectonics explains how continents drift and why earthquakes occur along fault lines.",
    "Renaissance art was characterized by a renewed interest in classical antiquity and a focus on humanism.",
    "Black holes form when massive stars collapse under their own gravity at the end of their life cycle.",
    "The water cycle describes the continuous movement of water on above and below the surface of the Earth.",
    "Shakespeare wrote approximately thirty seven plays during his literary career.",
    "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials.",
    "Genetic algorithms are optimization techniques inspired by the process of natural selection.",
    "The Great Wall of China was built over many centuries to protect the northern borders.",
]

# Collect hidden states at layer 35 (where manifold is resolved, curvature minimal)
ATLAS_LAYER = 35
atlas_points = []
atlas_probs = []  # top1 probability for each position (definedness label)

lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

for text in atlas_texts:
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        h = out.hidden_states[ATLAS_LAYER][0]  # [T, H]
        atlas_points.append(h.cpu().float())

        # Get definedness (top1 prob) for each position
        final_h = out.hidden_states[-1]
        logits = F.linear(final_norm(final_h), lm_head_weight)[0]
        probs = F.softmax(logits.float(), dim=-1)
        top1 = probs.max(dim=-1).values
        atlas_probs.append(top1.cpu())

atlas_points = torch.cat(atlas_points, dim=0)  # [N, H]
atlas_probs = torch.cat(atlas_probs, dim=0)    # [N]

print(f"Atlas: {atlas_points.shape[0]} points, layer {ATLAS_LAYER}", flush=True)

# PCA: find manifold basis
mean = atlas_points.mean(dim=0, keepdim=True)
centered = atlas_points - mean
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

# Top K components
var_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
for k in [5, 10, 14, 20]:
    print(f"  Top {k} components: {var_explained[k-1]*100:.1f}% variance")

# Use top 14 (matches manifold dim + KV coordinate)
K_MANIFOLD = 14
basis = Vt[:K_MANIFOLD]  # [K, H]

# Project atlas to manifold coordinates
atlas_coords = (centered @ basis.T)  # [N, K]

print(f"  Manifold coordinates: {atlas_coords.shape}", flush=True)

# ═══════════════════════════════════════════════════════
# Step 2: Define geometry measurements on the manifold
# ═══════════════════════════════════════════════════════

def manifold_density(coords, atlas_coords, k=5):
    """Local density: average distance to k nearest neighbors in manifold space."""
    dists = torch.cdist(coords.unsqueeze(0), atlas_coords.unsqueeze(0))[0]  # [N_query, N_atlas]
    knn_dists = dists.topk(k, largest=False, dim=-1).values  # [N_query, k]
    return knn_dists.mean(dim=-1)  # [N_query] — lower = denser


def manifold_curvature(coords, atlas_coords, k=10):
    """Local curvature estimate: variance of directions to neighbors.

    High variance = high curvature (surface bends a lot locally).
    Low variance = flat (surface is nearly planar locally).
    """
    dists = torch.cdist(coords.unsqueeze(0), atlas_coords.unsqueeze(0))[0]
    _, knn_idx = dists.topk(k, largest=False, dim=-1)  # [N_query, k]

    curvatures = []
    for i in range(coords.shape[0]):
        neighbors = atlas_coords[knn_idx[i]]  # [k, K]
        # Directions from point to neighbors
        directions = neighbors - coords[i].unsqueeze(0)  # [k, K]
        directions = F.normalize(directions, dim=-1)
        # Curvature proxy: how spread out are the directions?
        # Low spread = flat, high spread = curved
        mean_dir = directions.mean(dim=0, keepdim=True)
        cos_to_mean = (directions * mean_dir).sum(dim=-1)  # [k]
        curvature = 1.0 - cos_to_mean.mean()  # 0 = flat, 1 = maximally curved
        curvatures.append(curvature.item())

    return torch.tensor(curvatures)


# Compute geometry for all atlas points
print(f"\nComputing manifold geometry for atlas...", flush=True)
atlas_density = manifold_density(atlas_coords, atlas_coords, k=5)
atlas_curvature = manifold_curvature(atlas_coords, atlas_coords, k=10)

# ═══════════════════════════════════════════════════════
# Step 3: Correlate geometry with definedness
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MANIFOLD GEOMETRY → TOKEN DEFINEDNESS")
print("Direct measurement. No signals.")
print(f"{'='*60}")

# Correlations
density_np = atlas_density.numpy()
curvature_np = atlas_curvature.numpy()
top1_np = atlas_probs.numpy()

# Filter out padding/special tokens (very high/low prob)
valid = (top1_np > 0.001) & (top1_np < 0.999)
d_valid = density_np[valid]
c_valid = curvature_np[valid]
p_valid = top1_np[valid]

if len(d_valid) > 10:
    r_density = np.corrcoef(d_valid, p_valid)[0, 1]
    r_curvature = np.corrcoef(c_valid, p_valid)[0, 1]

    print(f"\n  Correlation with top1_prob (definedness):")
    print(f"    Manifold density (KNN):    r = {r_density:+.3f}")
    print(f"    Manifold curvature:        r = {r_curvature:+.3f}")
    print(f"    (Previous best signal was stab_depth: r = -0.367)")

    # Cohen's d for defined (p>0.8) vs branching (p<0.4)
    defined = p_valid > 0.8
    branching = p_valid < 0.4

    if defined.sum() > 5 and branching.sum() > 5:
        d_def = d_valid[defined]
        d_br = d_valid[branching]
        pooled = np.sqrt((d_def.std()**2 + d_br.std()**2) / 2) + 1e-10
        cohens_d_density = abs(d_def.mean() - d_br.mean()) / pooled

        c_def = c_valid[defined]
        c_br = c_valid[branching]
        pooled_c = np.sqrt((c_def.std()**2 + c_br.std()**2) / 2) + 1e-10
        cohens_d_curv = abs(c_def.mean() - c_br.mean()) / pooled_c

        print(f"\n  Defined ({defined.sum()}) vs Branching ({branching.sum()}):")
        print(f"    Density:   def={d_def.mean():.3f}±{d_def.std():.3f} "
              f"br={d_br.mean():.3f}±{d_br.std():.3f} Cohen's d={cohens_d_density:.2f}")
        print(f"    Curvature: def={c_def.mean():.3f}±{c_def.std():.3f} "
              f"br={c_br.mean():.3f}±{c_br.std():.3f} Cohen's d={cohens_d_curv:.2f}")

# ═══════════════════════════════════════════════════════
# Step 4: Test on NEW text (not in atlas)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GENERALIZATION: manifold read on held-out text")
print(f"{'='*60}")

test_texts = [
    "The future of artificial intelligence will be shaped by the interplay of three key factors.",
    "In quantum mechanics the wave function describes the probability amplitude of a particle.",
    "The stock market experienced significant volatility during the economic downturn of two thousand eight.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors in an environment.",
]

test_points = []
test_probs = []

for text in test_texts:
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        h = out.hidden_states[ATLAS_LAYER][0]
        test_points.append(h.cpu().float())

        final_h = out.hidden_states[-1]
        logits = F.linear(final_norm(final_h), lm_head_weight)[0]
        probs = F.softmax(logits.float(), dim=-1)
        top1 = probs.max(dim=-1).values
        test_probs.append(top1.cpu())

test_points = torch.cat(test_points, dim=0)
test_probs = torch.cat(test_probs, dim=0)

# Project to manifold
test_centered = test_points - mean
test_coords = test_centered @ basis.T  # [N_test, K]

# Measure geometry
test_density = manifold_density(test_coords, atlas_coords, k=5)
test_curvature = manifold_curvature(test_coords, atlas_coords, k=10)

# Correlations on test set
td = test_density.numpy()
tc = test_curvature.numpy()
tp = test_probs.numpy()

valid_t = (tp > 0.001) & (tp < 0.999)
if valid_t.sum() > 10:
    r_d_test = np.corrcoef(td[valid_t], tp[valid_t])[0, 1]
    r_c_test = np.corrcoef(tc[valid_t], tp[valid_t])[0, 1]

    print(f"\n  Held-out correlations with top1_prob:")
    print(f"    Manifold density:    r = {r_d_test:+.3f}")
    print(f"    Manifold curvature:  r = {r_c_test:+.3f}")

# Show examples: most dense vs most sparse in test set
print(f"\n  Examples (test set):")
print(f"  {'Density':>8} {'Curvature':>10} {'p1':>6} {'Token':>15}")

# Get tokens for display
all_test_tokens = []
for text in test_texts:
    ids = tokenizer(text, return_tensors='pt').input_ids[0]
    all_test_tokens.extend([tokenizer.decode(ids[i:i+1]) for i in range(len(ids))])

sorted_idx = td.argsort()
print("\n  Most DENSE (predicted defined):")
for i in sorted_idx[:6]:
    tok = all_test_tokens[i] if i < len(all_test_tokens) else "?"
    print(f"  {td[i]:>8.3f} {tc[i]:>10.3f} {tp[i]:>6.3f} '{tok}'")

print("\n  Most SPARSE (predicted branching):")
for i in sorted_idx[-6:]:
    tok = all_test_tokens[i] if i < len(all_test_tokens) else "?"
    print(f"  {td[i]:>8.3f} {tc[i]:>10.3f} {tp[i]:>6.3f} '{tok}'")

# ═══════════════════════════════════════════════════════
# Combined model: density + curvature as manifold read
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMBINED: density + curvature → routing signal")
print(f"{'='*60}")

# Simple linear combination: can density + curvature predict p1?
from numpy.linalg import lstsq

X_atlas = np.column_stack([d_valid, c_valid, np.ones(len(d_valid))])
coef, _, _, _ = lstsq(X_atlas, p_valid, rcond=None)

pred_atlas = X_atlas @ coef
r_combined_atlas = np.corrcoef(pred_atlas, p_valid)[0, 1]

X_test = np.column_stack([td[valid_t], tc[valid_t], np.ones(valid_t.sum())])
pred_test = X_test @ coef
r_combined_test = np.corrcoef(pred_test, tp[valid_t])[0, 1]

print(f"\n  Linear model: p1 ≈ {coef[0]:.3f}*density + {coef[1]:.3f}*curvature + {coef[2]:.3f}")
print(f"  Atlas R:       {r_combined_atlas:.3f}")
print(f"  Held-out R:    {r_combined_test:.3f}")
print(f"  (Compare: stab_depth alone was r=-0.367)")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/manifold_read.json", "w") as f:
    json.dump({
        "atlas_layer": ATLAS_LAYER,
        "k_manifold": K_MANIFOLD,
        "r_density_atlas": float(r_density) if 'r_density' in dir() else None,
        "r_curvature_atlas": float(r_curvature) if 'r_curvature' in dir() else None,
    }, f, indent=2)
print("Saved.", flush=True)
