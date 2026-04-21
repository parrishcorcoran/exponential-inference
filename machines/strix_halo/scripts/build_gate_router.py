"""Port unified-gate features to 14B and build the manifold router.

The unified-gate repo has proven features (R²=0.448 with logit lens on 0.6B).
These are MANIFOLD MEASUREMENTS:
- Neighborhood distances (KNN to recent states)
- State velocity/acceleration
- Hidden norm drift
- Phase-SVD coordinates (local SVD of trajectory)
- RG divergence at multiple scales
- Holographic boundary/bulk entropy

Port these to Qwen3-14B. Measure correlation with definedness.
If they work: this IS the manifold router.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import time

device = "cuda"

print("=" * 70)
print("UNIFIED-GATE FEATURES → 14B MANIFOLD ROUTER")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers

lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Feature computation (ported from unified-gate)
# ═══════════════════════════════════════════════════════

def compute_gate_features(hidden_seq, probs_seq):
    """Compute unified-gate features for a sequence.

    hidden_seq: [T, H] numpy — hidden states at last layer
    probs_seq: [T, V] numpy — output probabilities at each position

    Returns: [T, N_FEATURES] numpy
    """
    T = hidden_seq.shape[0]
    h = hidden_seq
    probs = probs_seq

    features = {}

    # ── Neighborhood: distances to recent states ──
    h_t = torch.from_numpy(h).float()
    nbr_min = np.zeros(T, dtype=np.float32)
    nbr_mean = np.zeros(T, dtype=np.float32)
    fifo_size = 50
    for i in range(1, T):
        lo = max(0, i - fifo_size)
        fifo = h_t[lo:i]
        cur = h_t[i:i+1]
        dists = torch.cdist(cur, fifo).squeeze(0)
        nbr_min[i] = float(dists.min())
        nbr_mean[i] = float(dists.mean())
    features["nbr_min_dist"] = nbr_min
    features["nbr_mean_dist"] = nbr_mean

    # ── Velocity / acceleration ──
    vel = np.zeros(T, dtype=np.float32)
    accel = np.zeros(T, dtype=np.float32)
    for i in range(1, T):
        vel[i] = np.linalg.norm(h[i] - h[i-1])
    for i in range(2, T):
        accel[i] = np.linalg.norm(h[i] - 2*h[i-1] + h[i-2])
    features["state_velocity"] = vel
    features["state_accel"] = accel

    # ── Hidden norm + drift ──
    norms = np.linalg.norm(h, axis=-1)
    norm_drift = np.zeros(T, dtype=np.float32)
    norm_drift[1:] = np.abs(norms[1:] - norms[:-1])
    features["hidden_norm"] = np.log1p(norms)
    features["norm_drift"] = norm_drift

    # ── Phase-SVD (local SVD of trajectory window) ──
    window = 20
    phase_residual = np.zeros(T, dtype=np.float32)
    phase_c1 = np.zeros(T, dtype=np.float32)
    for i in range(window, T):
        win = h[i-window:i]
        win_c = win - win.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(win_c, full_matrices=False)
            h_cur = h[i] - win.mean(axis=0)
            coords = h_cur @ Vt[:3].T
            phase_c1[i] = coords[0]
            reconstruct = coords @ Vt[:3]
            phase_residual[i] = np.linalg.norm(h_cur - reconstruct)
        except:
            pass
    features["phase_c1"] = phase_c1
    features["phase_residual"] = phase_residual

    # ── RG divergence at multiple scales ──
    rg_div1 = np.zeros(T, dtype=np.float32)
    rg_div3 = np.zeros(T, dtype=np.float32)
    rg_div9 = np.zeros(T, dtype=np.float32)
    for i in range(10, T):
        rg_div1[i] = np.linalg.norm(h[i] - h[i-1])
        rg_div3[i] = np.linalg.norm(h[i] - h[max(0,i-3):i].mean(axis=0))
        rg_div9[i] = np.linalg.norm(h[i] - h[max(0,i-9):i].mean(axis=0))
    features["rg_div1"] = rg_div1
    features["rg_div3"] = rg_div3
    features["rg_div9"] = rg_div9

    # ── Holographic: surface/bulk entropy ──
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=-1)
    conf = probs.max(axis=-1)

    surface_ent = np.zeros(T, dtype=np.float32)
    bulk_ent = np.zeros(T, dtype=np.float32)
    for i in range(50, T):
        surface_ent[i] = entropy[i-10:i].mean()
        bulk_ent[i] = entropy[i-50:i-10].mean()
    features["surface_ent"] = surface_ent
    features["bulk_ent"] = bulk_ent

    # ── Correlation length ──
    corr_len = np.zeros(T, dtype=np.float32)
    for i in range(50, T):
        early = norms[i-50:i-25]
        late = norms[i-25:i]
        if len(early) > 1 and len(late) > 1:
            minlen = min(len(early), len(late))
            c = np.corrcoef(early[:minlen], late[:minlen])[0, 1]
            corr_len[i] = c if np.isfinite(c) else 0
    features["corr_len"] = corr_len

    # ── Event horizon ──
    ev_horiz = np.full(T, 200.0, dtype=np.float32)
    last_confident = -1
    for i in range(T):
        if last_confident >= 0:
            d = np.linalg.norm(h[i] - h[last_confident])
            ev_horiz[i] = min(200, d)
        if conf[i] > 0.8:
            last_confident = i
    features["event_horizon"] = np.log1p(ev_horiz)

    # Stack all features
    feat_names = sorted(features.keys())
    feat_matrix = np.stack([features[n] for n in feat_names], axis=1)
    return feat_matrix, feat_names


# ═══════════════════════════════════════════════════════
# Collect features on diverse text
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world. Ancient Egyptians used geometry for surveying land after the annual flooding of the Nile. Greek mathematicians like Euclid and Archimedes laid the foundations of formal proof.",
    "Marine biology is the scientific study of organisms that live in the ocean and other saltwater environments. The ocean covers more than seventy percent of the Earth surface and contains an incredible diversity of life forms from microscopic plankton to enormous blue whales.",
    "The development of artificial intelligence has progressed through several distinct phases since its inception in the 1950s. Early systems relied on symbolic reasoning and hand crafted rules to solve specific problems. The introduction of machine learning shifted the focus toward statistical pattern recognition.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales where particles exhibit both wave and particle properties simultaneously. The uncertainty principle places fundamental limits on the precision of measurements.",
]

print(f"\nCollecting features on {len(texts)} sequences...", flush=True)

all_features = []
all_top1 = []

for text in texts:
    ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        out = model.model(ids, output_hidden_states=True)
        hidden_last = out.hidden_states[-1][0].cpu().float().numpy()  # [T, H]

        # Get output probs
        logits = F.linear(final_norm(out.hidden_states[-1]), lm_head_weight)[0]
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()  # [T, V]
        top1 = probs.max(axis=-1)  # [T]

    feat_matrix, feat_names = compute_gate_features(hidden_last, probs)
    all_features.append(feat_matrix)
    all_top1.append(top1)
    print(f"  {T} tokens processed", flush=True)

all_features = np.concatenate(all_features, axis=0)
all_top1 = np.concatenate(all_top1, axis=0)

print(f"\nTotal: {all_features.shape[0]} tokens, {all_features.shape[1]} features")
print(f"Features: {feat_names}")

# ═══════════════════════════════════════════════════════
# Correlate with definedness
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FEATURE CORRELATIONS WITH DEFINEDNESS (top1_prob)")
print("These are MANIFOLD MEASUREMENTS, not prediction signals")
print(f"{'='*60}")

# Filter valid positions (skip first 50 for windowed features, skip padding)
valid = (all_top1 > 0.001) & (all_top1 < 0.999) & (np.arange(len(all_top1)) % 70 > 50)
# Actually just skip the first 50 positions total per sequence isn't clean
# Let's use all positions where features are non-zero
valid = (all_top1 > 0.001) & (all_features.sum(axis=1) != 0)

print(f"\nValid positions: {valid.sum()}")
print(f"\n{'Feature':>20} {'r(top1)':>8} {'|r|':>6}")
print("-" * 40)

correlations = {}
for i, name in enumerate(feat_names):
    feat_vals = all_features[valid, i]
    top1_vals = all_top1[valid]
    if feat_vals.std() > 1e-10:
        r = np.corrcoef(feat_vals, top1_vals)[0, 1]
    else:
        r = 0.0
    correlations[name] = r
    print(f"{name:>20} {r:>+8.3f} {abs(r):>6.3f}")

# Sort by |r|
print(f"\nRanked by |correlation|:")
ranked = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for name, r in ranked[:10]:
    print(f"  {name:>20}: r = {r:>+.3f}")

# Combined model
print(f"\n{'='*60}")
print("LINEAR MODEL: all features → top1_prob")
print(f"{'='*60}")

from numpy.linalg import lstsq

X = all_features[valid]
y = all_top1[valid]

# Add intercept
X_aug = np.column_stack([X, np.ones(len(X))])
coef, residuals, rank, sv = lstsq(X_aug, y, rcond=None)

pred = X_aug @ coef
r_combined = np.corrcoef(pred, y)[0, 1]
r2 = r_combined ** 2

print(f"  Combined R: {r_combined:.3f}")
print(f"  Combined R²: {r2:.3f}")
print(f"  (Compare: unified-gate on 0.6B got R²=0.448 with logit lens)")

# Cross-validated: use first half to train, second half to test
mid = len(X) // 2
X_train, y_train = X[:mid], y[:mid]
X_test, y_test = X[mid:], y[mid:]

X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])

coef_cv, _, _, _ = lstsq(X_train_aug, y_train, rcond=None)
pred_cv = X_test_aug @ coef_cv
r_cv = np.corrcoef(pred_cv, y_test)[0, 1]
print(f"  Cross-validated R: {r_cv:.3f} (train on first half, test on second)")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/gate_router_14b.json", "w") as f:
    json.dump({"correlations": correlations, "r_combined": float(r_combined),
               "r_cv": float(r_cv), "feat_names": feat_names}, f, indent=2)
print("Saved.", flush=True)
