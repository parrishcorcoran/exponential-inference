"""Train small model on whale tokens → measure manifold dimension.

The question: does marine mammal communication live on a ~10D manifold
like human language? Or a different dimensionality?

Train a tiny transformer on whale token sequences.
Measure intrinsic dimension of hidden states with TwoNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from scipy.spatial.distance import cdist

device = "cuda"

print("=" * 70)
print("WHALE MANIFOLD: train model + measure intrinsic dimension")
print("=" * 70)

# Load tokenized whale data
DATA_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/data"
with open(f"{DATA_DIR}/whale_tokens.json") as f:
    data = json.load(f)

VOCAB = data["vocab_size"]  # 512
sequences = data["sequences"]
species = data["species"]
print(f"Data: {len(sequences)} sequences, vocab={VOCAB}")

# Pad/truncate to fixed length
SEQ_LEN = 64
train_ids = []
for seq in sequences:
    if len(seq) >= SEQ_LEN:
        train_ids.append(seq[:SEQ_LEN])
    else:
        train_ids.append(seq + [0] * (SEQ_LEN - len(seq)))

train_ids = torch.tensor(train_ids, dtype=torch.long, device=device)
print(f"Training: {train_ids.shape}")

# ═══════════════════════════════════════════════════════
# Small transformer for whale sequences
# ═══════════════════════════════════════════════════════

class WhaleTransformer(nn.Module):
    """Tiny transformer for whale token sequences."""
    def __init__(self, vocab_size, hidden=256, n_heads=8, n_layers=6, intermediate=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Embedding(SEQ_LEN, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=intermediate,
                dropout=0.1, activation='gelu', batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.hidden = hidden

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        all_hidden = [h.detach()] if return_hidden else None
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
            if return_hidden:
                all_hidden.append(h.detach())

        h = self.norm(h)
        logits = self.lm_head(h)

        if return_hidden:
            return logits, all_hidden
        return logits


HIDDEN = 256
model = WhaleTransformer(VOCAB, hidden=HIDDEN, n_heads=8, n_layers=6, intermediate=512).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params/1e6:.1f}M params (H={HIDDEN}, 6 layers, 8 heads)")

# ═══════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

N_STEPS = 2000
BATCH = 16

print(f"\nTraining {N_STEPS} steps...")
print(f"{'Step':>6} {'Loss':>8}")

losses = []
for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    logits = model(batch)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           batch[:, 1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % 200 == 0 or step == N_STEPS - 1:
        print(f"{step:>6} {loss.item():>8.4f}", flush=True)

# ═══════════════════════���═══════════════════════════════
# Measure manifold dimension with TwoNN
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MANIFOLD MEASUREMENT: TwoNN intrinsic dimension")
print(f"{'='*60}")


def twonn_dimension(X):
    """TwoNN estimator (Facco et al. 2017)."""
    N = X.shape[0]
    if N < 20:
        return float('nan')

    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)

    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    valid = r1 > 1e-10
    mu = r2[valid] / r1[valid]

    if len(mu) < 10:
        return float('nan')

    N_valid = len(mu)
    log_mu = np.log(np.sort(mu))
    d_mle = N_valid / np.sum(log_mu)

    return d_mle


# Collect hidden states from the trained model
print("\nCollecting hidden states...")
model.eval()

all_hidden = {l: [] for l in range(7)}  # 0=embedding, 1-6=layers

with torch.no_grad():
    for i in range(0, len(train_ids), 16):
        batch = train_ids[i:i+16]
        _, hidden_states = model(batch, return_hidden=True)

        for l in range(len(hidden_states)):
            # Take middle positions (avoid padding effects)
            h = hidden_states[l][:, SEQ_LEN//4:3*SEQ_LEN//4, :].reshape(-1, HIDDEN)
            all_hidden[l].append(h.cpu().float().numpy())

# Measure dimension at each layer
print(f"\n{'Layer':>6} {'Dimension':>10} {'N_points':>10}")
print("-" * 30)

dimensions = {}
for l in range(7):
    points = np.concatenate(all_hidden[l], axis=0)
    # Subsample if too many
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        points = points[idx]

    dim = twonn_dimension(points)
    dimensions[f"layer_{l}"] = dim
    label = "embed" if l == 0 else f"L{l}"
    print(f"{label:>6} {dim:>10.2f} {len(points):>10}")

# Per-species dimension
print(f"\n{'='*60}")
print("PER-SPECIES MANIFOLD DIMENSION (last layer)")
print(f"{'='*60}")

species_dims = {}
for sp in set(species):
    sp_indices = [i for i, s in enumerate(species) if s == sp and i < len(train_ids)]
    if len(sp_indices) < 20:
        continue

    sp_batch = train_ids[sp_indices[:min(100, len(sp_indices))]]
    with torch.no_grad():
        _, hidden_states = model(sp_batch, return_hidden=True)
        h_last = hidden_states[-1][:, SEQ_LEN//4:3*SEQ_LEN//4, :].reshape(-1, HIDDEN).cpu().float().numpy()

    if len(h_last) > 3000:
        idx = np.random.choice(len(h_last), 3000, replace=False)
        h_last = h_last[idx]

    dim = twonn_dimension(h_last)
    species_dims[sp] = dim
    print(f"  {sp:>35}: dim = {dim:.2f} ({len(sp_indices)} sequences)")

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

layer_dims = [dimensions[f"layer_{l}"] for l in range(7)]
print(f"  Overall manifold dimension: {np.nanmean(layer_dims):.2f} (mean across layers)")
print(f"  Embedding layer: {dimensions['layer_0']:.2f}")
print(f"  Final layer: {dimensions['layer_6']:.2f}")

if species_dims:
    sp_vals = [v for v in species_dims.values() if not np.isnan(v)]
    print(f"  Per-species range: {min(sp_vals):.2f} — {max(sp_vals):.2f}")
    print(f"  Per-species mean: {np.mean(sp_vals):.2f}")

print(f"\n  COMPARISON:")
print(f"    Human language (Qwen3 family): 9.07 — 10.89")
print(f"    Whale/dolphin sounds:          {dimensions.get('layer_6', 'N/A'):.2f}")

match = ""
whale_dim = dimensions.get('layer_6', 0)
if 7 < whale_dim < 13:
    match = "SIMILAR to human language!"
elif whale_dim < 7:
    match = "simpler than human language"
elif whale_dim > 13:
    match = "MORE complex than human language"
print(f"    → {match}")

with open(f"{DATA_DIR}/whale_manifold.json", "w") as f:
    json.dump({"dimensions": dimensions, "species_dims": species_dims,
               "training_loss": losses[-1]}, f, indent=2)
print(f"\nSaved.", flush=True)
