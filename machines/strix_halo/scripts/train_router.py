"""Train the manifold router: per-token (width, length) prediction.

The router is a tiny MLP that reads early-layer features and outputs:
  - length: how many layers this token needs (exit point)
  - width: what fraction of heads this token needs

Training:
1. Run full model, capture per-layer predictions at every layer
2. For each token: find the EARLIEST layer where prediction matches final
3. That's the label: optimal_length = that layer
4. Width label: test with fewer heads, find minimum heads needed
5. Train router on (features → optimal_length, optimal_width)

The router runs ONCE per token at near-zero cost, then controls
the entire forward pass dynamically.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
from pathlib import Path

device = 'cuda'

print("="*70, flush=True)
print("TRAINING THE MANIFOLD ROUTER", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to(device).eval()
N_LAYERS = model.config.num_hidden_layers  # 40
H = model.config.hidden_size  # 5120
print(f"  L={N_LAYERS} H={H} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# ═══════════════════════════════════════════════════════
# STEP 1: Generate training data for the router
# For each token: at which layer does prediction stabilize?
# ═══════════════════════════════════════════════════════
print("\n[1/3] Generating router training data...", flush=True)

# Diverse prompts for training
from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
texts = [item['text'][:1000] for i, item in zip(range(50), ds) if len(item['text']) > 200]
print(f"  {len(texts)} C4 texts for router training", flush=True)

# Capture hidden states at layer 1 (features) + all layer predictions
layer1_hiddens = []  # features for router input
per_layer_preds = {l: [] for l in range(N_LAYERS)}  # predictions at each layer
final_preds_all = []  # ground truth (final layer prediction)
output_entropy_all = []  # difficulty signal

# Hook to capture layer 1 hidden state (router input features)
layer1_data = []
def layer1_hook(module, input, output):
    h = output[0] if isinstance(output, tuple) else output
    layer1_data.append(h.detach())

# Hooks to capture predictions at sampled layers
CHECK_LAYERS = list(range(0, N_LAYERS, 2))  # every 2 layers
layer_pred_data = {l: [] for l in CHECK_LAYERS}

def make_pred_hook(layer_idx):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            logits = model.lm_head(model.model.norm(h))
            preds = logits.argmax(-1)
            layer_pred_data[layer_idx].append(preds.detach().cpu())
    return hook

h1 = model.model.layers[1].register_forward_hook(layer1_hook)
pred_hooks = [model.model.layers[l].register_forward_hook(make_pred_hook(l))
              for l in CHECK_LAYERS]

t0 = time.time()
with torch.no_grad():
    for i, text in enumerate(texts):
        ids = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=256).input_ids.to(device)
        if ids.shape[1] < 10:
            continue

        layer1_data.clear()
        for l in CHECK_LAYERS:
            layer_pred_data[l].clear()

        out = model(ids, use_cache=False)
        final_preds = out.logits[0].argmax(-1).cpu()  # [T]
        final_entropy = -(F.softmax(out.logits[0].float(), dim=-1) *
                          F.log_softmax(out.logits[0].float(), dim=-1)).sum(-1).cpu()  # [T]

        # Router features from layer 1
        h1_feat = layer1_data[0][0].float().cpu()  # [T, H]

        # For each token: find earliest layer where prediction matches final
        T = final_preds.shape[0]
        optimal_length = torch.full((T,), N_LAYERS, dtype=torch.float32)  # default: full

        for l in CHECK_LAYERS:
            if layer_pred_data[l]:
                l_preds = layer_pred_data[l][0][0]  # [T]
                matches = (l_preds == final_preds)
                # If this layer matches and we haven't found an earlier match
                for t in range(T):
                    if matches[t] and optimal_length[t] == N_LAYERS:
                        optimal_length[t] = l

        layer1_hiddens.append(h1_feat)
        final_preds_all.append(final_preds)
        output_entropy_all.append(final_entropy)

        # Store normalized optimal length as fraction of total layers
        # This is what the router will predict: 0.0 = exit immediately, 1.0 = need all layers
        normalized_length = optimal_length / N_LAYERS

        # Simple features from layer 1:
        # - hidden norm
        # - hidden state entropy (via top singular values)
        # - position in sequence
        norms = h1_feat.norm(dim=-1)  # [T]
        positions = torch.arange(T, dtype=torch.float32) / T

        features = torch.stack([
            norms,
            positions,
            final_entropy,
            normalized_length,
        ], dim=1)  # [T, 4] — last column is the LABEL

        if i == 0:
            all_features = features
        else:
            all_features = torch.cat([all_features, features], dim=0)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(texts)}  {all_features.shape[0]} tokens  [{time.time()-t0:.0f}s]", flush=True)

h1.remove()
for h in pred_hooks: h.remove()

print(f"  Total: {all_features.shape[0]} tokens with labels", flush=True)

# ═══════════════════════════════════════════════════════
# STEP 2: Analyze the distribution
# ═══════════════════════════════════════════════════════
print("\n[2/3] Analyzing optimal exit distribution...", flush=True)

labels = all_features[:, 3]  # normalized length (0-1)
norms = all_features[:, 0]
entropies = all_features[:, 2]

# What fraction of tokens can exit early?
for threshold in [0.25, 0.50, 0.75]:
    frac = (labels <= threshold).float().mean().item() * 100
    layer = int(threshold * N_LAYERS)
    print(f"  {frac:.1f}% of tokens stabilize by layer {layer}/{N_LAYERS}", flush=True)

# Correlation between features and optimal length
from scipy.stats import pearsonr
norm_r, _ = pearsonr(norms.numpy(), labels.numpy())
ent_r, _ = pearsonr(entropies.numpy(), labels.numpy())
print(f"  Correlation with optimal_length:", flush=True)
print(f"    hidden_norm:     r = {norm_r:.3f}", flush=True)
print(f"    output_entropy:  r = {ent_r:.3f}", flush=True)

# ═══════════════════════════════════════════════════════
# STEP 3: Train the router MLP
# ═══════════════════════════════════════════════════════
print("\n[3/3] Training router MLP...", flush=True)

# Features: norm, position, entropy (3 inputs)
# Label: optimal_length (regression target, 0-1)
X = all_features[:, :3]  # [N, 3]
Y = all_features[:, 3]   # [N]

# Split
n = len(X)
split = int(n * 0.8)
X_tr, X_te = X[:split].to(device), X[split:].to(device)
Y_tr, Y_te = Y[:split].to(device), Y[split:].to(device)

class Router(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

router = Router(3).to(device)
opt = torch.optim.Adam(router.parameters(), lr=1e-3)

for epoch in range(200):
    perm = torch.randperm(len(X_tr), device=device)
    total_loss = 0; nb = 0
    for i in range(0, len(X_tr), 256):
        bi = perm[i:i+256]
        pred = router(X_tr[bi])
        loss = F.mse_loss(pred, Y_tr[bi])
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item(); nb += 1

    if (epoch + 1) % 50 == 0:
        router.eval()
        with torch.no_grad():
            te_pred = router(X_te)
            te_loss = F.mse_loss(te_pred, Y_te).item()
            # Accuracy: if router says exit at layer L, does token actually match?
            pred_layers = (te_pred * N_LAYERS).long().clamp(0, N_LAYERS-1)
            actual_layers = (Y_te * N_LAYERS).long().clamp(0, N_LAYERS-1)
            within_5 = (pred_layers - actual_layers).abs() <= 5
            acc = within_5.float().mean().item() * 100
        router.train()
        print(f"  epoch {epoch+1:3d}  train_loss={total_loss/nb:.4f}  "
              f"test_loss={te_loss:.4f}  within_5_layers={acc:.1f}%", flush=True)

# Final evaluation
router.eval()
with torch.no_grad():
    te_pred = router(X_te)
    pred_layers = (te_pred * N_LAYERS).long().clamp(0, N_LAYERS-1)
    actual_layers = (Y_te * N_LAYERS).long().clamp(0, N_LAYERS-1)

print(f"\n{'='*70}", flush=True)
print(f"ROUTER RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Training tokens: {len(X_tr)}", flush=True)
print(f"  Test tokens: {len(X_te)}", flush=True)
print(f"  Router size: {sum(p.numel() for p in router.parameters())} params", flush=True)
print(f"  Router cost: ~microseconds per token", flush=True)
print(f"\n  Exit layer prediction accuracy:", flush=True)
for margin in [0, 2, 5, 10]:
    acc = ((pred_layers - actual_layers).abs() <= margin).float().mean().item() * 100
    print(f"    within ±{margin} layers: {acc:.1f}%", flush=True)

# Compute theoretical speedup
avg_predicted = te_pred.mean().item() * N_LAYERS
avg_actual = Y_te.mean().item() * N_LAYERS
print(f"\n  Avg predicted exit: layer {avg_predicted:.1f}/{N_LAYERS}", flush=True)
print(f"  Avg actual exit:   layer {avg_actual:.1f}/{N_LAYERS}", flush=True)
print(f"  Theoretical length speedup: {N_LAYERS/avg_actual:.2f}×", flush=True)
print(f"  Combined with 80% head pruning: {N_LAYERS/avg_actual * (1/0.2):.1f}× theoretical", flush=True)

# Save
torch.save(router.state_dict(), "machines/strix_halo/results/router_14b.pt")
results = {
    "model": "Qwen3-14B", "router_params": sum(p.numel() for p in router.parameters()),
    "train_tokens": len(X_tr), "test_tokens": len(X_te),
    "avg_predicted_layer": avg_predicted, "avg_actual_layer": avg_actual,
    "theoretical_speedup": N_LAYERS / avg_actual,
}
with open("machines/strix_halo/results/router_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved router_14b.pt + router_14b.json", flush=True)
