"""Train depth router end-to-end.

Router predicts depth. Model runs at that depth. Loss = did you get the right token.
KV rule: 1 head if within layers, 8 if outside.

End-to-end: router learns what depth each token needs by being
penalized when it picks wrong (too shallow = wrong token).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

device = "cuda"

print("=" * 70)
print("TRAIN DEPTH ROUTER — end-to-end, like MoE")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

N_LAYERS = model.config.num_hidden_layers
H = model.config.hidden_size
N_KV = model.config.num_key_value_heads

print(f"L={N_LAYERS} H={H} KV={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


class DepthRouter(nn.Module):
    """Predicts depth (0-1) from hidden state after layer 1.

    Output: depth fraction. Multiply by N_LAYERS to get exit layer.
    KV rule applied externally: depth ≤ 1.0 → 1 KV head, else → 8.
    """
    def __init__(self, hidden_dim, bottleneck=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        """h: [B, H] → depth: [B] in (0, 1)"""
        return self.net(h).squeeze(-1)


router = DepthRouter(H).to(device).float()
print(f"Router: {sum(p.numel() for p in router.parameters())/1e3:.0f}K params")

# ═══════════════════════════════════════════════════════
# Training: end-to-end
# For each token: router picks depth, model runs to that depth,
# check if the token at that depth matches the full-depth token.
# Loss: encourage minimum depth that still gets the right answer.
# ═══════════════════════════════════════════════════════

texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Marine biology studies organisms in the ocean covering more than seventy percent of Earth surface area.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales of existence.",
    "The French Revolution transformed society by uprooting centuries of tradition and absolute monarchy.",
    "Climate change driven by burning fossil fuels threatens ecosystems worldwide through rising temperatures.",
    "The human genome contains approximately three billion base pairs of DNA organized into chromosomes.",
    "Machine learning algorithms improve through experience without being explicitly programmed for tasks.",
    "The Amazon rainforest produces significant oxygen and houses incredible biodiversity across species.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in properties.",
] * 4

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=64, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

optimizer = torch.optim.AdamW(router.parameters(), lr=1e-3, weight_decay=0.01)

N_STEPS = 1000
BATCH = 4
DEPTH_PENALTY = 0.1  # encourage shallow when possible

lm_head_weight = model.lm_head.weight
final_norm_layer = model.model.norm

print(f"\nTraining {N_STEPS} steps, batch={BATCH}")
print(f"Loss = CE(predicted_token, true_token) + {DEPTH_PENALTY}*depth_penalty")
print(f"{'Step':>6} {'Loss':>8} {'CE':>7} {'Depth':>7} {'Avg_d':>7}")
print("-" * 40)

losses = []
for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]  # [B, T]

    with torch.no_grad():
        # Run full model to get hidden states at all layers
        out = model.model(batch, output_hidden_states=True)
        hidden_states = out.hidden_states  # (L+1) × [B, T, H]

        # Ground truth: full model's prediction
        full_logits = F.linear(final_norm_layer(hidden_states[-1]), lm_head_weight)
        full_pred = full_logits[:, :-1].argmax(-1)  # [B, T-1]

    # Router input: hidden state after layer 1 (has attention context)
    h_L1 = hidden_states[1].detach().float()  # [B, T, H]

    # Router predicts depth for each position
    # Use positions [0, T-2] (predicting next token at [1, T-1])
    h_input = h_L1[:, :-1].reshape(-1, H)  # [B*(T-1), H]
    pred_depth = router(h_input)  # [B*(T-1)] in (0, 1)

    # For each token: get the logits at the predicted depth
    # Use soft attention over layers (differentiable depth selection)
    # pred_depth * N_LAYERS = target layer (continuous)
    target_layer = pred_depth * (N_LAYERS - 1)  # [B*(T-1)]

    # Soft interpolation between layers for differentiability
    layer_below = target_layer.long().clamp(0, N_LAYERS - 1)
    layer_above = (layer_below + 1).clamp(0, N_LAYERS)
    frac = target_layer - layer_below.float()

    # Get logits at interpolated depth
    B_eff = h_input.shape[0]
    logits_at_depth = torch.zeros(B_eff, lm_head_weight.shape[0], device=device)

    with torch.no_grad():
        for i in range(B_eff):
            lb = layer_below[i].item() + 1  # +1 because hidden_states[0] is embedding
            ub = min(layer_above[i].item() + 1, N_LAYERS)
            f = frac[i].item()

            h_lb = hidden_states[lb][i // (batch.shape[1]-1), i % (batch.shape[1]-1)]
            h_ub = hidden_states[ub][i // (batch.shape[1]-1), i % (batch.shape[1]-1)]
            h_interp = (1 - f) * h_lb + f * h_ub
            h_normed = final_norm_layer(h_interp.unsqueeze(0).unsqueeze(0))
            logits_at_depth[i] = F.linear(h_normed, lm_head_weight)[0, 0]

    # Loss: CE against full model's prediction + depth penalty
    targets = full_pred.reshape(-1)  # [B*(T-1)]
    ce_loss = F.cross_entropy(logits_at_depth.float(), targets)
    depth_loss = pred_depth.mean()  # encourage shallow
    loss = ce_loss + DEPTH_PENALTY * depth_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())
    if step % 100 == 0 or step == N_STEPS - 1:
        print(f"{step:>6} {loss.item():>8.4f} {ce_loss.item():>7.4f} "
              f"{depth_loss.item():>7.4f} {pred_depth.mean().item()*N_LAYERS:>6.1f}L")

# ═══════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ROUTER EVALUATION")
print(f"{'='*60}")

with torch.no_grad():
    # Run on all training data
    out = model.model(train_ids[:8], output_hidden_states=True)
    h_L1 = out.hidden_states[1][:, :-1].reshape(-1, H).float()
    pred = router(h_L1)

    pred_layers = (pred * N_LAYERS).cpu().numpy()
    print(f"  Predicted depth distribution:")
    print(f"    Mean: {pred_layers.mean():.1f} layers")
    print(f"    Std:  {pred_layers.std():.1f}")
    print(f"    Min:  {pred_layers.min():.1f}")
    print(f"    Max:  {pred_layers.max():.1f}")

    # KV assignment
    within = pred <= 1.0
    print(f"\n  KV assignment:")
    print(f"    1 KV head (within {N_LAYERS}L): {within.sum().item()}/{len(pred)} "
          f"({within.float().mean()*100:.0f}%)")
    print(f"    {N_KV} KV heads (outside): {(~within).sum().item()}/{len(pred)} "
          f"({(~within).float().mean()*100:.0f}%)")

# Token-level examples
print(f"\n  Examples:")
test_text = "The future of artificial intelligence will be shaped by multiple factors"
ids = tokenizer(test_text, return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    out = model.model(ids, output_hidden_states=True)
    h_test = out.hidden_states[1][0].float()  # [T, H]
    depths = router(h_test)

tokens = [tokenizer.decode(ids[0, i:i+1]) for i in range(ids.shape[1])]
print(f"  {'Token':>15} {'Depth':>7} {'Layers':>7} {'KV':>4}")
for i in range(min(len(tokens), 15)):
    d = depths[i].item()
    n_l = int(d * N_LAYERS)
    kv = 1 if d <= 1.0 else N_KV
    print(f"  {tokens[i]:>15} {d:>7.3f} {n_l:>6}L {kv:>3}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "router_state": router.state_dict(),
    "config": {"input_dim": H, "bottleneck": 128, "n_layers": N_LAYERS, "n_kv": N_KV},
    "losses": losses,
}, os.path.join(SAVE_DIR, "depth_router_14b_e2e.pt"))
print(f"\nSaved. Final loss: {losses[-1]:.4f}", flush=True)
