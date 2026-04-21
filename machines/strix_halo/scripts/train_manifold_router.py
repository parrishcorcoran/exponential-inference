"""Train manifold router + tree head for Exponential Inference.

The router reads the EMBEDDING directly (manifold position) and outputs
routing decisions: (width, length, branching). No sharpness, no proxies.
The embedding IS the coordinates.

Phase 1: Collect labels from full model
  - Run full forward with logit lens
  - Per position: stabilization_depth (length), output_entropy (branching),
    logit_margin (width)

Phase 2: Train router + tree head jointly
  - Router: embedding → (width, length, branch) — tiny MLP
  - Tree head: hidden_state → K future tokens — residual blocks + shared lm_head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os

device = "cuda"

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
K = 4                    # Tree depth (future token positions)
HIDDEN = 5120
VOCAB = 151936
N_LAYERS = 40
N_HEADS = 40
BATCH_SIZE = 1           # Sequences per batch (full forward is expensive)
SEQ_LEN = 256            # Shorter for label collection (need full forward)
LR = 5e-4
ROUTER_LR = 1e-3
N_STEPS = 300
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"

print("=" * 70)
print("MANIFOLD ROUTER + TREE HEAD TRAINING")
print("Router reads embedding → outputs (width, length, branching)")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer + model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

for p in base.parameters():
    p.requires_grad = False

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training data — diverse text
# ═══════════════════════════════════════════════════════
corpus = [
    "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy.",
    "In computer science, a hash table is a data structure that implements an associative array abstract data type.",
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells that generate ATP.",
    "Machine learning focuses on developing computer programs that can access data and use it to learn for themselves.",
    "The French Revolution began in 1789 with the Storming of the Bastille and fundamentally changed France.",
    "Quantum computing harnesses superposition and entanglement to process information exponentially faster.",
    "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
    "The Amazon rainforest covers 5.5 million square kilometers and is home to 10 percent of all species on Earth.",
    "Cryptography is the practice of techniques for secure communication in the presence of adversarial behavior.",
    "The Standard Model describes three of four fundamental forces and classifies all known elementary particles.",
    "Photosynthesis transforms light energy into chemical energy in green plants using water and carbon dioxide.",
    "The human immune system is a complex network of cells, tissues, and organs that defend against pathogens.",
    "Plate tectonics describes the movement of lithospheric plates on the asthenosphere driven by mantle convection.",
    "The Krebs cycle is a series of chemical reactions used by aerobic organisms to release stored energy.",
    "Fibonacci numbers appear in biological settings, such as branching in trees and the arrangement of leaves.",
    "The double helix structure of DNA was first described by Watson and Crick based on X-ray crystallography.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape from inside.",
    "The periodic table organizes chemical elements by atomic number, electron configuration, and chemical properties.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors and punishing undesired ones.",
    "The Cambrian explosion was a rapid diversification of animal life that occurred approximately 541 million years ago.",
    "Graph neural networks extend deep learning to graph-structured data for molecular property prediction.",
    "Superconductors conduct electricity with zero resistance below a critical temperature unique to each material.",
    "The Riemann hypothesis concerns the distribution of prime numbers and remains unproven since 1859.",
    "CRISPR-Cas9 is a molecular tool that allows precise editing of DNA sequences in living organisms.",
    "The cosmic microwave background radiation is the thermal radiation left over from the Big Bang.",
    "Transformer architectures use self-attention to process sequential data without recurrence or convolution.",
    "Entropy in thermodynamics measures the number of microscopic configurations consistent with macroscopic state.",
    "The Navier-Stokes equations describe the motion of viscous fluid substances in three dimensions.",
    "Bayesian inference updates probability estimates as new evidence becomes available using Bayes theorem.",
    "The central dogma of molecular biology describes the flow of genetic information from DNA to RNA to protein.",
    "Topological insulators conduct electricity on their surface but act as insulators in their interior.",
    "The halting problem proves that no general algorithm can determine whether an arbitrary program will terminate.",
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects like merging black holes.",
    "Monte Carlo methods use random sampling to obtain numerical results for problems that are deterministic in principle.",
    "The endosymbiotic theory explains how mitochondria and chloroplasts originated as free-living prokaryotes.",
    "Attention mechanisms allow neural networks to focus on relevant parts of the input when producing output.",
    "The uncertainty principle states that position and momentum of a particle cannot both be precisely determined.",
    "Convolutional neural networks use learnable filters to detect spatial hierarchies in grid-like data such as images.",
    "The Drake equation estimates the number of active communicative civilizations in the Milky Way galaxy.",
    "Protein folding is the process by which a polypeptide chain assumes its functional three-dimensional structure.",
]

print(f"Tokenizing {len(corpus)} training sequences...", flush=True)
train_ids = []
for text in corpus:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=SEQ_LEN, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}", flush=True)

# ═══════════════════════════════════════════════════════
# Phase 1: Collect manifold labels from full model
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 1: Collecting manifold labels (logit lens + entropy)")
print(f"{'='*60}")

all_embeddings = []      # [N, T, H] — router input
all_hidden_states = []   # [N, T, H] — tree head input (last layer)
all_stab_depth = []      # [N, T] — stabilization depth (length label)
all_entropy = []         # [N, T] — output entropy (branching label)
all_margin = []          # [N, T] — logit margin (width label)
all_targets = []         # [N, T] — target token ids

lm_head_weight = base.lm_head.weight  # [V, H]
final_norm = base.model.norm

for seq_idx in range(len(train_ids)):
    ids = train_ids[seq_idx:seq_idx+1]

    with torch.no_grad():
        # Full forward with all hidden states
        out = base.model(ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple of (L+1) × [1, T, H]

        # Embedding = hidden_states[0] (before any layers)
        embedding = hidden_states[0]  # [1, T, H]

        # Final hidden state
        final_hidden = hidden_states[-1]  # [1, T, H]

        # Logit lens: apply lm_head at every layer
        per_layer_argmax = []
        for layer_idx in range(1, len(hidden_states)):  # skip embedding
            h = hidden_states[layer_idx]
            h_normed = final_norm(h)
            logits_layer = F.linear(h_normed, lm_head_weight)
            per_layer_argmax.append(logits_layer[0].argmax(-1))  # [T]

        per_layer_argmax = torch.stack(per_layer_argmax)  # [L, T]
        final_argmax = per_layer_argmax[-1]  # [T]

        # Stabilization depth: last layer that disagrees with final
        agrees = (per_layer_argmax == final_argmax.unsqueeze(0))  # [L, T]
        # For each position, find the last disagreeing layer
        disagrees = ~agrees  # [L, T]
        L = disagrees.shape[0]
        layer_indices = torch.arange(L, device=device).unsqueeze(1)  # [L, 1]
        # Last disagreeing layer (0 if all agree)
        last_disagree = (disagrees * layer_indices).max(dim=0).values  # [T]
        stab_depth = (last_disagree + 1).float() / L  # normalized 0-1

        # Output entropy from final logits
        final_logits = F.linear(final_norm(final_hidden), lm_head_weight)
        probs = F.softmax(final_logits[0], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(-1)  # [T]
        # Normalize entropy to [0, 1] range (max entropy = log(vocab))
        entropy_norm = entropy / (torch.log(torch.tensor(float(VOCAB), device=device)))

        # Logit margin: top1 - top2 gap (higher = more certain = fewer heads needed)
        top2 = final_logits[0].topk(2, dim=-1).values  # [T, 2]
        margin = (top2[:, 0] - top2[:, 1]).float()
        # Normalize margin (empirically ~0-20 range for bf16)
        margin_norm = (margin / 20.0).clamp(0, 1)

    all_embeddings.append(embedding[0].cpu())
    all_hidden_states.append(final_hidden[0].cpu())
    all_stab_depth.append(stab_depth.cpu())
    all_entropy.append(entropy_norm.cpu())
    all_margin.append(margin_norm.cpu())
    all_targets.append(ids[0].cpu())

    if (seq_idx + 1) % 10 == 0:
        print(f"  {seq_idx+1}/{len(train_ids)} sequences processed", flush=True)

# Stack all
all_embeddings = torch.stack(all_embeddings).to(device)      # [N, T, H]
all_hidden_states = torch.stack(all_hidden_states).to(device) # [N, T, H]
all_stab_depth = torch.stack(all_stab_depth).to(device)       # [N, T]
all_entropy = torch.stack(all_entropy).to(device)             # [N, T]
all_margin = torch.stack(all_margin).to(device)               # [N, T]
all_targets = torch.stack(all_targets).to(device)             # [N, T]

print(f"\nLabel stats:")
print(f"  stab_depth: mean={all_stab_depth.mean():.3f} std={all_stab_depth.std():.3f}")
print(f"  entropy:    mean={all_entropy.mean():.3f} std={all_entropy.std():.3f}")
print(f"  margin:     mean={all_margin.mean():.3f} std={all_margin.std():.3f}")
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Free the base model's hidden state cache
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Phase 2: Define router + tree head
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: Training router + tree head")
print(f"{'='*60}")


class ManifoldRouter(nn.Module):
    """Reads embedding (manifold position) → outputs routing decisions.

    Input: hidden state after embedding [B, T, H]
    Output: 3 values per position:
      - width:  [0,1] → fraction of heads to use (0=min, 1=all)
      - length: [0,1] → fraction of layers to use (0=early exit, 1=all)
      - branch: [0,1] → branching factor (0=single path, 1=max branches)
    """
    def __init__(self, hidden_dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck, 3, bias=False),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(self, embedding):
        return self.net(embedding)  # [B, T, 3]


class TreeResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class TreeHead(nn.Module):
    def __init__(self, hidden_dim, k, lm_head_weight):
        super().__init__()
        self.blocks = nn.ModuleList([TreeResBlock(hidden_dim) for _ in range(k)])
        self.lm_head_weight = lm_head_weight  # frozen, shared

    def forward(self, hidden_states):
        logits = []
        for block in self.blocks:
            h = block(hidden_states)
            l = F.linear(h, self.lm_head_weight)
            logits.append(l)
        return logits


# Build
router = ManifoldRouter(HIDDEN).to(device).to(torch.bfloat16)
tree_head = TreeHead(HIDDEN, K, base.lm_head.weight).to(device).to(torch.bfloat16)

router_params = sum(p.numel() for p in router.parameters())
tree_params = sum(p.numel() for p in tree_head.blocks.parameters())
print(f"Router: {router_params/1e3:.1f}K params", flush=True)
print(f"Tree head: {tree_params/1e6:.1f}M params ({K} blocks)", flush=True)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════
# Router loss: predict manifold labels from embedding
# Tree head loss: predict future tokens from final hidden state
# Joint training

router_optimizer = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR, weight_decay=0.01)
tree_optimizer = torch.optim.AdamW(tree_head.blocks.parameters(), lr=LR, weight_decay=0.01)

N_seqs = all_embeddings.shape[0]

print(f"\nTraining {N_STEPS} steps...", flush=True)
print(f"{'Step':>6} {'R_loss':>8} {'T_loss':>8} {'w_mae':>7} {'l_mae':>7} {'b_mae':>7}", flush=True)
print("-" * 60, flush=True)

router_losses = []
tree_losses = []

for step in range(N_STEPS):
    # Random batch
    idx = torch.randint(0, N_seqs, (BATCH_SIZE,))

    # ── Router training ──
    emb = all_embeddings[idx]          # [B, T, H]
    stab = all_stab_depth[idx]         # [B, T]
    ent = all_entropy[idx]             # [B, T]
    marg = all_margin[idx]             # [B, T]

    route_out = router(emb)            # [B, T, 3]
    # Targets: width ← 1-margin (low margin = need more heads),
    #          length ← stab_depth, branch ← entropy
    width_target = 1.0 - marg          # high margin = few heads needed
    length_target = stab               # high stab = need more layers
    branch_target = ent                # high entropy = more branches

    targets = torch.stack([width_target, length_target, branch_target], dim=-1)
    router_loss = F.mse_loss(route_out.float(), targets.float())

    router_optimizer.zero_grad()
    router_loss.backward()
    router_optimizer.step()

    # Per-output MAE for monitoring
    with torch.no_grad():
        w_mae = (route_out[..., 0].float() - width_target.float()).abs().mean().item()
        l_mae = (route_out[..., 1].float() - length_target.float()).abs().mean().item()
        b_mae = (route_out[..., 2].float() - branch_target.float()).abs().mean().item()

    # ── Tree head training ──
    hidden = all_hidden_states[idx]    # [B, T, H]
    tgt_ids = all_targets[idx]         # [B, T]

    tree_logits = tree_head(hidden.detach())
    tree_loss = 0.0
    for k_idx in range(K):
        offset = k_idx + 2
        if offset >= tgt_ids.shape[1]:
            continue
        logits_k = tree_logits[k_idx][:, :-offset, :]
        targets_k = tgt_ids[:, offset:]
        tree_loss = tree_loss + F.cross_entropy(
            logits_k.reshape(-1, VOCAB).float(), targets_k.reshape(-1)
        )
    tree_loss = tree_loss / K

    tree_optimizer.zero_grad()
    tree_loss.backward()
    torch.nn.utils.clip_grad_norm_(tree_head.blocks.parameters(), 1.0)
    tree_optimizer.step()

    router_losses.append(router_loss.item())
    tree_losses.append(tree_loss.item())

    if step % 25 == 0 or step == N_STEPS - 1:
        print(f"{step:>6} {router_loss.item():>8.5f} {tree_loss.item():>8.4f} "
              f"{w_mae:>7.4f} {l_mae:>7.4f} {b_mae:>7.4f}", flush=True)

print(f"\nTraining complete.", flush=True)

# ═══════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

# Test router on a sequence
val_text = "Transformer models use self-attention to process sequences without recurrence."
val_ids = tokenizer(val_text, return_tensors='pt').input_ids.to(device)

with torch.no_grad():
    val_out = base.model(val_ids, output_hidden_states=True)
    val_emb = val_out.hidden_states[0]  # embedding
    val_hidden = val_out.hidden_states[-1]  # final

    # Router predictions
    route = router(val_emb)  # [1, T, 3]
    width_pred = route[0, :, 0]
    length_pred = route[0, :, 1]
    branch_pred = route[0, :, 2]

    # Decode tokens for display
    tokens = [tokenizer.decode(val_ids[0, i:i+1]) for i in range(val_ids.shape[1])]

print(f"\nRouter predictions per token:")
print(f"{'Token':>15} {'Width':>7} {'Length':>7} {'Branch':>7} {'Heads':>6} {'Layers':>7}")
print("-" * 65)
for i in range(min(20, len(tokens))):
    w = width_pred[i].item()
    l = length_pred[i].item()
    b = branch_pred[i].item()
    n_heads = max(2, int(w * N_HEADS))
    n_layers = max(5, int(l * N_LAYERS))
    print(f"{tokens[i]:>15} {w:>7.3f} {l:>7.3f} {b:>7.3f} {n_heads:>5}h {n_layers:>6}L")

# Tree head accuracy
with torch.no_grad():
    tree_logits = tree_head(val_hidden)
    base_logits = F.linear(final_norm(val_hidden), lm_head_weight)
    base_acc = (base_logits[0, :-1].argmax(-1) == val_ids[0, 1:]).float().mean().item()
    print(f"\nBase model t+1: {base_acc*100:.1f}%")
    for k_idx in range(K):
        offset = k_idx + 2
        if offset >= val_ids.shape[1]:
            break
        preds = tree_logits[k_idx][0, :-offset].argmax(-1)
        acc = (preds == val_ids[0, offset:]).float().mean().item()
        print(f"Tree head t+{offset}: {acc*100:.1f}%")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, "manifold_router_final.pt")
torch.save({
    "router_state": router.state_dict(),
    "tree_blocks_state": tree_head.blocks.state_dict(),
    "config": {
        "K": K, "hidden": HIDDEN, "vocab": VOCAB,
        "router_bottleneck": 256,
        "n_layers": N_LAYERS, "n_heads": N_HEADS,
    },
    "router_losses": router_losses,
    "tree_losses": tree_losses,
    "label_stats": {
        "stab_depth_mean": all_stab_depth.mean().item(),
        "entropy_mean": all_entropy.mean().item(),
        "margin_mean": all_margin.mean().item(),
    },
}, save_path)
print(f"\nSaved: {save_path}", flush=True)

results_path = "machines/strix_halo/results/manifold_router.json"
with open(results_path, "w") as f:
    json.dump({
        "router_params": router_params,
        "tree_params": tree_params,
        "router_final_loss": router_losses[-1],
        "tree_final_loss": tree_losses[-1],
        "n_steps": N_STEPS,
    }, f, indent=2)
print(f"Saved: {results_path}", flush=True)
