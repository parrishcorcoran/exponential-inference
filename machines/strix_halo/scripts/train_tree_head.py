"""Train tree prediction head for Exponential Inference.

One small head on frozen Qwen3-14B. Learns to predict tokens at t+1, t+2, ... t+K
from the hidden state at position t. At inference time, the manifold determines
the tree shape — which predictions to use and where to branch.

Architecture per lookahead position:
  hidden_state → ResBlock(5120→5120) → lm_head → token prediction

The ResBlock is a single residual MLP. The lm_head is SHARED with the base model
(no extra 5120×151936 weight — that's the whole point).

Memory budget:
  - Base model (frozen, bf16): ~29.5 GB
  - K=4 ResBlocks: 4 × (5120×5120 + 5120×5120) × 2 bytes ≈ 400 MB
  - Training overhead (grads, optimizer): ~2 GB
  - Total: ~32 GB of 89 GB available
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os

PYTHON_ENV = "/home/cpinchington/MedusaBitNet/.venv/bin/python"
device = "cuda"

# ═══════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════
K = 4                    # Predict K future tokens (tree depth)
HIDDEN = 5120
VOCAB = 151936
BATCH_SIZE = 2           # Sequences per batch
SEQ_LEN = 512            # Tokens per sequence
LR = 1e-3
N_STEPS = 500
SAVE_EVERY = 100
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"

print("=" * 70)
print(f"TREE HEAD TRAINING — K={K} lookahead positions")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# Model + Head
# ═══════════════════════════════════════════════════════
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

print("Loading base model (frozen)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

# Freeze everything
for p in base.parameters():
    p.requires_grad = False

print(f"Base model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


class TreeResBlock(nn.Module):
    """Single residual block: hidden → hidden. Shared lm_head projects to vocab."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        # Residual MLP: x + fc2(SiLU(fc1(LN(x))))
        h = self.ln(x)
        h = F.silu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class TreeHead(nn.Module):
    """K residual blocks, each predicting a future token position.

    Block 0: hidden[t] → predict token[t+2]  (t+1 is already predicted by lm_head)
    Block 1: hidden[t] → predict token[t+3]
    ...
    Block K-1: hidden[t] → predict token[t+K+1]

    All blocks share the base model's lm_head for the final projection.
    """
    def __init__(self, hidden_dim, k, lm_head):
        super().__init__()
        self.blocks = nn.ModuleList([TreeResBlock(hidden_dim) for _ in range(k)])
        self.lm_head = lm_head  # Shared, frozen

    def forward(self, hidden_states):
        """
        hidden_states: [B, T, H]
        Returns: list of K logit tensors, each [B, T, V]
        """
        logits = []
        for block in self.blocks:
            h = block(hidden_states)
            # lm_head is frozen but we need grads to flow through the block
            # The lm_head weight doesn't need grad, but the matmul output does
            l = F.linear(h, self.lm_head.weight)
            logits.append(l)
        return logits


# Build head
tree_head = TreeHead(HIDDEN, K, base.lm_head).to(device).to(torch.bfloat16)
n_params = sum(p.numel() for p in tree_head.blocks.parameters())
print(f"Tree head: {K} blocks, {n_params/1e6:.1f}M trainable params", flush=True)
print(f"VRAM after head: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training data: use model's own tokenizer on diverse text
# ═══════════════════════════════════════════════════════
# Simple: generate training sequences from prompts
# In production you'd use a real corpus. For now, use a mix of prompts.
training_prompts = [
    "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. Einstein published this theory in 1915, fundamentally changing our understanding of the universe. One of its key predictions was the bending of light around massive objects, which was confirmed during a solar eclipse in 1919.",
    "In computer science, a hash table is a data structure that implements an associative array, a structure that can map keys to values. A hash table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.",
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. Mitochondria have their own DNA, which is separate from the nuclear DNA.",
    "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
    "The French Revolution began in 1789 with the Storming of the Bastille and ended in the late 1790s with the ascent of Napoleon Bonaparte. During this period, French citizens razed and redesigned their country's political landscape, uprooting centuries-old institutions such as absolute monarchy.",
    "Quantum computing harnesses quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical bits, quantum bits or qubits can exist in multiple states simultaneously, enabling certain computations to be performed exponentially faster.",
    "The human genome contains approximately 3 billion base pairs of DNA, organized into 23 pairs of chromosomes. The Human Genome Project, completed in 2003, was an international scientific research project with the goal of determining the sequence of nucleotide base pairs.",
    "Neural networks are computing systems inspired by biological neural networks that constitute animal brains. These systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules.",
    "The Amazon rainforest covers approximately 5.5 million square kilometers and is home to roughly 10 percent of all species on Earth. It produces about 20 percent of the world's oxygen and plays a critical role in regulating the global climate.",
    "Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior. Modern cryptography exists at the intersection of mathematics, computer science, electrical engineering, and communication science.",
    "The Standard Model of particle physics describes three of the four known fundamental forces and classifies all known elementary particles. It was developed throughout the latter half of the 20th century and was confirmed by the discovery of the Higgs boson in 2012.",
    "Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis, plants capture light energy and use it to convert water, carbon dioxide, and minerals into oxygen and energy-rich organic compounds.",
] * 4  # Repeat for more data

print(f"Tokenizing {len(training_prompts)} training sequences...", flush=True)
train_ids = []
for prompt in training_prompts:
    toks = tokenizer(prompt, return_tensors='pt', truncation=True,
                     max_length=SEQ_LEN, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape} ({train_ids.shape[0]} seqs × {train_ids.shape[1]} tokens)", flush=True)

# ═══════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(tree_head.blocks.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_STEPS)

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nTraining {N_STEPS} steps, batch_size={BATCH_SIZE}...", flush=True)
print(f"{'Step':>6} {'Loss':>8} {'L1':>7} {'L2':>7} {'L3':>7} {'L4':>7} {'tok/s':>7} {'VRAM':>6}", flush=True)
print("-" * 70, flush=True)

losses_log = []
t_start = time.time()

for step in range(N_STEPS):
    # Random batch
    idx = torch.randint(0, len(train_ids), (BATCH_SIZE,))
    input_ids = train_ids[idx]

    t0 = time.time()

    # Forward through frozen base — get hidden states
    with torch.no_grad():
        outputs = base.model(input_ids, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # [B, T, H]

    # Tree head predictions
    tree_logits = tree_head(hidden.detach())  # K × [B, T, V]

    # Loss: for each head k, predict token at position t+k+2
    # (t+1 is the base model's job, tree heads predict further ahead)
    total_loss = 0.0
    per_head_loss = []
    for k in range(K):
        offset = k + 2  # head 0 predicts t+2, head 1 predicts t+3, ...
        if offset >= input_ids.shape[1]:
            continue
        # Logits at positions [0, ..., T-offset-1] predict tokens at [offset, ..., T-1]
        logits_k = tree_logits[k][:, :-offset, :]  # [B, T-offset, V]
        targets_k = input_ids[:, offset:]            # [B, T-offset]
        loss_k = F.cross_entropy(logits_k.reshape(-1, VOCAB), targets_k.reshape(-1))
        total_loss = total_loss + loss_k
        per_head_loss.append(loss_k.item())

    total_loss = total_loss / K

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(tree_head.blocks.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    dt = time.time() - t0
    tps = BATCH_SIZE * SEQ_LEN / dt

    losses_log.append({
        "step": step, "loss": total_loss.item(),
        "per_head": per_head_loss, "tps": tps
    })

    if step % 25 == 0 or step == N_STEPS - 1:
        head_str = " ".join(f"{l:.3f}" for l in per_head_loss)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>6} {total_loss.item():>8.4f} {head_str} {tps:>7.0f} {vram:>5.1f}G", flush=True)

    if (step + 1) % SAVE_EVERY == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"tree_head_step{step+1}.pt")
        torch.save({
            "step": step + 1,
            "blocks_state": tree_head.blocks.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {"K": K, "hidden": HIDDEN, "vocab": VOCAB},
            "losses": losses_log,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}", flush=True)

elapsed = time.time() - t_start
print(f"\nTraining complete: {elapsed:.0f}s ({N_STEPS/elapsed:.1f} steps/s)", flush=True)

# Final save
final_path = os.path.join(SAVE_DIR, "tree_head_final.pt")
torch.save({
    "step": N_STEPS,
    "blocks_state": tree_head.blocks.state_dict(),
    "config": {"K": K, "hidden": HIDDEN, "vocab": VOCAB},
    "losses": losses_log,
}, final_path)
print(f"Saved final: {final_path}", flush=True)

# ═══════════════════════════════════════════════════════
# Quick validation: does the head predict future tokens?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION: tree head accuracy on held-out text")
print(f"{'='*60}")

val_text = "Transformer models process sequences using self-attention mechanisms that allow each token to attend to all other tokens in the sequence. This enables the model to capture long-range dependencies without the sequential bottleneck of recurrent architectures."
val_ids = tokenizer(val_text, return_tensors='pt').input_ids.to(device)

with torch.no_grad():
    val_hidden = base.model(val_ids).last_hidden_state
    val_logits = tree_head(val_hidden)

    # Base model accuracy (t+1)
    base_logits = base.lm_head(val_hidden)
    base_preds = base_logits[0, :-1].argmax(-1)
    base_acc = (base_preds == val_ids[0, 1:]).float().mean().item()

    print(f"Base model (t+1): {base_acc*100:.1f}% accuracy")

    for k in range(K):
        offset = k + 2
        if offset >= val_ids.shape[1]:
            break
        preds = val_logits[k][0, :-offset].argmax(-1)
        targets = val_ids[0, offset:]
        acc = (preds == targets).float().mean().item()
        print(f"Tree head {k} (t+{offset}): {acc*100:.1f}% accuracy")

# Show example predictions
print(f"\nExample predictions from position 5:")
pos = 5
actual_tokens = [tokenizer.decode(val_ids[0, pos+1+k:pos+2+k]) for k in range(K+1)]
print(f"  Actual: {actual_tokens}")

with torch.no_grad():
    h = base.model(val_ids).last_hidden_state
    base_pred = tokenizer.decode(base.lm_head(h)[0, pos].argmax(-1).unsqueeze(0))
    tree_preds = [tokenizer.decode(val_logits[k][0, pos].argmax(-1).unsqueeze(0)) for k in range(K)]
    print(f"  Base t+1: '{base_pred}'")
    for k, p in enumerate(tree_preds):
        print(f"  Tree t+{k+2}: '{p}'")

# Save results
results = {
    "K": K, "n_params": n_params, "n_steps": N_STEPS,
    "final_loss": losses_log[-1]["loss"],
    "base_acc": base_acc,
    "elapsed_s": elapsed,
}
with open("machines/strix_halo/results/tree_head.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to tree_head.json", flush=True)
