"""Adaptive Depth Model: Qwen3-4B base + extension block.

Stock Qwen3-4B (36 layers) as the base.
Shared extension block (trained) for layers 37+.
Manifold measurement determines depth per token.

Training:
1. Freeze base model (36 layers, stock weights)
2. Train ONE extension block to continue the rotation past layer 36
3. KL distillation: teacher is the base model at 36 layers
   (for now — later, teacher is larger model or self-supervised)
4. Extension learns to refine: each additional application
   improves the reconstruction

At inference:
- Easy tokens: exit at layer 10-20 (base model layers, no extension needed)
- Normal tokens: full 36 layers (base model as-is)
- Hard tokens: 36 base + N extension steps (more angular coverage)
- Depth determined by manifold: velocity of hidden state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import copy

device = "cuda"

print("=" * 70)
print("ADAPTIVE DEPTH: Qwen3-4B base + trained extension block")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

print("Loading base model...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in base.parameters():
    p.requires_grad = False

N_LAYERS = base.config.num_hidden_layers  # 36
H = base.config.hidden_size              # 2560

print(f"Base: {N_LAYERS} layers, H={H}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Extension block: initialized from last layer of base model
# This layer already knows the "end of rotation" behavior
# Training teaches it to continue rotating past the base depth
# ═══════════════════════════════════════════════════════

print("Building extension block (from base layer 35)...", flush=True)
extension = copy.deepcopy(base.model.layers[N_LAYERS - 1])  # last layer
extension_norm = copy.deepcopy(base.model.norm)  # final norm for logit lens checks

# Unfreeze extension
for p in extension.parameters():
    p.requires_grad = True
for p in extension_norm.parameters():
    p.requires_grad = True

ext_params = sum(p.numel() for p in extension.parameters() if p.requires_grad)
print(f"Extension: {ext_params/1e6:.0f}M trainable params")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def run_base_layers(base_model, input_ids, n_layers=None):
    """Run base model through n_layers (default: all).
    Returns hidden state and KV cache."""
    if n_layers is None:
        n_layers = N_LAYERS

    h = base_model.model.embed_tokens(input_ids)
    B, T, D = h.shape
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = base_model.model.rotary_emb(h, pos)

    position_embeddings = (cos, sin)

    for i in range(n_layers):
        layer = base_model.model.layers[i]
        layer_out = layer(h, position_embeddings=position_embeddings)
        if isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out

    return h, cos, sin, position_embeddings


def run_extension(h, ext_block, position_embeddings, n_steps):
    """Run extension block n_steps times."""
    for step in range(n_steps):
        layer_out = ext_block(h, position_embeddings=position_embeddings)
        if isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out
    return h


# ═══════════════════════════════════════════════════════
# Training: teach extension to improve upon base model output
#
# Strategy: base model at 36 layers produces good output.
# The extension should REFINE it — each step brings the output
# closer to the teacher's (or improves quality).
#
# Loss: cross-entropy on the training text. The base model
# already gets most tokens right. The extension improves the rest.
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TRAINING EXTENSION BLOCK")
print(f"{'='*60}")

texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world.",
    "Marine biology studies organisms in the ocean covering more than seventy percent of Earth surface area.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes the behavior of matter and energy at the smallest scales of existence.",
    "The French Revolution transformed French society by uprooting centuries of tradition and absolute monarchy.",
    "Climate change driven by burning fossil fuels threatens ecosystems worldwide through rising temperatures.",
    "The human genome contains approximately three billion base pairs of DNA organized into chromosomes.",
    "Machine learning algorithms improve through experience without being explicitly programmed for each task.",
    "The Amazon rainforest produces significant oxygen and houses incredible biodiversity across many species.",
    "Cryptography enables secure communication by transforming messages into encrypted ciphertext safely.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy distributions.",
    "Evolution explains how populations change over generations through variation inheritance and selection.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "Photosynthesis converts light energy into chemical energy using water and carbon dioxide in plants.",
    "The Internet connected billions of devices through standardized protocols enabling instant communication.",
] * 8

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=64, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

optimizer = torch.optim.AdamW(
    list(extension.parameters()) + list(extension_norm.parameters()),
    lr=5e-5, weight_decay=0.01
)

lm_head = base.lm_head

N_TRAIN = 500
BATCH = 4
EXT_STEPS = [1, 2, 4, 8]  # train with variable extension depth

print(f"Training {N_TRAIN} steps, batch={BATCH}")
print(f"Extension steps per training step: randomly from {EXT_STEPS}")
print(f"{'Step':>6} {'CE_loss':>8} {'ext_n':>6} {'VRAM':>6}")
print("-" * 35)

losses = []
for step in range(N_TRAIN):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    # Random extension depth for this step
    n_ext = EXT_STEPS[step % len(EXT_STEPS)]

    with torch.no_grad():
        # Run base model (frozen) to get hidden state after 36 layers
        h_base, cos, sin, pos_emb = run_base_layers(base, batch)

    # Run extension (trainable)
    h_ext = run_extension(h_base.detach(), extension, pos_emb, n_ext)

    # Final norm + lm_head
    h_out = extension_norm(h_ext)
    logits = lm_head(h_out)

    # Cross-entropy loss: predict next token
    # Shift: logits[:-1] predicts tokens[1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, logits.shape[-1]).float(),
                           shift_labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 50 == 0 or step == N_TRAIN - 1:
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>6} {loss.item():>8.4f} {n_ext:>5} {vram:>5.1f}G", flush=True)

# ═══════════════════════════════════════════════════════
# Validation: compare base (36L) vs base+extension at various depths
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION: base vs base+extension at different depths")
print(f"{'='*60}")

val_prompts = [
    "The future of artificial intelligence will",
    "Water freezes at zero degrees and boils at",
    "The most important discovery in physics was",
]

for prompt in val_prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\n  '{prompt}'")

    # Base model (36 layers, stock)
    with torch.no_grad():
        out = base.generate(ids, max_new_tokens=40, do_sample=False)
    base_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  Base (36L):  {base_text[:70]}")

    # Base + extension at various depths
    for n_ext in [0, 2, 4, 8, 16]:
        with torch.no_grad():
            gen = ids.clone()
            for _ in range(40):
                h, cos, sin, pos_emb = run_base_layers(base, gen)
                if n_ext > 0:
                    h = run_extension(h, extension, pos_emb, n_ext)
                h_out = extension_norm(h) if n_ext > 0 else base.model.norm(h)
                logits = lm_head(h_out)
                next_tok = logits[0, -1:].argmax(-1)
                gen = torch.cat([gen, next_tok.unsqueeze(0)], dim=-1)

        text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        total_layers = N_LAYERS + n_ext
        print(f"  +{n_ext}ext ({total_layers}L): {text[:70]}")

# ═══════════════════════════════════════════════════════
# Measure: per-token velocity at different depths
# (Does the extension reduce velocity → reaching resolution?)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VELOCITY: does extension bring tokens to resolution?")
print(f"{'='*60}")

test_text = "The theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy"
ids = tokenizer(test_text, return_tensors='pt').input_ids.to(device)

with torch.no_grad():
    h, cos, sin, pos_emb = run_base_layers(base, ids)
    h_prev = h.clone()

    print(f"  {'Depth':>6} {'Velocity':>10} {'Mean norm':>10}")

    for ext_step in range(20):
        h = run_extension(h, extension, pos_emb, 1)
        vel = (h - h_prev).norm(dim=-1).mean().item()
        norm = h.norm(dim=-1).mean().item()
        h_prev = h.clone()
        total = N_LAYERS + ext_step + 1
        print(f"  {total:>5}L {vel:>10.2f} {norm:>10.1f}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "extension_state": extension.state_dict(),
    "extension_norm_state": extension_norm.state_dict(),
    "config": {"base_model": "Qwen/Qwen3-4B", "n_base_layers": N_LAYERS,
               "hidden": H, "ext_params": ext_params},
    "losses": losses,
}, os.path.join(SAVE_DIR, "adaptive_depth_4b.pt"))
print(f"\nSaved. Final CE: {losses[-1]:.4f}", flush=True)
