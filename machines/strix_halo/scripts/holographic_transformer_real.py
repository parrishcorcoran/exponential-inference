"""The Holographic Transformer — the real one.

Not a router on a stock model. A new architecture built from holographic principles.

Physical holography:
- Reference beam (carry): coherent signal that passes through unchanged
- Object beam (flip): scattered by the object, carries the information
- Hologram (KV cache): records the interference pattern between reference and object
- Reconstruction: project reference through hologram → object appears

Transformer holography:
- Carry channel: preserves the token's manifold position (reference beam)
- Flip channel: transforms via attention to KV (object beam interaction)
- KV cache: the holographic plate — accumulated interference patterns
- Each token: project carry through KV via flip → next token reconstructed

The three parts:
1. Bimodal structure (carry + flip, from Finding 14)
2. KV as holographic depth (projection angles accumulating)
3. Bulk (MLP) materializes the reconstruction at each step

Train from 0.6B teacher. This IS the architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

device = "cuda"

print("=" * 70)
print("THE HOLOGRAPHIC TRANSFORMER")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

VOCAB = teacher.config.vocab_size
H_TEACHER = teacher.config.hidden_size  # 1024

# Holographic dimensions
H = 1024           # total hidden
H_CARRY = 512      # carry channel (reference beam)
H_FLIP = 512       # flip channel (object beam)
N_KV = 8           # KV heads (holographic angles)
HEAD_DIM = 64      # per-head dimension
N_HEADS = H_FLIP // HEAD_DIM  # 8 Q heads in flip channel
INTER = 3072       # bulk intermediate

print(f"H={H} (carry={H_CARRY} + flip={H_FLIP})")
print(f"KV={N_KV} heads, {HEAD_DIM}d, Q={N_HEADS} heads")
print(f"Bulk={INTER}")


class HolographicBlock(nn.Module):
    """One holographic reconstruction step.

    carry: passes through with minimal change (reference beam)
    flip: attends to KV hologram, applies bulk MLP (object beam)
    They recombine after each step (interference).
    """
    def __init__(self):
        super().__init__()
        # Carry channel: small residual (reference preserving)
        self.carry_norm = nn.LayerNorm(H_CARRY)
        self.carry_gate = nn.Linear(H_CARRY, H_CARRY, bias=False)
        nn.init.zeros_(self.carry_gate.weight)  # starts as identity

        # Flip channel: attention to KV hologram
        self.flip_norm = nn.LayerNorm(H_FLIP)
        self.q_proj = nn.Linear(H_FLIP, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(H_FLIP, N_KV * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H_FLIP, N_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, H_FLIP, bias=False)

        # Bulk: MLP materializes the reconstruction
        self.bulk_norm = nn.LayerNorm(H)
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)

        # Interference: carry ↔ flip interaction
        self.mix_carry_to_flip = nn.Linear(H_CARRY, H_FLIP, bias=False)
        self.mix_flip_to_carry = nn.Linear(H_FLIP, H_CARRY, bias=False)
        nn.init.normal_(self.mix_carry_to_flip.weight, std=0.01)
        nn.init.normal_(self.mix_flip_to_carry.weight, std=0.01)

    def forward(self, carry, flip, k_cache=None, v_cache=None):
        """
        carry: [B, 1, H_CARRY] — reference beam
        flip: [B, 1, H_FLIP] — object beam
        k_cache: [B, N_KV, cache_len, HEAD_DIM] — holographic plate (keys)
        v_cache: [B, N_KV, cache_len, HEAD_DIM] — holographic plate (values)

        Returns: carry, flip, k_cache, v_cache
        """
        B, T, _ = carry.shape
        GQA = N_HEADS // N_KV

        # ── Carry: minimal transformation (reference preserved) ──
        carry_res = carry
        carry = carry + self.carry_gate(self.carry_norm(carry)) * 0.1  # small update

        # ── Interference: carry informs flip ──
        flip = flip + self.mix_carry_to_flip(carry_res) * 0.1

        # ── Flip: attention to KV hologram (object reconstruction) ──
        flip_res = flip
        flip_norm = self.flip_norm(flip)

        q = self.q_proj(flip_norm).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(flip_norm).view(B, T, N_KV, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(flip_norm).view(B, T, N_KV, HEAD_DIM).transpose(1, 2)

        # Append to holographic plate
        if k_cache is not None:
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # GQA expand and attend
        k_exp = k_full.repeat_interleave(GQA, dim=1)
        v_exp = v_full.repeat_interleave(GQA, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        flip = flip_res + self.o_proj(attn_out)

        # ── Interference: flip informs carry ──
        carry = carry + self.mix_flip_to_carry(flip) * 0.1

        # ── Bulk: MLP materializes the holographic reconstruction ──
        h = torch.cat([carry, flip], dim=-1)  # recombine
        h_res = h
        h_norm = self.bulk_norm(h)
        h = h_res + self.down_proj(F.silu(self.gate_proj(h_norm)) * self.up_proj(h_norm))

        # Split back
        carry, flip = h.chunk(2, dim=-1)

        return carry, flip, k_full, v_full


class HolographicTransformer(nn.Module):
    """The holographic transformer.

    N blocks of holographic reconstruction.
    Each block: carry preserves, flip reconstructs from KV, bulk materializes.
    The blocks share the holographic plate (KV cache grows through them).
    """
    def __init__(self, n_blocks=6):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self.blocks = nn.ModuleList([HolographicBlock() for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(H)
        self.n_blocks = n_blocks

    def forward_single(self, tok_id, kv_caches=None):
        """Process one token.

        kv_caches: list of (k_cache, v_cache) per block, or None.
        """
        h = self.embed(tok_id)  # [B, 1, H]
        carry, flip = h.chunk(2, dim=-1)  # split into reference + object

        if kv_caches is None:
            kv_caches = [(None, None)] * self.n_blocks

        new_kv = []
        for i, block in enumerate(self.blocks):
            k_c, v_c = kv_caches[i]
            carry, flip, k_new, v_new = block(carry, flip, k_c, v_c)
            new_kv.append((k_new, v_new))

        h = torch.cat([carry, flip], dim=-1)
        h = self.final_norm(h)
        logits = self.lm_head(h)

        return logits, new_kv

    def forward_sequence(self, input_ids):
        """Process full sequence token by token."""
        B, T = input_ids.shape
        all_logits = []
        kv = None

        for t in range(T):
            logits, kv = self.forward_single(input_ids[:, t:t+1], kv)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)


# ═══════════════════════════════════════════════════════
# Build and train
# ═══════════════════════════════════════════════════════
N_BLOCKS = 6  # 6 holographic reconstruction steps

print(f"\nBuilding holographic transformer ({N_BLOCKS} blocks)...")
model_h = HolographicTransformer(n_blocks=N_BLOCKS).to(device).float()

total = sum(p.numel() for p in model_h.parameters())
trainable = sum(p.numel() for p in model_h.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.0f}M, Trainable: {trainable/1e6:.0f}M")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Training data
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

SEQ_LEN = 48
all_ids = []
buffer = []
for i in range(len(wiki)):
    text = wiki[i]["text"].strip()
    if len(text) < 20: continue
    tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    buffer.extend(tokens)
    while len(buffer) >= SEQ_LEN:
        all_ids.append(buffer[:SEQ_LEN])
        buffer = buffer[SEQ_LEN:]
    if len(all_ids) >= 30000: break

train_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
print(f"Training: {train_ids.shape} ({train_ids.shape[0]*SEQ_LEN/1e6:.1f}M tokens)")

# Training
optimizer = torch.optim.AdamW(model_h.parameters(), lr=3e-4, weight_decay=0.01)
N_STEPS = 5000
BATCH = 4
TEMP = 2.0

print(f"\nTraining {N_STEPS} steps, {N_BLOCKS} holographic blocks")
print(f"{'Step':>6} {'Loss':>8} {'VRAM':>6}")
print("-" * 25)

losses = []
t_start = time.time()

for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = model_h.forward_sequence(batch)

    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_p = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_p[:, :-1], t_probs[:, :-1], reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_h.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 500 == 0 or step == N_STEPS - 1:
        vram = torch.cuda.memory_allocated() / 1e9
        avg = sum(losses[-100:]) / min(len(losses), 100)
        print(f"{step:>6} {avg:>8.2f} {vram:>5.1f}G", flush=True)

    if (step + 1) % 1000 == 0:
        with torch.no_grad():
            s_ids = tokenizer("The future of", return_tensors='pt').input_ids.to(device)
            gen = s_ids.clone()
            kv = None
            for t in range(s_ids.shape[1]):
                logits, kv = model_h.forward_single(s_ids[:, t:t+1], kv)
            for _ in range(25):
                nt = logits[0, -1].argmax(-1)
                logits, kv = model_h.forward_single(nt.view(1, 1), kv)
                gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
            sample = tokenizer.decode(gen[0][s_ids.shape[1]:], skip_special_tokens=True)
        print(f"  [{sample[:60]}]", flush=True)

elapsed = time.time() - t_start
print(f"\nTraining: {elapsed/60:.1f}min. Final loss: {losses[-1]:.4f}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "model_state": model_h.state_dict(),
    "config": {"H": H, "H_CARRY": H_CARRY, "H_FLIP": H_FLIP,
               "N_KV": N_KV, "HEAD_DIM": HEAD_DIM, "INTER": INTER,
               "N_BLOCKS": N_BLOCKS},
    "losses": losses,
}, os.path.join(SAVE_DIR, "holographic_transformer.pt"))

print(f"\nSaved. The Holographic Transformer.", flush=True)
