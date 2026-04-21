"""Train continuous rotation model — Universal Transformer with manifold-adaptive depth.

One shared rotation block (attention + MLP) applied N times.
N is adaptive per token, determined by manifold measurement.
KL distillation from Qwen3-4B teacher.

The rotation block learns the universal curve — one set of weights
applied as many times as needed. Defined tokens: 2-5 steps.
Branching tokens: 10-20 steps. Adaptive.

Architecture:
- Shared attention + MLP block (~76M params)
- Time embedding: tells the block what position on the arc it's at
- Adaptive halting: stop when hidden state stabilizes (ACT-style)
- Total params: ~76M trainable (vs 3.6B in teacher)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

device = "cuda"

print("=" * 70)
print("CONTINUOUS ROTATION MODEL — adaptive depth, shared weights")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

print("Loading teacher...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

print(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Config
H = 2560
N_HEADS = 32
N_KV = 8
HEAD_DIM = H // N_HEADS  # 80
GQA = N_HEADS // N_KV    # 4
INTERMEDIATE = 9728
MAX_STEPS = 36  # max rotation steps (match teacher depth)
VOCAB = teacher.config.vocab_size

# Get actual dimensions from teacher weights
Q_DIM = teacher.model.layers[0].self_attn.q_proj.weight.shape[0]  # 4096
KV_DIM = teacher.model.layers[0].self_attn.k_proj.weight.shape[0]  # 1024
HEAD_DIM = Q_DIM // N_HEADS  # 128
KV_HEAD_DIM = KV_DIM // N_KV  # 128

print(f"H={H} heads={N_HEADS} hd={HEAD_DIM} kv={N_KV} kvhd={KV_HEAD_DIM}")
print(f"Q_DIM={Q_DIM} KV_DIM={KV_DIM} inter={INTERMEDIATE}")


class RotationBlock(nn.Module):
    """One rotation step: attention + MLP + time embedding.

    Applied repeatedly (weight-shared). Time embedding tells the block
    where on the arc it is (step 0 = high curvature, step N = flat).
    """
    def __init__(self):
        super().__init__()
        # Attention (match teacher shapes exactly)
        self.q_proj = nn.Linear(H, Q_DIM, bias=False)
        self.k_proj = nn.Linear(H, KV_DIM, bias=False)
        self.v_proj = nn.Linear(H, KV_DIM, bias=False)
        self.o_proj = nn.Linear(Q_DIM, H, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM)
        self.k_norm = nn.RMSNorm(KV_HEAD_DIM)
        self.attn_norm = nn.RMSNorm(H)

        # MLP (same shape as teacher)
        self.mlp_norm = nn.RMSNorm(H)
        self.gate_proj = nn.Linear(H, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(H, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, H, bias=False)

        # Time embedding: scalar t ∈ [0, 1] → H-dim modulation
        self.time_proj = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, H),
        )

        # Halting: predict probability of stopping at this step
        self.halt_proj = nn.Linear(H, 1)

    def forward(self, h, t_frac, cos, sin):
        """
        h: [B, T, H]
        t_frac: float in [0, 1] — position on the rotation arc
        cos, sin: rotary embeddings
        Returns: h_new, halt_prob
        """
        B, T, D = h.shape

        # Time modulation
        t_input = torch.tensor([[t_frac]], device=h.device, dtype=h.dtype)
        t_emb = self.time_proj(t_input).unsqueeze(0)  # [1, 1, H]

        # Attention
        residual = h
        h_norm = self.attn_norm(h + t_emb)

        q = self.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, N_KV, KV_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, N_KV, KV_HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary — use half of head_dim
        rd = HEAD_DIM // 2
        cos_r = cos[..., :rd].unsqueeze(1)
        sin_r = sin[..., :rd].unsqueeze(1)
        q1, q2 = q[..., :rd], q[..., rd:]
        q = torch.cat([q1*cos_r - q2*sin_r, q2*cos_r + q1*sin_r], -1)
        # K uses same head_dim as Q in Qwen3 (KV_HEAD_DIM == HEAD_DIM)
        krd = KV_HEAD_DIM // 2
        cos_k = cos[..., :krd].unsqueeze(1)
        sin_k = sin[..., :krd].unsqueeze(1)
        k1, k2 = k[..., :krd], k[..., krd:]
        k = torch.cat([k1*cos_k - k2*sin_k, k2*cos_k + k1*sin_k], -1)

        # GQA
        k = k.repeat_interleave(GQA, dim=1)
        v = v.repeat_interleave(GQA, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, Q_DIM)
        h = residual + self.o_proj(attn_out)

        # MLP
        residual = h
        h_mlp = self.mlp_norm(h)
        h = residual + self.down_proj(F.silu(self.gate_proj(h_mlp)) * self.up_proj(h_mlp))

        # Halt probability
        halt_logit = self.halt_proj(h.mean(dim=1))  # [B, 1]
        halt_prob = torch.sigmoid(halt_logit)

        return h, halt_prob


class ContinuousRotationModel(nn.Module):
    """Continuous rotation: one shared block applied adaptively."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.rotary_emb = teacher.model.rotary_emb  # reuse teacher's rotary
        self.rotation = RotationBlock()
        self.final_norm = nn.RMSNorm(H)
        self.lm_head = nn.Linear(H, VOCAB, bias=False)

        # Initialize embedding and lm_head from teacher
        self.embed.weight.data.copy_(teacher.model.embed_tokens.weight.data)
        self.lm_head.weight.data.copy_(teacher.lm_head.weight.data)
        # Freeze them
        self.embed.weight.requires_grad = False
        self.lm_head.weight.requires_grad = False

    def forward(self, input_ids, n_steps=None):
        """
        If n_steps is None: use adaptive halting (ACT).
        If n_steps is int: run exactly that many steps (for training).
        """
        h = self.embed(input_ids)
        B, T, D = h.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(h, pos)

        if n_steps is not None:
            # Run all steps like normal layers, but with time embedding
            # providing the curvature. No scaling — just run through.
            for step in range(n_steps):
                t_frac = step / max(n_steps - 1, 1)
                h, _ = self.rotation(h, t_frac, cos, sin)
        else:
            # Adaptive halting (ACT)
            cum_halt = torch.zeros(B, 1, device=device)
            remainders = torch.zeros(B, 1, device=device)
            n_updates = torch.zeros(B, 1, device=device)
            halted = torch.zeros(B, 1, dtype=torch.bool, device=device)

            for step in range(MAX_STEPS):
                t_frac = step / (MAX_STEPS - 1)
                h_new, halt_prob = self.rotation(h, t_frac, cos, sin)

                still_running = ~halted.squeeze(-1)
                cum_halt[still_running] += halt_prob[still_running]

                # Halt tokens where cumulative halt >= 1
                new_halted = (cum_halt >= 1.0).squeeze(-1) & still_running
                if new_halted.any():
                    remainders[new_halted] = 1.0 - (cum_halt[new_halted] - halt_prob.squeeze(-1)[new_halted].unsqueeze(-1))

                halted = halted | new_halted.unsqueeze(-1)
                n_updates[still_running] += 1

                # Update hidden state (weighted by running mask)
                mask = still_running.float().view(B, 1, 1)
                h = h * (1 - mask) + h_new * mask

                if halted.all():
                    break

        h = self.final_norm(h)
        return self.lm_head(h)


print("Building continuous rotation model...", flush=True)
student = ContinuousRotationModel()

# Initialize rotation block from teacher's MIDDLE layer (layer 18)
# This layer already knows how to rotate correctly
print("Initializing rotation block from teacher layer 18...", flush=True)
src_layer = teacher.model.layers[18]
with torch.no_grad():
    student.rotation.q_proj.weight.copy_(src_layer.self_attn.q_proj.weight)
    student.rotation.k_proj.weight.copy_(src_layer.self_attn.k_proj.weight)
    student.rotation.v_proj.weight.copy_(src_layer.self_attn.v_proj.weight)
    student.rotation.o_proj.weight.copy_(src_layer.self_attn.o_proj.weight)
    student.rotation.q_norm.weight.copy_(src_layer.self_attn.q_norm.weight)
    student.rotation.k_norm.weight.copy_(src_layer.self_attn.k_norm.weight)
    student.rotation.attn_norm.weight.copy_(src_layer.input_layernorm.weight)
    student.rotation.mlp_norm.weight.copy_(src_layer.post_attention_layernorm.weight)
    student.rotation.gate_proj.weight.copy_(src_layer.mlp.gate_proj.weight)
    student.rotation.up_proj.weight.copy_(src_layer.mlp.up_proj.weight)
    student.rotation.down_proj.weight.copy_(src_layer.mlp.down_proj.weight)

student = student.to(torch.bfloat16).to(device)

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
print(f"Params: {total/1e6:.0f}M total, {trainable/1e6:.0f}M trainable")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training: curriculum — start with more steps, anneal down
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations.",
    "Marine biology studies organisms in the ocean covering more than seventy percent of Earth surface.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s.",
    "Quantum mechanics describes matter and energy at smallest scales where particles exhibit wave properties.",
    "The French Revolution transformed French society by uprooting centuries of tradition and absolute monarchy.",
    "Climate change is driven by burning fossil fuels which releases carbon dioxide into the atmosphere.",
    "The human genome contains approximately three billion base pairs organized into twenty three chromosomes.",
    "Machine learning algorithms improve through experience without being explicitly programmed for each task.",
    "The Amazon rainforest produces significant oxygen and houses incredible biodiversity across many species.",
    "Cryptography enables secure communication by transforming readable messages into encrypted ciphertext.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
    "General relativity describes gravity as curvature of spacetime caused by mass and energy distributions.",
    "Evolution explains how populations change over generations through variation inheritance and selection.",
    "The Internet revolutionized global communication by connecting billions of devices through protocols.",
    "Photosynthesis converts light energy into chemical energy in green plants using water and carbon dioxide.",
] * 8

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=64, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

N_STEPS = 1000
BATCH = 4
TEMP = 2.0

# Curriculum: start with 36 steps (match teacher depth), anneal to fewer
def get_n_steps(training_step, total_steps):
    """Fixed: always run all steps. Learn the rotation first. Optimize later."""
    return MAX_STEPS

print(f"\nTraining {N_STEPS} steps, batch={BATCH}")
print(f"Fixed {MAX_STEPS} steps (like normal model, but curved). Initialized from teacher layer 18.")
print(f"{'Step':>6} {'KL':>8} {'n_rot':>6} {'VRAM':>6}")
print("-" * 30)

losses = []
for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    n_rot = get_n_steps(step, N_STEPS)

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = student(batch, n_steps=n_rot)

    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_p = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_p, t_probs, reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 100 == 0 or step == N_STEPS - 1:
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>6} {loss.item():>8.4f} {n_rot:>5} {vram:>5.1f}G", flush=True)

# ═══════════════════════════════════════════════════════
# Validation: generate and compare
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

for n_test_steps in [36, 24, 12, 6]:
    print(f"\n  n_steps={n_test_steps}:")
    for prompt in ["The future of artificial intelligence will",
                   "Water freezes at zero degrees and"]:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            gen = ids.clone()
            for _ in range(30):
                logits = student(gen, n_steps=n_test_steps)
                gen = torch.cat([gen, logits[0, -1:].argmax(-1).unsqueeze(0)], dim=-1)
        text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"    [{text[:60]}]")

# Teacher comparison
print(f"\n  Teacher:")
for prompt in ["The future of artificial intelligence will",
               "Water freezes at zero degrees and"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = teacher.generate(ids, max_new_tokens=30, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"    [{text[:60]}]")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "state_dict": student.state_dict(),
    "config": {"H": H, "N_HEADS": N_HEADS, "N_KV": N_KV, "HEAD_DIM": HEAD_DIM,
               "INTERMEDIATE": INTERMEDIATE, "MAX_STEPS": MAX_STEPS, "VOCAB": VOCAB},
    "losses": losses,
}, os.path.join(SAVE_DIR, "continuous_rotation_4b.pt"))
print(f"\nSaved. Final KL: {losses[-1]:.4f}", flush=True)
