"""Geodesic Model v2: attention + ODE flow.

Per token:
1. Embed
2. One attention against KV cache (provides coordinate/context)
3. ODE flow (continuous annealing from bulk to boundary)
4. Token appears at boundary
5. Store 1 KV head from final state

Teacher: Qwen3-4B. Student: attention + ODE dynamics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import time
import os

device = "cuda"

print("=" * 70)
print("GEODESIC v2: attention → ODE flow → boundary")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

H = teacher.config.hidden_size  # 2560
N_HEADS = 32
HEAD_DIM = teacher.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS  # 128
N_KV_TEACHER = teacher.config.num_key_value_heads  # 8
VOCAB = teacher.config.vocab_size

print(f"H={H}, heads={N_HEADS}, hd={HEAD_DIM}, kv={N_KV_TEACHER}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# ═══════════════════════════════════════════════════════
# Student: Attention (1 KV head) + ODE dynamics
# ═══════════════════════════════════════════════════════

class ContextAttention(nn.Module):
    """Single attention layer: provides coordinate from KV context.
    1 KV head (we know the position precisely).
    Full Q heads for spatial resolution.
    """
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.n_kv = 1  # ONE KV head

        q_dim = N_HEADS * HEAD_DIM
        kv_dim = 1 * HEAD_DIM  # 1 KV head

        self.q_proj = nn.Linear(H, q_dim, bias=False)
        self.k_proj = nn.Linear(H, kv_dim, bias=False)
        self.v_proj = nn.Linear(H, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, H, bias=False)
        self.norm = nn.LayerNorm(H)

    def forward(self, h, k_cache=None, v_cache=None):
        """
        h: [B, 1, H] current token
        k_cache: [B, 1, cache_len, HD] or None (first token)
        v_cache: [B, 1, cache_len, HD] or None

        Returns: context-aware h, new_k, new_v (for cache)
        """
        B, T, D = h.shape
        h = h.float()
        h_norm = self.norm(h)

        q = self.q_proj(h_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)

        # Append to cache
        if k_cache is not None:
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)
        else:
            k_full = k
            v_full = v

        # GQA: expand KV for all Q heads
        k_exp = k_full.expand(B, self.n_heads, -1, self.head_dim)
        v_exp = v_full.expand(B, self.n_heads, -1, self.head_dim)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(attn_out)

        return h + out, k_full, v_full


class BulkDynamics(nn.Module):
    """f(t, h) — ODE dynamics. The gradient field in the bulk.

    The bulk IS the holographic projection medium. It needs to be
    substantial — BitNet showed 66% of params is bulk and necessary.
    Gate + up + down structure matches the teacher's SwiGLU MLP.
    """
    def __init__(self):
        super().__init__()
        INTER = 9728  # match teacher's intermediate size exactly

        self.norm = nn.LayerNorm(H)
        # SwiGLU MLP (same structure as teacher's MLP)
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)
        # Time conditioning
        self.time_mix = nn.Linear(1, H, bias=False)

        # Small init on output for gentle initial dynamics
        nn.init.normal_(self.down_proj.weight, std=0.01)

    def forward(self, t, h):
        h = h.float()
        t_emb = self.time_mix(t.float().reshape(1, 1)).expand(h.shape[0], -1)
        h_in = self.norm(h + t_emb)
        # SwiGLU: same as transformer MLP
        return self.down_proj(F.silu(self.gate_proj(h_in)) * self.up_proj(h_in))


class GeodesicV2(nn.Module):
    """Full model: embed → attend → ODE → lm_head."""
    def __init__(self, teacher_model):
        super().__init__()
        self.embed = teacher_model.model.embed_tokens
        self.lm_head = teacher_model.lm_head
        for p in self.embed.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

        self.attention = ContextAttention()
        self.dynamics = BulkDynamics()
        self.final_norm = nn.LayerNorm(H)

    def forward_single(self, tok_id, k_cache=None, v_cache=None):
        """Process one token. Returns logits, updated k_cache, v_cache."""
        h = self.embed(tok_id).float()  # [B, 1, H]

        # Step 1: Attention (get coordinate from context)
        h_ctx, k_cache, v_cache = self.attention(h, k_cache, v_cache)

        # Step 2: ODE flow (anneal to boundary)
        B, T, D = h_ctx.shape
        h_flat = h_ctx.reshape(B * T, D)
        t_span = torch.tensor([0.0, 1.0], device=h_flat.device)
        h_traj = odeint(self.dynamics, h_flat, t_span,
                        method='euler', options={'step_size': 0.1})
        h_final = h_traj[-1].reshape(B, T, D)

        # Step 3: Project to vocab
        h_normed = self.final_norm(h_final.float())
        logits = self.lm_head(h_normed.to(self.lm_head.weight.dtype))

        return logits, k_cache, v_cache

    def forward_sequence(self, input_ids):
        """Process full sequence (for training). Teacher-forcing."""
        B, T = input_ids.shape
        all_logits = []
        k_cache = None
        v_cache = None

        for t in range(T):
            tok = input_ids[:, t:t+1]  # [B, 1]
            logits, k_cache, v_cache = self.forward_single(tok, k_cache, v_cache)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)  # [B, T, V]


# ═══════════════════════════════════════════════════════
# Build and train
# ═══════════════════════════════════════════════════════
print("Building model...", flush=True)
student = GeodesicV2(teacher).to(device)
student.dynamics = student.dynamics.float()
student.final_norm = student.final_norm.float()
student.attention = student.attention.float()

# V3: Initialize attention from teacher's first layer
# The teacher's layer 0 already knows how to read context correctly
print("Initializing attention from teacher layer 0...", flush=True)
src = teacher.model.layers[0].self_attn
with torch.no_grad():
    # Q: teacher has [Q_DIM, H], we have [N_HEADS*HEAD_DIM, H]
    # They should be same shape since we use same N_HEADS and HEAD_DIM
    student.attention.q_proj.weight.copy_(src.q_proj.weight.float())
    # K/V: teacher has 8 KV heads, we have 1. Take first KV head.
    k_full = src.k_proj.weight.view(N_KV_TEACHER, HEAD_DIM, H)
    student.attention.k_proj.weight.copy_(k_full[0].reshape(HEAD_DIM, H).float())
    v_full = src.v_proj.weight.view(N_KV_TEACHER, HEAD_DIM, H)
    student.attention.v_proj.weight.copy_(v_full[0].reshape(HEAD_DIM, H).float())
    # O: teacher has [H, Q_DIM], same shape
    student.attention.o_proj.weight.copy_(src.o_proj.weight.float())
    # Norm from teacher's input layernorm
    student.attention.norm.weight.copy_(teacher.model.layers[0].input_layernorm.weight.float())
print("Attention initialized from teacher.", flush=True)

# Also initialize dynamics MLP from teacher's middle layer MLP
# The bulk medium should start from a working state
print("Initializing dynamics MLP from teacher layer 18...", flush=True)
src_mlp = teacher.model.layers[18].mlp
with torch.no_grad():
    student.dynamics.gate_proj.weight.copy_(src_mlp.gate_proj.weight.float())
    student.dynamics.up_proj.weight.copy_(src_mlp.up_proj.weight.float())
    # down_proj stays small-init (so initial dh/dt is small)
    # student.dynamics.down_proj.weight.copy_(src_mlp.down_proj.weight.float())
    student.dynamics.norm.weight.copy_(teacher.model.layers[18].post_attention_layernorm.weight.float())
print("Dynamics MLP initialized from teacher.", flush=True)

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.1f}M params")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Training data — much more diverse for 24h run
texts = [
    "The history of mathematics spans thousands of years and includes contributions from civilizations.",
    "Marine biology studies organisms in the ocean covering seventy percent of Earth surface.",
    "Artificial intelligence has progressed through several phases since the 1950s.",
    "Quantum mechanics describes matter and energy at the smallest scales.",
    "Climate change threatens ecosystems worldwide through rising temperatures.",
    "Machine learning algorithms improve through experience without explicit programming.",
    "Neural networks learn representations through layers of nonlinear transformations.",
    "General relativity describes gravity as curvature of spacetime.",
    "The stock market operates through complex interactions between buyers and sellers worldwide.",
    "DNA carries the genetic instructions for the development and function of living organisms.",
    "The Renaissance was a period of cultural rebirth in Europe beginning in the fourteenth century.",
    "Volcanic eruptions release magma ash and gases from deep within the Earth to the surface.",
    "The Internet transformed communication by connecting billions of devices through standard protocols.",
    "Antibiotics revolutionized medicine by enabling the treatment of bacterial infections effectively.",
    "Plate tectonics explains the movement of large plates on Earth creating earthquakes and mountains.",
    "Photosynthesis converts sunlight water and carbon dioxide into glucose and oxygen in plants.",
    "The Pythagorean theorem states that in a right triangle the square of the hypotenuse equals.",
    "Democracy originated in ancient Athens where citizens participated directly in government decisions.",
    "Superconductors carry electrical current with zero resistance below their critical temperature.",
    "The human brain contains approximately eighty six billion neurons connected by trillions of synapses.",
    "Coffee originated in Ethiopia and became one of the most popular beverages in the world.",
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects like black holes.",
    "Shakespeare wrote approximately thirty seven plays that are considered masterpieces of English literature.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Coral reefs are among the most diverse ecosystems supporting thousands of marine species globally.",
    "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward signals.",
    "The water cycle describes continuous movement of water through evaporation condensation and precipitation.",
    "Encryption protects digital information by transforming readable data into seemingly random ciphertext.",
    "Evolution by natural selection explains how species adapt to their environments over many generations.",
    "The periodic table arranges elements by atomic number revealing recurring patterns in chemical properties.",
    "Black holes form when massive stars exhaust their nuclear fuel and collapse under their own gravity.",
    "The printing press invented by Gutenberg around 1440 democratized access to knowledge across Europe.",
    "Fibonacci numbers appear throughout nature in the arrangement of leaves petals and spiral shells.",
    "Thermodynamics governs the relationships between heat work and energy in physical and chemical systems.",
    "The Great Wall of China stretches thousands of kilometers across northern China built over centuries.",
    "Protein folding determines the three dimensional structure and function of biological molecules.",
    "The Industrial Revolution transformed manufacturing through mechanization steam power and factory systems.",
    "Bayesian statistics updates probability estimates as new evidence becomes available using prior knowledge.",
    "The Amazon River carries more water than any other river flowing through South America to the Atlantic.",
    "Transistors are the fundamental building blocks of all modern electronic devices and computer processors.",
] * 4

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=48, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

# 24h run: at 0.32 steps/s = ~27,000 steps per day
N_STEPS = 25000
BATCH = 2

# Learning rate warmup + cosine decay
warmup_steps = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_STEPS - warmup_steps)

print(f"\nTraining {N_STEPS} steps (~24h), batch={BATCH}")
print(f"LR: warmup {warmup_steps} steps → cosine decay")
print(f"{'Step':>7} {'Loss':>8} {'LR':>10} {'VRAM':>6} {'tok/s':>7}")
print("-" * 45)

losses = []
t_start = time.time()
best_loss = float('inf')

for step in range(N_STEPS):
    # LR warmup
    if step < warmup_steps:
        lr = 1e-4 * (step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    else:
        scheduler.step()

    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    # Teacher logits
    with torch.no_grad():
        t_logits = teacher(batch).logits

    # Student logits
    s_logits = student.forward_sequence(batch)

    # CE against teacher's predictions
    targets = t_logits[:, :-1].argmax(-1)
    s_shift = s_logits[:, :-1].float()
    loss = F.cross_entropy(s_shift.reshape(-1, VOCAB), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()

    losses.append(loss.item())

    if step % 500 == 0 or step == N_STEPS - 1:
        elapsed_so_far = time.time() - t_start
        steps_per_sec = (step + 1) / elapsed_so_far
        eta_h = (N_STEPS - step) / steps_per_sec / 3600
        lr_now = optimizer.param_groups[0]['lr']
        vram = torch.cuda.memory_allocated() / 1e9
        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        print(f"{step:>7} {avg_loss:>8.4f} {lr_now:>10.6f} {vram:>5.1f}G {steps_per_sec:>6.2f} ETA:{eta_h:.1f}h",
              flush=True)

    # Checkpoint every 2500 steps
    if (step + 1) % 2500 == 0:
        avg_recent = sum(losses[-500:]) / min(len(losses), 500)
        ckpt_path = os.path.join(SAVE_DIR, f"geodesic_step{step+1}.pt")
        torch.save({
            "attention_state": student.attention.state_dict(),
            "dynamics_state": student.dynamics.state_dict(),
            "norm_state": student.final_norm.state_dict(),
            "step": step + 1, "loss": avg_recent, "losses": losses[-1000:],
        }, ckpt_path)
        print(f"  Checkpoint: {ckpt_path} (loss={avg_recent:.4f})", flush=True)

        # Quick generation sample
        with torch.no_grad():
            sample_ids = tokenizer("The future of", return_tensors='pt').input_ids.to(device)
            gen = sample_ids.clone()
            kc, vc = None, None
            for t_idx in range(sample_ids.shape[1]):
                logits, kc, vc = student.forward_single(sample_ids[:, t_idx:t_idx+1], kc, vc)
            for _ in range(20):
                nt = logits[0, -1].argmax(-1)
                logits, kc, vc = student.forward_single(nt.view(1, 1), kc, vc)
                gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
            sample = tokenizer.decode(gen[0][sample_ids.shape[1]:], skip_special_tokens=True)
            print(f"  Sample: [{sample[:60]}]", flush=True)

elapsed = time.time() - t_start
print(f"\nTraining complete: {elapsed/3600:.1f}h ({N_STEPS/elapsed:.2f} steps/s)")

# ═══════════════════════════════════════════════════════
# Validation: generate
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("GENERATION")
print(f"{'='*60}")

for prompt in ["The future of artificial intelligence",
               "Water freezes at zero degrees"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # Teacher
    with torch.no_grad():
        t_out = teacher.generate(ids, max_new_tokens=20, do_sample=False)
    t_text = tokenizer.decode(t_out[0][ids.shape[1]:], skip_special_tokens=True)

    # Student
    with torch.no_grad():
        gen = ids.clone()
        k_cache, v_cache = None, None
        # Process prompt
        for t in range(ids.shape[1]):
            logits, k_cache, v_cache = student.forward_single(
                ids[:, t:t+1], k_cache, v_cache)
        # Generate
        next_tok = logits[0, -1].argmax(-1)
        for _ in range(20):
            logits, k_cache, v_cache = student.forward_single(
                next_tok.view(1, 1), k_cache, v_cache)
            next_tok = logits[0, -1].argmax(-1)
            gen = torch.cat([gen, next_tok.view(1, 1)], dim=-1)
    s_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"\n  '{prompt}'")
    print(f"  Teacher: {t_text[:60]}")
    print(f"  Student: {s_text[:60]}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "attention_state": student.attention.state_dict(),
    "dynamics_state": student.dynamics.state_dict(),
    "norm_state": student.final_norm.state_dict(),
    "config": {"H": H, "n_heads": N_HEADS, "head_dim": HEAD_DIM,
               "n_kv": 1, "ode_method": "euler", "step_size": 0.1},
    "losses": losses,
}, os.path.join(SAVE_DIR, "geodesic_v2_4b.pt"))
print(f"\nSaved. Final loss: {losses[-1]:.4f}", flush=True)
