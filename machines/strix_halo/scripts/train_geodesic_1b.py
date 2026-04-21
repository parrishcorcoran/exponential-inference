"""Geodesic model at manifold scale — ~1B params.

Teacher: Qwen3-0.6B (same manifold, right-sized bulk)
Student: 1 attention + ODE dynamics at H=1024, intermediate=3072

The 0.6B model proves the manifold fits in this bulk.
The geodesic learns to navigate it in one continuous pass
instead of 28 discrete layers.

24-hour training run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import time
import os

device = "cuda"

print("=" * 70)
print("GEODESIC 1B — right-sized for the manifold")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

H = teacher.config.hidden_size          # 1024
N_HEADS = teacher.config.num_attention_heads  # 16
N_KV = teacher.config.num_key_value_heads     # 8
INTER = teacher.config.intermediate_size      # 3072
N_LAYERS = teacher.config.num_hidden_layers   # 28
VOCAB = teacher.config.vocab_size
HEAD_DIM = teacher.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS

print(f"Teacher: H={H}, heads={N_HEADS}, kv={N_KV}, hd={HEAD_DIM}, inter={INTER}, L={N_LAYERS}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


class ContextAttention(nn.Module):
    """1 KV head attention. Provides starting coordinate from context."""
    def __init__(self):
        super().__init__()
        q_dim = N_HEADS * HEAD_DIM
        kv_dim = 1 * HEAD_DIM  # 1 KV head

        self.q_proj = nn.Linear(H, q_dim, bias=False)
        self.k_proj = nn.Linear(H, kv_dim, bias=False)
        self.v_proj = nn.Linear(H, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, H, bias=False)
        self.norm = nn.LayerNorm(H)

    def forward(self, h, k_cache=None, v_cache=None):
        B, T, D = h.shape
        h_norm = self.norm(h)

        q = self.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, 1, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, 1, HEAD_DIM).transpose(1, 2)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        k_exp = k.expand(B, N_HEADS, -1, HEAD_DIM)
        v_exp = v.expand(B, N_HEADS, -1, HEAD_DIM)

        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)

        return h + self.o_proj(attn_out), k, v


class BulkDynamics(nn.Module):
    """f(t, h) — the gradient field. SwiGLU MLP at manifold scale."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.gate_proj = nn.Linear(H, INTER, bias=False)
        self.up_proj = nn.Linear(H, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, H, bias=False)
        self.time_mix = nn.Linear(1, H, bias=False)

        nn.init.normal_(self.down_proj.weight, std=0.01)

    def forward(self, t, h):
        h = h.float()
        t_emb = self.time_mix(t.float().reshape(1, 1)).expand(h.shape[0], -1)
        h_in = self.norm(h + t_emb)
        return self.down_proj(F.silu(self.gate_proj(h_in)) * self.up_proj(h_in))


class Geodesic1B(nn.Module):
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
        h = self.embed(tok_id).float()
        h_ctx, k_cache, v_cache = self.attention(h, k_cache, v_cache)

        B, T, D = h_ctx.shape
        h_flat = h_ctx.reshape(B * T, D)
        t_span = torch.tensor([0.0, 1.0], device=device)
        h_traj = odeint(self.dynamics, h_flat, t_span,
                        method='euler', options={'step_size': 0.1})
        h_final = h_traj[-1].reshape(B, T, D)

        h_normed = self.final_norm(h_final)
        logits = self.lm_head(h_normed.to(self.lm_head.weight.dtype))
        return logits, k_cache, v_cache

    def forward_sequence(self, input_ids):
        B, T = input_ids.shape
        all_logits = []
        k_cache, v_cache = None, None
        for t in range(T):
            logits, k_cache, v_cache = self.forward_single(
                input_ids[:, t:t+1], k_cache, v_cache)
            all_logits.append(logits)
        return torch.cat(all_logits, dim=1)


print("Building model...", flush=True)
student = Geodesic1B(teacher).to(device)
student.dynamics = student.dynamics.float()
student.final_norm = student.final_norm.float()
student.attention = student.attention.float()

# Initialize from teacher
print("Initializing from teacher...", flush=True)
src_attn = teacher.model.layers[0].self_attn
src_mlp = teacher.model.layers[14].mlp  # middle layer
with torch.no_grad():
    student.attention.q_proj.weight.copy_(src_attn.q_proj.weight.float())
    k_full = src_attn.k_proj.weight.view(N_KV, HEAD_DIM, H)
    student.attention.k_proj.weight.copy_(k_full[0].reshape(HEAD_DIM, H).float())
    v_full = src_attn.v_proj.weight.view(N_KV, HEAD_DIM, H)
    student.attention.v_proj.weight.copy_(v_full[0].reshape(HEAD_DIM, H).float())
    student.attention.o_proj.weight.copy_(src_attn.o_proj.weight.float())
    student.attention.norm.weight.copy_(teacher.model.layers[0].input_layernorm.weight.float())

    student.dynamics.gate_proj.weight.copy_(src_mlp.gate_proj.weight.float())
    student.dynamics.up_proj.weight.copy_(src_mlp.up_proj.weight.float())
    # down_proj stays small init
    student.dynamics.norm.weight.copy_(teacher.model.layers[14].post_attention_layernorm.weight.float())

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
print(f"Total: {total/1e6:.0f}M, Trainable: {trainable/1e6:.1f}M")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Training data
texts = [
    "The history of mathematics spans thousands of years and includes contributions from civilizations.",
    "Marine biology studies organisms in the ocean covering seventy percent of Earth surface.",
    "Artificial intelligence has progressed through several phases since the 1950s.",
    "Quantum mechanics describes matter and energy at the smallest scales.",
    "Climate change threatens ecosystems worldwide through rising temperatures.",
    "Machine learning algorithms improve through experience without explicit programming.",
    "Neural networks learn representations through layers of nonlinear transformations.",
    "General relativity describes gravity as curvature of spacetime.",
    "The stock market operates through interactions between buyers and sellers.",
    "DNA carries genetic instructions for development and function of organisms.",
    "The Renaissance was a period of cultural rebirth beginning in the fourteenth century.",
    "Volcanic eruptions release magma and gases from deep within the Earth.",
    "The Internet transformed communication by connecting billions of devices.",
    "Antibiotics revolutionized medicine by treating bacterial infections effectively.",
    "Plate tectonics explains movement creating earthquakes and mountains.",
    "Photosynthesis converts sunlight into glucose and oxygen in plants.",
    "Democracy originated in ancient Athens where citizens participated directly.",
    "The human brain contains eighty six billion neurons connected by synapses.",
    "Coffee originated in Ethiopia and became popular worldwide over centuries.",
    "Gravitational waves are ripples in spacetime caused by massive objects.",
    "The speed of light is approximately three hundred thousand kilometers per second.",
    "Coral reefs support thousands of marine species in tropical waters.",
    "Evolution explains how species adapt to environments over many generations.",
    "Black holes form when massive stars collapse under their own gravity.",
    "Fibonacci numbers appear in nature in leaves petals and spiral shells.",
    "The printing press democratized access to knowledge across Europe.",
    "Protein folding determines structure and function of biological molecules.",
    "Reinforcement learning trains agents by maximizing cumulative reward signals.",
    "The water cycle moves water through evaporation condensation and precipitation.",
    "Encryption protects digital information by transforming data into ciphertext.",
] * 8

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=48, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training data: {train_ids.shape}")

# Training
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

N_STEPS = 25000
BATCH = 4  # 0.6B is small, can afford larger batch
warmup_steps = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_STEPS - warmup_steps)

SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nTraining {N_STEPS} steps, batch={BATCH}")
print(f"{'Step':>7} {'Loss':>8} {'LR':>10} {'VRAM':>6} {'s/step':>7} {'ETA_h':>6}")
print("-" * 50)

losses = []
t_start = time.time()

for step in range(N_STEPS):
    if step < warmup_steps:
        lr = 1e-4 * (step + 1) / warmup_steps
        for pg in optimizer.param_groups: pg['lr'] = lr
    else:
        scheduler.step()

    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = student.forward_sequence(batch)

    targets = t_logits[:, :-1].argmax(-1)
    s_shift = s_logits[:, :-1].float()
    loss = F.cross_entropy(s_shift.reshape(-1, VOCAB), targets.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 500 == 0 or step == N_STEPS - 1:
        elapsed = time.time() - t_start
        spd = elapsed / (step + 1)
        eta = (N_STEPS - step) * spd / 3600
        lr_now = optimizer.param_groups[0]['lr']
        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>7} {avg_loss:>8.4f} {lr_now:>10.6f} {vram:>5.1f}G {spd:>6.2f}s {eta:>5.1f}h",
              flush=True)

    if (step + 1) % 2500 == 0:
        avg = sum(losses[-500:]) / min(len(losses), 500)
        torch.save({
            "attention_state": student.attention.state_dict(),
            "dynamics_state": student.dynamics.state_dict(),
            "norm_state": student.final_norm.state_dict(),
            "step": step + 1, "loss": avg, "losses": losses[-2000:],
        }, os.path.join(SAVE_DIR, f"geodesic_1b_step{step+1}.pt"))

        with torch.no_grad():
            sample_ids = tokenizer("The future of", return_tensors='pt').input_ids.to(device)
            gen = sample_ids.clone()
            kc, vc = None, None
            for t_idx in range(sample_ids.shape[1]):
                logits, kc, vc = student.forward_single(sample_ids[:, t_idx:t_idx+1], kc, vc)
            for _ in range(25):
                nt = logits[0, -1].argmax(-1)
                logits, kc, vc = student.forward_single(nt.view(1, 1), kc, vc)
                gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
            sample = tokenizer.decode(gen[0][sample_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Ckpt step{step+1} loss={avg:.4f}: [{sample[:60]}]", flush=True)

# Final validation
print(f"\n{'='*60}")
print("FINAL VALIDATION")
print(f"{'='*60}")

for prompt in ["The future of artificial intelligence",
               "Water freezes at zero degrees",
               "The most important concept in physics"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        t_out = teacher.generate(ids, max_new_tokens=30, do_sample=False)
    t_text = tokenizer.decode(t_out[0][ids.shape[1]:], skip_special_tokens=True)

    with torch.no_grad():
        gen = ids.clone()
        kc, vc = None, None
        for t_idx in range(ids.shape[1]):
            logits, kc, vc = student.forward_single(ids[:, t_idx:t_idx+1], kc, vc)
        for _ in range(30):
            nt = logits[0, -1].argmax(-1)
            logits, kc, vc = student.forward_single(nt.view(1, 1), kc, vc)
            gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
    s_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"\n  '{prompt}'")
    print(f"  Teacher: {t_text[:60]}")
    print(f"  Student: {s_text[:60]}")

elapsed = time.time() - t_start
print(f"\nTotal: {elapsed/3600:.1f}h. Final loss: {losses[-1]:.4f}", flush=True)
