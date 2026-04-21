"""Train 128-head model via KL distillation from Qwen3-14B.

Architecture:
- 128 Q heads × 40 dim = 5120 (same hidden dim as teacher)
- 32 KV heads × 40 dim (GQA ratio 4:1)
- MLP: FROZEN from teacher (bulk doesn't change)
- Only attention weights trained (Q, K, V, O projections + norms)
- KL divergence against teacher's output distribution

This gives us fine-grained head routing: 128 independent heads where
the manifold router selects which subset to use per token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import copy

device = "cuda"

print("=" * 70)
print("128-HEAD DISTILLATION — from Qwen3-14B teacher")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

# Load teacher
print("Loading teacher (frozen)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

print(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Student: 128 heads × 40 dim, frozen MLP from teacher
# ═══════════════════════════════════════════════════════

STUDENT_HEADS = 128
STUDENT_HEAD_DIM = 40
STUDENT_KV_HEADS = 32
H = 5120
N_LAYERS = 40

print(f"\nStudent: {STUDENT_HEADS} Q heads × {STUDENT_HEAD_DIM}d, "
      f"{STUDENT_KV_HEADS} KV heads, GQA {STUDENT_HEADS//STUDENT_KV_HEADS}:1")


class StudentAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, STUDENT_HEADS * STUDENT_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(H, STUDENT_KV_HEADS * STUDENT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H, STUDENT_KV_HEADS * STUDENT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(STUDENT_HEADS * STUDENT_HEAD_DIM, H, bias=False)
        self.q_norm = nn.RMSNorm(STUDENT_HEAD_DIM)
        self.k_norm = nn.RMSNorm(STUDENT_HEAD_DIM)

    def forward(self, h_norm, cos, sin):
        B, T, D = h_norm.shape
        gqa = STUDENT_HEADS // STUDENT_KV_HEADS

        q = self.q_proj(h_norm).view(B, T, STUDENT_HEADS, STUDENT_HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, STUDENT_KV_HEADS, STUDENT_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, STUDENT_KV_HEADS, STUDENT_HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotary (head_dim = 40, so rd = 20)
        rd = STUDENT_HEAD_DIM // 2
        cos_r = cos[..., :rd].unsqueeze(1)
        sin_r = sin[..., :rd].unsqueeze(1)
        q1, q2 = q[..., :rd], q[..., rd:]
        q = torch.cat([q1*cos_r - q2*sin_r, q2*cos_r + q1*sin_r], -1)
        k1, k2 = k[..., :rd], k[..., rd:]
        k = torch.cat([k1*cos_r - k2*sin_r, k2*cos_r + k1*sin_r], -1)

        # GQA
        k = k.repeat_interleave(gqa, dim=1)
        v = v.repeat_interleave(gqa, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


class Student128(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        # Shared frozen components
        self.embed_tokens = teacher_model.model.embed_tokens
        self.rotary_emb = teacher_model.model.rotary_emb
        self.final_norm = teacher_model.model.norm
        self.lm_head = teacher_model.lm_head

        for p in self.embed_tokens.parameters(): p.requires_grad = False
        for p in self.final_norm.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

        # Trainable attention + copied layernorms
        self.attentions = nn.ModuleList([StudentAttention() for _ in range(N_LAYERS)])
        self.input_norms = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(N_LAYERS):
            layer = teacher_model.model.layers[i]
            self.input_norms.append(copy.deepcopy(layer.input_layernorm))
            self.post_norms.append(copy.deepcopy(layer.post_attention_layernorm))
            self.mlps.append(layer.mlp)
            for p in layer.mlp.parameters(): p.requires_grad = False

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        B, T, D = h.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(h, pos_ids)

        for i in range(N_LAYERS):
            residual = h
            h_norm = self.input_norms[i](h)
            h = residual + self.attentions[i](h_norm, cos, sin)
            residual = h
            h = residual + self.mlps[i](self.post_norms[i](h))

        return self.lm_head(self.final_norm(h))


print("Building student...", flush=True)
student = Student128(teacher).to(torch.bfloat16).to(device)

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
total = sum(p.numel() for p in student.parameters())
print(f"Params: {total/1e9:.2f}B total, {trainable/1e6:.0f}M trainable (attention only)")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training
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
    "The Amazon rainforest produces significant oxygen and houses incredible biodiversity across millions of species.",
    "Cryptography enables secure communication by transforming readable messages into encrypted ciphertext.",
    "Neural networks learn hierarchical representations through multiple layers of nonlinear transformations.",
    "The periodic table organizes chemical elements by atomic number revealing patterns in their properties.",
] * 4

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=64, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"\nTraining data: {train_ids.shape}")

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=2e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)

N_STEPS = 500
BATCH = 2
TEMP = 2.0

print(f"Training {N_STEPS} steps, batch={BATCH}, T={TEMP}")
print(f"{'Step':>6} {'KL':>8} {'tok/s':>7} {'VRAM':>6}")
print("-" * 35)

losses = []
for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = student(batch)

    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_probs = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if step % 50 == 0 or step == N_STEPS - 1:
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>6} {loss.item():>8.4f} {'—':>7} {vram:>5.1f}G", flush=True)

# ═══════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

for prompt in ["The future of artificial intelligence will",
               "Water freezes at zero degrees and boils at"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        t_out = teacher.generate(ids, max_new_tokens=30, do_sample=False)
    t_text = tokenizer.decode(t_out[0][ids.shape[1]:], skip_special_tokens=True)

    with torch.no_grad():
        gen = ids.clone()
        for _ in range(30):
            logits = student(gen)
            gen = torch.cat([gen, logits[0, -1:].argmax(-1).unsqueeze(0)], dim=-1)
    s_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"\n  '{prompt}'")
    print(f"  Teacher: {t_text[:70]}")
    print(f"  Student: {s_text[:70]}")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "state_dict": {k: v for k, v in student.state_dict().items() if "attention" in k},
    "config": {"n_heads": STUDENT_HEADS, "head_dim": STUDENT_HEAD_DIM,
               "n_kv": STUDENT_KV_HEADS, "n_layers": N_LAYERS, "hidden": H},
    "losses": losses,
}, os.path.join(SAVE_DIR, "student_128head_v2.pt"))
print(f"\nSaved. Final loss: {losses[-1]:.4f}", flush=True)
