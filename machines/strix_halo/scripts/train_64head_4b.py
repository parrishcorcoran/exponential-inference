"""Train 64-head model from Qwen3-4B teacher.

Teacher: Qwen3-4B (32 heads × 80 dim, 8 KV, 36 layers, hidden=2560)
Student: 64 heads × 40 dim, 16 KV heads, same MLP (frozen)

Same tokenizer = same manifold. 4B is above manifold floor.
64 heads gives fine-grained routing. Manifold router selects which heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import copy

device = "cuda"

print("=" * 70)
print("64-HEAD DISTILLATION from Qwen3-4B")
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
N_LAYERS = 36
STUDENT_HEADS = 64
STUDENT_HEAD_DIM = 40  # 64 × 40 = 2560
STUDENT_KV = 16        # GQA 4:1
TEACHER_HEADS = 32
TEACHER_HD = 80

print(f"Student: {STUDENT_HEADS} Q × {STUDENT_HEAD_DIM}d, {STUDENT_KV} KV, GQA {STUDENT_HEADS//STUDENT_KV}:1")


class StudentAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, STUDENT_HEADS * STUDENT_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(H, STUDENT_KV * STUDENT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H, STUDENT_KV * STUDENT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(STUDENT_HEADS * STUDENT_HEAD_DIM, H, bias=False)
        self.q_norm = nn.RMSNorm(STUDENT_HEAD_DIM)
        self.k_norm = nn.RMSNorm(STUDENT_HEAD_DIM)

    def forward(self, h_norm, cos, sin):
        B, T, _ = h_norm.shape
        gqa = STUDENT_HEADS // STUDENT_KV

        q = self.q_proj(h_norm).view(B, T, STUDENT_HEADS, STUDENT_HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, STUDENT_KV, STUDENT_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, STUDENT_KV, STUDENT_HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        rd = STUDENT_HEAD_DIM // 2
        cos_r = cos[..., :rd].unsqueeze(1)
        sin_r = sin[..., :rd].unsqueeze(1)
        q1, q2 = q[..., :rd], q[..., rd:]
        q = torch.cat([q1*cos_r - q2*sin_r, q2*cos_r + q1*sin_r], -1)
        k1, k2 = k[..., :rd], k[..., rd:]
        k = torch.cat([k1*cos_r - k2*sin_r, k2*cos_r + k1*sin_r], -1)

        k = k.repeat_interleave(gqa, dim=1)
        v = v.repeat_interleave(gqa, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class Student64(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.embed = teacher_model.model.embed_tokens
        self.rotary = teacher_model.model.rotary_emb
        self.final_norm = teacher_model.model.norm
        self.lm_head = teacher_model.lm_head

        for p in self.embed.parameters(): p.requires_grad = False
        for p in self.final_norm.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

        self.attns = nn.ModuleList([StudentAttn() for _ in range(N_LAYERS)])
        self.in_norms = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(N_LAYERS):
            layer = teacher_model.model.layers[i]
            self.in_norms.append(copy.deepcopy(layer.input_layernorm))
            self.post_norms.append(copy.deepcopy(layer.post_attention_layernorm))
            self.mlps.append(layer.mlp)
            for p in layer.mlp.parameters(): p.requires_grad = False

    def forward(self, input_ids):
        h = self.embed(input_ids)
        B, T, _ = h.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary(h, pos)

        for i in range(N_LAYERS):
            r = h
            h = r + self.attns[i](self.in_norms[i](h), cos, sin)
            r = h
            h = r + self.mlps[i](self.post_norms[i](h))

        return self.lm_head(self.final_norm(h))


print("Building student...", flush=True)
student = Student64(teacher).to(torch.bfloat16).to(device)

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.0f}M params (attention + norms)")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training data — longer sequences, more diverse
# ═══════════════════════════════════════════════════════
texts = [
    "The history of mathematics spans thousands of years and includes contributions from many civilizations around the world including ancient Egypt Greece India China and the Islamic world.",
    "Marine biology is the scientific study of organisms that live in the ocean and other saltwater environments covering more than seventy percent of the Earth surface with incredible diversity.",
    "Artificial intelligence has progressed through several distinct phases since its inception in the 1950s moving from symbolic reasoning to statistical learning to deep neural networks.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales where classical physics breaks down and probabilistic descriptions become necessary.",
    "The French Revolution which began in 1789 fundamentally transformed French society by abolishing feudalism establishing civil rights and creating a republic based on popular sovereignty.",
    "Climate change driven primarily by human activities threatens ecosystems worldwide through rising temperatures extreme weather events and sea level rise affecting billions of people.",
    "The human genome project completed in 2003 mapped all human genes providing unprecedented insight into human biology disease mechanisms and potential therapeutic targets.",
    "Machine learning algorithms automatically improve through experience by finding patterns in data without being explicitly programmed for each specific task they encounter.",
    "The Amazon rainforest spanning nine countries in South America contains roughly ten percent of all known species and plays a critical role in regulating global climate patterns.",
    "Modern cryptography relies on mathematical problems that are easy to compute in one direction but extremely difficult to reverse providing security for digital communications worldwide.",
    "Neural networks inspired by biological brains learn hierarchical representations of data through layers of interconnected nodes with adjustable weights trained by gradient descent.",
    "The periodic table first organized by Mendeleev in 1869 arranges chemical elements by atomic number revealing repeating patterns in their physical and chemical properties.",
    "Plate tectonics theory explains how the Earth lithosphere is divided into large plates that move slowly over the asthenosphere causing earthquakes volcanic eruptions and mountain formation.",
    "General relativity published by Einstein in 1915 describes gravity not as a force but as the curvature of spacetime caused by mass and energy distributions.",
    "Evolution by natural selection proposed by Darwin explains how populations of organisms change over generations through variation inheritance and differential reproductive success.",
    "The Internet revolutionized global communication by connecting billions of devices through standardized protocols enabling instant exchange of information across the entire world.",
] * 8

train_ids = []
for text in texts:
    toks = tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=64, padding='max_length').input_ids[0]
    train_ids.append(toks)
train_ids = torch.stack(train_ids).to(device)
print(f"Training: {train_ids.shape}")

# ═══════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=3e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

N_STEPS = 1000
BATCH = 4
TEMP = 2.0

print(f"\nTraining {N_STEPS} steps, batch={BATCH}, T={TEMP}")
print(f"{'Step':>6} {'KL':>8} {'VRAM':>6}")
print("-" * 25)

losses = []
for step in range(N_STEPS):
    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = student(batch)

    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_p = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_p, t_probs, reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if step % 100 == 0 or step == N_STEPS - 1:
        print(f"{step:>6} {loss.item():>8.4f} {torch.cuda.memory_allocated()/1e9:>5.1f}G", flush=True)

# ═══════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

for prompt in ["The future of artificial intelligence will",
               "Water freezes at zero degrees and boils at",
               "The most fundamental concept in physics is"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        t_out = teacher.generate(ids, max_new_tokens=40, do_sample=False)
    t_text = tokenizer.decode(t_out[0][ids.shape[1]:], skip_special_tokens=True)

    with torch.no_grad():
        gen = ids.clone()
        for _ in range(40):
            logits = student(gen)
            gen = torch.cat([gen, logits[0, -1:].argmax(-1).unsqueeze(0)], dim=-1)
    s_text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)

    # Token match
    t_ids = t_out[0][ids.shape[1]:].tolist()
    s_ids = gen[0][ids.shape[1]:].tolist()
    match = sum(1 for a, b in zip(t_ids, s_ids) if a == b)
    pct = match / max(len(t_ids), 1) * 100

    print(f"\n  '{prompt}'")
    print(f"  Teacher: {t_text[:70]}")
    print(f"  Student: {s_text[:70]}")
    print(f"  Match: {match}/{min(len(t_ids),len(s_ids))} = {pct:.0f}%")

# Save
SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save({
    "attn_states": {k: v for k, v in student.state_dict().items()
                    if "attn" in k or "norm" in k},
    "config": {"n_heads": STUDENT_HEADS, "head_dim": STUDENT_HEAD_DIM,
               "n_kv": STUDENT_KV, "n_layers": N_LAYERS, "hidden": H},
    "losses": losses,
}, os.path.join(SAVE_DIR, "student_64head_4b.pt"))
print(f"\nSaved. Final KL: {losses[-1]:.4f}", flush=True)
