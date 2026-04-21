"""Geodesic model trained from 32B teacher on WikiText-103.

32B teacher (64 layers, 5120 hidden) → 13.9M student (ODE, 1024 hidden)
Loss computed in vocab space — hidden dims don't need to match.
WikiText-103: 135M tokens of real text.

24-hour training run.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from datasets import load_dataset
import time
import os

device = "cuda"

print("=" * 70)
print("GEODESIC 32B TEACHER — real data, 24h training")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Use 32B teacher but 0.6B tokenizer/embedding (same tokenizer family)
# Actually both use the same tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

print("Loading 32B teacher...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

T_VRAM = torch.cuda.memory_allocated() / 1e9
print(f"Teacher loaded. VRAM: {T_VRAM:.1f} GB", flush=True)

# Student uses 0.6B embedding/lm_head (same vocab, smaller hidden)
print("Loading 0.6B for student embedding...", flush=True)
small = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = small.config.hidden_size          # 1024
N_HEADS = small.config.num_attention_heads  # 16
N_KV_S = small.config.num_key_value_heads   # 8
INTER = small.config.intermediate_size      # 3072
VOCAB = small.config.vocab_size             # 151936
HEAD_DIM = small.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS

print(f"Student: H={H}, heads={N_HEADS}, hd={HEAD_DIM}, inter={INTER}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Student model (same architecture as before)
# ═══════════════════════════════════════════════════════

class ContextAttention(nn.Module):
    def __init__(self):
        super().__init__()
        q_dim = N_HEADS * HEAD_DIM
        self.q_proj = nn.Linear(H, q_dim, bias=False)
        self.k_proj = nn.Linear(H, HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(H, HEAD_DIM, bias=False)
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


class Geodesic32B(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding and lm_head from 0.6B (same vocab, right hidden size)
        self.embed = small.model.embed_tokens
        self.lm_head = small.lm_head
        for p in self.embed.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

        self.attention = ContextAttention()
        self.dynamics = BulkDynamics()
        self.final_norm = nn.LayerNorm(H)

    def forward_sequence(self, input_ids):
        """Process sequence token by token (for training)."""
        B, T = input_ids.shape
        all_logits = []
        k_cache, v_cache = None, None

        for t in range(T):
            h = self.embed(input_ids[:, t:t+1]).float()
            h_ctx, k_cache, v_cache = self.attention(h, k_cache, v_cache)

            BT, TT, D = h_ctx.shape
            h_flat = h_ctx.reshape(BT * TT, D)
            t_span = torch.tensor([0.0, 1.0], device=device)
            h_traj = odeint(self.dynamics, h_flat, t_span,
                            method='euler', options={'step_size': 0.1})
            h_final = h_traj[-1].reshape(BT, TT, D)

            h_normed = self.final_norm(h_final)
            logits = self.lm_head(h_normed.to(self.lm_head.weight.dtype))
            all_logits.append(logits)

        return torch.cat(all_logits, dim=1)

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
        return self.lm_head(h_normed.to(self.lm_head.weight.dtype)), k_cache, v_cache


print("Building student...", flush=True)
student = Geodesic32B().to(device)
student.dynamics = student.dynamics.float()
student.final_norm = student.final_norm.float()
student.attention = student.attention.float()

# Initialize attention from 0.6B teacher layer 0
src = small.model.layers[0].self_attn
with torch.no_grad():
    student.attention.q_proj.weight.copy_(src.q_proj.weight.float())
    k_full = src.k_proj.weight.view(N_KV_S, HEAD_DIM, H)
    student.attention.k_proj.weight.copy_(k_full[0].reshape(HEAD_DIM, H).float())
    v_full = src.v_proj.weight.view(N_KV_S, HEAD_DIM, H)
    student.attention.v_proj.weight.copy_(v_full[0].reshape(HEAD_DIM, H).float())
    student.attention.o_proj.weight.copy_(src.o_proj.weight.float())
    student.attention.norm.weight.copy_(small.model.layers[0].input_layernorm.weight.float())
    # Dynamics from 0.6B middle layer
    src_mlp = small.model.layers[14].mlp
    student.dynamics.gate_proj.weight.copy_(src_mlp.gate_proj.weight.float())
    student.dynamics.up_proj.weight.copy_(src_mlp.up_proj.weight.float())
    student.dynamics.norm.weight.copy_(small.model.layers[14].post_attention_layernorm.weight.float())

# Free 0.6B model (keep only embed + lm_head which are shared)
del small
torch.cuda.empty_cache()

trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"Student: {trainable/1e6:.1f}M trainable")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Load WikiText-103
# ═══════════════════════════════════════════════════════
print("\nLoading WikiText-103...", flush=True)
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Pre-tokenize into chunks
SEQ_LEN = 64
print(f"Tokenizing into {SEQ_LEN}-token chunks...", flush=True)

all_ids = []
buffer = []
for i in range(len(wiki)):
    text = wiki[i]["text"].strip()
    if len(text) < 20:
        continue
    tokens = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    buffer.extend(tokens)

    while len(buffer) >= SEQ_LEN:
        all_ids.append(buffer[:SEQ_LEN])
        buffer = buffer[SEQ_LEN:]

    if len(all_ids) >= 50000:  # cap at 50K sequences for memory
        break

train_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
print(f"Training data: {train_ids.shape} ({train_ids.shape[0] * SEQ_LEN / 1e6:.1f}M tokens)")
print(f"VRAM after data: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01
)

# Estimate steps for 24h
# 32B forward is slow — maybe 3-5s per step
# 24h * 3600 / 4s = ~21,000 steps
N_STEPS = 20000
BATCH = 1  # 32B is big, batch=1 to fit
TEMP = 2.0
warmup_steps = 300

SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nTraining {N_STEPS} steps, batch={BATCH}, teacher=32B")
print(f"{'Step':>7} {'Loss':>8} {'LR':>10} {'VRAM':>6} {'s/step':>7} {'ETA_h':>6}")
print("-" * 50)

losses = []
t_start = time.time()

for step in range(N_STEPS):
    if step < warmup_steps:
        lr = 1e-4 * (step + 1) / warmup_steps
        for pg in optimizer.param_groups: pg['lr'] = lr

    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    # Teacher logits (32B — the good stuff)
    with torch.no_grad():
        t_logits = teacher(batch).logits  # [B, T, V_teacher]

    # Student logits
    s_logits = student.forward_sequence(batch)  # [B, T, V_student]

    # KL divergence in vocab space
    # Both have same vocab size (151936) — same tokenizer
    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_probs = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_probs[:, :-1], t_probs[:, :-1], reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 200 == 0 or step == N_STEPS - 1:
        elapsed = time.time() - t_start
        spd = elapsed / (step + 1)
        eta = (N_STEPS - step) * spd / 3600
        lr_now = optimizer.param_groups[0]['lr']
        avg_loss = sum(losses[-50:]) / min(len(losses), 50)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>7} {avg_loss:>8.4f} {lr_now:>10.6f} {vram:>5.1f}G {spd:>6.1f}s {eta:>5.1f}h",
              flush=True)

    if (step + 1) % 2000 == 0:
        avg = sum(losses[-500:]) / min(len(losses), 500)
        torch.save({
            "attention_state": student.attention.state_dict(),
            "dynamics_state": student.dynamics.state_dict(),
            "norm_state": student.final_norm.state_dict(),
            "step": step + 1, "loss": avg, "losses": losses[-2000:],
        }, os.path.join(SAVE_DIR, f"geodesic_32b_step{step+1}.pt"))

        # Generation sample
        with torch.no_grad():
            sample_ids = tokenizer("The future of", return_tensors='pt').input_ids.to(device)
            gen = sample_ids.clone()
            kc, vc = None, None
            for t_idx in range(sample_ids.shape[1]):
                logits, kc, vc = student.forward_single(sample_ids[:, t_idx:t_idx+1], kc, vc)
            for _ in range(30):
                nt = logits[0, -1].argmax(-1)
                logits, kc, vc = student.forward_single(nt.view(1, 1), kc, vc)
                gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
            sample = tokenizer.decode(gen[0][sample_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Ckpt {step+1} loss={avg:.4f}: [{sample[:60]}]", flush=True)

elapsed = time.time() - t_start
print(f"\nTotal: {elapsed/3600:.1f}h. Final loss: {losses[-1]:.4f}", flush=True)

# Final generation
print(f"\nFINAL GENERATION:")
for prompt in ["The future of artificial intelligence",
               "Water freezes at zero degrees",
               "The most important discovery in science"]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        gen = ids.clone()
        kc, vc = None, None
        for t_idx in range(ids.shape[1]):
            logits, kc, vc = student.forward_single(ids[:, t_idx:t_idx+1], kc, vc)
        for _ in range(30):
            nt = logits[0, -1].argmax(-1)
            logits, kc, vc = student.forward_single(nt.view(1, 1), kc, vc)
            gen = torch.cat([gen, nt.view(1, 1)], dim=-1)
    text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  '{prompt}' → {text[:60]}")

print("\nDone.", flush=True)
