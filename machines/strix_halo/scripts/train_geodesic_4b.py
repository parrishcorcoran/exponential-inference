"""Geodesic model — 4B teacher — fast iteration — proper KV heads.

Fixes from 32B run:
- 8 KV heads (not 1 — need proper context resolution)
- 32 Q heads (spatial coverage)
- 4B teacher for fast iteration (~5s/step vs 37s)
- WikiText-103 real data
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
print("GEODESIC 0.6B — fastest iteration, prove it can copy teacher")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

print("Loading 0.6B teacher...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

VOCAB = teacher.config.vocab_size
print(f"Teacher: H={teacher.config.hidden_size}, L={teacher.config.num_hidden_layers}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Student — same heads/KV as teacher, but MORE bulk for the hologram
# Heads/KV match teacher exactly. Intermediate is larger for proof of concept.
H_S = teacher.config.hidden_size                    # 1024
INTER_S = 24576                                      # 8x teacher — real bulk
N_HEADS_S = teacher.config.num_attention_heads      # 16
N_KV_S = teacher.config.num_key_value_heads         # 8
HEAD_DIM_S = teacher.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS_S  # 128
GQA_S = N_HEADS_S // N_KV_S                        # 2

print(f"Student: H={H_S}, inter={INTER_S}, Q={N_HEADS_S}, KV={N_KV_S}, hd={HEAD_DIM_S}")
print(f"Same heads/KV as teacher. Bulk={INTER_S} (8x teacher, proof of concept)")


class ContextAttention(nn.Module):
    """EXACT same dimensions as teacher's attention layer."""
    def __init__(self):
        super().__init__()
        # Match teacher exactly: Q_DIM=2048, KV_DIM=1024, H=1024
        Q_DIM = N_HEADS_S * HEAD_DIM_S   # 16*128 = 2048
        KV_DIM = N_KV_S * HEAD_DIM_S     # 8*128 = 1024

        self.q_proj = nn.Linear(H_S, Q_DIM, bias=False)
        self.k_proj = nn.Linear(H_S, KV_DIM, bias=False)
        self.v_proj = nn.Linear(H_S, KV_DIM, bias=False)
        self.o_proj = nn.Linear(Q_DIM, H_S, bias=False)
        self.norm = nn.LayerNorm(H_S)

    def forward(self, h, k_cache=None, v_cache=None):
        B, T, D = h.shape
        h_norm = self.norm(h)
        q = self.q_proj(h_norm).view(B, T, N_HEADS_S, HEAD_DIM_S).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, N_KV_S, HEAD_DIM_S).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, N_KV_S, HEAD_DIM_S).transpose(1, 2)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # GQA expand
        k_exp = k.repeat_interleave(GQA_S, dim=1)
        v_exp = v.repeat_interleave(GQA_S, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return h + self.o_proj(attn_out), k, v


class BulkDynamics(nn.Module):
    """SwiGLU MLP — the holographic projection medium."""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(H_S)
        self.gate_proj = nn.Linear(H_S, INTER_S, bias=False)
        self.up_proj = nn.Linear(H_S, INTER_S, bias=False)
        self.down_proj = nn.Linear(INTER_S, H_S, bias=False)
        self.time_mix = nn.Linear(1, H_S, bias=False)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, t, h):
        h = h.float()
        t_emb = self.time_mix(t.float().reshape(1, 1)).expand(h.shape[0], -1)
        h_in = self.norm(h + t_emb)
        return self.down_proj(F.silu(self.gate_proj(h_in)) * self.up_proj(h_in))


class GeodesicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H_S)
        self.lm_head = nn.Linear(H_S, VOCAB, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

        self.attention = ContextAttention()
        self.dynamics = BulkDynamics()
        self.final_norm = nn.LayerNorm(H_S)

    def forward_sequence(self, input_ids):
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
            logits = self.lm_head(self.final_norm(h_final))
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
        return self.lm_head(self.final_norm(h_final)), k_cache, v_cache


print("Building student...", flush=True)
student = GeodesicModel().to(device).float()

total = sum(p.numel() for p in student.parameters())
trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.0f}M, Trainable: {trainable/1e6:.0f}M")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# WikiText-103
print("\nLoading WikiText-103...", flush=True)
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

SEQ_LEN = 64
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
    if len(all_ids) >= 50000:
        break

train_ids = torch.tensor(all_ids, dtype=torch.long, device=device)
print(f"Training: {train_ids.shape} ({train_ids.shape[0]*SEQ_LEN/1e6:.1f}M tokens)")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Training
optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=0.01)

N_STEPS = 15000
BATCH = 2
TEMP = 2.0
warmup = 500

SAVE_DIR = "/home/cpinchington/Exponential-Inference/machines/strix_halo/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nTraining {N_STEPS} steps, batch={BATCH}, 4B teacher, 8 KV heads")
print(f"{'Step':>7} {'Loss':>8} {'LR':>10} {'VRAM':>6} {'s/step':>7} {'ETA_h':>6}")
print("-" * 50)

losses = []
t_start = time.time()

for step in range(N_STEPS):
    if step < warmup:
        lr = 3e-4 * (step + 1) / warmup
        for pg in optimizer.param_groups: pg['lr'] = lr

    idx = torch.randint(0, len(train_ids), (BATCH,))
    batch = train_ids[idx]

    with torch.no_grad():
        t_logits = teacher(batch).logits

    s_logits = student.forward_sequence(batch)

    t_probs = F.softmax(t_logits.float() / TEMP, dim=-1)
    s_log_probs = F.log_softmax(s_logits.float() / TEMP, dim=-1)
    loss = F.kl_div(s_log_probs[:, :-1], t_probs[:, :-1], reduction='batchmean') * (TEMP ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())

    if step % 100 == 0 or step == N_STEPS - 1:
        elapsed = time.time() - t_start
        spd = elapsed / (step + 1)
        eta = (N_STEPS - step) * spd / 3600
        lr_now = optimizer.param_groups[0]['lr']
        avg = sum(losses[-50:]) / min(len(losses), 50)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"{step:>7} {avg:>8.2f} {lr_now:>10.6f} {vram:>5.1f}G {spd:>6.1f}s {eta:>5.1f}h",
              flush=True)

    if (step + 1) % 1000 == 0:
        avg = sum(losses[-200:]) / min(len(losses), 200)
        torch.save({
            "model_state": student.state_dict(),
            "step": step + 1, "loss": avg,
        }, os.path.join(SAVE_DIR, f"geodesic_4b_step{step+1}.pt"))

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
        print(f"  Ckpt {step+1}: [{sample[:60]}]", flush=True)

elapsed = time.time() - t_start
print(f"\nTotal: {elapsed/3600:.1f}h. Final loss: {losses[-1]:.4f}", flush=True)
