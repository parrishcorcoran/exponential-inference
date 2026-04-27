"""Joint training: Eagle + Medusa heads + learned combiner.

All three components train under one CE loss against the model's greedy
argmax at offsets 1..K. The combiner mixes Eagle (autoregressive) and
Medusa (parallel) per-offset logits.

This is the integrated drafter you sketched — Eagle and Medusa as parts
of a single fused score, all aligned to produce the Jacobi fixed point
(= greedy decode) directly.
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
N_OFFSETS = 5
TRAIN_STEPS = 1500
EVAL_EVERY = 100
LR = 2e-4
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_combined_drafter_train.json")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(self.weight.dtype)


class EagleDrafter(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_mult=4):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.head_dim = head_dim
        self.fc_in = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        ffn_dim = ffn_mult * d_model
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)
    def forward(self, h_prev, tok_embeds):
        x = self.fc_in(torch.cat([h_prev, tok_embeds], dim=-1))
        B, S, _ = x.shape
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device, dtype=scores.dtype), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ V).transpose(1, 2).reshape(B, S, -1)
        x = x + self.o_proj(attn_out)
        x_norm2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(x_norm2)) * self.up(x_norm2))
        return x


class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])
    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


class CombinedDrafter(nn.Module):
    def __init__(self, eagle, medusa_heads, n_offsets):
        super().__init__()
        self.eagle = eagle
        self.medusa = medusa_heads
        self.n_offsets = n_offsets
        # Learned mix per offset (logit, sigmoid → α)
        self.alpha_logits = nn.Parameter(torch.zeros(n_offsets))

    def forward(self, h_t, tok_t, embed_layer, lm_head_weight):
        """Returns combined logits [B, n_offsets, vocab]."""
        # Eagle: autoregressive K-step
        h_curr, tok_curr = h_t, tok_t
        eagle_logits = []
        for k in range(self.n_offsets):
            te = embed_layer(tok_curr).detach().float()
            pred_h = self.eagle(h_curr, te)
            log = F.linear(pred_h.to(lm_head_weight.dtype), lm_head_weight).float()
            eagle_logits.append(log)
            h_curr = pred_h
            tok_curr = log.argmax(-1)
        eagle_logits = torch.cat(eagle_logits, dim=1)  # [B, n_offsets, vocab]

        # Medusa: parallel
        medusa_logits = []
        for head in self.medusa:
            log = head(h_t, lm_head_weight).float()
            medusa_logits.append(log)
        medusa_logits = torch.cat(medusa_logits, dim=1)  # [B, n_offsets, vocab]

        # Combine: per-offset learned mix
        alpha = torch.sigmoid(self.alpha_logits).view(1, self.n_offsets, 1)
        combined = alpha * eagle_logits + (1 - alpha) * medusa_logits
        return combined, eagle_logits, medusa_logits


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def epoch_iter(tokens, seq_len, device):
    while True:
        n = (len(tokens) - 1) // seq_len
        idx = list(range(n)); random.shuffle(idx)
        for i in idx:
            start = i * seq_len
            window = tokens[start:start + seq_len + 1]
            if len(window) < seq_len + 1: continue
            yield torch.tensor([window], dtype=torch.long, device=device)


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
vocab_size = model.config.vocab_size
embed_layer = model.model.embed_tokens
lm_head_weight = model.lm_head.weight.detach()

# Load components — warm starts
print("Loading Eagle drafter...")
eagle = EagleDrafter(d_model, n_attn_heads, head_dim).to(device).to(torch.float32)
eagle.load_state_dict(torch.load(CKPT_DIR / "eagle_drafter.pt", map_location=device))
eagle.train()

print(f"Loading {N_OFFSETS} Medusa heads...")
medusa_heads = nn.ModuleList()
for k in range(1, N_OFFSETS + 1):
    h = MedusaHead(d_model, n_layers=1).to(device).to(torch.float32)
    h.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h.train()
    medusa_heads.append(h)

drafter = CombinedDrafter(eagle, medusa_heads, N_OFFSETS).to(device).to(torch.float32)
total_params = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
print(f"  Total trainable params: {total_params/1e6:.2f}M")

opt = torch.optim.AdamW(drafter.parameters(), lr=LR, weight_decay=0.01)

print("Loading tokens...")
train_tokens = load_owt(tokenizer, SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"\n{'='*60}\nJOINT TRAINING — Eagle + Medusa + combiner\n{'='*60}\n")
step = 0
history = []

for batch in epoch_iter(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=False)
        h_final = out.hidden_states[-1].float()  # [1, S, d]
        targets = out.logits.argmax(-1)  # [1, S] — model's greedy at each position

    # Pick random anchor positions; train at each
    n_anchors = 4
    anchors = random.sample(range(20, SEQ_LEN - N_OFFSETS - 5), n_anchors)
    losses = []
    eagle_acc_per_off = [0.0] * N_OFFSETS
    medusa_acc_per_off = [0.0] * N_OFFSETS
    combined_acc_per_off = [0.0] * N_OFFSETS

    for t in anchors:
        h_t = h_final[:, t:t+1]
        tok_t = inp[:, t:t+1]

        combined, eagle_log, medusa_log = drafter(h_t, tok_t, embed_layer, lm_head_weight)
        # Targets: model's greedy at positions t+1..t+N_OFFSETS
        # baseline_top1[t+k-1] = model's prediction at position t+k-1, predicting token at t+k
        # Hmm, careful: targets[t+k-1] is the model's argmax of logit at position t+k-1, which predicts token at t+k.
        # So target for offset k from h_t is targets[0, t+k-1].
        target_offsets = targets[0, t:t+N_OFFSETS].unsqueeze(0)  # [1, n_offsets]

        loss = F.cross_entropy(combined.reshape(-1, vocab_size), target_offsets.reshape(-1))
        losses.append(loss)

        with torch.no_grad():
            eagle_pred = eagle_log.argmax(-1)
            medusa_pred = medusa_log.argmax(-1)
            combined_pred = combined.argmax(-1)
            for k in range(N_OFFSETS):
                eagle_acc_per_off[k] += (eagle_pred[0, k] == target_offsets[0, k]).float().item() / n_anchors
                medusa_acc_per_off[k] += (medusa_pred[0, k] == target_offsets[0, k]).float().item() / n_anchors
                combined_acc_per_off[k] += (combined_pred[0, k] == target_offsets[0, k]).float().item() / n_anchors

    total_loss = sum(losses) / len(losses)
    opt.zero_grad(); total_loss.backward()
    torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        alphas = torch.sigmoid(drafter.alpha_logits).tolist()
        print(f"  step {step:>4}: loss={total_loss.item():.3f}", flush=True)
        print(f"    eagle acc:    " + " ".join(f"{a:.2f}" for a in eagle_acc_per_off))
        print(f"    medusa acc:   " + " ".join(f"{a:.2f}" for a in medusa_acc_per_off))
        print(f"    combined acc: " + " ".join(f"{a:.2f}" for a in combined_acc_per_off))
        print(f"    α (eagle wt): " + " ".join(f"{a:.2f}" for a in alphas))
        history.append({"step": step, "loss": round(total_loss.item(), 4),
                        "eagle_acc": [round(a, 4) for a in eagle_acc_per_off],
                        "medusa_acc": [round(a, 4) for a in medusa_acc_per_off],
                        "combined_acc": [round(a, 4) for a in combined_acc_per_off],
                        "alpha": [round(a, 4) for a in alphas]})

torch.save({"eagle": drafter.eagle.state_dict(),
            "medusa": [h.state_dict() for h in drafter.medusa],
            "alpha_logits": drafter.alpha_logits.detach()},
           CKPT_DIR / "combined_drafter.pt")
print(f"\nSaved combined drafter")

# Final eval
drafter.eval()
e_acc, m_acc, c_acc = [0]*N_OFFSETS, [0]*N_OFFSETS, [0]*N_OFFSETS
n_eval = 0
ANCHORS = [40, 80, 120, 160, 200]
for seq_idx, vbatch in enumerate(epoch_iter(val_tokens, SEQ_LEN, device)):
    if seq_idx >= 10: break
    vinp = vbatch[:, :SEQ_LEN]
    with torch.no_grad():
        out = model(vinp, output_hidden_states=True, use_cache=False)
        h_final = out.hidden_states[-1].float()
        targets = out.logits.argmax(-1)
    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        tok_t = vinp[:, t:t+1]
        with torch.no_grad():
            combined, eagle_log, medusa_log = drafter(h_t, tok_t, embed_layer, lm_head_weight)
            target_off = targets[0, t:t+N_OFFSETS]
            for k in range(N_OFFSETS):
                e_acc[k] += (eagle_log[0, k].argmax() == target_off[k]).item()
                m_acc[k] += (medusa_log[0, k].argmax() == target_off[k]).item()
                c_acc[k] += (combined[0, k].argmax() == target_off[k]).item()
        n_eval += 1

print(f"\n{'='*60}\nFINAL VAL — n_anchors={n_eval}\n{'='*60}")
print(f"  {'offset':<8}{'eagle':<10}{'medusa':<10}{'combined':<10}")
for k in range(N_OFFSETS):
    print(f"  t+{k+1:<6}{e_acc[k]/n_eval:<10.3f}{m_acc[k]/n_eval:<10.3f}{c_acc[k]/n_eval:<10.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_offsets": N_OFFSETS, "train_steps": TRAIN_STEPS,
               "history": history,
               "final_eagle_acc": [e/n_eval for e in e_acc],
               "final_medusa_acc": [m/n_eval for m in m_acc],
               "final_combined_acc": [c/n_eval for c in c_acc]}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
