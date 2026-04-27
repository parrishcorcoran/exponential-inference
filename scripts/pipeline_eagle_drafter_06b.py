"""Train an Eagle-style drafter on Qwen3-0.6B.

Architecture: 1 transformer-style block (causal self-attention + FFN, RMSNorm).
Input at position p: concat(h_{p-1}, embed(tok_p)) -> Linear -> d_model.
Output: predicted h_p (matching the main model's final-norm hidden state at p).

Training loss:
  - MSE(predicted h_p, true h_p)
  - + lambda * cross_entropy(LM_head(predicted h_p), tok_{p+1})

The MSE keeps the drafter on the hidden-state manifold; the CE pushes the
drafter toward producing hiddens that decode to the right next token via the
*shared LM head*. This is the EAGLE-1 trick: borrow the LM head's quality.

Inference (in a separate test script): autoregressive — feed predicted h back in.
"""
import math
import json
import gc
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


def empty_cache():
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()


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


def iter_batches(tokens, seq_len, batch_size, device):
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < seq_len + 1: continue
        batch.append(window)
        if len(batch) == batch_size:
            yield torch.tensor(batch, dtype=torch.long, device=device)
            batch = []


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
    """Single transformer block that maps (prev_hidden, current_token_embed) -> predicted next hidden."""
    def __init__(self, d_model, n_heads, head_dim, ffn_mult=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Project (h, embed) [2*d_model] -> d_model
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
        # h_prev: [B, S, d], tok_embeds: [B, S, d]
        x = self.fc_in(torch.cat([h_prev, tok_embeds], dim=-1))
        # Causal self-attention
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
        # SwiGLU FFN
        x_norm2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(x_norm2)) * self.up(x_norm2))
        return x


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
STEPS = 1500  # more than Medusa heads since this is a richer model
EVAL_EVERY = 100
LR = 5e-4
LAMBDA_CE = 0.1
CKPT_DIR = Path("checkpoints/qwen_06b")
CKPT_PATH = CKPT_DIR / "eagle_drafter.pt"
RESULTS_PATH = Path("results/pipeline_eagle_drafter_06b.json")

print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"Loading {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
vocab_size = model.config.vocab_size
embed_layer = model.model.embed_tokens
lm_head_weight = model.lm_head.weight.detach()

print(f"  d_model={d_model}, n_attn_heads={n_attn_heads}, head_dim={head_dim}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

drafter = EagleDrafter(d_model, n_heads=n_attn_heads, head_dim=head_dim).to(device).to(torch.float32)
drafter_params = sum(p.numel() for p in drafter.parameters())
print(f"  Drafter params: {drafter_params/1e6:.1f}M")

opt = torch.optim.AdamW(drafter.parameters(), lr=LR, weight_decay=0.01)
drafter.train()
step = 0
history = []

print("\nTraining...", flush=True)
for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
    if step >= STEPS:
        break

    with torch.no_grad():
        out = model(batch, use_cache=False, output_hidden_states=True)
        h_all = out.hidden_states[-1].float()  # [1, S, d] — final norm output
        tok_embeds_all = embed_layer(batch).detach().float()  # [1, S, d]

    # Drafter input at position p: h_{p-1} and embed(tok_p)
    # Output target: h_p (predict h_p given h_{p-1} and tok_p)
    # Use slicing: h_prev = h_all[:, :-1], tok_embed = tok_embeds_all[:, 1:]  (token at pos p, hidden at p-1)
    # But we want position-aligned: input tuple (h_{p-1}, embed(tok_p)) predicts h_p
    # So at index p in our input arrays, we put (h_{p-1}, embed(tok_p)) and target h_p.
    # Equivalent: for arrays h_seq[p] = h_{p-1}, tok_seq[p] = embed(tok_p), target_h[p] = h_p
    # Using indices 1..S-1: h_prev = h_all[:, :-1] (h_0..h_{S-2}), tok_in = tok_embeds_all[:, 1:] (embed_1..embed_{S-1})
    # target = h_all[:, 1:] (h_1..h_{S-1})
    # And to predict NEXT token via LM head: tok_target_for_ce = batch[:, 2:] (tok_2..tok_{S-1}) — but only S-2 positions
    h_prev = h_all[:, :-1]                         # [1, S-1, d]
    tok_in = tok_embeds_all[:, 1:]                 # [1, S-1, d] (embed of tok_p, p=1..S-1)
    target_h = h_all[:, 1:]                        # [1, S-1, d] (true h_p for p=1..S-1)

    pred_h = drafter(h_prev, tok_in)               # [1, S-1, d]

    loss_mse = F.mse_loss(pred_h, target_h)

    # CE loss: LM_head(pred_h[p]) should predict tok_{p+1}, i.e., batch[p+1] in original indexing
    # pred_h is [1, S-1, d] for predicted h at positions 1..S-1.
    # LM_head(pred_h[t]) predicts token at position t+1 in original indexing.
    # In pred_h indexing: pred_h[i] corresponds to original position i+1, so it predicts token at i+2.
    # So next_tok_targets = batch[2:] (original positions 2..S-1), aligning with pred_h[:S-2]
    pred_logits = F.linear(pred_h[:, :-1].to(lm_head_weight.dtype), lm_head_weight)  # [1, S-2, vocab]
    next_tok_targets = batch[:, 2:]                                                   # [1, S-2]
    loss_ce = F.cross_entropy(pred_logits.float().reshape(-1, vocab_size), next_tok_targets.reshape(-1))

    loss = loss_mse + LAMBDA_CE * loss_ce

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = pred_logits.argmax(-1)
            acc = (preds == next_tok_targets).float().mean().item()
            cos = F.cosine_similarity(pred_h.reshape(-1, d_model), target_h.reshape(-1, d_model), dim=-1).mean().item()
        print(f"  step {step:>4}: loss={loss.item():.4f} mse={loss_mse.item():.4f} ce={loss_ce.item():.4f} cos_h={cos:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({
            "step": step, "loss": round(loss.item(), 4),
            "mse": round(loss_mse.item(), 4),
            "ce": round(loss_ce.item(), 4),
            "cos_h": round(cos, 4),
            "tok_acc": round(acc, 4),
        })

# Final val eval
drafter.eval()
val_accs, val_cos = [], []
val_count = 0
for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
    if val_count >= 20: break
    with torch.no_grad():
        out = model(vbatch, use_cache=False, output_hidden_states=True)
        h_all = out.hidden_states[-1].float()
        tok_embeds_all = embed_layer(vbatch).detach().float()
        h_prev = h_all[:, :-1]
        tok_in = tok_embeds_all[:, 1:]
        target_h = h_all[:, 1:]
        pred_h = drafter(h_prev, tok_in)
        cos = F.cosine_similarity(pred_h.reshape(-1, d_model), target_h.reshape(-1, d_model), dim=-1).mean().item()
        val_cos.append(cos)
        pred_logits = F.linear(pred_h[:, :-1].to(lm_head_weight.dtype), lm_head_weight)
        next_tok_targets = vbatch[:, 2:]
        acc = (pred_logits.argmax(-1) == next_tok_targets).float().mean().item()
        val_accs.append(acc)
    val_count += 1

final_acc = sum(val_accs) / len(val_accs)
final_cos = sum(val_cos) / len(val_cos)
print(f"\n  EAGLE DRAFTER FINAL: val_tok_acc={final_acc:.3f} val_cos_h={final_cos:.3f}", flush=True)

torch.save(drafter.state_dict(), CKPT_PATH)

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT, "device": device,
        "drafter_params_M": round(drafter_params / 1e6, 2),
        "steps": STEPS,
        "val_tok_acc": round(final_acc, 4),
        "val_cos_h": round(final_cos, 4),
        "history": history,
    }, f, indent=2)
print(f"\nSaved {CKPT_PATH} and {RESULTS_PATH}")

del model; gc.collect(); empty_cache()
