"""Q-only decoder ceiling test.

Captures Q at layer 14 (post q_norm, post RoPE) via a forward hook, trains
a small decoder Q -> token. Reports the real-Q ceiling for comparison with
K's 0.56 ceiling.

This is the analog of v1's K-decoder ceiling test. No Q-prediction yet —
just "how much token information does Q at this position carry?"

If real-Q > real-K, Q is the more informative stream.
If real-Q ≈ real-K, the streams are likely complementary and K+V+Q wins.
"""
import json
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


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


def iter_batches(tokens, seq_len, device):
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < seq_len + 1: continue
        yield torch.tensor([window], dtype=torch.long, device=device)


class QDecoder(nn.Module):
    def __init__(self, d_q, d_model):
        super().__init__()
        self.proj = nn.Linear(d_q, d_model, bias=False)
    def forward(self, Q_flat, lm_head_weight):
        h = self.proj(Q_flat)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
TRAIN_STEPS = 1000
EVAL_EVERY = 100
LR = 5e-4
N_VAL_BATCHES = 20
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_q_only.json")


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
d_q = n_attn_heads * head_dim
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

print(f"  d_model={d_model}, n_attn_heads={n_attn_heads}, head_dim={head_dim}, d_q={d_q}")

# ─── Hook layer-14 to capture Q ─────────────────────────────────────────────
attn_layer = model.model.layers[TARGET_LAYER].self_attn
captured_Q = {}
orig_forward = attn_layer.forward

def capturing_forward(hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    self = attn_layer
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    captured_Q["q"] = query_states.detach()
    return orig_forward(hidden_states, position_embeddings, attention_mask,
                        past_key_values, cache_position, **kwargs)

attn_layer.forward = capturing_forward

print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

decoder = QDecoder(d_q, d_model).to(device).to(torch.float32)
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.01)

print(f"\nTrain Q-decoder on real Q at layer {TARGET_LAYER}...")
decoder.train()
step = 0
history = []
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]
    captured_Q.clear()
    with torch.no_grad():
        _ = model(inp, use_cache=True, output_hidden_states=False)
        Q = captured_Q["q"].float()  # [1, n_attn, S, hd]
        S = Q.shape[2]
        Q_flat = Q.permute(0, 2, 1, 3).reshape(1, S, d_q)
    target_toks = inp

    logits = decoder(Q_flat, lm_head_weight)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
        print(f"  step {step:>4}: loss={loss.item():.3f} train_acc={acc:.3f}", flush=True)
        history.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})

# Val
decoder.eval()
correct = 0; total = 0
top5_correct = 0
val_count = 0
for vbatch in iter_batches(val_tokens, SEQ_LEN, device):
    if val_count >= N_VAL_BATCHES: break
    vinp = vbatch[:, :SEQ_LEN]
    captured_Q.clear()
    with torch.no_grad():
        _ = model(vinp, use_cache=True, output_hidden_states=False)
        Q = captured_Q["q"].float()
        Sv = Q.shape[2]
        Q_flat = Q.permute(0, 2, 1, 3).reshape(1, Sv, d_q)
        logits = decoder(Q_flat, lm_head_weight)
        preds = logits.argmax(-1)
        top5 = logits.topk(5, dim=-1).indices  # [1, S, 5]
    correct += (preds == vinp).float().sum().item()
    top5_correct += (top5 == vinp.unsqueeze(-1)).any(-1).float().sum().item()
    total += vinp.numel()
    val_count += 1

attn_layer.forward = orig_forward

t1 = correct / total
t5 = top5_correct / total

print(f"\n{'='*70}")
print(f"Q-DECODER CEILING — Qwen3-0.6B layer {TARGET_LAYER}")
print(f"{'='*70}")
print(f"  Real-Q top-1 token acc: {t1:.3f}")
print(f"  Real-Q top-5 token acc: {t5:.3f}")
print(f"  Reference: real-K top-1 = 0.560 (v1 K-decoder)")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "real_Q_top1": round(t1, 4), "real_Q_top5": round(t5, 4),
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
