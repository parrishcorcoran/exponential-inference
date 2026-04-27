"""Quick test: 1 KV head + 1 Q head + 1 decoder reading (K, V, Q) at offset 1.

Captures the layer-14 Q (post q_norm, post RoPE) via a forward hook on
Qwen3Attention. Trains a Q-Medusa head with MSE against captured Q. The
decoder takes [K_pred ; V_pred ; Q_pred] concatenated -> token.

Joint loss = MSE(K) + MSE(V) + MSE(Q) + CE(decoder([K;V;Q]) -> token).

If Q carries information complementary to K and V at the same position, the
decoder's top-1 / top-5 should jump above the K-only joint focused result
(0.21 / 0.46).
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


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads; self.head_dim = head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


class QMedusaHead(nn.Module):
    """Predict Q (post-norm, post-RoPE) at layer 14 at offset+1."""
    def __init__(self, d_model, n_attn_heads, head_dim):
        super().__init__()
        self.n_attn_heads = n_attn_heads; self.head_dim = head_dim
        self.q_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, n_attn_heads * head_dim, bias=False),
        )
    def forward(self, h):
        q = self.q_pred(h).view(h.shape[0], h.shape[1], self.n_attn_heads, self.head_dim)
        return q


class KVQDecoder(nn.Module):
    """Reads (K, V, Q) flat-concatenated, projects to d_model, LM head."""
    def __init__(self, d_kv, d_q, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv + d_kv + d_q, d_model, bias=False)
    def forward(self, K_flat, V_flat, Q_flat, lm_head_weight):
        x = torch.cat([K_flat, V_flat, Q_flat], dim=-1)
        h = self.proj(x)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
TRAIN_STEPS = 1000
EVAL_EVERY = 100
LR = 2e-4
N_EVAL_SEQS = 20
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_kvq.json")


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
d_kv = n_kv_heads * head_dim
d_q = n_attn_heads * head_dim
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

print(f"  d_model={d_model}, n_attn_heads={n_attn_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
print(f"  d_kv={d_kv}, d_q={d_q}")

# ─── Hook layer-14 attention to capture Q (post q_norm, post RoPE) ─────────
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
    captured_Q["q"] = query_states.detach()  # [bsz, n_attn, S, head_dim]
    return orig_forward(hidden_states, position_embeddings, attention_mask,
                        past_key_values, cache_position, **kwargs)

attn_layer.forward = capturing_forward

# ─── Heads + decoder ───────────────────────────────────────────────────────
print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

# Warm start KV head from joint-trained head 1
print("Init KV head from kv_medusa_head_joint_one_1.pt (warm start)...")
kv_head = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
kv_head.load_state_dict(torch.load(CKPT_DIR / "kv_medusa_head_joint_one_1.pt", map_location=device))
kv_head.train()

q_head = QMedusaHead(d_model, n_attn_heads, head_dim).to(device).to(torch.float32)
q_head.train()

decoder = KVQDecoder(d_kv, d_q, d_model).to(device).to(torch.float32)
decoder.train()

opt = torch.optim.AdamW([
    {"params": kv_head.parameters(), "lr": LR},
    {"params": q_head.parameters(), "lr": LR},
    {"params": decoder.parameters(), "lr": LR},
], weight_decay=0.01)

print(f"\n{'='*60}")
print(f"K + V + Q decoder — offset {OFFSET}, {TRAIN_STEPS} steps")
print(f"{'='*60}\n")

step = 0
history = []
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    captured_Q.clear()
    with torch.no_grad():
        out = model(inp, use_cache=True, output_hidden_states=True)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()  # [1, n_kv, S, hd]
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()
        actual_q = captured_Q["q"].float()  # [1, n_attn, S, hd]

    h_in = h_final[:, :-OFFSET]  # [1, S-1, d]
    target_k = actual_k[:, :, OFFSET:].permute(0, 2, 1, 3).float()  # [1, S-1, n_kv, hd]
    target_v = actual_v[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_q = actual_q[:, :, OFFSET:].permute(0, 2, 1, 3).float()  # [1, S-1, n_attn, hd]
    target_toks = inp[:, OFFSET:]

    ml = min(h_in.shape[1], target_k.shape[1], target_q.shape[1], target_toks.shape[1])
    h_in, target_k, target_v, target_q, target_toks = (
        h_in[:, :ml], target_k[:, :ml], target_v[:, :ml], target_q[:, :ml], target_toks[:, :ml])

    pred_k, pred_v = kv_head(h_in)
    pred_q = q_head(h_in)

    loss_mse = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v) + F.mse_loss(pred_q, target_q)

    pred_k_flat = pred_k.reshape(1, ml, d_kv)
    pred_v_flat = pred_v.reshape(1, ml, d_kv)
    pred_q_flat = pred_q.reshape(1, ml, d_q)
    logits = decoder(pred_k_flat, pred_v_flat, pred_q_flat, lm_head_weight)
    loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))

    loss = loss_mse + loss_ce
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(kv_head.parameters()) + list(q_head.parameters()) + list(decoder.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
            cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean().item()
            cos_q = F.cosine_similarity(pred_q.reshape(-1, head_dim), target_q.reshape(-1, head_dim), dim=-1).mean().item()
        print(f"  step {step:>4}: loss={loss.item():.3f} mse={loss_mse.item():.3f} ce={loss_ce.item():.3f} "
              f"cos_k={cos_k:.3f} cos_q={cos_q:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({"step": step, "mse": round(loss_mse.item(), 4),
                        "ce": round(loss_ce.item(), 4), "cos_k": round(cos_k, 4),
                        "cos_q": round(cos_q, 4), "acc": round(acc, 4)})

# Save
torch.save(kv_head.state_dict(), CKPT_DIR / f"kv_head_kvq_{OFFSET}.pt")
torch.save(q_head.state_dict(), CKPT_DIR / f"q_head_kvq_{OFFSET}.pt")
torch.save(decoder.state_dict(), CKPT_DIR / "decoder_kvq.pt")

# ─── Eval ─────────────────────────────────────────────────────────────────
kv_head.eval(); q_head.eval(); decoder.eval()
match_top1 = match_top5 = total = 0

for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    captured_Q.clear()
    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        with torch.no_grad():
            pk, pv = kv_head(h_t)
            pq = q_head(h_t)
            pk_flat = pk.reshape(1, 1, d_kv)
            pv_flat = pv.reshape(1, 1, d_kv)
            pq_flat = pq.reshape(1, 1, d_q)
            logits = decoder(pk_flat, pv_flat, pq_flat, lm_head_weight)
            top1 = logits.argmax(-1).item()
            top5 = set(logits.topk(5, dim=-1).indices[0, 0].tolist())
        true_tok = baseline_toks[t + OFFSET].item()
        if top1 == true_tok: match_top1 += 1
        if true_tok in top5: match_top5 += 1
        total += 1

# Restore forward
attn_layer.forward = orig_forward

t1 = match_top1 / total
t5 = match_top5 / total

print(f"\n{'='*70}")
print(f"K+V+Q DECODER — offset {OFFSET}, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  top-1: {t1:.3f}   top-5: {t5:.3f}   (n={total})")
print(f"  Reference: K-only joint focused (1×1) — top-1 0.21 / top-5 0.46")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER, "offset": OFFSET,
               "top1": round(t1, 4), "top5": round(t5, 4), "n": total,
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
