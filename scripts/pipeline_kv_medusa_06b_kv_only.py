"""K+V joint decoder — adds V to the K-only decoder, no Q.

Single offset (1). Single KV head, single decoder reading [K_pred ; V_pred].
Joint loss: MSE(K) + MSE(V) + CE(decoder([K;V]) -> token).

Reference: K-only joint focused was top-1 0.21 / top-5 0.46.
"""
import json
from pathlib import Path
import random

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


class KVDecoder(nn.Module):
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.proj = nn.Linear(2 * d_kv, d_model, bias=False)
    def forward(self, K_flat, V_flat, lm_head_weight):
        x = torch.cat([K_flat, V_flat], dim=-1)
        h = self.proj(x)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
TRAIN_STEPS = 1500
EVAL_EVERY = 100
LR = 2e-4
N_EVAL_SEQS = 20
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_kv_only.json")


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)
d_kv = n_kv_heads * head_dim
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

print(f"  d_model={d_model}, d_kv={d_kv}")
print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

# Warm start KV head
print("Init KV head from kv_medusa_head_joint_one_1.pt (warm start)...")
kv_head = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
kv_head.load_state_dict(torch.load(CKPT_DIR / "kv_medusa_head_joint_one_1.pt", map_location=device))
kv_head.train()

decoder = KVDecoder(d_kv, d_model).to(device).to(torch.float32)
decoder.train()

opt = torch.optim.AdamW([
    {"params": kv_head.parameters(), "lr": LR},
    {"params": decoder.parameters(), "lr": LR},
], weight_decay=0.01)

print(f"\n{'='*60}")
print(f"K + V decoder — offset {OFFSET}, {TRAIN_STEPS} steps")
print(f"{'='*60}\n")

step = 0
history = []
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    with torch.no_grad():
        out = model(inp, use_cache=True, output_hidden_states=True)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()

    h_in = h_final[:, :-OFFSET]
    target_k = actual_k[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_v = actual_v[:, :, OFFSET:].permute(0, 2, 1, 3).float()
    target_toks = inp[:, OFFSET:]

    ml = min(h_in.shape[1], target_k.shape[1], target_toks.shape[1])
    h_in, target_k, target_v, target_toks = h_in[:, :ml], target_k[:, :ml], target_v[:, :ml], target_toks[:, :ml]

    pred_k, pred_v = kv_head(h_in)
    loss_mse = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)

    pred_k_flat = pred_k.reshape(1, ml, d_kv)
    pred_v_flat = pred_v.reshape(1, ml, d_kv)
    logits = decoder(pred_k_flat, pred_v_flat, lm_head_weight)
    loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))

    loss = loss_mse + loss_ce
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(kv_head.parameters()) + list(decoder.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
            cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean().item()
            cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim), target_v.reshape(-1, head_dim), dim=-1).mean().item()
        print(f"  step {step:>4}: loss={loss.item():.3f} mse={loss_mse.item():.3f} ce={loss_ce.item():.3f} "
              f"cos_k={cos_k:.3f} cos_v={cos_v:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({"step": step, "mse": round(loss_mse.item(), 4),
                        "ce": round(loss_ce.item(), 4), "cos_k": round(cos_k, 4),
                        "cos_v": round(cos_v, 4), "acc": round(acc, 4)})

torch.save(kv_head.state_dict(), CKPT_DIR / f"kv_head_kv_only_{OFFSET}.pt")
torch.save(decoder.state_dict(), CKPT_DIR / "decoder_kv_only.pt")

# ─── Eval ─────────────────────────────────────────────────────────────────
kv_head.eval(); decoder.eval()
match_top1 = match_top5 = total = 0

for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        with torch.no_grad():
            pk, pv = kv_head(h_t)
            pk_flat = pk.reshape(1, 1, d_kv)
            pv_flat = pv.reshape(1, 1, d_kv)
            logits = decoder(pk_flat, pv_flat, lm_head_weight)
            top1 = logits.argmax(-1).item()
            top5 = set(logits.topk(5, dim=-1).indices[0, 0].tolist())
        true_tok = baseline_toks[t + OFFSET].item()
        if top1 == true_tok: match_top1 += 1
        if true_tok in top5: match_top5 += 1
        total += 1

t1 = match_top1 / total
t5 = match_top5 / total

print(f"\n{'='*70}")
print(f"K+V DECODER — offset {OFFSET}, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  top-1: {t1:.3f}   top-5: {t5:.3f}   (n={total})")
print(f"  Reference: K-only joint focused (1×1) — top-1 0.21 / top-5 0.46")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER, "offset": OFFSET,
               "top1": round(t1, 4), "top5": round(t5, 4), "n": total,
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
