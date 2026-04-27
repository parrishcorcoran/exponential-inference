"""Focused joint training: 1 KV-Medusa head (offset 1) + 1 K-decoder.

The previous joint training trained 10 heads with random-offset rotation. Each
head got ~10% of updates and the shared decoder got pulled across 10 different
K-distributions — too noisy to converge.

This script: just offset 1, just one head, just one decoder. Full 2000 steps
of dedicated joint MSE+CE training. The clean question: can the loss push the
head's K predictions into the token-decodable subspace at all?

If yes -> expand to per-offset heads later.
If no  -> the K-manifold's token-discriminative structure is in a different
         direction than h_t can predict, and we know to stop chasing this path.
"""
import json
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


class KDecoder(nn.Module):
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv, d_model, bias=False)
    def forward(self, K_flat, lm_head_weight):
        h = self.proj(K_flat)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
TRAIN_STEPS = 2000
EVAL_EVERY = 100
LR_HEAD = 2e-4
LR_DECODER = 2e-4
ALPHA_MSE = 1.0
BETA_CE = 1.0
N_EVAL_SEQS = 20
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_joint_one.json")


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

print(f"  d_model={d_model}, d_kv={d_kv}, target_layer={TARGET_LAYER}, offset={OFFSET}")
print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

# Warm start
print(f"Loading KV-Medusa head_{OFFSET} (warm start)...")
head = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
head.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{OFFSET}.pt", map_location=device))
head.train()

print("Loading v1 K-decoder (warm start)...")
decoder = KDecoder(d_kv, d_model).to(device).to(torch.float32)
decoder.load_state_dict(torch.load(CKPT_DIR / "k_decoder.pt", map_location=device))
decoder.train()

opt = torch.optim.AdamW([
    {"params": head.parameters(), "lr": LR_HEAD},
    {"params": decoder.parameters(), "lr": LR_DECODER},
], weight_decay=0.01)

print(f"\n{'='*60}")
print(f"FOCUSED JOINT — 1 head (offset {OFFSET}) + 1 decoder")
print(f"  loss = {ALPHA_MSE} * MSE + {BETA_CE} * CE  for {TRAIN_STEPS} steps")
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

    pred_k, pred_v = head(h_in)
    loss_mse = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)

    pred_k_flat = pred_k.reshape(1, ml, d_kv)
    logits = decoder(pred_k_flat, lm_head_weight)
    loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))
    loss = ALPHA_MSE * loss_mse + BETA_CE * loss_ce

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(list(head.parameters()) + list(decoder.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
            cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim),
                                        target_k.reshape(-1, head_dim), dim=-1).mean().item()
        print(f"  step {step:>4}: loss={loss.item():.3f} mse={loss_mse.item():.3f} ce={loss_ce.item():.3f} "
              f"cos_k={cos_k:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({"step": step,
                        "mse": round(loss_mse.item(), 4),
                        "ce": round(loss_ce.item(), 4),
                        "cos_k": round(cos_k, 4),
                        "acc": round(acc, 4)})

# Save
torch.save(head.state_dict(), CKPT_DIR / f"kv_medusa_head_joint_one_{OFFSET}.pt")
torch.save(decoder.state_dict(), CKPT_DIR / "k_decoder_joint_one.pt")

# ─── Eval at offset 1 ──────────────────────────────────────────────────────
print(f"\n{'='*60}\nEVAL — offset {OFFSET}\n{'='*60}")
head.eval(); decoder.eval()

match_top1 = 0
match_top5 = 0
total = 0
cos_list = []

n_done = 0
for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + OFFSET >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]
        with torch.no_grad():
            pk, _ = head(h_t)
            K_flat_pred = pk.reshape(1, 1, d_kv)
            logits = decoder(K_flat_pred, lm_head_weight)
            top1 = logits.argmax(-1).item()
            top5 = set(logits.topk(5, dim=-1).indices[0, 0].tolist())
            ak = actual_k[:, :, t + OFFSET, :]
            cos = F.cosine_similarity(pk[0, 0].reshape(-1, head_dim),
                                      ak[0].reshape(-1, head_dim), dim=-1).mean().item()
            cos_list.append(cos)

        true_tok = baseline_toks[t + OFFSET].item()
        if top1 == true_tok: match_top1 += 1
        if true_tok in top5: match_top5 += 1
        total += 1
    n_done += 1

# Real-K ceiling
real_correct = 0; real_n = 0
val_count = 0
for vbatch in iter_batches(val_tokens, SEQ_LEN, device):
    if val_count >= 10: break
    vinp = vbatch[:, :SEQ_LEN]
    with torch.no_grad():
        out = model(vinp, use_cache=True, output_hidden_states=False)
        K_real = out.past_key_values.layers[TARGET_LAYER].keys.float()
        Sv = K_real.shape[2]
        K_flat = K_real.permute(0, 2, 1, 3).reshape(1, Sv, d_kv)
        logits = decoder(K_flat, lm_head_weight)
        preds = logits.argmax(-1)
        real_correct += (preds == vinp).float().sum().item()
        real_n += vinp.numel()
    val_count += 1
real_acc = real_correct / real_n

mean_cos = sum(cos_list) / len(cos_list)
top1_rate = match_top1 / total
top5_rate = match_top5 / total

print(f"\n  Real-K ceiling (post-train decoder): {real_acc:.3f}")
print(f"  Predicted-K cos_k:                    {mean_cos:.3f}")
print(f"  Predicted-K top-1 token acc:          {top1_rate:.3f}")
print(f"  Predicted-K top-5 token acc:          {top5_rate:.3f}")
print(f"  Reference: standard Medusa head 1:    0.318")
print(f"  Reference: v1 decoder on predicted K (offset 1): 0.120")
print(f"  Reference: v2 noise-aug:                          0.180")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER, "offset": OFFSET,
               "real_K_ceiling": round(real_acc, 4),
               "predicted_cos_k": round(mean_cos, 4),
               "top1": round(top1_rate, 4),
               "top5": round(top5_rate, 4),
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
