"""Anneal shape forward: continue fine-tune from offset-1-shaped model with
offsets 1 AND 2 in the loss. Tests whether the shape pressure extends.

Loads qwen_06b_shaped.pt (cos_k 0.91 at offset 1) and continues with N=2.
If cos_k at offset 2 climbs from stock 0.75 toward 0.85+, the shape generalizes.
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
SEQ_LEN = 128
TARGET_LAYER = 14
N_OFFSETS = 2
TRAIN_STEPS = 800
EVAL_EVERY = 25
LR_MODEL = 5e-6
LR_HEAD = 2e-4
LAMBDA_K = 1.0
LAMBDA_V = 0.5
LAMBDA_DEC = 1.0
CKPT_DIR = Path("checkpoints/qwen_06b")
LOAD_PATH = CKPT_DIR / "qwen_06b_shaped.pt"
SAVE_PATH = CKPT_DIR / "qwen_06b_shaped_h2.pt"
RESULTS_PATH = Path("results/pipeline_shape_finetune_06b_h2.json")


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
).to(device)

# Load shaped checkpoint
print(f"Loading shaped checkpoint from {LOAD_PATH}...")
shaped = torch.load(LOAD_PATH, map_location=device)
model.load_state_dict(shaped["model"])

# Freeze embeddings + lm_head
for p in model.parameters():
    p.requires_grad = True
for p in model.model.embed_tokens.parameters():
    p.requires_grad = False
for p in model.lm_head.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)
d_kv = n_kv_heads * head_dim
vocab_size = model.config.vocab_size

# Init KV-Medusa heads — offset 1 from shaped, offset 2 fresh (or warm from stock-trained)
kv_heads = nn.ModuleList()
h1 = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
h1.load_state_dict(shaped["kv_heads"][0])
h1.train()
kv_heads.append(h1)

h2 = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
# Try warm-starting from stock-trained offset 2 head if exists
stock_h2 = CKPT_DIR / "kv_medusa_head_2.pt"
if stock_h2.exists():
    h2.load_state_dict(torch.load(stock_h2, map_location=device))
    print(f"  Offset 2 head warm-started from stock-trained checkpoint")
else:
    print(f"  Offset 2 head: fresh init")
h2.train()
kv_heads.append(h2)

# Single shared decoder
decoder = KDecoder(d_kv, d_model).to(device).to(torch.float32)
decoder.load_state_dict(shaped["decoder"])
decoder.train()

opt = torch.optim.AdamW([
    {"params": model.parameters(), "lr": LR_MODEL, "weight_decay": 0.0},
    {"params": kv_heads.parameters(), "lr": LR_HEAD, "weight_decay": 0.01},
    {"params": decoder.parameters(), "lr": LR_HEAD, "weight_decay": 0.01},
])

print(f"\n{'='*60}\nANNEAL SHAPE → offset 2 (loss includes offsets 1+2)\n{'='*60}\n")

print("Loading tokens...")
train_tokens = load_owt(tokenizer, SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

step = 0
history = []

for batch in epoch_iter(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    out = model(inp, output_hidden_states=True, use_cache=True, return_dict=True)
    h_final = out.hidden_states[-1]
    actual_k = out.past_key_values.layers[TARGET_LAYER].keys
    actual_v = out.past_key_values.layers[TARGET_LAYER].values
    logits_lm = out.logits

    targets_lm = inp[:, 1:]
    loss_lm = F.cross_entropy(logits_lm[:, :-1, :].reshape(-1, vocab_size).float(),
                              targets_lm.reshape(-1))

    loss_K_total = 0.0; loss_V_total = 0.0; loss_dec_total = 0.0
    cos_k_log, cos_v_log, dec_acc_log = [], [], []

    for offset_idx in range(N_OFFSETS):
        offset = offset_idx + 1
        h_in = h_final[:, :-offset].float()
        target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).float()
        target_v = actual_v[:, :, offset:].permute(0, 2, 1, 3).float()
        target_toks = inp[:, offset:]

        ml = min(h_in.shape[1], target_k.shape[1], target_toks.shape[1])
        h_in = h_in[:, :ml]
        target_k = target_k[:, :ml]; target_v = target_v[:, :ml]; target_toks = target_toks[:, :ml]

        pred_k, pred_v = kv_heads[offset_idx](h_in)
        cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean()
        cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim), target_v.reshape(-1, head_dim), dim=-1).mean()
        loss_K_total = loss_K_total + (1 - cos_k)
        loss_V_total = loss_V_total + (1 - cos_v)

        pred_k_flat = pred_k.reshape(1, ml, d_kv)
        dec_logits = decoder(pred_k_flat, model.lm_head.weight.detach())
        loss_dec = F.cross_entropy(dec_logits.reshape(-1, vocab_size), target_toks.reshape(-1))
        loss_dec_total = loss_dec_total + loss_dec

        with torch.no_grad():
            cos_k_log.append(cos_k.item())
            cos_v_log.append(cos_v.item())
            preds = dec_logits.argmax(-1)
            dec_acc_log.append((preds == target_toks).float().mean().item())

    loss = loss_lm + LAMBDA_K * loss_K_total + LAMBDA_V * loss_V_total + LAMBDA_DEC * loss_dec_total

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(kv_heads.parameters()) + list(decoder.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        print(f"  step {step:>4}: lm={loss_lm.item():.3f} "
              f"K={loss_K_total.item():.3f} V={loss_V_total.item():.3f} dec={loss_dec_total.item():.3f}")
        print(f"    cos_k: t+1={cos_k_log[0]:.3f}  t+2={cos_k_log[1]:.3f}")
        print(f"    cos_v: t+1={cos_v_log[0]:.3f}  t+2={cos_v_log[1]:.3f}")
        print(f"    dec_acc: t+1={dec_acc_log[0]:.3f}  t+2={dec_acc_log[1]:.3f}", flush=True)
        history.append({"step": step,
                        "lm_loss": round(loss_lm.item(), 4),
                        "cos_k": [round(c, 4) for c in cos_k_log],
                        "cos_v": [round(c, 4) for c in cos_v_log],
                        "dec_acc": [round(a, 4) for a in dec_acc_log]})

print("\nSaving...")
torch.save({"model": model.state_dict(),
            "kv_heads": [h.state_dict() for h in kv_heads],
            "decoder": decoder.state_dict()}, SAVE_PATH)

# Final val
model.eval(); [h.eval() for h in kv_heads]; decoder.eval()
val_cos_k = [[] for _ in range(N_OFFSETS)]
val_cos_v = [[] for _ in range(N_OFFSETS)]
val_dec_acc = [[] for _ in range(N_OFFSETS)]
val_count = 0
for vbatch in epoch_iter(val_tokens, SEQ_LEN, device):
    if val_count >= 20: break
    vinp = vbatch[:, :SEQ_LEN]
    with torch.no_grad():
        out = model(vinp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()
        actual_v = out.past_key_values.layers[TARGET_LAYER].values.float()
        for offset_idx in range(N_OFFSETS):
            offset = offset_idx + 1
            h_in = h_final[:, :-offset]
            tk = actual_k[:, :, offset:].permute(0, 2, 1, 3)
            tv = actual_v[:, :, offset:].permute(0, 2, 1, 3)
            tt = vinp[:, offset:]
            ml = min(h_in.shape[1], tk.shape[1], tt.shape[1])
            pk, pv = kv_heads[offset_idx](h_in[:, :ml])
            val_cos_k[offset_idx].append(F.cosine_similarity(pk.reshape(-1, head_dim), tk[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            val_cos_v[offset_idx].append(F.cosine_similarity(pv.reshape(-1, head_dim), tv[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            dec_logits = decoder(pk.reshape(1, ml, d_kv), model.lm_head.weight.detach())
            val_dec_acc[offset_idx].append((dec_logits.argmax(-1) == tt[:, :ml]).float().mean().item())
    val_count += 1

print(f"\n{'='*60}\nFINAL VAL\n{'='*60}")
print(f"  {'offset':<8}{'cos_k':<10}{'cos_v':<10}{'dec_acc':<10}")
final = []
for k in range(N_OFFSETS):
    ck = sum(val_cos_k[k]) / len(val_cos_k[k])
    cv = sum(val_cos_v[k]) / len(val_cos_v[k])
    da = sum(val_dec_acc[k]) / len(val_dec_acc[k])
    print(f"  t+{k+1:<6}{ck:<10.3f}{cv:<10.3f}{da:<10.3f}")
    final.append({"offset": k+1, "cos_k": round(ck, 4), "cos_v": round(cv, 4), "dec_acc": round(da, 4)})

print(f"\n  Stock model reference:    cos_k 0.77 / 0.75; cos_v 0.41 / 0.34")
print(f"  Shaped (offset 1 only):   cos_k 0.91 / —;    cos_v 0.86 / —")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_offsets": N_OFFSETS, "train_steps": TRAIN_STEPS,
               "load_path": str(LOAD_PATH), "save_path": str(SAVE_PATH),
               "final": final, "history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
