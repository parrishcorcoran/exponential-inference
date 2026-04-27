"""Joint training: 10 KV-Medusa heads + K-decoder with MSE + CE loss.

The diagnosis from v2: predicted K (cos ~0.75 to real K) had 56% token-decode
info available in *real K* but only 18% recoverable in *predicted K*. That gap
is because the KV-Medusa heads were trained with MSE only — they preserve
attention-pattern geometry but discard the token-discriminative directions of
the K-manifold.

Fix: joint loss = α * MSE(K_pred, K_real) + β * CE(decoder(K_pred), token_real).
The CE term pulls K predictions into the subspace where decoder(K) → correct
token. The MSE term keeps them on the attention manifold.

Warm start: existing 10 KV-Medusa heads (cos~0.75) and v1 K-decoder (56% on
real K). Both fine-tune jointly.
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
    """v1 architecture: single linear projection to d_model + frozen LM head."""
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv, d_model, bias=False)
    def forward(self, K_flat, lm_head_weight):
        h = self.proj(K_flat)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
N_OFFSETS = 10
TRAIN_STEPS = 2000
EVAL_EVERY = 200
LR_HEAD = 2e-4
LR_DECODER = 1e-4
ALPHA_MSE = 1.0
BETA_CE = 1.0
N_EVAL_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_joint.json")


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

print(f"  d_model={d_model}, d_kv={d_kv}, target_layer={TARGET_LAYER}")
print("Loading tokens...")
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

# Warm start
print(f"Loading {N_OFFSETS} KV-Medusa heads (warm start)...")
kv_heads = []
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.train()  # trainable
    kv_heads.append(h)

print("Loading v1 K-decoder (warm start)...")
decoder = KDecoder(d_kv, d_model).to(device).to(torch.float32)
decoder.load_state_dict(torch.load(CKPT_DIR / "k_decoder.pt", map_location=device))
decoder.train()

# Optimizer: heads get higher LR than decoder
head_params = [p for h in kv_heads for p in h.parameters()]
opt = torch.optim.AdamW([
    {"params": head_params, "lr": LR_HEAD, "weight_decay": 0.01},
    {"params": list(decoder.parameters()), "lr": LR_DECODER, "weight_decay": 0.01},
])

print(f"\n{'='*60}")
print(f"JOINT TRAIN: heads (LR {LR_HEAD}) + decoder (LR {LR_DECODER})")
print(f"  loss = {ALPHA_MSE} * MSE + {BETA_CE} * CE   for {TRAIN_STEPS} steps")
print(f"{'='*60}\n")

step = 0
history = []
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    with torch.no_grad():
        out = model(inp, use_cache=True, output_hidden_states=True)
        h_final = out.hidden_states[-1].float()  # [1, S, d]
        actual_k = out.past_key_values.layers[TARGET_LAYER].keys.float()  # [1, n_kv, S, head_dim]

    # Random offset per step
    offset = random.randint(1, N_OFFSETS)
    head = kv_heads[offset - 1]

    h_in = h_final[:, :-offset]  # [1, S-off, d]
    target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).float()  # [1, S-off, n_kv, head_dim]
    target_v = out.past_key_values.layers[TARGET_LAYER].values[:, :, offset:].permute(0, 2, 1, 3).float()
    target_toks = inp[:, offset:]

    ml = min(h_in.shape[1], target_k.shape[1], target_toks.shape[1])
    h_in = h_in[:, :ml]
    target_k = target_k[:, :ml]
    target_v = target_v[:, :ml]
    target_toks = target_toks[:, :ml]

    pred_k, pred_v = head(h_in)  # [1, ml, n_kv, head_dim]
    loss_mse = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)

    # K-decoder applied to predicted K
    pred_k_flat = pred_k.reshape(1, ml, d_kv)
    logits = decoder(pred_k_flat, lm_head_weight)  # [1, ml, vocab]
    loss_ce = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))

    loss = ALPHA_MSE * loss_mse + BETA_CE * loss_ce

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(head_params + list(decoder.parameters()), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
            cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim),
                                        target_k.reshape(-1, head_dim), dim=-1).mean().item()
        print(f"  step {step:>4}: offset={offset:>2} loss={loss.item():.3f} "
              f"mse={loss_mse.item():.3f} ce={loss_ce.item():.3f} "
              f"cos_k={cos_k:.3f} tok_acc={acc:.3f}", flush=True)
        history.append({"step": step, "offset": offset,
                        "mse": round(loss_mse.item(), 4),
                        "ce": round(loss_ce.item(), 4),
                        "cos_k": round(cos_k, 4),
                        "acc": round(acc, 4)})

# Save
for k in range(1, N_OFFSETS + 1):
    torch.save(kv_heads[k - 1].state_dict(), CKPT_DIR / f"kv_medusa_head_joint_{k}.pt")
torch.save(decoder.state_dict(), CKPT_DIR / "k_decoder_joint.pt")

# ─── Per-offset eval ───────────────────────────────────────────────────────
print(f"\n{'='*60}\nPER-OFFSET EVAL — joint-trained heads + decoder\n{'='*60}")
for h in kv_heads: h.eval()
decoder.eval()

per_offset_match = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_total = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_top5 = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_cos = {k: [] for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        actual_k_full = out.past_key_values.layers[TARGET_LAYER].keys.float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]

        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                pk, _ = kv_heads[k - 1](h_t)  # [1, 1, n_kv, head_dim]
                K_flat_pred = pk.reshape(1, 1, d_kv)
                logits = decoder(K_flat_pred, lm_head_weight)
                top1 = logits.argmax(-1).item()
                top5_set = set(logits.topk(5, dim=-1).indices[0, 0].tolist())

                # cos_k vs real
                ak = actual_k_full[:, :, t + k, :].permute(0, 1, 2)  # [1, n_kv, head_dim]
                cos = F.cosine_similarity(pk[0, 0].reshape(-1, head_dim),
                                          ak[0].reshape(-1, head_dim), dim=-1).mean().item()
                per_offset_cos[k].append(cos)

            true_tok = baseline_toks[t + k].item()
            if top1 == true_tok: per_offset_match[k] += 1
            if true_tok in top5_set: per_offset_top5[k] += 1
            per_offset_total[k] += 1

    n_done += 1
    print(f"  seq {n_done}/{N_EVAL_SEQS} done", flush=True)

# Real-K decoder ceiling
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

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HRR-UNBINDING JOINT — joint-trained heads + decoder, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  Real-K decoder ceiling (post-joint-train): top-1 = {real_acc:.3f}")
print(f"\n  {'offset':<8}{'cos_k':<10}{'top-1':<10}{'top-5':<10}")

results = []
for k in range(1, N_OFFSETS + 1):
    n = per_offset_total[k]
    a1 = per_offset_match[k] / n if n else 0
    a5 = per_offset_top5[k] / n if n else 0
    cos_avg = sum(per_offset_cos[k]) / len(per_offset_cos[k]) if per_offset_cos[k] else 0
    print(f"  t+{k:<6}{cos_avg:<10.3f}{a1:<10.3f}{a5:<10.3f}")
    results.append({"offset": k, "n": n, "cos_k": round(cos_avg, 4),
                    "top1": round(a1, 4), "top5": round(a5, 4)})

def chain(rates):
    out = 1.0; prod = 1.0
    for r in rates:
        prod *= r; out += prod
    return out

ch1 = chain([r["top1"] for r in results])
ch5 = chain([r["top5"] for r in results])
print(f"\n  Chained tokens/step (top-1): {ch1:.3f}")
print(f"  Chained tokens/step (top-5): {ch5:.3f}")
print(f"  vs v1 (decoder on predicted K): top-1 chain ~ 1.12")
print(f"  vs v2 (noise-aug):              top-1 chain ~ 1.19")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "real_K_top1_post_joint": round(real_acc, 4),
               "alpha_mse": ALPHA_MSE, "beta_ce": BETA_CE,
               "results": results,
               "chained_top1": round(ch1, 4),
               "chained_top5": round(ch5, 4),
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
