"""HRR unbinding test: decode tokens directly from K-cache.

Hypothesis (Plate-style HRR): the layer-14 K vector at position p is the
*unbinding key* for the bound state at that position. After 13 transformer
layers process the input embedding of tok_p, K_proj produces a 1024-dim
address whose direction encodes which content was just bound — i.e., which
token is at this position.

If the K vector preserves enough token signature, a learned linear projection
K -> d_model -> LM_head should recover tok_p with high accuracy.

Then at inference we use our 10 trained KV-Medusa heads to predict K at
positions t+1..t+10 from h_t, decode each predicted K via the classifier,
and read off 10 future tokens — *with no extra forward passes*.

Pipeline:
  1. Train a KDecoder (Linear 1024 -> 1024) on (real_K_at_p, tok_p) pairs,
     using the frozen LM head as the final readout.
  2. Eval on real K's: ceiling.
  3. Eval on KV-Medusa predicted K's at offsets 1..10: real per-offset accuracy.
"""
import math
import json
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
    import random
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
    """Maps K (post-norm, post-RoPE, layer 14) -> token logits via frozen LM head."""
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.proj = nn.Linear(d_kv, d_model, bias=False)

    def forward(self, K_flat, lm_head_weight):
        # K_flat: [..., d_kv]
        h = self.proj(K_flat)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
N_OFFSETS = 10
TRAIN_STEPS = 1000
EVAL_EVERY = 100
LR = 5e-4
N_EVAL_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
CKPT_PATH = CKPT_DIR / "k_decoder.pt"
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_unbind.json")


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

# ─── 1. Train KDecoder on real K -> token pairs ────────────────────────────
print("\n" + "="*60)
print("STAGE 1: Train K-decoder on real K's")
print("="*60)

decoder = KDecoder(d_kv, d_model).to(device).to(torch.float32)
print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.01)

decoder.train()
step = 0
history = []
for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break

    with torch.no_grad():
        inp = batch[:, :SEQ_LEN]  # [1, S]
        out = model(inp, use_cache=True, output_hidden_states=False)
        K_real = out.past_key_values.layers[TARGET_LAYER].keys.float()  # [1, n_kv, S, head_dim]
        S = K_real.shape[2]
        K_flat = K_real.permute(0, 2, 1, 3).reshape(1, S, d_kv)  # [1, S, d_kv]
    target_toks = inp  # [1, S]

    logits = decoder(K_flat, lm_head_weight)  # [1, S, vocab]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks.reshape(-1))

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks).float().mean().item()
        print(f"  step {step:>4}: loss={loss.item():.4f} train_acc={acc:.3f}", flush=True)
        history.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})

# Eval ceiling: K-decoder on REAL K's
decoder.eval()
real_acc_acc = 0; real_n = 0
val_count = 0
for vbatch in iter_batches(val_tokens, SEQ_LEN, device):
    if val_count >= 20: break
    with torch.no_grad():
        vinp = vbatch[:, :SEQ_LEN]
        out = model(vinp, use_cache=True, output_hidden_states=False)
        K_real = out.past_key_values.layers[TARGET_LAYER].keys.float()
        Sv = K_real.shape[2]
        K_flat = K_real.permute(0, 2, 1, 3).reshape(1, Sv, d_kv)
        logits = decoder(K_flat, lm_head_weight)
        preds = logits.argmax(-1)
        real_acc_acc += (preds == vinp).float().sum().item()
        real_n += vinp.numel()
    val_count += 1

real_K_acc = real_acc_acc / real_n
print(f"\n  CEILING — K-decoder on REAL layer-14 K's:  top-1 token acc = {real_K_acc:.3f}")

torch.save(decoder.state_dict(), CKPT_PATH)

# ─── 2. Apply K-decoder to PREDICTED K's from KV-Medusa heads ──────────────
print("\n" + "="*60)
print("STAGE 2: K-decoder on KV-Medusa PREDICTED K's (offsets 1..10)")
print("="*60)

print("Loading KV-Medusa heads 1..10...")
kv_heads = []
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.eval(); kv_heads.append(h)

per_offset_match = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_total = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=True)
        h_final = out.hidden_states[-1].float()
        # Baseline tokens: input token at each position (the "input" we want to recover)
        baseline_toks = inp[0]  # [seq]

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]  # [1, 1, d]

        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                pk, pv = kv_heads[k - 1](h_t)  # [1, 1, n_kv, head_dim]
                K_flat_pred = pk.reshape(1, 1, d_kv)
                logits = decoder(K_flat_pred, lm_head_weight)  # [1, 1, vocab]
                pred_tok = logits.argmax(-1).item()

            true_tok = baseline_toks[t + k].item()
            if pred_tok == true_tok:
                per_offset_match[k] += 1
            per_offset_total[k] += 1
    n_done += 1
    print(f"  seq {n_done}/{N_EVAL_SEQS} done", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HRR-UNBINDING TOKEN EXTRACTION — Qwen3-0.6B")
print(f"{'='*70}")
print(f"  CEILING (real K's):  top-1 acc = {real_K_acc:.3f}")
print(f"\n  {'offset':<8}{'tok acc (predicted K)':<25}{'std-Medusa for ref':<22}")
std_medusa = {1: 0.318, 2: 0.052, 3: 0.025, 4: 0.026, 5: 0.019}

results = []
for k in range(1, N_OFFSETS + 1):
    acc = per_offset_match[k] / per_offset_total[k] if per_offset_total[k] else 0
    ref = std_medusa.get(k, "—")
    print(f"  t+{k:<6}{acc:<25.3f}{ref:<22}")
    results.append({"offset": k, "n": per_offset_total[k],
                    "tok_acc_predicted_K": round(acc, 4)})

def chain(rates):
    out = 1.0; prod = 1.0
    for r in rates:
        prod *= r; out += prod
    return out

ch = chain([r["tok_acc_predicted_K"] for r in results])
print(f"\n  Chained tokens/step (1 + a1 + a1*a2 + ...): {ch:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "real_K_top1_acc": round(real_K_acc, 4),
               "results": results,
               "chained_tokens_per_step": round(ch, 4),
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
