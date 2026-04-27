"""HRR unbinding v2: train K-decoder on PREDICTED K's (matched noise distribution).

The v1 decoder had 56% accuracy on real K but collapsed to 0-12% on KV-Medusa
predicted K's (cos~0.75). It learned a sharp partition that doesn't tolerate
the angular perturbation it sees at inference.

Fix: instead of training on real K -> token, train on **predicted K -> true
token**, where the predicted K comes from running our existing 10 KV-Medusa
heads on h_t. This forces the decoder to learn a noise-robust readout — each
token's basin of attraction in K-space gets wide enough to swallow the real
KV-Medusa prediction noise (the snap-to-value mechanism Plate/Vaswani
exploit via FFT correlation / softmax).

At each training step we randomly pick an offset k in [1, 10], compute
predicted K via head_k(h_{t}) for valid positions, and use the actual token
at the corresponding future position as the target.
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


class KDecoderRobust(nn.Module):
    """Wider decoder with nonlinearity — more capacity to learn noise-robust basins."""
    def __init__(self, d_kv, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_kv, d_model * 2, bias=False)
        self.fc2 = nn.Linear(d_model * 2, d_model, bias=False)
    def forward(self, K_flat, lm_head_weight):
        h = F.silu(self.fc1(K_flat))
        h = self.fc2(h)
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight).float()


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
N_OFFSETS = 10
TRAIN_STEPS = 1500
EVAL_EVERY = 150
LR = 5e-4
N_EVAL_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
CKPT_DIR = Path("checkpoints/qwen_06b")
CKPT_PATH = CKPT_DIR / "k_decoder_v2.pt"
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_unbind_v2.json")


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

print(f"Loading {N_OFFSETS} KV-Medusa heads (frozen)...")
kv_heads = []
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.eval()
    for p in h.parameters():
        p.requires_grad = False
    kv_heads.append(h)

# ─── Train robust decoder on predicted K's ─────────────────────────────────
print("\n" + "="*60)
print("STAGE 1: Train robust K-decoder on PREDICTED K's at random offsets")
print("="*60)

decoder = KDecoderRobust(d_kv, d_model).to(device).to(torch.float32)
print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.01)

decoder.train()
step = 0
history = []

for batch in iter_batches(train_tokens, SEQ_LEN, device):
    if step >= TRAIN_STEPS: break
    inp = batch[:, :SEQ_LEN]

    with torch.no_grad():
        out = model(inp, use_cache=False, output_hidden_states=True)
        h_final = out.hidden_states[-1].float()  # [1, S, d]

    # Pick a random offset and use the corresponding head
    offset = random.randint(1, N_OFFSETS)
    head = kv_heads[offset - 1]

    with torch.no_grad():
        h_in = h_final[:, :-offset]  # [1, S-offset, d]
        pred_k, _ = head(h_in)  # [1, S-offset, n_kv, head_dim]
    pred_k_flat = pred_k.reshape(1, -1, d_kv)

    target_toks = inp[:, offset:]  # [1, S-offset]
    ml = min(pred_k_flat.shape[1], target_toks.shape[1])

    logits = decoder(pred_k_flat[:, :ml], lm_head_weight)  # [1, ml, vocab]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_toks[:, :ml].reshape(-1))

    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step()
    step += 1

    if step % EVAL_EVERY == 0:
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == target_toks[:, :ml]).float().mean().item()
        print(f"  step {step:>4}: offset={offset} loss={loss.item():.4f} train_acc={acc:.3f}", flush=True)
        history.append({"step": step, "offset": offset, "loss": round(loss.item(), 4),
                        "acc": round(acc, 4)})

torch.save(decoder.state_dict(), CKPT_PATH)
print(f"\nSaved {CKPT_PATH}")

# ─── Eval per-offset on PREDICTED K's ──────────────────────────────────────
print("\n" + "="*60)
print("STAGE 2: Per-offset eval — robust decoder on predicted K's")
print("="*60)

decoder.eval()
per_offset_match = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_total = {k: 0 for k in range(1, N_OFFSETS + 1)}
per_offset_top5 = {k: 0 for k in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_EVAL_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=False)
        h_final = out.hidden_states[-1].float()
        baseline_toks = inp[0]

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]

        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                pk, _ = kv_heads[k - 1](h_t)
                K_flat_pred = pk.reshape(1, 1, d_kv)
                logits = decoder(K_flat_pred, lm_head_weight)  # [1, 1, vocab]
                top1 = logits.argmax(-1).item()
                top5_set = set(logits.topk(5, dim=-1).indices[0, 0].tolist())

            true_tok = baseline_toks[t + k].item()
            if top1 == true_tok:
                per_offset_match[k] += 1
            if true_tok in top5_set:
                per_offset_top5[k] += 1
            per_offset_total[k] += 1
    n_done += 1
    print(f"  seq {n_done}/{N_EVAL_SEQS} done", flush=True)

# ─── Also eval on real K's for ceiling reference ───────────────────────────
print("\nMeasuring ceiling on REAL K's for reference...")
real_correct = 0; real_n = 0
val_count = 0
for vbatch in iter_batches(val_tokens, SEQ_LEN, device):
    if val_count >= 20: break
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
real_K_acc = real_correct / real_n

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HRR-UNBINDING v2 — robust K-decoder, Qwen3-0.6B")
print(f"{'='*70}")
print(f"  Decoder trained on PREDICTED K (matched noise)")
print(f"  Real-K ceiling (this decoder): top-1 = {real_K_acc:.3f}")
print(f"\n  {'offset':<8}{'top-1':<10}{'top-5':<10}{'std-Medusa for ref'}")
std_medusa = {1: 0.318, 2: 0.052, 3: 0.025, 4: 0.026, 5: 0.019}

results = []
for k in range(1, N_OFFSETS + 1):
    n = per_offset_total[k]
    a = per_offset_match[k] / n if n else 0
    a5 = per_offset_top5[k] / n if n else 0
    ref = std_medusa.get(k, "—")
    print(f"  t+{k:<6}{a:<10.3f}{a5:<10.3f}{ref}")
    results.append({"offset": k, "n": n, "top1": round(a, 4), "top5": round(a5, 4)})

def chain(rates):
    out = 1.0; prod = 1.0
    for r in rates:
        prod *= r; out += prod
    return out

ch1 = chain([r["top1"] for r in results])
ch5 = chain([r["top5"] for r in results])
print(f"\n  Chained tokens/step (top-1): {ch1:.3f}")
print(f"  Chained tokens/step (top-5): {ch5:.3f}  (would be the tree-Medusa K=5 ceiling)")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "real_K_top1_ceiling": round(real_K_acc, 4),
               "results": results,
               "chained_top1": round(ch1, 4),
               "chained_top5": round(ch5, 4),
               "training_history": history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
