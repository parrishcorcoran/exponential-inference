"""Extend KV-Medusa to 30+ heads, then stack regular Medusa on top.

Part 1: Train KV-Medusa heads t+11 through t+30 (we have t+1 to t+10)
  → Does acceptance stay at 99% at longer offsets?
  → Can we draft 30 tokens per step?

Part 2: Train regular Medusa heads on the KV-Medusa-enabled model
  → Does KV-Medusa improve regular Medusa acceptance?
  → How many useful regular Medusa heads can we get?
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda"
CHECKPOINT = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
STEPS_PER_HEAD = 300  # fast — they plateau quickly
LR = 5e-4

print("=" * 60)
print("KV-MEDUSA EXTEND TO 30 HEADS + REGULAR MEDUSA ON TOP")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

# Tokens
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
toks = []
for item in ds:
    t = item.get("text", "")
    if not t.strip(): continue
    toks.extend(tokenizer.encode(t, add_special_tokens=False))
    if len(toks) >= SEQ_LEN * 3000: break
train_tokens = toks[:SEQ_LEN * 2500]
val_tokens = toks[SEQ_LEN * 2500:]

def iter_batches(tokens, seq_len, device, n=999):
    import random
    idxs = list(range((len(tokens) - 1) // seq_len))
    random.shuffle(idxs)
    for i in idxs[:n]:
        s = i * seq_len
        w = tokens[s:s + seq_len + 1]
        if len(w) < seq_len + 1: continue
        yield torch.tensor([w], dtype=torch.long, device=device)

print("\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters(): p.requires_grad = False

d = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = d // model.config.num_attention_heads
L = model.config.num_hidden_layers
TARGET_LAYER = L // 2
lm_head_weight = model.lm_head.weight.detach()

class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        d_kv = n_kv_heads * head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, d_kv, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False), nn.SiLU(),
            nn.Linear(d_model // 2, d_kv, bias=False),
        )
    def forward(self, h):
        k = self.k_pred(h).view(*h.shape[:2], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(*h.shape[:2], self.n_kv_heads, self.head_dim)
        return k, v

class MedusaHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
        )
    def forward(self, h, lm_w):
        h2 = h + self.mlp(h)
        return F.linear(h2.to(lm_w.dtype), lm_w)

# ═══════════════════════════════════════════════════════
# PART 1: Extend KV-Medusa to t+30
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART 1: KV-Medusa heads t+11 through t+30")
print(f"{'='*60}", flush=True)

kv_results = []

# Load existing heads t+1 to t+10
for i in range(1, 11):
    path = f"checkpoints/qwen_halo/kv_medusa_head_{i}.pt"
    if Path(path).exists():
        kv_results.append({"offset": i, "status": "loaded"})

for offset in range(11, 31):
    head = KVMedusaHead(d, n_kv_heads, head_dim).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0

    for batch in iter_batches(train_tokens, SEQ_LEN, device, STEPS_PER_HEAD):
        if step >= STEPS_PER_HEAD: break
        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-offset].float()
            lc = out.past_key_values.layers[TARGET_LAYER]
            ak = lc.keys[:, :, offset:].permute(0, 2, 1, 3).float()
            av = lc.values[:, :, offset:].permute(0, 2, 1, 3).float()

        ml = min(h.shape[1], ak.shape[1])
        if ml < 1: continue
        h = h[:, :ml]; ak = ak[:, :ml]; av = av[:, :ml]

        pk, pv = head(h)
        loss = F.mse_loss(pk, ak) + F.mse_loss(pv, av)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); step += 1

    # Eval
    head.eval()
    cos_ks = []
    for vb in iter_batches(val_tokens, SEQ_LEN, device, 10):
        with torch.no_grad():
            out = model(vb, use_cache=True, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-offset].float()
            lc = out.past_key_values.layers[TARGET_LAYER]
            ak = lc.keys[:, :, offset:].permute(0, 2, 1, 3).float()
            ml = min(h.shape[1], ak.shape[1])
            if ml < 1: continue
            pk, _ = head(h[:, :ml])
            cos = F.cosine_similarity(pk.reshape(-1, head_dim), ak[:, :ml].reshape(-1, head_dim), dim=-1).mean().item()
            cos_ks.append(cos)

    avg_cos = sum(cos_ks) / len(cos_ks) if cos_ks else 0
    print(f"  t+{offset:>2}: K_cos={avg_cos:.3f}", flush=True)
    kv_results.append({"offset": offset, "k_cos": round(avg_cos, 4)})

    torch.save(head.state_dict(), f"checkpoints/qwen_halo/kv_medusa_head_{offset}.pt")
    del head, opt; torch.cuda.empty_cache()

print(f"\n  KV-Medusa summary (t+11 to t+30):")
for r in kv_results:
    if "k_cos" in r:
        status = "GOOD" if r["k_cos"] > 0.7 else "OK" if r["k_cos"] > 0.5 else "WEAK"
        print(f"    t+{r['offset']:>2}: {r['k_cos']:.3f} [{status}]")

# ═══════════════════════════════════════════════════════
# PART 2: Regular Medusa heads on top
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART 2: Regular Medusa heads (token prediction)")
print(f"{'='*60}", flush=True)

medusa_results = []

for offset in range(1, 11):
    head = MedusaHead(d).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    head.train()
    step = 0

    for batch in iter_batches(train_tokens, SEQ_LEN, device, STEPS_PER_HEAD):
        if step >= STEPS_PER_HEAD: break
        with torch.no_grad():
            out = model(batch[:, :-offset], use_cache=False, output_hidden_states=True)
            h = out.hidden_states[-1].detach()
        targets = batch[:, offset:]
        ml = min(h.shape[1], targets.shape[1])
        logits = head(h[:, :ml].float(), lm_head_weight.float())
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[:, :ml].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); step += 1

    # Eval accuracy
    head.eval()
    accs = []
    for vb in iter_batches(val_tokens, SEQ_LEN, device, 10):
        with torch.no_grad():
            out = model(vb[:, :-offset], use_cache=False, output_hidden_states=True)
            h = out.hidden_states[-1]
            targets = vb[:, offset:]
            ml = min(h.shape[1], targets.shape[1])
            logits = head(h[:, :ml].float(), lm_head_weight.float())
            preds = logits.argmax(-1)
            accs.append((preds == targets[:, :ml]).float().mean().item())

    acc = sum(accs) / len(accs) if accs else 0
    print(f"  Medusa head t+{offset:>2}: acc={acc:.3f} ({acc*100:.1f}%)", flush=True)
    medusa_results.append({"offset": offset, "acc": round(acc, 4)})

    torch.save(head.state_dict(), f"checkpoints/qwen_halo/medusa_head_v2_{offset}.pt")
    del head, opt; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Combined potential
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COMBINED KV-MEDUSA + REGULAR MEDUSA")
print(f"{'='*60}")

good_kv = sum(1 for r in kv_results if r.get("k_cos", 0) > 0.7)
useful_medusa = sum(1 for r in medusa_results if r["acc"] > 0.05)

print(f"  KV-Medusa heads with K_cos > 0.7: {good_kv}")
print(f"  Regular Medusa heads with acc > 5%: {useful_medusa}")
print(f"  Combined draft pool: up to {good_kv} positions with predicted KV")

# Estimate tokens per step with combined system
accs = [r["acc"] for r in medusa_results]
expected = 1.0
product = 1.0
for a in accs:
    product *= a
    expected += product
    if product < 0.01: break
print(f"  Regular Medusa expected tok/step: {expected:.2f}")
print(f"  With KV-Medusa enabling {good_kv} positions: up to {good_kv} tok/step")

Path("results").mkdir(exist_ok=True)
with open("results/kv_medusa_extended.json", "w") as f:
    json.dump({
        "kv_medusa_results": kv_results,
        "medusa_results": medusa_results,
        "good_kv_heads": good_kv,
        "useful_medusa_heads": useful_medusa,
    }, f, indent=2)
print(f"\nSaved results/kv_medusa_extended.json", flush=True)

del model; torch.cuda.empty_cache()
