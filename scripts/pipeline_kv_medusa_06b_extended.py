"""KV-Medusa 0.6B — extend to offsets 11-30.

Hypothesis (HRR-driven): the layer-14 K-cache lies on a manifold whose effective
dimension supports much more than 10 offsets. Plate's HRR capacity bound says a
d-dim space can store ~d/k distinguishable bound items. With d_kv=1024 and our
measured per-position cos~0.75, the prediction horizon should support 30-60
offsets before quality degrades. We measured 1-10 and saw zero decay; extending
to 11-30 tests whether the manifold actually carries that capacity.

Trains one head per offset, 300 steps each, same recipe as the 1-10 run.
Saves to checkpoints/qwen_06b/kv_medusa_head_{11..30}.pt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

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


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )

    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256          # need long enough sequences for offset 30
START_OFFSET = 11
END_OFFSET = 30
STEPS_PER_HEAD = 300
EVAL_EVERY = 100
LR = 5e-4
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_extended.json")

print(f"device={device} dtype={dtype}")
print(f"Training KV-Medusa heads {START_OFFSET}..{END_OFFSET}, {STEPS_PER_HEAD} steps each")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading tokens...", flush=True)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"Loading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)
TARGET_LAYER = model.config.num_hidden_layers // 2

print(f"  d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, target_layer={TARGET_LAYER}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

all_results = []

for offset in range(START_OFFSET, END_OFFSET + 1):
    print(f"\n{'='*60}\n  KV-MEDUSA HEAD {offset} (target layer {TARGET_LAYER})\n{'='*60}", flush=True)

    head = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
    head_params = sum(p.numel() for p in head.parameters())

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0
    history = []

    for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
        if step >= STEPS_PER_HEAD: break

        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-offset].detach().float()
            actual_k = out.past_key_values.layers[TARGET_LAYER].keys
            actual_v = out.past_key_values.layers[TARGET_LAYER].values
            target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).detach().float()
            target_v = actual_v[:, :, offset:].permute(0, 2, 1, 3).detach().float()

        ml = min(h.shape[1], target_k.shape[1])
        h, target_k, target_v = h[:, :ml], target_k[:, :ml], target_v[:, :ml]

        pred_k, pred_v = head(h)
        loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean().item()
                cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim), target_v.reshape(-1, head_dim), dim=-1).mean().item()
            print(f"  step {step:>4}: loss={loss.item():.4f} cos_k={cos_k:.3f} cos_v={cos_v:.3f}", flush=True)
            history.append({"step": step, "loss": round(loss.item(), 4),
                            "cos_k": round(cos_k, 4), "cos_v": round(cos_v, 4)})

    # Final val
    head.eval()
    val_cos_k, val_cos_v = [], []
    val_count = 0
    for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
        if val_count >= 10: break
        with torch.no_grad():
            out = model(vbatch, use_cache=True, output_hidden_states=True)
            h_val = out.hidden_states[-1][:, :-offset].float()
            lc = out.past_key_values.layers[TARGET_LAYER]
            ak = lc.keys[:, :, offset:].permute(0, 2, 1, 3).float()
            av = lc.values[:, :, offset:].permute(0, 2, 1, 3).float()
            ml = min(h_val.shape[1], ak.shape[1])
            pk, pv = head(h_val[:, :ml])
            val_cos_k.append(F.cosine_similarity(pk.reshape(-1, head_dim), ak[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            val_cos_v.append(F.cosine_similarity(pv.reshape(-1, head_dim), av[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
        val_count += 1

    fck = sum(val_cos_k) / len(val_cos_k); fcv = sum(val_cos_v) / len(val_cos_v)
    print(f"\n  HEAD {offset} FINAL: cos_k={fck:.3f} cos_v={fcv:.3f}", flush=True)

    all_results.append({"offset": offset, "final_cos_k": round(fck, 4),
                        "final_cos_v": round(fcv, 4), "history": history})

    torch.save(head.state_dict(), CKPT_DIR / f"kv_medusa_head_{offset}.pt")
    del head, opt; empty_cache()

print(f"\n{'='*60}\nKV-MEDUSA EXTENDED SUMMARY (offsets {START_OFFSET}-{END_OFFSET})\n{'='*60}")
for r in all_results:
    quality = "GOOD" if r["final_cos_k"] > 0.7 else "OK" if r["final_cos_k"] > 0.5 else "WEAK"
    print(f"  t+{r['offset']:>2}: cos_k={r['final_cos_k']:.3f} cos_v={r['final_cos_v']:.3f}  [{quality}]")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER,
               "results": all_results}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
