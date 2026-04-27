"""Train 5 *conditional* KV-Medusa heads on Qwen3-0.6B.

Each head takes (h_t, embed(token_at_t+offset)) and predicts the layer-14 K, V
that the model would have computed for that token at that position. Same MSE
loss as the unconditional KV-Medusa heads, but the input now carries which
token is being placed at the future position.

Hypothesis: at inference, conditioning on the *candidate* token from a tree
branch lets the head predict KV consistent with that specific branch — so
verify can fairly evaluate non-natural branches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32


def empty_cache():
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()


def load_owt_tokens(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
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


class ConditionalKVMedusaHead(nn.Module):
    """Predict layer-14 K, V at offset, conditioned on (h_t, embed(token_at_t+offset))."""
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )

    def forward(self, h, token_embeds):
        # h: [B, S, d], token_embeds: [B, S, d]
        x = torch.cat([h, token_embeds], dim=-1)
        k = self.k_pred(x).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(x).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
MAX_OFFSETS = 5
STEPS_PER_HEAD = 300
EVAL_EVERY = 50
LR = 5e-4
TARGET_LAYER = 14
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_conditional.json")


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading tokens...")
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"Loading {CHECKPOINT}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)

embed_layer = model.model.embed_tokens

for p in model.parameters():
    p.requires_grad = False

print(f"  d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, target_layer={TARGET_LAYER}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

all_results = []

for offset in range(1, MAX_OFFSETS + 1):
    print(f"\n{'='*60}\n  CONDITIONAL KV-MEDUSA HEAD {offset}\n{'='*60}", flush=True)

    head = ConditionalKVMedusaHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {head_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0
    history = []

    for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
        if step >= STEPS_PER_HEAD: break

        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-offset].detach().float()  # [1, S-off, d]

            target_tokens = batch[:, offset:]  # [1, S-off]
            token_embeds = embed_layer(target_tokens).detach().float()  # [1, S-off, d]

            actual_k = out.past_key_values.layers[TARGET_LAYER].keys
            actual_v = out.past_key_values.layers[TARGET_LAYER].values
            target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).detach().float()
            target_v = actual_v[:, :, offset:].permute(0, 2, 1, 3).detach().float()

        ml = min(h.shape[1], target_k.shape[1], token_embeds.shape[1])
        h, token_embeds = h[:, :ml], token_embeds[:, :ml]
        target_k, target_v = target_k[:, :ml], target_v[:, :ml]

        pred_k, pred_v = head(h, token_embeds)
        loss = F.mse_loss(pred_k, target_k) + F.mse_loss(pred_v, target_v)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        step += 1

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim),
                                            target_k.reshape(-1, head_dim), dim=-1).mean().item()
                cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim),
                                            target_v.reshape(-1, head_dim), dim=-1).mean().item()
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
            tt = vbatch[:, offset:]
            te = embed_layer(tt).float()
            lc = out.past_key_values.layers[TARGET_LAYER]
            ak = lc.keys[:, :, offset:].permute(0, 2, 1, 3).float()
            av = lc.values[:, :, offset:].permute(0, 2, 1, 3).float()
            ml = min(h_val.shape[1], ak.shape[1], te.shape[1])
            pk, pv = head(h_val[:, :ml], te[:, :ml])
            val_cos_k.append(F.cosine_similarity(pk.reshape(-1, head_dim),
                                                 ak[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            val_cos_v.append(F.cosine_similarity(pv.reshape(-1, head_dim),
                                                 av[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
        val_count += 1

    fck = sum(val_cos_k) / len(val_cos_k); fcv = sum(val_cos_v) / len(val_cos_v)
    print(f"\n  CONDITIONAL HEAD {offset} FINAL: cos_k={fck:.3f} cos_v={fcv:.3f}", flush=True)

    all_results.append({"offset": offset, "final_cos_k": round(fck, 4),
                        "final_cos_v": round(fcv, 4),
                        "head_params_M": round(head_params/1e6, 2),
                        "history": history})

    torch.save(head.state_dict(), CKPT_DIR / f"kv_medusa_cond_head_{offset}.pt")
    del head, opt; empty_cache()

print(f"\n{'='*60}\nCONDITIONAL KV-MEDUSA SUMMARY (vs unconditional)\n{'='*60}")
print(f"  {'offset':<8}{'cond cos_k':<14}{'cond cos_v':<14}")
for r in all_results:
    print(f"  t+{r['offset']:<6}{r['final_cos_k']:<14.3f}{r['final_cos_v']:<14.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "device": device, "target_layer": TARGET_LAYER,
               "results": all_results}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")

del model; gc.collect(); empty_cache()
