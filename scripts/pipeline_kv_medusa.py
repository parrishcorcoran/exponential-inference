"""KV-Medusa: predict future KV cache entries, not just tokens.

Standard Medusa predicts future token logits. KV-Medusa predicts
future K and V cache vectors. This enables:
  - Drafting 30-60 tokens ahead without forward passes
  - ONE batched verification pass at the end
  - Massively parallel speculative decoding

With KV at rank 256, each prediction target is just 256 dims.
Compare with token prediction which targets vocab_size (151K+).
KV prediction should be MUCH easier.

Architecture per head:
  KVMedusaHead:
    k_pred: MLP(d_model → d_kv)  # predict K at t+offset
    v_pred: MLP(d_model → d_kv)  # predict V at t+offset

Train on the KV-256 base model's actual KV cache outputs.
Measure reconstruction error and token prediction accuracy
when using predicted KV for attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import gc
from pathlib import Path

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_owt_tokens(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(encoded)
            continue
        toks.extend(encoded)
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
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t
            batch = []


class KVMedusaHead(nn.Module):
    """Predict future K and V cache entries from hidden state."""
    def __init__(self, d_model, d_kv, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Predict K and V for all KV heads at a future position
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
        """h: [batch, seq, d_model] → predicted K,V: [batch, seq, n_kv_heads, head_dim]"""
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


CHECKPOINT = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
MAX_OFFSETS = 10  # predict t+1 through t+10
STEPS_PER_HEAD = 500
EVAL_EVERY = 100
LR = 5e-4

print("=" * 60)
print("KV-MEDUSA: predict future K/V cache entries")
print(f"  {MAX_OFFSETS} heads (t+1 through t+{MAX_OFFSETS})")
print(f"  {STEPS_PER_HEAD} steps each")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("\nLoading tokens...", flush=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"\nLoading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = d_model // model.config.num_attention_heads
d_kv = n_kv_heads * head_dim
L = model.config.num_hidden_layers

# Freeze base model
for p in model.parameters():
    p.requires_grad = False

print(f"  d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, d_kv={d_kv}")
print(f"  KV-Medusa target dim: {d_kv} per head per layer", flush=True)

# Choose a representative layer for KV prediction (middle of model)
TARGET_LAYER = L // 2
print(f"  Training on layer {TARGET_LAYER} (middle)", flush=True)

all_results = []

for offset in range(1, MAX_OFFSETS + 1):
    print(f"\n{'='*60}")
    print(f"  KV-MEDUSA HEAD {offset} — predict KV at t+{offset}, layer {TARGET_LAYER}")
    print(f"{'='*60}", flush=True)

    head = KVMedusaHead(d_model, d_kv, n_kv_heads, head_dim).to(device).to(torch.float32)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {head_params/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0
    history = []

    for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
        if step >= STEPS_PER_HEAD:
            break

        # Get hidden states AND actual KV cache
        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            # Hidden state at each position (before the offset)
            h = out.hidden_states[-1][:, :-offset].detach().float()  # [1, seq-offset, d]

            # Actual K, V at target layer
            pkv = out.past_key_values
            layer_cache = pkv.layers[TARGET_LAYER]
            actual_k = layer_cache.keys   # [1, n_kv_heads, seq, head_dim]
            actual_v = layer_cache.values

            # Target: KV at positions offset through end
            target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).detach().float()  # [1, seq-offset, n_kv_heads, head_dim]
            target_v = actual_v[:, :, offset:].permute(0, 2, 1, 3).detach().float()

        # Trim to match
        min_len = min(h.shape[1], target_k.shape[1])
        h = h[:, :min_len]
        target_k = target_k[:, :min_len]
        target_v = target_v[:, :min_len]

        # Predict KV
        pred_k, pred_v = head(h)

        # Loss: MSE on K and V
        loss_k = F.mse_loss(pred_k, target_k)
        loss_v = F.mse_loss(pred_v, target_v)
        loss = loss_k + loss_v

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        step += 1

        if step % EVAL_EVERY == 0:
            # Cosine similarity as quality metric
            with torch.no_grad():
                cos_k = F.cosine_similarity(pred_k.reshape(-1, head_dim), target_k.reshape(-1, head_dim), dim=-1).mean().item()
                cos_v = F.cosine_similarity(pred_v.reshape(-1, head_dim), target_v.reshape(-1, head_dim), dim=-1).mean().item()

                # Relative error
                rel_k = (pred_k - target_k).norm() / target_k.norm()
                rel_v = (pred_v - target_v).norm() / target_v.norm()

            print(f"  step {step:>4}: loss={loss.item():.4f} cos_k={cos_k:.3f} cos_v={cos_v:.3f} rel_k={rel_k:.3f} rel_v={rel_v:.3f}", flush=True)
            history.append({
                "step": step, "loss": round(loss.item(), 5),
                "cos_k": round(cos_k, 4), "cos_v": round(cos_v, 4),
                "rel_k": round(rel_k.item(), 4), "rel_v": round(rel_v.item(), 4),
            })

    # Final eval on validation
    head.eval()
    val_cos_k, val_cos_v, val_rel_k, val_rel_v = [], [], [], []
    val_count = 0
    for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
        if val_count >= 10: break
        with torch.no_grad():
            out = model(vbatch, use_cache=True, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-offset].float()
            pkv2 = out.past_key_values
            lc2 = pkv2.layers[TARGET_LAYER]
            ak = lc2.keys[:, :, offset:].permute(0, 2, 1, 3).float()
            av = lc2.values[:, :, offset:].permute(0, 2, 1, 3).float()
            ml = min(h.shape[1], ak.shape[1])
            pk, pv = head(h[:, :ml])
            val_cos_k.append(F.cosine_similarity(pk.reshape(-1, head_dim), ak[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            val_cos_v.append(F.cosine_similarity(pv.reshape(-1, head_dim), av[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
            val_rel_k.append(((pk - ak[:, :ml]).norm() / ak[:, :ml].norm()).item())
            val_rel_v.append(((pv - av[:, :ml]).norm() / av[:, :ml].norm()).item())
        val_count += 1

    final_cos_k = sum(val_cos_k) / len(val_cos_k)
    final_cos_v = sum(val_cos_v) / len(val_cos_v)
    final_rel_k = sum(val_rel_k) / len(val_rel_k)
    final_rel_v = sum(val_rel_v) / len(val_rel_v)

    print(f"\n  KV-HEAD {offset} FINAL: cos_k={final_cos_k:.3f} cos_v={final_cos_v:.3f} rel_k={final_rel_k:.3f} rel_v={final_rel_v:.3f}", flush=True)

    all_results.append({
        "offset": offset,
        "final_cos_k": round(final_cos_k, 4),
        "final_cos_v": round(final_cos_v, 4),
        "final_rel_k": round(final_rel_k, 4),
        "final_rel_v": round(final_rel_v, 4),
        "head_params_M": round(head_params / 1e6, 1),
        "history": history,
    })

    torch.save(head.state_dict(), f"checkpoints/qwen_halo/kv_medusa_head_{offset}.pt")
    del head, opt; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("KV-MEDUSA SUMMARY")
print(f"{'='*60}")
print(f"  Layer: {TARGET_LAYER}")
for r in all_results:
    quality = "GOOD" if r["final_cos_k"] > 0.8 else "OK" if r["final_cos_k"] > 0.5 else "WEAK"
    print(f"  t+{r['offset']:>2}: cos_k={r['final_cos_k']:.3f} cos_v={r['final_cos_v']:.3f} rel_err={r['final_rel_k']:.3f}/{r['final_rel_v']:.3f}  [{quality}]")

# How many draft tokens can we trust?
good_heads = sum(1 for r in all_results if r["final_cos_k"] > 0.7 and r["final_cos_v"] > 0.7)
ok_heads = sum(1 for r in all_results if r["final_cos_k"] > 0.5 and r["final_cos_v"] > 0.5)
print(f"\n  Heads with cos > 0.7: {good_heads} (high quality drafts)")
print(f"  Heads with cos > 0.5: {ok_heads} (usable drafts)")
print(f"  Potential draft tokens per step: {ok_heads}")

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_kv_medusa.json", "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "target_layer": TARGET_LAYER,
        "d_kv": d_kv,
        "results": all_results,
        "good_heads": good_heads,
        "ok_heads": ok_heads,
    }, f, indent=2)
print(f"\nSaved results/pipeline_kv_medusa.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
