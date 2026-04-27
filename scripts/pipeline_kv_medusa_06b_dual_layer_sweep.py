"""Dual-layer sweep — V only, fast.

For each PAIR of layers (L1, L2), train a probe that takes concat(h[L1], h[L2])
and predicts V at target layer 14, offset +1.

Hypothesis: maybe a pair like (throat, mouth) predicts V better than either
alone — throat = compressed essence, mouth = fresh features.

V was the uniform-rank axis in finding 18, so it should be the cleanest test
of "do dual layers add information." Less training needed than K or Q.

Setup:
  - All pairs (L1, L2) with L1 < L2 over 28 layers = C(28,2) = 378 pairs
  - 50 steps per pair (down from 200)
  - Hidden states cached per batch
  - Output: 28x28 V cosine matrix + best pair

Estimated runtime on Strix (cuda + bf16): ~15-20 minutes.
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


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
STEPS_PER_PAIR = 50
LR = 5e-4
BATCH_CACHE_SIZE = 4
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_dual_layer_sweep_v.json")


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


def iter_batches(tokens, seq_len, device, n=999999):
    nb = (len(tokens) - 1) // seq_len
    idx = list(range(nb)); random.shuffle(idx)
    for i in idx[:n]:
        s = i * seq_len
        w = tokens[s:s + seq_len + 1]
        if len(w) < seq_len + 1: continue
        yield torch.tensor([w], dtype=torch.long, device=device)


class PairVHead(nn.Module):
    """V-only probe: concat(h[L1], h[L2]) -> V at target layer."""
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads; self.head_dim = head_dim
        d_in = 2 * d_model
        hidden = d_model // 2
        self.v = nn.Sequential(
            nn.Linear(d_in, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, n_kv_heads * head_dim, bias=False),
        )

    def forward(self, h1, h2):
        h = torch.cat([h1, h2], dim=-1)
        b, s = h.shape[0], h.shape[1]
        return self.v(h).view(b, s, self.n_kv_heads, self.head_dim)


def collect_cache(model, batches_iter, n_batches, target_layer, offset):
    """Run model on n_batches, cache hidden_states + V target at offset."""
    cache = []
    for _ in range(n_batches):
        try:
            batch = next(batches_iter)
        except StopIteration:
            break
        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            hidden_states = [h.detach().float() for h in out.hidden_states]
            av = out.past_key_values.layers[target_layer].values.detach().float()
        ml = hidden_states[0].shape[1] - offset
        h_per_layer = [h[:, :ml] for h in hidden_states]
        target_v = av[:, :, offset:].permute(0, 2, 1, 3)[:, :ml]
        cache.append({"h_per_layer": h_per_layer, "target_v": target_v})
    return cache


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
n_layers = model.config.num_hidden_layers

print("Loading tokens...", flush=True)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 600)
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 60, skip_tokens=SEQ_LEN * 600)
print(f"  train_tokens={len(train_tokens)}, val_tokens={len(val_tokens)}")

pairs = [(L1, L2) for L1 in range(n_layers) for L2 in range(L1 + 1, n_layers)]
print(f"  d_model={d_model}, n_layers={n_layers}, total pairs={len(pairs)}")

results = []
for pair_idx, (L1, L2) in enumerate(pairs):
    head = PairVHead(d_model, n_kv_heads, head_dim).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()

    train_iter = iter_batches(train_tokens, SEQ_LEN, device)
    step = 0
    while step < STEPS_PER_PAIR:
        cache = collect_cache(model, train_iter, BATCH_CACHE_SIZE, TARGET_LAYER, OFFSET)
        if not cache: break
        for c in cache:
            if step >= STEPS_PER_PAIR: break
            h1 = c["h_per_layer"][L1]
            h2 = c["h_per_layer"][L2]
            pv = head(h1, h2)
            loss = F.mse_loss(pv, c["target_v"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); step += 1

    # Val cosine
    head.eval()
    cs_v = []
    val_iter = iter_batches(val_tokens, SEQ_LEN, device)
    val_count = 0
    while val_count < 6:
        cache = collect_cache(model, val_iter, 4, TARGET_LAYER, OFFSET)
        if not cache: break
        with torch.no_grad():
            for c in cache:
                if val_count >= 6: break
                h1 = c["h_per_layer"][L1]
                h2 = c["h_per_layer"][L2]
                pv = head(h1, h2)
                cs_v.append(F.cosine_similarity(pv.reshape(-1, head_dim),
                                                c["target_v"].reshape(-1, head_dim), dim=-1).mean().item())
                val_count += 1

    cv = sum(cs_v) / max(len(cs_v), 1)
    print(f"  [{pair_idx+1:>3}/{len(pairs)}] L1={L1:>2} L2={L2:>2}  cos_v={cv:.3f}", flush=True)
    results.append({"L1": L1, "L2": L2, "cos_v": round(cv, 4)})

    del head, opt
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()

    if (pair_idx + 1) % 20 == 0 or pair_idx == len(pairs) - 1:
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "checkpoint": CHECKPOINT,
                "target_layer": TARGET_LAYER,
                "offset": OFFSET,
                "axis": "V",
                "n_layers": n_layers,
                "steps_per_pair": STEPS_PER_PAIR,
                "pairs_completed": pair_idx + 1,
                "total_pairs": len(pairs),
                "results": results,
            }, f, indent=2)

# Summary
print(f"\n{'='*60}\nDUAL-LAYER SWEEP (V) SUMMARY\n{'='*60}")
top10 = sorted(results, key=lambda r: r["cos_v"], reverse=True)[:10]
print("  Top 10 pairs for V:")
for r in top10:
    print(f"    L1={r['L1']:>2} L2={r['L2']:>2}  cos_v={r['cos_v']:.3f}")

best_v = top10[0]
print(f"\n  Best pair for V: (L{best_v['L1']}, L{best_v['L2']})  cos_v={best_v['cos_v']:.3f}")

# Compare to single-layer baseline if present
single_path = Path("results/pipeline_kv_medusa_06b_layer_sweep.json")
if single_path.exists():
    single = json.load(open(single_path))
    s_best = max(single["results"], key=lambda r: r["cos_v"])
    print(f"\n  Single-layer baseline V: L{s_best['layer']}  cos_v={s_best['cos_v']:.3f}")
    delta = best_v["cos_v"] - s_best["cos_v"]
    print(f"  Δcos_v (dual − single) = {delta:+.3f}")
    if delta > 0.02:
        print("  → Dual layers carry V information that no single layer does.")
    elif delta < 0.005:
        print("  → Single layer suffices; dual adds no V signal.")
    else:
        print("  → Marginal gain.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "target_layer": TARGET_LAYER,
        "offset": OFFSET,
        "axis": "V",
        "n_layers": n_layers,
        "steps_per_pair": STEPS_PER_PAIR,
        "pairs_completed": len(results),
        "total_pairs": len(pairs),
        "results": results,
        "best_v_pair": [best_v["L1"], best_v["L2"]],
        "best_v_cos": best_v["cos_v"],
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
