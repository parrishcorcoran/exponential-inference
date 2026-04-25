"""Stage 141 — KV cache rank anneal on 14B.

Stage 140 found K and V cache are uniformly ranked (EVR-95 ~125).
Now slowly anneal the KV projection ranks down with fine-tuning
to find the true compression floor.

This is different from weight rank anneal (stage 120) — here we're
compressing the KV PROJECTIONS (k_proj, v_proj) which directly
determine the KV cache dimensionality.

Schedule: 1024 → 768 → 512 → 384 → 256 → 192 → 128 → 96 → 64 → 48 → 32 → 16
Fine-tune norms + KV 200 steps between each.
"""
import torch
import torch.nn.functional as F
import math
import json
import time
from pathlib import Path

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
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
        if len(window) < 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


@torch.no_grad()
def eval_ppl(model, val_tokens, seq_len, device, n_batches=15):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_sample(model, tokenizer, prompt, n=40):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def compress_kv_all_layers(model, rank):
    """SVD truncate k_proj + v_proj on ALL layers to given rank."""
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            W = proj.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = max(min(rank, len(S)), 1)
            proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def finetune_kv_norms(model, train_tokens, seq_len, device, steps=200, lr=5e-5):
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train()
    step = 0
    for inp, tgt in iter_batches(train_tokens, seq_len, 1, device):
        if step >= steps:
            break
        opt.zero_grad()
        loss = F.cross_entropy(
            model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(),
            tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
    del opt
    for p in model.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"

RANK_SCHEDULE = [768, 512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16]

print("=" * 60)
print("STAGE 141 — KV CACHE RANK ANNEAL (14B)")
print(f"  Schedule: {RANK_SCHEDULE}")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

print("\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
baseline_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Baseline: ppl={baseline_ppl:.1f}  [{baseline_text[:60]}]", flush=True)

history = []

for rank in RANK_SCHEDULE:
    t0 = time.time()
    print(f"\n{'─'*50}")
    print(f"  KV → rank {rank} (all layers)", flush=True)

    compress_kv_all_layers(model, rank)

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-tune: ppl={pre_ppl:.1f}", flush=True)

    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    elapsed = time.time() - t0

    # KV cache size at this rank
    # Original: 2 * L * seq * d_kv * 2 bytes = 2*40*seq*1024*2
    # At rank r: the projections output d_kv dims but the effective info is rank r
    # Cache compression = d_kv / rank (for both K and V)
    cache_compression = 1024.0 / rank

    print(f"  post-tune: ppl={post_ppl:.1f} (Δ={post_ppl-baseline_ppl:+.1f})")
    print(f"  cache compression: {cache_compression:.1f}x")
    print(f"  [{text[:60]}]")
    print(f"  elapsed={elapsed:.0f}s", flush=True)

    history.append({
        "rank": rank, "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "cache_compression": cache_compression,
        "text": text[:80], "elapsed": elapsed,
    })

    if post_ppl > baseline_ppl * 4:
        print(f"\n  ⚠ STOPPED: ppl {post_ppl:.1f} > 4x baseline")
        break

# Summary
print(f"\n{'='*60}")
print("KV CACHE ANNEAL COMPLETE")
print(f"{'='*60}")
print(f"  Baseline: {baseline_ppl:.1f}")
for h in history:
    status = "✓" if h["post_ppl"] <= baseline_ppl * 1.5 else "⚠" if h["post_ppl"] <= baseline_ppl * 3 else "✗"
    print(f"  {status} rank {h['rank']:>4}: {h['post_ppl']:.1f} (Δ={h['post_ppl']-baseline_ppl:+.1f}) cache={h['cache_compression']:.0f}x")

# Find sweet spot
good = [h for h in history if h["post_ppl"] <= baseline_ppl * 1.2]
if good:
    best = min(good, key=lambda h: h["rank"])
    print(f"\n  Sweet spot: rank {best['rank']} → {best['cache_compression']:.0f}x cache compression at {best['post_ppl']:.1f} ppl")

Path("results").mkdir(exist_ok=True)
with open("results/stage141_cache_anneal.json", "w") as f:
    json.dump({"baseline_ppl": baseline_ppl, "history": history}, f, indent=2)
print(f"\nSaved results/stage141_cache_anneal.json", flush=True)
