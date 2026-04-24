"""Stage 120 — Slow anneal throat to rank 1 with fine-tuning.

Progressive factorization of throat layers (L7-14) from full rank
down to rank 1, with norm+KV fine-tuning at each step.

Schedule (per projection in throat):
  1280 → 640 → 320 → 160 → 80 → 40 → 20 → 10 → 5 → 2 → 1

Each step: SVD truncate → fine-tune 200 steps → eval → next step.
The model adapts at each stage instead of being shocked to rank 1.
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


def svd_truncate_layer(layer, rank):
    """SVD truncate all KV + attention projections in one layer."""
    for name in ("k_proj", "v_proj", "q_proj", "o_proj"):
        proj = getattr(layer.self_attn, name)
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(min(rank, len(S)), 1)
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def finetune(model, train_tokens, seq_len, device, steps=200, lr=5e-5):
    """Fine-tune norms + KV projections in throat."""
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
        # Also tune KV in throat layers (L7-14)
        elif any(f"layers.{i}." in name for i in range(7, 15)):
            if "k_proj" in name or "v_proj" in name:
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


MODEL = "checkpoints/qwen_halo/wormhole_compressed"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"

# Anneal schedule: halving each step down to 1
RANK_SCHEDULE = [640, 320, 160, 80, 40, 20, 10, 5, 2, 1]

print("=" * 60)
print("STAGE 120 — THROAT ANNEAL TO RANK 1")
print(f"  Schedule: {RANK_SCHEDULE}")
print(f"  Throat layers: L7-L14")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

print("\nLoading wormhole model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Baseline: ppl={teacher_ppl:.1f}  [{teacher_text[:60]}]", flush=True)

history = []

for rank in RANK_SCHEDULE:
    t0 = time.time()
    print(f"\n{'─'*50}")
    print(f"  Throat → rank {rank}", flush=True)

    # Truncate all throat layers to this rank
    for i in range(7, 15):
        svd_truncate_layer(model.model.layers[i], rank)

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-tune: ppl={pre_ppl:.1f}", flush=True)

    # Fine-tune
    finetune(model, train_tokens, SEQ_LEN, device, steps=200)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    elapsed = time.time() - t0

    print(f"  post-tune: ppl={post_ppl:.1f} (Δ={post_ppl-teacher_ppl:+.1f})")
    print(f"  [{text[:60]}]")
    print(f"  elapsed={elapsed:.0f}s", flush=True)

    history.append({
        "rank": rank, "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "text": text[:80], "elapsed": elapsed,
    })

    if post_ppl > teacher_ppl * 4:
        print(f"\n  ⚠ STOPPED: ppl {post_ppl:.1f} > 4x baseline")
        break

# Summary
print(f"\n{'='*60}")
print("THROAT ANNEAL COMPLETE")
print(f"{'='*60}")
print(f"  Baseline: {teacher_ppl:.1f}")
for h in history:
    print(f"  rank {h['rank']:>4}: {h['pre_ppl']:.1f} → {h['post_ppl']:.1f}")
if history:
    print(f"  Final: rank {history[-1]['rank']}, ppl {history[-1]['post_ppl']:.1f}")

Path("results").mkdir(exist_ok=True)
with open("results/stage120_throat_anneal.json", "w") as f:
    json.dump({"baseline_ppl": teacher_ppl, "history": history}, f, indent=2)
print(f"\nSaved results/stage120_throat_anneal.json", flush=True)
