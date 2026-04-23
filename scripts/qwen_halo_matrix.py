"""Qwen Halo — Orthogonality Matrix.

Test each compression lever ALL THE WAY DOWN independently.
Then cross-test: at each lever's floor, try adding every other lever.

This reveals:
- Which axes are orthogonal (independent budgets)
- Which axes share budget (coupled)
- The TRUE floor of each axis

Levers tested:
  A. KV rank: 512 → 384 → 256 → 128 → 64 → 32 → 16
  B. Weight quant: Q8 → Q6 → Q4 (no QAT fine-tune — measure raw impact)
  C. Embed quant: Q8 → Q6 → Q4

Phase 1: Each lever solo (fresh model each time)
Phase 2: At KV floor, try B and C
Phase 3: At weight floor, try A and C
"""
import torch
import torch.nn.functional as F
import math
import json
import time
import copy
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
def eval_ppl(model, val_tokens, seq_len, device):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= 10: break
    return total / max(n, 1)


def generate_sample(model, tokenizer, prompt, n=30):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def finetune_kv(model, train_tokens, seq_len, device, steps=300, lr=5e-5):
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable: return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train()
    step = 0
    for inp, tgt in iter_batches(train_tokens, seq_len, 1, device):
        if step >= steps: break
        opt.zero_grad()
        loss = F.cross_entropy(model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); step += 1
    del opt
    for p in model.parameters(): p.requires_grad = False
    torch.cuda.empty_cache()


def compress_kv(model, rank):
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            W = proj.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = min(rank, len(S))
            proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def quantize_weights(model, n_bits):
    levels = 2 ** n_bits; half = levels // 2
    for layer in model.model.layers:
        for name in ["q_proj", "o_proj"]:
            w = getattr(layer.self_attn, name).weight
            s = w.float().abs().max().item() / max(half - 1, 1)
            if s > 1e-10: w.data = (w.float() / s).round().clamp(-half+1, half-1).mul(s).to(w.dtype)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            w = getattr(layer.mlp, name).weight
            s = w.float().abs().max().item() / max(half - 1, 1)
            if s > 1e-10: w.data = (w.float() / s).round().clamp(-half+1, half-1).mul(s).to(w.dtype)


def quantize_embed(model, n_bits):
    w = model.get_input_embeddings().weight
    levels = 2 ** n_bits; half = levels // 2
    s = w.float().abs().max().item() / max(half - 1, 1)
    if s > 1e-10: w.data = (w.float() / s).round().clamp(-half+1, half-1).mul(s).to(w.dtype)


def load_fresh_model(model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()


print("=" * 70)
print("QWEN HALO — ORTHOGONALITY MATRIX")
print("=" * 70)

MODEL = "Qwen/Qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
PROMPT = "The theory of general relativity describes gravity as"
SEQ_LEN = 128

train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 20, split="validation")

results = {}

# ═══════════════════════════════════════════════════════
# PHASE 1: Each lever solo, all the way down
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 1: Each lever SOLO, all the way down")
print(f"{'='*60}")

# A. KV rank solo
print(f"\n--- LEVER A: KV rank (solo) ---")
model = load_fresh_model(MODEL)
teacher_ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Teacher: ppl={math.exp(teacher_ce):.1f}  [{teacher_text[:50]}]")

kv_results = []
for rank in [512, 384, 256, 128, 64, 32, 16]:
    compress_kv(model, rank)
    pre = eval_ppl(model, val_tokens, SEQ_LEN, device)
    finetune_kv(model, train_tokens, SEQ_LEN, device, steps=300)
    post = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  KV {rank:>4}: {math.exp(pre):.1f}→{math.exp(post):.1f}  [{text[:50]}]")
    kv_results.append({"rank": rank, "pre_ppl": math.exp(pre), "post_ppl": math.exp(post), "text": text[:80]})
    if post > teacher_ce * 4:
        print(f"  ⚠ KV floor reached at rank {rank}")
        break

results["kv_solo"] = kv_results
del model; torch.cuda.empty_cache()

# B. Weight quant solo — WITH fine-tuning (QAT-aware)
print(f"\n--- LEVER B: Weight quant (solo, with fine-tune) ---")
model = load_fresh_model(MODEL)
wt_results = []
for bits in [8, 6, 4]:
    quantize_weights(model, bits)
    pre = eval_ppl(model, val_tokens, SEQ_LEN, device)
    # Fine-tune norms + any small params to recover (full body OOMs)
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True; trainable.append(p)
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
        model.train(); step = 0
        for inp, tgt in iter_batches(train_tokens, SEQ_LEN, 1, device):
            if step >= 300: break
            opt.zero_grad()
            loss = F.cross_entropy(model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(), tgt.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); step += 1
        del opt
        for p in model.parameters(): p.requires_grad = False
        torch.cuda.empty_cache()
    post = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  Q{bits}: {math.exp(pre):.1f}→{math.exp(post):.1f}  [{text[:50]}]")
    wt_results.append({"bits": bits, "pre_ppl": math.exp(pre), "post_ppl": math.exp(post), "text": text[:80]})
    if post > teacher_ce * 4:
        print(f"  ⚠ Weight floor at Q{bits}")
        break

results["weight_solo"] = wt_results
del model; torch.cuda.empty_cache()

# C. Embed quant solo
print(f"\n--- LEVER C: Embed quant (solo) ---")
model = load_fresh_model(MODEL)
em_results = []
for bits in [8, 6, 4, 2]:
    quantize_embed(model, bits)
    ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  Embed Q{bits}: ppl={math.exp(ce):.1f}  [{text[:50]}]")
    em_results.append({"bits": bits, "ppl": math.exp(ce), "text": text[:80]})
    if ce > teacher_ce * 4:
        print(f"  ⚠ Embed floor at Q{bits}")
        break
    del model; torch.cuda.empty_cache()
    model = load_fresh_model(MODEL)

results["embed_solo"] = em_results
del model; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# PHASE 2: Cross-test — at KV floor, try other levers
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: KV compressed → try stacking other levers")
print(f"{'='*60}")

model = load_fresh_model(MODEL)
# Compress KV to best working level
best_kv = kv_results[-2]["rank"] if len(kv_results) > 1 else 512  # one above floor
print(f"  Compressing KV to {best_kv}...")
for rank in [r["rank"] for r in kv_results if r["rank"] >= best_kv]:
    compress_kv(model, rank)
    finetune_kv(model, train_tokens, SEQ_LEN, device, steps=300)

kv_base_ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  KV {best_kv} base: ppl={math.exp(kv_base_ce):.1f}")

# Now try weight quant on top
cross_results = []
for bits in [8, 6, 4]:
    # Save state before quant
    quantize_weights(model, bits)
    ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  + Q{bits} weights: ppl={math.exp(ce):.1f}  [{text[:50]}]")
    cross_results.append({"on_top_of": f"kv_{best_kv}", "axis": "weights", "bits": bits,
                          "ppl": math.exp(ce), "text": text[:80]})
    if ce > teacher_ce * 4:
        break
    # Reload KV-compressed model for next test
    del model; torch.cuda.empty_cache()
    model = load_fresh_model(MODEL)
    for rank in [r["rank"] for r in kv_results if r["rank"] >= best_kv]:
        compress_kv(model, rank)
        finetune_kv(model, train_tokens, SEQ_LEN, device, steps=300)

results["kv_then_weights"] = cross_results

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ORTHOGONALITY MATRIX SUMMARY")
print(f"{'='*60}")

print(f"\n  Teacher ppl: {math.exp(teacher_ce):.1f}")

print(f"\n  SOLO floors:")
if kv_results:
    best = min(kv_results, key=lambda x: x["post_ppl"])
    print(f"    KV: best={best['rank']} ppl={best['post_ppl']:.1f}")
if wt_results:
    best = min(wt_results, key=lambda x: x["ppl"])
    print(f"    Weights: best=Q{best['bits']} ppl={best['ppl']:.1f}")
if em_results:
    best = min(em_results, key=lambda x: x["ppl"])
    print(f"    Embed: best=Q{best['bits']} ppl={best['ppl']:.1f}")

print(f"\n  CROSS-TEST (KV compressed → add weights):")
for r in cross_results:
    print(f"    KV {r['on_top_of']} + Q{r['bits']}: ppl={r['ppl']:.1f}")

# Orthogonality check
if wt_results and cross_results:
    solo_q8 = next((w["ppl"] for w in wt_results if w["bits"] == 8), None)
    cross_q8 = next((c["ppl"] for c in cross_results if c["bits"] == 8), None)
    if solo_q8 and cross_q8:
        ratio = cross_q8 / solo_q8
        if ratio < 1.5:
            print(f"\n  → ORTHOGONAL: Q8 barely changes with KV compression (ratio={ratio:.2f})")
        elif ratio < 3:
            print(f"\n  → PARTIALLY COUPLED: Q8 somewhat worse with KV (ratio={ratio:.2f})")
        else:
            print(f"\n  → SHARED BUDGET: Q8 much worse with KV (ratio={ratio:.2f})")

Path("results").mkdir(exist_ok=True)
with open("results/orthogonality_matrix.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved results/orthogonality_matrix.json", flush=True)
