"""Stage 116 — Annealed KV compression + bathtub stack.

Progressive KV rank annealing with fine-tune at each step,
then stack the additive axes (weight Q5-mid, MLP 90%-mid, embed Q6).

Two variants:
  A. Global KV anneal (same rank all layers) → then stack
  B. Bathtub KV anneal (low rank middle, high rank edges) → then stack

Bathtub profile showed KV sensitivity is NOT bathtub-shaped:
early layers are most sensitive, late layers IMPROVE with compression.
So bathtub-aware KV should keep edges at higher rank.

Anneal schedule: 512 → 384 → 256 → 192 → 128 → 96 → 64
Fine-tune: norms + KV projections, 150 steps each
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=20):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_sample(model, tokenizer, prompt, n=30):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def compress_kv_layer(layer, rank):
    for name in ("k_proj", "v_proj"):
        proj = getattr(layer.self_attn, name)
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, len(S))
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)


def compress_kv_global(model, rank):
    for layer in model.model.layers:
        compress_kv_layer(layer, rank)
    torch.cuda.empty_cache()


def compress_kv_bathtub(model, edge_rank, mid_rank, edge_width):
    L = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        edge = i < edge_width or i >= L - edge_width
        r = edge_rank if edge else mid_rank
        compress_kv_layer(layer, r)
    torch.cuda.empty_cache()


def finetune_kv(model, train_tokens, seq_len, device, steps=150, lr=5e-5):
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable:
        return
    n_train = sum(p.numel() for p in trainable)
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


def apply_post_hoc_stack(model, edge_width):
    """Apply the additive stack: weight Q5-mid + MLP 90%-mid + embed Q6."""
    L = len(model.model.layers)
    for i in range(L):
        edge = i < edge_width or i >= L - edge_width
        layer = model.model.layers[i]
        bits = 6 if edge else 5
        half = 2 ** (bits - 1)
        for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                              (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
            for name in names:
                w = getattr(parent, name).weight
                W = w.data.float()
                scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
                w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)
        if not edge:
            # MLP 90%
            for mname in ["gate_proj", "up_proj"]:
                w = getattr(layer.mlp, mname).weight
                keep = int(w.shape[0] * 0.90)
                w.data[keep:] = 0
            w = layer.mlp.down_proj.weight
            keep = int(w.shape[1] * 0.90)
            w.data[:, keep:] = 0
    # Embed Q6
    w = model.get_input_embeddings().weight
    half = 32
    s = w.float().abs().max().item() / (half - 1)
    w.data = (w.float() / s).round().clamp(-half + 1, half - 1).mul(s).to(w.dtype)
    torch.cuda.empty_cache()


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 128
EDGE_WIDTH = 7
PROMPT = "The theory of general relativity describes gravity as"

print("=" * 60)
print("STAGE 116 — Annealed KV + bathtub stack")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

# ═══════════════════════════════════════════════════════
# Variant A: Global KV anneal → then stack
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VARIANT A: Global KV anneal → then post-hoc stack")
print(f"{'='*60}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Teacher: ppl={teacher_ppl:.1f}  [{teacher_text[:60]}]")

kv_schedule = [512, 384, 256, 192, 128]
history_a = []

for rank in kv_schedule:
    print(f"\n  KV anneal → rank {rank}...", end="", flush=True)
    compress_kv_global(model, rank)
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre={pre_ppl:.1f}", end="", flush=True)

    finetune_kv(model, train_tokens, SEQ_LEN, device, steps=150)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  post={post_ppl:.1f}  [{text[:50]}]")

    history_a.append({"rank": rank, "pre_ppl": pre_ppl, "post_ppl": post_ppl, "text": text[:80]})

    if post_ppl > teacher_ppl * 5:
        print(f"  ⚠ Stopping: quality >5x teacher")
        break

# Now apply post-hoc stack on top
print(f"\n  Applying post-hoc stack (Q5-mid + MLP 90% + E6)...")
kv_only_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
apply_post_hoc_stack(model, EDGE_WIDTH)
stacked_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
stacked_text = generate_sample(model, tokenizer, PROMPT)
print(f"  KV-annealed only:   ppl={kv_only_ppl:.1f}")
print(f"  + post-hoc stack:   ppl={stacked_ppl:.1f}  [{stacked_text[:60]}]")
print(f"  Total Δ from teacher: {stacked_ppl - teacher_ppl:+.1f}")

result_a = {
    "variant": "A_global_kv_anneal",
    "kv_schedule": kv_schedule,
    "history": history_a,
    "kv_only_ppl": kv_only_ppl,
    "stacked_ppl": stacked_ppl,
    "stacked_text": stacked_text[:80],
    "delta": stacked_ppl - teacher_ppl,
}

del model; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Variant B: Bathtub KV anneal → then stack
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VARIANT B: Bathtub KV anneal (edge=2x mid rank) → then stack")
print(f"{'='*60}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

# Bathtub schedule: edges get 2x the rank of middle
bathtub_kv_schedule = [
    (512, 512),   # both start at 512
    (512, 384),   # mid drops first
    (512, 256),
    (384, 192),
    (384, 128),
    (256, 96),
    (256, 64),
]

history_b = []
for edge_rank, mid_rank in bathtub_kv_schedule:
    print(f"\n  KV bathtub → edge={edge_rank} mid={mid_rank}...", end="", flush=True)
    compress_kv_bathtub(model, edge_rank, mid_rank, EDGE_WIDTH)
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre={pre_ppl:.1f}", end="", flush=True)

    finetune_kv(model, train_tokens, SEQ_LEN, device, steps=150)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  post={post_ppl:.1f}  [{text[:50]}]")

    history_b.append({
        "edge_rank": edge_rank, "mid_rank": mid_rank,
        "pre_ppl": pre_ppl, "post_ppl": post_ppl, "text": text[:80]
    })

    if post_ppl > teacher_ppl * 5:
        print(f"  ⚠ Stopping: quality >5x teacher")
        break

# Apply post-hoc stack
print(f"\n  Applying post-hoc stack (Q5-mid + MLP 90% + E6)...")
kv_only_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
apply_post_hoc_stack(model, EDGE_WIDTH)
stacked_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
stacked_text = generate_sample(model, tokenizer, PROMPT)
print(f"  KV-annealed only:   ppl={kv_only_ppl:.1f}")
print(f"  + post-hoc stack:   ppl={stacked_ppl:.1f}  [{stacked_text[:60]}]")
print(f"  Total Δ from teacher: {stacked_ppl - teacher_ppl:+.1f}")

result_b = {
    "variant": "B_bathtub_kv_anneal",
    "kv_schedule": bathtub_kv_schedule,
    "history": history_b,
    "kv_only_ppl": kv_only_ppl,
    "stacked_ppl": stacked_ppl,
    "stacked_text": stacked_text[:80],
    "delta": stacked_ppl - teacher_ppl,
}

del model; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STAGE 116 SUMMARY")
print(f"{'='*60}")
print(f"  Teacher: ppl={teacher_ppl:.1f}")
print(f"\n  A (global KV anneal → stack):")
print(f"    KV-only: ppl={result_a['kv_only_ppl']:.1f}")
print(f"    + Q5-mid + MLP 90% + E6: ppl={result_a['stacked_ppl']:.1f} (Δ={result_a['delta']:+.1f})")
print(f"\n  B (bathtub KV anneal → stack):")
print(f"    KV-only: ppl={result_b['kv_only_ppl']:.1f}")
print(f"    + Q5-mid + MLP 90% + E6: ppl={result_b['stacked_ppl']:.1f} (Δ={result_b['delta']:+.1f})")

if result_b['stacked_ppl'] < result_a['stacked_ppl']:
    print(f"\n  → BATHTUB KV wins by {result_a['stacked_ppl'] - result_b['stacked_ppl']:.1f} ppl")
else:
    print(f"\n  → GLOBAL KV wins by {result_b['stacked_ppl'] - result_a['stacked_ppl']:.1f} ppl")

# Additivity check
solo_stack_delta = 2.0  # from stage 115: Q5-mid + MLP 90% + E6
kv_only_delta_a = result_a['kv_only_ppl'] - teacher_ppl
total_delta_a = result_a['stacked_ppl'] - teacher_ppl
expected_additive_a = kv_only_delta_a + solo_stack_delta
print(f"\n  Additivity check (variant A):")
print(f"    KV-only Δ: {kv_only_delta_a:+.1f}")
print(f"    Stack-only Δ: {solo_stack_delta:+.1f}")
print(f"    Expected additive: {expected_additive_a:+.1f}")
print(f"    Actual combined: {total_delta_a:+.1f}")
ratio = total_delta_a / expected_additive_a if expected_additive_a > 0 else float('inf')
print(f"    Coupling ratio: {ratio:.2f}x (1.0 = perfectly additive)")

Path("results").mkdir(exist_ok=True)
with open("results/stage116_annealed_kv_stack.json", "w") as f:
    json.dump({
        "teacher_ppl": teacher_ppl, "L": L, "edge_width": EDGE_WIDTH,
        "variant_a": result_a, "variant_b": result_b,
    }, f, indent=2, default=str)
print(f"\nSaved results/stage116_annealed_kv_stack.json", flush=True)
