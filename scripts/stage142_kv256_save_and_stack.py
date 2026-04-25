"""Stage 142 (Strix) — Save KV-256 model, then stack per-layer K + uniform V + K bits + V bits.

Step 1: Anneal KV to rank 256 (3 steps: 768→512→256) with FT. Save model.
Step 2: On saved model, measure per-layer K rank sensitivity to find the floor per layer.
Step 3: On saved model, reduce V uniformly and find floor.
Step 4: Stack Q4 on K cache.
Step 5: Stack Q4 on V cache.

Each step saves the model for the next.
"""
import torch
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


def compress_kv_all(model, rank):
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            W = proj.weight.data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = max(min(rank, len(S)), 1)
            proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def compress_k_layer(layer, rank):
    """SVD truncate k_proj only on one layer."""
    proj = layer.self_attn.k_proj
    W = proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(min(rank, len(S)), 1)
    proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)


def compress_v_all(model, rank):
    """SVD truncate v_proj only on all layers."""
    for layer in model.model.layers:
        proj = layer.self_attn.v_proj
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(min(rank, len(S)), 1)
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def finetune_kv_norms(model, train_tokens, seq_len, device, steps=200, lr=5e-5):
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable: return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train(); step = 0
    for inp, tgt in iter_batches(train_tokens, seq_len, 1, device):
        if step >= steps: break
        opt.zero_grad()
        loss = F.cross_entropy(
            model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(),
            tgt.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); step += 1
    del opt
    for p in model.parameters(): p.requires_grad = False
    torch.cuda.empty_cache()


def quantize_kv_cache_inplace(model, k_bits, v_bits):
    """Quantize k_proj and v_proj weights to simulate cache quantization."""
    for layer in model.model.layers:
        for name, bits in [("k_proj", k_bits), ("v_proj", v_bits)]:
            if bits >= 16: continue
            w = getattr(layer.self_attn, name).weight
            W = w.data.float()
            half = 2 ** (bits - 1)
            scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
            w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"

print("=" * 60)
print("STAGE 142 (Strix) — KV-256 SAVE + STACK ALL CACHE LEVERS")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

results = {}

# ═══════════════════════════════════════════════════════
# STEP 1: Anneal to KV-256 and save
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1: Anneal to KV-256 and save")
print(f"{'='*60}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Baseline: ppl={baseline_ppl:.1f}", flush=True)
results["baseline_ppl"] = baseline_ppl

for rank in [768, 512, 384, 256]:
    print(f"  KV → {rank}...", end="", flush=True)
    compress_kv_all(model, rank)
    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  ppl={ppl:.1f}", flush=True)

kv256_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
kv256_text = generate_sample(model, tokenizer, PROMPT)
print(f"\n  KV-256 saved: ppl={kv256_ppl:.1f}  [{kv256_text[:60]}]", flush=True)
results["kv256_ppl"] = kv256_ppl

save_path = Path("checkpoints/qwen_halo/kv256_base")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  Saved to {save_path}", flush=True)
del model; gc.collect(); torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# STEP 2: Per-layer K rank sensitivity
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 2: Per-layer K rank sensitivity (on KV-256 base)")
print(f"{'='*60}")

model = AutoModelForCausalLM.from_pretrained(
    str(save_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

# For each layer, compress K to rank 64 and measure damage
# (K is already at 256 from step 1)
k_sensitivity = []
for i in range(L):
    # Save original
    k_orig = model.model.layers[i].self_attn.k_proj.weight.data.clone()

    # Compress K to rank 64 (from current ~256)
    compress_k_layer(model.model.layers[i], 64)
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=5)
    damage = ppl - kv256_ppl

    # Restore
    model.model.layers[i].self_attn.k_proj.weight.data = k_orig

    k_sensitivity.append({"layer": i, "ppl": round(ppl, 2), "damage": round(damage, 2)})
    if i % 4 == 0 or i == L - 1:
        print(f"  L{i:>2} K→64: ppl={ppl:.1f} damage={damage:+.2f}", flush=True)

results["k_per_layer_sensitivity"] = k_sensitivity

# Find which layers are cheapest to compress further
sorted_by_damage = sorted(k_sensitivity, key=lambda x: x["damage"])
cheap_layers = [s["layer"] for s in sorted_by_damage[:10]]
expensive_layers = [s["layer"] for s in sorted_by_damage[-5:]]
print(f"\n  Cheapest to compress: {cheap_layers}")
print(f"  Most expensive:      {expensive_layers}")

# ═══════════════════════════════════════════════════════
# STEP 3: V rank uniform reduction
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 3: V rank uniform reduction (on KV-256 base)")
print(f"{'='*60}")

# Reload fresh KV-256
del model; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    str(save_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

v_results = []
for v_rank in [192, 128, 96, 64]:
    print(f"  V → {v_rank}...", end="", flush=True)
    compress_v_all(model, v_rank)
    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    print(f"  ppl={ppl:.1f}  [{text[:50]}]", flush=True)
    v_results.append({"v_rank": v_rank, "ppl": round(ppl, 2), "text": text[:80]})

results["v_uniform_anneal"] = v_results

# ═══════════════════════════════════════════════════════
# STEP 4: K bits (Q4) on KV-256 base
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 4: K quantization (on KV-256 base)")
print(f"{'='*60}")

del model; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    str(save_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

for k_bits in [8, 6, 4]:
    quantize_kv_cache_inplace(model, k_bits=k_bits, v_bits=16)
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  K Q{k_bits}: ppl={ppl:.1f} (Δ={ppl-kv256_ppl:+.1f})", flush=True)
    results[f"k_q{k_bits}"] = {"ppl": round(ppl, 2), "delta": round(ppl - kv256_ppl, 2)}

    # Fine-tune and remeasure
    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)
    ppl_ft = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  K Q{k_bits} + FT: ppl={ppl_ft:.1f} (Δ={ppl_ft-kv256_ppl:+.1f})", flush=True)
    results[f"k_q{k_bits}_ft"] = {"ppl": round(ppl_ft, 2), "delta": round(ppl_ft - kv256_ppl, 2)}

    # Reload for next test
    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        str(save_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

# ═══════════════════════════════════════════════════════
# STEP 5: V bits (Q4) on KV-256 base
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 5: V quantization (on KV-256 base)")
print(f"{'='*60}")

for v_bits in [8, 6, 4]:
    quantize_kv_cache_inplace(model, k_bits=16, v_bits=v_bits)
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  V Q{v_bits}: ppl={ppl:.1f} (Δ={ppl-kv256_ppl:+.1f})", flush=True)
    results[f"v_q{v_bits}"] = {"ppl": round(ppl, 2), "delta": round(ppl - kv256_ppl, 2)}

    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)
    ppl_ft = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  V Q{v_bits} + FT: ppl={ppl_ft:.1f} (Δ={ppl_ft-kv256_ppl:+.1f})", flush=True)
    results[f"v_q{v_bits}_ft"] = {"ppl": round(ppl_ft, 2), "delta": round(ppl_ft - kv256_ppl, 2)}

    del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        str(save_path), dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STAGE 142 COMPLETE — ALL CACHE LEVERS MEASURED")
print(f"{'='*60}")
print(f"  Baseline: {baseline_ppl:.1f}")
print(f"  KV-256 base: {kv256_ppl:.1f}")
print(f"\n  Per-layer K sensitivity: cheapest={cheap_layers[:5]}, expensive={expensive_layers}")
print(f"  V uniform anneal: {v_results}")
for k in ["k_q8_ft", "k_q6_ft", "k_q4_ft"]:
    if k in results: print(f"  {k}: ppl={results[k]['ppl']}")
for k in ["v_q8_ft", "v_q6_ft", "v_q4_ft"]:
    if k in results: print(f"  {k}: ppl={results[k]['ppl']}")

Path("results").mkdir(exist_ok=True)
with open("results/stage142_kv_stack.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/stage142_kv_stack.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
