"""Magnitude annealing on 4B — streaming OWT to avoid overfit.

Slowly reduce weight magnitude (×0.99 per step).
Inverse-law FT: more training when compression gets harder.
Streaming OpenWebText — fresh tokens every batch, no overfit.

Each step:
  1. Scale all weights by 0.99
  2. Compute ppl_ratio = current / thermostat
  3. FT steps = base / (1 - ppl_ratio)  [inverse law]
  4. Fine-tune norms on FRESH streaming tokens
  5. Eval on held-out set
  6. Thermostat check
"""
import torch
import torch.nn.functional as F
import math
import json
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda"
MODEL = "Qwen/Qwen3-4B"
SEQ_LEN = 256
SQUEEZE = 0.99
MAX_STEPS = 200
BASE_FT = 200
THERMOSTAT = 2.0

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "Machine learning models are trained by",
    "The French Revolution began in 1789 when",
]

print("=" * 60)
print("MAGNITUDE ANNEALING — 4B, streaming OWT, inverse-law FT")
print(f"  Squeeze: {SQUEEZE}/step | Thermostat: {THERMOSTAT}x")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Streaming train — fresh tokens every time, no overfit
print("\nSetting up streaming OWT...", flush=True)
train_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
train_iter = iter(train_ds)

# Fixed val set for consistent eval
val_toks = []
val_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
skip = 0
for item in val_ds:
    skip += 1
    if skip < 50000: continue  # skip first 50K articles for val
    t = item.get("text", "")
    if not t.strip(): continue
    val_toks.extend(tokenizer.encode(t, add_special_tokens=False))
    if len(val_toks) >= SEQ_LEN * 200: break
val_tokens = val_toks[:SEQ_LEN * 200]
print(f"  Val: {len(val_tokens)} tokens (fixed)", flush=True)

def get_fresh_batch(train_iter, tokenizer, seq_len, device):
    """Get one fresh training batch from streaming OWT."""
    toks = []
    while len(toks) < seq_len + 1:
        try:
            item = next(train_iter)
        except StopIteration:
            train_iter = iter(load_dataset("Skylion007/openwebtext", split="train", streaming=True))
            item = next(train_iter)
        t = item.get("text", "")
        if t.strip():
            toks.extend(tokenizer.encode(t, add_special_tokens=False))
    toks = toks[:seq_len + 1]
    t = torch.tensor([toks], dtype=torch.long, device=device)
    return t[:, :-1], t[:, 1:], train_iter

def eval_ppl(model, val_tokens, seq_len, device, n=15):
    model.eval()
    total = 0; c = 0
    import random
    idxs = list(range((len(val_tokens)-1)//seq_len))
    random.shuffle(idxs)
    with torch.no_grad():
        for i in idxs[:n]:
            s = i * seq_len
            w = val_tokens[s:s+seq_len+1]
            if len(w) < 2: continue
            t = torch.tensor([w], dtype=torch.long, device=device)
            logits = model(t[:, :-1], use_cache=False).logits
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), t[:, 1:].reshape(-1)).item()
            c += 1
    return math.exp(total / max(c, 1))

print(f"\nLoading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
thermostat_limit = baseline_ppl * THERMOSTAT
print(f"  Baseline: {baseline_ppl:.1f}")
print(f"  Thermostat: {thermostat_limit:.1f}", flush=True)

# Initial magnitude
def avg_magnitude(model):
    total = 0; count = 0
    for name, p in model.named_parameters():
        if "norm" not in name.lower() and "embed" not in name.lower() and "lm_head" not in name.lower():
            total += p.data.float().abs().mean().item()
            count += 1
    return total / max(count, 1)

init_mag = avg_magnitude(model)
print(f"  Initial magnitude: {init_mag:.6f}", flush=True)

history = []

for step in range(1, MAX_STEPS + 1):
    # Scale weights
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name.lower() and "embed" not in name.lower() and "lm_head" not in name.lower():
                p.data *= SQUEEZE

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)

    # Inverse-law FT
    ppl_ratio = min(pre_ppl / thermostat_limit, 0.98)
    ft_steps = int(BASE_FT / max(1 - ppl_ratio, 0.05))
    ft_steps = min(ft_steps, 5000)

    # Fine-tune on STREAMING tokens (fresh each time)
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)

    if trainable:
        opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
        model.train()
        for ft in range(ft_steps):
            inp, tgt, train_iter = get_fresh_batch(train_iter, tokenizer, SEQ_LEN, device)
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        del opt
        for p in model.parameters(): p.requires_grad = False
        torch.cuda.empty_cache()

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    mag = avg_magnitude(model) / init_mag

    if step % 5 == 0 or step <= 3:
        ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  step {step:>3}: pre={pre_ppl:.1f}→post={post_ppl:.1f} | mag={mag:.3f} | ft={ft_steps} | [{text[:50]}]", flush=True)
    else:
        print(f"  step {step:>3}: pre={pre_ppl:.1f}→post={post_ppl:.1f} | mag={mag:.3f} | ft={ft_steps}", flush=True)

    history.append({
        "step": step, "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
        "magnitude": round(mag, 4), "ft_steps": ft_steps,
    })

    if post_ppl > thermostat_limit:
        print(f"\n  THERMOSTAT: {post_ppl:.1f} > {thermostat_limit:.1f}", flush=True)
        break

    if step % 20 == 0:
        save_path = Path(f"checkpoints/pipeline/mag_4b_s{step}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# Final
print(f"\n{'='*60}")
print(f"MAGNITUDE ANNEAL COMPLETE")
print(f"  Baseline: {baseline_ppl:.1f}")
if history:
    print(f"  Final ppl: {history[-1]['post_ppl']:.1f}")
    print(f"  Final magnitude: {history[-1]['magnitude']:.3f} of original")
    print(f"  Steps: {len(history)}")

print(f"\nCoherence:")
for p in PROMPTS:
    ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{p[:30]}...] → {text[:60]}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/magnitude_anneal_4b.json", "w") as f:
    json.dump({"baseline": baseline_ppl, "history": history}, f, indent=2)
print(f"\nSaved results/magnitude_anneal_4b.json", flush=True)

del model; torch.cuda.empty_cache()
