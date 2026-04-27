"""Magnitude annealing: slowly reduce weight magnitude, let the model find balance.

Scale all weights by 0.99 per step. Fine-tune norms after each step.
Thermostat controlled. Inverse-law FT scaling: more FT when compression
gets harder.

FT steps = base_steps / (1 - ppl_ratio)
  where ppl_ratio = current_ppl / thermostat_limit
  - Easy (ppl_ratio ~0.5): FT = 200 steps
  - Hard (ppl_ratio ~0.9): FT = 2000 steps
  - Very hard (ppl_ratio ~0.95): FT = 4000 steps

This gives the model exactly as much training as it needs at each
compression level. No wasted compute on easy steps, maximum effort
on hard steps.
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
CHECKPOINT = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
SQUEEZE = 0.99  # 1% magnitude reduction per step
MAX_STEPS = 200
BASE_FT_STEPS = 200
THERMOSTAT = 2.0  # 2x baseline

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "Machine learning models are trained by",
    "The French Revolution began in 1789 when",
]

def load_owt_tokens(tokenizer, max_tokens, skip_tokens=0):
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

def iter_batches(tokens, seq_len, device, n=999):
    import random
    idxs = list(range((len(tokens)-1)//seq_len))
    random.shuffle(idxs)
    for i in idxs[:n]:
        s = i * seq_len
        w = tokens[s:s+seq_len+1]
        if len(w) < seq_len+1: continue
        yield torch.tensor([w], dtype=torch.long, device=device)

@torch.no_grad()
def eval_ppl(model, val_tokens, seq_len, device, n=15):
    model.eval()
    total = 0; c = 0
    for batch in iter_batches(val_tokens, seq_len, device, n):
        logits = model(batch[:, :-1], use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1)).item()
        c += 1
    return math.exp(total / max(c, 1))

def finetune_norms(model, train_tokens, seq_len, device, steps):
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable: return
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
    model.train(); step = 0
    for batch in iter_batches(train_tokens, seq_len, device, steps):
        if step >= steps: break
        opt.zero_grad()
        logits = model(batch[:, :-1], use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); step += 1
    del opt
    for p in model.parameters(): p.requires_grad = False
    torch.cuda.empty_cache()

print("=" * 60)
print("MAGNITUDE ANNEALING — inverse law FT scaling")
print(f"  Squeeze: {SQUEEZE}/step | Thermostat: {THERMOSTAT}x")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print("\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
thermostat_limit = baseline_ppl * THERMOSTAT
print(f"  Baseline: {baseline_ppl:.1f}")
print(f"  Thermostat: {thermostat_limit:.1f}", flush=True)

# Track total magnitude
def total_magnitude(model):
    total = 0
    count = 0
    for name, p in model.named_parameters():
        if "norm" not in name.lower() and "embed" not in name.lower():
            total += p.data.float().abs().mean().item()
            count += 1
    return total / max(count, 1)

initial_mag = total_magnitude(model)
print(f"  Initial avg magnitude: {initial_mag:.6f}", flush=True)

history = []

for step in range(1, MAX_STEPS + 1):
    # Scale all non-norm, non-embed weights by squeeze factor
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name.lower() and "embed" not in name.lower() and "lm_head" not in name.lower():
                p.data *= SQUEEZE

    # Eval
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)

    # Inverse-law FT scaling
    ppl_ratio = pre_ppl / thermostat_limit
    ppl_ratio = min(ppl_ratio, 0.98)  # cap to avoid division by zero
    ft_steps = int(BASE_FT_STEPS / max(1 - ppl_ratio, 0.05))
    ft_steps = min(ft_steps, 5000)  # cap at 5000

    # Fine-tune norms
    finetune_norms(model, train_tokens, SEQ_LEN, device, ft_steps)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    current_mag = total_magnitude(model)
    mag_ratio = current_mag / initial_mag

    if step % 5 == 0 or step <= 3:
        ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  step {step:>3}: pre={pre_ppl:.1f} → post={post_ppl:.1f} | mag={mag_ratio:.3f} | ft={ft_steps} | [{text[:50]}]", flush=True)
    else:
        print(f"  step {step:>3}: pre={pre_ppl:.1f} → post={post_ppl:.1f} | mag={mag_ratio:.3f} | ft={ft_steps}", flush=True)

    history.append({
        "step": step, "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
        "magnitude_ratio": round(mag_ratio, 4), "ft_steps": ft_steps,
    })

    if post_ppl > thermostat_limit:
        print(f"\n  ⚠ THERMOSTAT: {post_ppl:.1f} > {thermostat_limit:.1f}", flush=True)
        break

    # Checkpoint every 20 steps
    if step % 20 == 0:
        save_path = Path(f"checkpoints/pipeline/magnitude_s{step}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# Final
print(f"\n{'='*60}")
print(f"MAGNITUDE ANNEAL COMPLETE")
print(f"  Baseline: {baseline_ppl:.1f}")
if history:
    print(f"  Final: {history[-1]['post_ppl']:.1f}")
    print(f"  Magnitude: {history[-1]['magnitude_ratio']:.3f} of original")
    print(f"  Steps: {len(history)}")

# Coherence
print(f"\nCoherence:")
for p in PROMPTS:
    ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{p[:30]}...] → {text[:60]}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/magnitude_anneal.json", "w") as f:
    json.dump({"baseline": baseline_ppl, "history": history}, f, indent=2)
print(f"\nSaved results/magnitude_anneal.json", flush=True)

del model; torch.cuda.empty_cache()
