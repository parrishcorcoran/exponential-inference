"""Stage 167: post-hoc binary quantization baseline on Qwen3-0.6B.

Take base Qwen3-0.6B (FP), apply naive post-hoc binary quantization to
every linear weight (sign × per-channel scale = mean(|W_row|)), measure
val CE. Establishes our own "what does naive Q1 PTQ cost on Qwen3-0.6B"
baseline.

Compare against:
  - Bonsai-8B's 11% benchmark drop (different size, but same family)
  - Strix's nGPT τ=1.0 conversion (+0.031 nats, ~0.8% perplexity)
  - The eventual nGPT + binary compound

If naive Q1 on Qwen3-0.6B is, say, +5 nats CE (~17000% perplexity blowup),
that's the gap our recipe needs to close. If it's ~+1 nat (~170%
perplexity), it's already kind of usable.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 64
RESULTS_PATH = Path("results/stage167_post_hoc_binary_baseline.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32

print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False


def load_owt(tokenizer, max_tokens):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def lm_ce(model, val_tokens):
    losses = []
    for i in range(N_VAL_CHUNKS):
        start = i * SEQ_LEN
        window = val_tokens[start:start + SEQ_LEN + 1]
        if len(window) < SEQ_LEN + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="mean")
        losses.append(loss.item())
    return sum(losses) / len(losses)


print("Loading val tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * (N_VAL_CHUNKS + 5))

print("\nBaseline (no quantization)...")
base_ce = lm_ce(model, val_tokens)
print(f"  base val CE: {base_ce:.4f}  (ppl {math.exp(base_ce):.1f})")


def binary_quantize_inplace(model):
    """Replace each target Linear's weight with sign(W) * mean(|W_row|).
    Returns originals so we can restore."""
    saved = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        if not any(m in name for m in TARGET_NAMES): continue
        W = mod.weight.data
        saved[name] = W.clone()
        # Per-row scale: alpha[i] = mean(|W[i, :]|)
        alpha = W.abs().mean(dim=-1, keepdim=True)  # [out, 1]
        W_bin = torch.sign(W) * alpha  # broadcast scale per row
        mod.weight.data.copy_(W_bin)
    return saved


def restore(model, saved):
    for name, mod in model.named_modules():
        if name in saved:
            mod.weight.data.copy_(saved[name])


print("\nApplying naive post-hoc binary (sign × per-row alpha = mean(|W_row|))...")
saved = binary_quantize_inplace(model)
print(f"  Quantized {len(saved)} linears")
binary_ce = lm_ce(model, val_tokens)
print(f"  binary val CE: {binary_ce:.4f}  (ppl {math.exp(binary_ce):.1f})  delta {binary_ce - base_ce:+.3f}")

# Compare to nGPT-then-binary path: would expect MUCH LOWER delta after Strix's
# Stage 1 + Stage 2 + binary QAT pipeline lands.
restore(model, saved)
restored_ce = lm_ce(model, val_tokens)
print(f"\n  sanity check restored CE: {restored_ce:.4f}  (should match {base_ce:.4f})")

# Also try with global alpha (single scalar per linear matrix, not per row)
print("\nNow trying with GLOBAL alpha (single scalar per linear, like simplest Q1)...")
saved2 = {}
for name, mod in model.named_parameters():
    if "weight" not in name: continue
mods_to_quantize = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    mods_to_quantize.append((name, mod))

for name, mod in mods_to_quantize:
    W = mod.weight.data
    saved2[name] = W.clone()
    alpha = W.abs().mean()  # single scalar
    mod.weight.data.copy_(torch.sign(W) * alpha)

global_alpha_ce = lm_ce(model, val_tokens)
print(f"  global alpha val CE: {global_alpha_ce:.4f}  (ppl {math.exp(global_alpha_ce):.1f})  delta {global_alpha_ce - base_ce:+.3f}")
restore(model, saved2)

print(f"\n{'='*70}\nSUMMARY (Qwen3-0.6B post-hoc Q1 quantization)")
print(f"{'='*70}")
print(f"  Base FP val CE:                  {base_ce:.4f}  (ppl {math.exp(base_ce):.1f})")
print(f"  Binary, per-row alpha:           {binary_ce:.4f}  (ppl {math.exp(binary_ce):.1f})  Δ {binary_ce-base_ce:+.3f}")
print(f"  Binary, global alpha:            {global_alpha_ce:.4f}  (ppl {math.exp(global_alpha_ce):.1f})  Δ {global_alpha_ce-base_ce:+.3f}")
print(f"\n  vs Strix's nGPT τ=1.0 conversion: +0.031 nats (the conversion alone)")
print(f"  vs Bonsai-8B 1-bit on benchmarks: 11% avg drop (different model, real eval)")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "base_ce": float(base_ce),
        "base_ppl": float(math.exp(base_ce)),
        "binary_per_row_alpha": {
            "ce": float(binary_ce),
            "ppl": float(math.exp(binary_ce)),
            "delta_ce": float(binary_ce - base_ce),
        },
        "binary_global_alpha": {
            "ce": float(global_alpha_ce),
            "ppl": float(math.exp(global_alpha_ce)),
            "delta_ce": float(global_alpha_ce - base_ce),
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
