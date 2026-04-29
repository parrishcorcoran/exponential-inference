"""Stage 169: Build a synthetic nGPT-mini, freeze weights, raise magnitude.

We don't have Strix's baked τ=1.0 checkpoint locally. Instead build a
synthetic version: project Qwen3-0.6B base weight rows to exact unit
norm (no fine-tune), then add a learnable per-channel α, freeze the
unit-norm body, train ONLY α on streaming OWT.

Test 1: does α recovery alone recover quality lost to projection?
Test 2: with α at near-recovered, apply binary quant to W (sign / sqrt(d))
        and continue training α — does it survive binary?

Stages measured:
  T0: base quality (FP)
  T1: projected to unit norm (no training, no α)
  T2: projected + α initialized to row norms (no training)
  T3: projected + α trained for N steps
  T4: projected + α + W made binary (no further training)
  T5: projected + α + W binary + further α training

If T3 → T0 (recovers), magnitude bridge works.
If T5 → T0 (recovers), the full 3-stage compound works in principle
even with synthetic Stage 1.
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
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR = 2e-5
GRAD_CLIP = 1.0
STAGE2_STEPS = 500
STAGE3_REFINE_STEPS = 500
RESULTS_PATH = Path("results/stage169_alpha_recovery_then_quant.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt(tokenizer, max_tokens, skip=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def lm_ce(model, val_tokens):
    losses = []
    model.eval()
    for i in range(N_VAL_CHUNKS):
        start = i * SEQ_LEN
        window = val_tokens[start:start + SEQ_LEN + 1]
        if len(window) < SEQ_LEN + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            losses.append(F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

print("\nLoading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# ─── T0: base ───
T0 = lm_ce(model, val_tokens)
print(f"\nT0  base FP                                                  CE={T0:.4f}  ppl={math.exp(T0):.2f}")


# ─── Find target linears, save originals, capture row norms (= α_init candidate) ───
target_mods = []
original_W = {}
row_norms = {}  # original ||W_row|| → α_init
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    original_W[name] = mod.weight.data.clone()
    row_norms[name] = mod.weight.data.norm(dim=-1, keepdim=True).clone()  # [out, 1]
print(f"\nTargeted {len(target_mods)} linears.")


# ─── T1: project rows to unit norm, no α ───
print("\nProjecting weight rows to unit norm (synthetic τ=1.0)...")
for name, mod in target_mods:
    rn = row_norms[name].to(mod.weight.dtype)
    mod.weight.data = mod.weight.data / rn.clamp(min=1e-8)

T1 = lm_ce(model, val_tokens)
print(f"T1  unit-norm rows, no α                                     CE={T1:.4f}  ppl={math.exp(T1):.2f}  Δ={T1-T0:+.3f}")


# ─── T2: add α (per-channel scale) initialized to original row norms ───
# y = α ⊙ (W_unit @ x) + b   with α[i] = original_row_norm[i] should reproduce base
# Implement by patching forward of each target Linear
class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight  # share
        self.bias = original_module.bias
        # alpha is learnable, shape [out_features]
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone().to(self.weight.device).to(torch.float32))

    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


def attach_alpha(model, target_names, row_norms):
    """Replace target Linear forwards with alpha-multiplied versions."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    alphas = {}
    for full_name, mod in target_names:
        parent, child_attr = parent_lookup[full_name]
        alpha_init = row_norms[full_name]
        new_layer = AlphaLinear(mod, alpha_init)
        setattr(parent, child_attr, new_layer)
        alphas[full_name] = new_layer.alpha
    return alphas


print("\nAttaching α scales to all target linears...")
alphas = attach_alpha(model, target_mods, row_norms)
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  trainable α params: {trainable:,} / {total:,}  ({100*trainable/total:.4f}%)")

T2 = lm_ce(model, val_tokens)
print(f"T2  unit-norm + α=row_norm (no training)                     CE={T2:.4f}  ppl={math.exp(T2):.2f}  Δ={T2-T0:+.3f}")


# ─── T3: train α only ───
print(f"\nTraining α for {STAGE2_STEPS} steps...")
opt = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


it = iter_train()
model.train()
for step in range(STAGE2_STEPS):
    opt.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = model(ids[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_([a for a in alphas.values()], GRAD_CLIP)
    opt.step()
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{STAGE2_STEPS}  train loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

T3 = lm_ce(model, val_tokens)
print(f"T3  unit-norm + α (trained {STAGE2_STEPS} steps)                  CE={T3:.4f}  ppl={math.exp(T3):.2f}  Δ={T3-T0:+.3f}")


# ─── T4: binary quantize W on top, no further training ───
print("\nApplying binary quant: W ← sign(W) / sqrt(d_in)...")
for name, mod in target_mods:
    if hasattr(mod, "weight"):
        w = mod.weight.data
        d_in = w.shape[-1]
        w_bin = torch.sign(w) / math.sqrt(d_in)
        mod.weight.data = w_bin

T4 = lm_ce(model, val_tokens)
print(f"T4  binary W + α (no further training)                       CE={T4:.4f}  ppl={math.exp(T4):.2f}  Δ={T4-T0:+.3f}")


# ─── T5: refine α to compensate for binary ───
print(f"\nRefining α for {STAGE3_REFINE_STEPS} more steps with binary W...")
opt2 = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)
model.train()
for step in range(STAGE3_REFINE_STEPS):
    opt2.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = model(ids[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_([a for a in alphas.values()], GRAD_CLIP)
    opt2.step()
    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{STAGE3_REFINE_STEPS}  train loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

T5 = lm_ce(model, val_tokens)
print(f"T5  binary W + α (refined {STAGE3_REFINE_STEPS} steps)              CE={T5:.4f}  ppl={math.exp(T5):.2f}  Δ={T5-T0:+.3f}")


# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY (Qwen3-0.6B, synthetic τ=1.0 + α + binary)\n{'='*70}")
results = [
    ("T0  base FP",                          T0),
    ("T1  unit-norm, no α",                  T1),
    ("T2  unit-norm + α=row_norm (no train)", T2),
    (f"T3  unit-norm + α (trained {STAGE2_STEPS}s)",   T3),
    ("T4  binary W + α (no train)",          T4),
    (f"T5  binary W + α (refined {STAGE3_REFINE_STEPS}s)", T5),
]
for label, ce in results:
    ppl = math.exp(min(ce, 30))
    print(f"  {label:<50} CE={ce:8.4f}  ppl={ppl:>14.2f}  Δ={ce-T0:+.3f}")

print(f"\n  Naive post-hoc binary baseline (Stage 167):              Δ=+13.34 (per-row)")
print(f"  Bonsai 1-bit on Qwen3-8B base (different size):           ~Δ=+0.7 (~11% benchmarks)")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "stage2_steps": STAGE2_STEPS,
        "stage3_refine_steps": STAGE3_REFINE_STEPS,
        "T0_base_ce": float(T0),
        "T1_unit_norm_no_alpha_ce": float(T1),
        "T2_unit_norm_alpha_init_ce": float(T2),
        "T3_alpha_trained_ce": float(T3),
        "T4_binary_no_refine_ce": float(T4),
        "T5_binary_alpha_refined_ce": float(T5),
        "delta_T1": float(T1 - T0),
        "delta_T2": float(T2 - T0),
        "delta_T3": float(T3 - T0),
        "delta_T4": float(T4 - T0),
        "delta_T5": float(T5 - T0),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
