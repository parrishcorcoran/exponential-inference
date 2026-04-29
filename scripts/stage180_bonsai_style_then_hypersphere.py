"""Stage 180: Does hypersphere projection help/hurt/wash on top of
Bonsai-style per-group binary quantization?

We can't run Bonsai's actual 8B weights on Mac. But we can test the
PRINCIPLE on Qwen3-0.6B:
  T0: base Qwen3-0.6B FP
  T1: apply Bonsai-style per-128-group absmax binary quant
  T2: project T1 rows to unit norm, multiply by α=T1_row_norm to preserve
  T3: train α from T2 for some steps

If T2 ≈ T1: hypersphere + α with init=row_norm is mathematically
equivalent to keeping the per-group structure (but compressed to 1 α
per row instead of 32). The hypersphere step is lossless re-encoding.

If T2 << T1 (worse): hypersphere conflicts with per-group structure.
The 32 per-group scales were doing real work that 1 α can't capture.

If T2 >> T1 (better): unlikely, but would mean hypersphere fixes
something binary alone broke.

If T3 < T0 (training α improves below base): same magic as Stage 169 —
adding learnable α gives extra degrees of freedom.
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
TRAIN_STEPS = 500
GROUP_SIZE = 128
RESULTS_PATH = Path("results/stage180_bonsai_style_then_hypersphere.json")
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
        s = i * SEQ_LEN
        window = val_tokens[s:s + SEQ_LEN + 1]
        if len(window) < SEQ_LEN + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            losses.append(F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def bonsai_style_quantize(W, group_size=128):
    """Per-group symmetric absmax binary quantization, Bonsai-style.
    For each group of `group_size` weights along input dim, compute
    scale = mean(|w|), bias = -scale (symmetric), so weight values
    become {-scale, +scale} based on sign.
    """
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        # Pad with the actual group_size dividing — fall back to row scale
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    W_grouped = W.reshape(out_features, n_groups, group_size)
    # Per-group mean abs (symmetric absmax/absmean)
    scales = W_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)  # [out, n_groups, 1]
    W_q = torch.sign(W_grouped) * scales  # symmetric: -scale, +scale
    return W_q.reshape(out_features, in_features)


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

print("Loading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# ─── T0: base ───
T0 = lm_ce(model, val_tokens)
print(f"\nT0  base FP                                                 CE={T0:.4f}  ppl={math.exp(T0):.2f}")


# ─── T1: Bonsai-style per-group binary quantization ───
target_mods = []
original_W = {}
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    original_W[name] = mod.weight.data.clone()

print(f"\nApplying Bonsai-style per-{GROUP_SIZE}-group binary quant to {len(target_mods)} linears...")
for name, mod in target_mods:
    W = original_W[name].float()
    W_q = bonsai_style_quantize(W, GROUP_SIZE)
    mod.weight.data = W_q.to(mod.weight.dtype)

T1 = lm_ce(model, val_tokens)
print(f"T1  Bonsai-style per-128-group binary                       CE={T1:.4f}  ppl={math.exp(T1):.2f}  Δ={T1-T0:+.3f}")


# ─── T2: hypersphere projection on top ───
print(f"\nProjecting Bonsai-binary rows to unit norm + α=row_norm bridge...")

# Save Bonsai row norms (these become α inits)
bonsai_row_norms = {}
for name, mod in target_mods:
    rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clone()
    bonsai_row_norms[name] = rn

# Project to unit norm
for name, mod in target_mods:
    rn = bonsai_row_norms[name].to(mod.weight.dtype)
    mod.weight.data = mod.weight.data / rn.clamp(min=1e-8)


class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight
        self.bias = original_module.bias
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone().to(self.weight.device).to(torch.float32))
    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


parent_lookup = {}
for name, mod in model.named_modules():
    for child_name, child_mod in mod.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (mod, child_name)
alphas = {}
for full_name, mod in target_mods:
    parent, child_attr = parent_lookup[full_name]
    new_layer = AlphaLinear(mod, bonsai_row_norms[full_name])
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

T2 = lm_ce(model, val_tokens)
print(f"T2  Bonsai-binary + hypersphere projection + α=row_norm     CE={T2:.4f}  ppl={math.exp(T2):.2f}  Δ={T2-T0:+.3f}  Δ_vs_T1={T2-T1:+.3f}")


# ─── T3: train α ───
print(f"\nTraining α for {TRAIN_STEPS} steps...")
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
for step in range(TRAIN_STEPS):
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
        print(f"  step {step+1}/{TRAIN_STEPS}  loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

T3 = lm_ce(model, val_tokens)
print(f"T3  + α trained {TRAIN_STEPS} steps                                CE={T3:.4f}  ppl={math.exp(T3):.2f}  Δ={T3-T0:+.3f}  Δ_vs_T1={T3-T1:+.3f}")


# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY: Bonsai-style binary + hypersphere on Qwen3-0.6B\n{'='*70}")
results = [
    ("T0  base FP",                                      T0, 0.0,    None),
    ("T1  Bonsai-style per-128-group binary",            T1, T1-T0,  None),
    ("T2  + hypersphere projection + α=row_norm",        T2, T2-T0,  T2-T1),
    (f"T3  + α trained {TRAIN_STEPS} steps",                  T3, T3-T0,  T3-T1),
]
for label, ce, delta, delta_vs_t1 in results:
    ppl = math.exp(min(ce, 30))
    extra = f"  Δ_vs_T1={delta_vs_t1:+.3f}" if delta_vs_t1 is not None else ""
    print(f"  {label:<55} CE={ce:8.4f}  ppl={ppl:>14.2f}  Δ={delta:+.3f}{extra}")

print(f"\nINTERPRETATION:")
if abs(T2 - T1) < 0.05:
    print(f"  T2 ≈ T1 (Δ_vs_T1={T2-T1:+.4f}): hypersphere projection is LOSSLESS re-encoding of per-group binary.")
    print(f"  Per-group magnitude info is mathematically equivalent to row_norm × unit-norm decomposition.")
elif T2 < T1:
    print(f"  T2 < T1 (Δ_vs_T1={T2-T1:+.4f}): hypersphere on top IMPROVES Bonsai-style binary.")
    print(f"  Our recipe genuinely adds value — hypersphere shape is structurally better.")
else:
    print(f"  T2 > T1 (Δ_vs_T1={T2-T1:+.4f}): hypersphere on top DEGRADES Bonsai-style binary.")
    print(f"  Per-group structure conflicts with hypersphere projection.")
print(f"")
if T3 < T1:
    print(f"  T3 < T1 (Δ_vs_T1={T3-T1:+.4f}): training α below Bonsai-style binary recovers further.")
elif T3 < T0:
    print(f"  T3 < T0: full pipeline beats baseline — combined recipe works")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "group_size": GROUP_SIZE,
        "train_steps": TRAIN_STEPS,
        "T0_base_ce": float(T0),
        "T1_bonsai_style_binary_ce": float(T1),
        "T2_plus_hypersphere_alpha_init_ce": float(T2),
        "T3_plus_alpha_trained_ce": float(T3),
        "delta_T1_vs_T0": float(T1 - T0),
        "delta_T2_vs_T0": float(T2 - T0),
        "delta_T2_vs_T1": float(T2 - T1),
        "delta_T3_vs_T0": float(T3 - T0),
        "delta_T3_vs_T1": float(T3 - T1),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
