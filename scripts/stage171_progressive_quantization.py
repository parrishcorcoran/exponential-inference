"""Stage 171: Progressive precision anneal — drop one bit at a time.

Instead of FP → binary in one step (catastrophic), descend gradually:
  base FP → unit-norm (synth τ=1.0) → α trained
  → 256 levels → α refined
  → 16 levels → α refined
  → 4 levels → α refined
  → 3 levels (ternary) → α refined
  → 2 levels (binary) → α refined

User's intuition: each precision step is a small perturbation the model
can absorb. The catastrophic gap of FP→binary becomes a series of small,
recoverable gaps.

Quantization scheme: per-row absmax-symmetric. Each row's weights scaled
by max(|W_row|), quantized to N symmetric levels in [-1, 1], then scaled
back. Ternary uses BitNet-style threshold (|w| < gamma → 0, else sign).
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
STAGE2_STEPS = 500       # initial α training at FP unit-norm
REFINE_STEPS = 200       # per-quantization-step α refinement
RESULTS_PATH = Path("results/stage171_progressive_quantization.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
QUANT_SCHEDULE = [256, 16, 8, 4, 3, 2]  # levels per row, descending


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


def quantize_row_symmetric(W, n_levels):
    """Per-row absmax-symmetric quantization to n_levels distinct values.
    For 2: binary {-1, +1} × scale.
    For 3: ternary {-1, 0, +1} × scale (BitNet style — zero at low magnitude).
    For ≥4: linear quantization to n_levels symmetric levels.
    Scale per row preserves the absmax of original row."""
    row_max = W.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    W_scaled = W / row_max  # in [-1, 1]
    if n_levels == 2:
        W_q = torch.sign(W_scaled)
    elif n_levels == 3:
        # BitNet b1.58 style: threshold ~ mean(|W|), values below → 0
        gamma = W_scaled.abs().mean(dim=-1, keepdim=True)
        W_q = torch.where(W_scaled.abs() < gamma * 0.7,
                          torch.zeros_like(W_scaled),
                          torch.sign(W_scaled))
    else:
        # Linear quantization: map [-1, 1] to {-(n-1)/2, ..., (n-1)/2}, round
        half = (n_levels - 1) / 2
        W_q = torch.round(W_scaled * half) / half
    return W_q * row_max


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
print(f"\nT0  base FP                                   CE={T0:.4f}  ppl={math.exp(T0):.2f}")


# ─── Stage 1+2: unit-norm + α ───
target_mods = []
original_W = {}
row_norms = {}
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    original_W[name] = mod.weight.data.clone()
    row_norms[name] = mod.weight.data.norm(dim=-1, keepdim=True).clone()

print(f"\nProjecting weight rows to unit norm (synth τ=1.0)...")
for name, mod in target_mods:
    rn = row_norms[name].to(mod.weight.dtype)
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


def attach_alpha(model, target_list, row_norms):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    alphas = {}
    target_layers = {}
    for full_name, mod in target_list:
        parent, child_attr = parent_lookup[full_name]
        new_layer = AlphaLinear(mod, row_norms[full_name])
        setattr(parent, child_attr, new_layer)
        alphas[full_name] = new_layer.alpha
        target_layers[full_name] = new_layer
    return alphas, target_layers


print("Attaching α scales...")
alphas, target_layers = attach_alpha(model, target_mods, row_norms)
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_alpha(steps, label):
    opt = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)
    model.train()
    for step in range(steps):
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
            print(f"  [{label}] step {step+1}/{steps}  train loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)


it = iter_train()
print(f"\n--- Stage 1+2: train α at FP unit-norm for {STAGE2_STEPS} steps ---")
train_alpha(STAGE2_STEPS, "Stage 1+2")
T_unit_norm_alpha = lm_ce(model, val_tokens)
print(f"  CE={T_unit_norm_alpha:.4f}  ppl={math.exp(T_unit_norm_alpha):.2f}  Δ={T_unit_norm_alpha-T0:+.3f}")


# ─── Progressive quantization ───
trajectory = [
    {"label": "T0_base_FP",            "ce": float(T0),                  "delta": 0.0,  "n_levels": "FP"},
    {"label": "Tunit_alpha_trained",   "ce": float(T_unit_norm_alpha),   "delta": float(T_unit_norm_alpha - T0), "n_levels": "FP"},
]
print(f"\n{'='*70}")
print("Progressive quantization with α refinement between each step:")
print(f"{'='*70}")

# Start from current state (unit-norm + trained α) — keep that as substrate W
substrate_W = {}
for name, layer in target_layers.items():
    substrate_W[name] = layer.weight.data.clone()  # this is the unit-norm W

for n_levels in QUANT_SCHEDULE:
    bits = math.log2(n_levels)
    print(f"\n--- Quantize each row to {n_levels} levels ({bits:.2f} bits) ---")
    for name, layer in target_layers.items():
        layer.weight.data = quantize_row_symmetric(substrate_W[name], n_levels).to(layer.weight.dtype)

    pre_ce = lm_ce(model, val_tokens)
    print(f"  pre-refine  CE={pre_ce:.4f}  ppl={math.exp(min(pre_ce, 30)):.2f}  Δ={pre_ce-T0:+.3f}")
    train_alpha(REFINE_STEPS, f"{n_levels}lvl")
    post_ce = lm_ce(model, val_tokens)
    print(f"  post-refine CE={post_ce:.4f}  ppl={math.exp(min(post_ce, 30)):.2f}  Δ={post_ce-T0:+.3f}")
    trajectory.append({
        "label": f"T_{n_levels}lvl_pre_refine",
        "n_levels": n_levels,
        "bits": bits,
        "ce": float(pre_ce),
        "delta": float(pre_ce - T0),
        "phase": "pre-refine",
    })
    trajectory.append({
        "label": f"T_{n_levels}lvl_post_refine",
        "n_levels": n_levels,
        "bits": bits,
        "ce": float(post_ce),
        "delta": float(post_ce - T0),
        "phase": "post-refine",
    })


# ─── Summary ───
print(f"\n{'='*70}")
print("PROGRESSIVE QUANTIZATION TRAJECTORY")
print(f"{'='*70}")
print(f"  {'state':<28} {'bits':<8} {'CE':<10} {'ppl':<14} {'Δ vs base':<10}")
for t in trajectory:
    bits = t.get('bits', 'FP')
    bits_str = f"{bits:.2f}" if isinstance(bits, float) else str(bits)
    ppl = math.exp(min(t['ce'], 30))
    print(f"  {t['label']:<28} {bits_str:<8} {t['ce']:<10.4f} {ppl:<14.2f} {t['delta']:+10.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "stage2_steps": STAGE2_STEPS,
        "refine_steps": REFINE_STEPS,
        "quant_schedule": QUANT_SCHEDULE,
        "trajectory": trajectory,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
