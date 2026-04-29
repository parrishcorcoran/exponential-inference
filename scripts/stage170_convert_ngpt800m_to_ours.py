"""Stage 170: Convert nGPT_800m to OUR full-normalization shape.

The community nGPT (p2o6e100/nGPT_800m) normalizes input projections only:
  qkv_proj, gate_proj, up_proj, embed, lm_head:  unit norm
  o_proj:    mean ~0.96, CV ~0.30  (NOT normalized)
  down_proj: mean ~2.0,  CV ~0.075 (NOT normalized)

OUR recipe normalizes everything including o_proj and down_proj. Test:
project these two to unit norm in nGPT_800m, add α scales to recover,
train α only. Does it improve, hurt, or wash?

This directly tests whether our "normalize everything" extension to nGPT
provides benefit on a model that's ALREADY in nGPT-shape.

Stages measured:
  T0: nGPT_800m as-released (asymmetric normalization)
  T1: + o_proj and down_proj projected to unit norm (no α, no train)
  T2: + α scales attached, init=row_norm of original (no train)
  T3: + α trained for N steps
  T4 (optional): the FULL normalize ours-style + apply binary to W
  T5 (optional): refine α post-binary
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "p2o6e100/nGPT_800m"
SEQ_LEN = 128
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR = 2e-5
GRAD_CLIP = 1.0
STAGE2_STEPS = 500
RESULTS_PATH = Path("results/stage170_convert_ngpt800m_to_ours.json")
# nGPT already has qkv/gate/up at unit norm; we re-normalize o_proj and down_proj
TARGET_NAMES = ("o_proj", "down_proj")


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


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

# Use the model's own tokenizer for consistency
print("\nLoading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# ─── T0: baseline (nGPT_800m as-is) ───
T0 = lm_ce(model, val_tokens)
print(f"\nT0  nGPT_800m as-released                                  CE={T0:.4f}  ppl={math.exp(T0):.2f}")


# ─── Find o_proj and down_proj layers ───
target_mods = []
original_W = {}
row_norms = {}
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    original_W[name] = mod.weight.data.clone()
    row_norms[name] = mod.weight.data.norm(dim=-1, keepdim=True).clone()

print(f"\nTargeted {len(target_mods)} linears (o_proj + down_proj only).")
# Show stats on what we're about to project
print(f"\n{'name':<55} {'mean':<8} {'std':<8} {'cv':<8}")
for n, m in target_mods[:6]:  # first few
    rn = row_norms[n].squeeze(-1)
    print(f"  {n:<55} {rn.mean().item():<8.3f} {rn.std().item():<8.3f} {(rn.std()/rn.mean()).item():<8.4f}")
print(f"  ... ({len(target_mods) - 6} more)")


# ─── T1: project to unit norm, no α ───
print("\nProjecting o_proj and down_proj rows to unit norm...")
for name, mod in target_mods:
    rn = row_norms[name].to(mod.weight.dtype)
    mod.weight.data = mod.weight.data / rn.clamp(min=1e-8)

T1 = lm_ce(model, val_tokens)
print(f"T1  + o_proj & down_proj projected (no α)                  CE={T1:.4f}  ppl={math.exp(T1):.2f}  Δ={T1-T0:+.3f}")


# ─── T2: add α with initialization = original row norms ───
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


def attach_alpha(model, target_names_list, row_norms):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    alphas = {}
    for full_name, mod in target_names_list:
        if full_name not in parent_lookup:
            print(f"  WARN: {full_name} not in parent_lookup")
            continue
        parent, child_attr = parent_lookup[full_name]
        new_layer = AlphaLinear(mod, row_norms[full_name])
        setattr(parent, child_attr, new_layer)
        alphas[full_name] = new_layer.alpha
    return alphas


print("\nAttaching α scales (initialized to original row norms)...")
alphas = attach_alpha(model, target_mods, row_norms)
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  trainable α params: {trainable:,} / {total:,}")

T2 = lm_ce(model, val_tokens)
print(f"T2  + α=row_norm (no training)                             CE={T2:.4f}  ppl={math.exp(T2):.2f}  Δ={T2-T0:+.3f}")


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
print(f"T3  + α trained {STAGE2_STEPS} steps                              CE={T3:.4f}  ppl={math.exp(T3):.2f}  Δ={T3-T0:+.3f}")


# ─── Verify the geometry is now FULL unit-norm ───
print("\nVerifying final geometry of o_proj and down_proj...")
final_cv_o = []
final_cv_d = []
for name, mod in target_mods:
    if hasattr(mod, "weight"):
        w = mod.weight.data.float()
    else:
        w = mod.weight.float() if hasattr(mod, 'weight') else None
    # need to access weight from AlphaLinear
    for full_name, alpha_layer in [(n, mod) for n, mod in model.named_modules() if isinstance(mod, AlphaLinear)]:
        if full_name == name:
            w = alpha_layer.weight.data.float()
            break
    if w is not None and w.ndim == 2:
        rn = w.norm(dim=-1)
        cv = (rn.std() / rn.mean()).item()
        if "o_proj" in name: final_cv_o.append(cv)
        elif "down_proj" in name: final_cv_d.append(cv)

print(f"  Final o_proj CV (target ~0.0): mean across layers = {sum(final_cv_o)/max(len(final_cv_o),1):.4f}")
print(f"  Final down_proj CV (target ~0.0): mean across layers = {sum(final_cv_d)/max(len(final_cv_d),1):.4f}")


# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY: nGPT_800m → OURS (full normalization)\n{'='*70}")
print(f"  Started: nGPT_800m with input-only normalization")
print(f"  Goal:    extend normalization to o_proj and down_proj (our recipe)")
print(f"  Result:")
results = [
    ("T0  nGPT_800m as-released",                T0),
    ("T1  o_proj+down_proj projected, no α",     T1),
    ("T2  + α=row_norm (no train)",              T2),
    (f"T3  + α trained {STAGE2_STEPS} steps",            T3),
]
for label, ce in results:
    ppl = math.exp(min(ce, 30))
    print(f"  {label:<50} CE={ce:8.4f}  ppl={ppl:>14.2f}  Δ={ce-T0:+.3f}")

print(f"\n  Headline: extending nGPT to FULL normalization changes quality by Δ={T3-T0:+.4f} nats after α recovery.")
print(f"  If Δ ≈ 0: full normalization is free (validates our recipe extension)")
print(f"  If Δ < 0: full normalization IMPROVES nGPT (extension is strict win)")
print(f"  If Δ > 0: full normalization costs vs nGPT (their input-only is optimal)")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "stage2_steps": STAGE2_STEPS,
        "T0_ngpt_baseline_ce": float(T0),
        "T1_projected_no_alpha_ce": float(T1),
        "T2_alpha_init_no_train_ce": float(T2),
        "T3_alpha_trained_ce": float(T3),
        "delta_T1": float(T1 - T0),
        "delta_T2": float(T2 - T0),
        "delta_T3": float(T3 - T0),
        "final_o_proj_cv_avg": float(sum(final_cv_o)/max(len(final_cv_o),1)),
        "final_down_proj_cv_avg": float(sum(final_cv_d)/max(len(final_cv_d),1)),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
