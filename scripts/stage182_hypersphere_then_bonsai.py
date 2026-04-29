"""Stage 182: ABA pair to Stage 180. Hypersphere FIRST, then Bonsai-style
binary on top.

Stage 180:  base → Bonsai-binary → hypersphere + α → train α
Stage 182:  base → hypersphere + α → train α → Bonsai-binary → train α

If both paths converge to similar quality: encoding is order-independent,
the two methods capture equivalent magnitude info.

If they differ: order matters. Tells us which structure is "primary" and
which is corrective.

Stage 169 already established the first half (T3 = Δ -0.121 nats).
This stage continues from there: apply Bonsai-style per-group binary to
the unit-norm + α-trained substrate, measure damage, train α more.
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
PRE_BINARY_STEPS = 500    # alpha training before binary (replicates Stage 169 T3)
POST_BINARY_STEPS = 500   # alpha training after binary
GROUP_SIZE = 128
RESULTS_PATH = Path("results/stage182_hypersphere_then_bonsai.json")
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
    """Per-group symmetric absmax binary."""
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    W_grouped = W.reshape(out_features, n_groups, group_size)
    scales = W_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    W_q = torch.sign(W_grouped) * scales
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


# ─── Stage 1: project to unit norm ───
target_mods = []
row_norms_init = {}
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))
    rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clone()
    row_norms_init[name] = rn

print(f"\nProjecting to unit norm + attaching α=row_norm...")
for name, mod in target_mods:
    rn = row_norms_init[name].to(mod.weight.dtype)
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
target_layers = {}
for full_name, mod in target_mods:
    parent, child_attr = parent_lookup[full_name]
    new_layer = AlphaLinear(mod, row_norms_init[full_name])
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha
    target_layers[full_name] = new_layer

for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

T1 = lm_ce(model, val_tokens)
print(f"T1  unit-norm + α=row_norm (no train)                       CE={T1:.4f}  ppl={math.exp(T1):.2f}  Δ={T1-T0:+.3f}")


# ─── T2: train α 500 steps ───
opt = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


it = iter_train()
print(f"\nTraining α for {PRE_BINARY_STEPS} steps...")
model.train()
for step in range(PRE_BINARY_STEPS):
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
        print(f"  step {step+1}/{PRE_BINARY_STEPS}  loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

T2 = lm_ce(model, val_tokens)
print(f"T2  + α trained {PRE_BINARY_STEPS} steps (= Stage 169 T3)              CE={T2:.4f}  ppl={math.exp(T2):.2f}  Δ={T2-T0:+.3f}")


# ─── T3: apply Bonsai-style per-group binary on top of unit-norm weights ───
print(f"\nApplying Bonsai-style per-{GROUP_SIZE}-group binary to unit-norm W...")
for full_name, layer in target_layers.items():
    W_unit = layer.weight.data.float()
    W_q = bonsai_style_quantize(W_unit, GROUP_SIZE)
    layer.weight.data = W_q.to(layer.weight.dtype)

T3 = lm_ce(model, val_tokens)
print(f"T3  + Bonsai-style binary on unit-norm W (no extra train)   CE={T3:.4f}  ppl={math.exp(T3):.2f}  Δ={T3-T0:+.3f}")


# ─── T4: refine α post-binary ───
print(f"\nRefining α for {POST_BINARY_STEPS} more steps...")
opt2 = torch.optim.AdamW([a for a in alphas.values()], lr=LR, weight_decay=0.0)
model.train()
for step in range(POST_BINARY_STEPS):
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
        print(f"  step {step+1}/{POST_BINARY_STEPS}  loss={loss.item()*GRAD_ACCUM:.4f}", flush=True)

T4 = lm_ce(model, val_tokens)
print(f"T4  + α refined {POST_BINARY_STEPS} steps post-binary              CE={T4:.4f}  ppl={math.exp(T4):.2f}  Δ={T4-T0:+.3f}")


# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY: hypersphere → Bonsai-binary (ABA pair to Stage 180)\n{'='*70}")
results = [
    ("T0  base FP",                                                         T0),
    ("T1  unit-norm + α=row_norm (no train)",                               T1),
    (f"T2  + α trained {PRE_BINARY_STEPS} steps (replicates Stage 169 T3)",      T2),
    ("T3  + Bonsai-style binary on top",                                    T3),
    (f"T4  + α refined {POST_BINARY_STEPS} steps",                              T4),
]
for label, ce in results:
    ppl = math.exp(min(ce, 30))
    print(f"  {label:<60} CE={ce:8.4f}  ppl={ppl:>14.2f}  Δ={ce-T0:+.3f}")

print(f"\nABA COMPARISON (with Stage 180):")
print(f"  Stage 180 (Bonsai then hypersphere): T3 will land at Δ={'?'}")
print(f"  Stage 182 (hypersphere then Bonsai): T4 = Δ={T4-T0:+.3f}")
print(f"")
print(f"  If similar: encodings are commutative — same destination either way")
print(f"  If different: order matters; preference informs the production recipe")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "group_size": GROUP_SIZE,
        "pre_binary_steps": PRE_BINARY_STEPS,
        "post_binary_steps": POST_BINARY_STEPS,
        "T0": float(T0), "T1": float(T1), "T2": float(T2),
        "T3": float(T3), "T4": float(T4),
        "delta_T1": float(T1-T0), "delta_T2": float(T2-T0),
        "delta_T3": float(T3-T0), "delta_T4": float(T4-T0),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
