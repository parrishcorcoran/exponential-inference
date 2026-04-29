"""Stage 188: full BitNet-shape preconditioning before binary.

User reframe (2026-04-29): "If we artificially raise up what is raised
up in binary or ternary then we should be able to reach the capacity
through compensation mechanisms. We are giving it the info in forms
other than quantization."

Stage 185 mapped what trained low-bit models look like:
  - body row-norm flat (CV ~0.3)         ← Stage 184 already does
  - body row-norm boosted ~2×            ← α-bridge handles
  - RMSNorm gain MAX 1.01 (BitNet) vs 192 (Qwen)  ← NEW HERE
  - embedding row-norm 2.5× (BitNet)     ← NEW HERE
  - lm_head row-norm 1.6× (Bonsai, decoupled)     ← NEW HERE

This stage installs ALL of those axes simultaneously, then runs the
same compensation training as Stage 184. Plateau comparison:

  Stage 184 (body prep only):   +3.854 nats above base
  Stage 187 (+ weak outlier reg): +3.878 (reg too weak to test)
  Stage 188 (full BitNet-shape):  ?

Predictions per the RG framing:
  Plateau drops < +2:  pre-positioning displaces the system into a
    binary-friendly attractor; compensation closes the rest. Big win.
  Plateau drops modestly < +3.5:  partial — we're closer to attractor
    but not at it. Each axis contributing some.
  Plateau ≈ +3.85:  preconditioning didn't help; binary's hard floor
    is structural beyond what FP DOF redistribution can buy.
  Plateau > +3.85:  we displaced the system AWAY from its attractor.
    The outlier channels were load-bearing relevant operators; can't
    just delete them without QAT-driven relocation.

Knobs (design choices to be revised based on result):
  GAIN_CAP        absolute cap on |RMSNorm gain|  default 5.0
  EMBED_SCALE     embedding row-norm boost factor default 2.5
  LMHEAD_TEMP     learnable temperature on lm_head logits, init 2.5
                  (compensates the embed scaling on the tied weight)
"""
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR = 5e-4
GRAD_CLIP = 1.0
TRAIN_STEPS = 600
MEASURE_EVERY = 50
GROUP_SIZE = 128

# BitNet-shape preconditioning parameters
GAIN_CAP = 5.0          # hard cap on |RMSNorm gain|, clamped each step
EMBED_SCALE = 2.5       # multiplicative boost on embedding row norms
LMHEAD_TEMP_INIT = 2.5  # divider on lm_head logits to neutralize embed scale

RESULTS_PATH = Path("results/stage188_bitnet_shape_precondition.json")
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
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    W_grouped = W.reshape(out_features, n_groups, group_size)
    scales = W_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return (torch.sign(W_grouped) * scales).reshape(out_features, in_features)


class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight
        self.bias = original_module.bias
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone()
                                  .to(self.weight.device).to(torch.float32))
    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


def measure_norm_gains(model):
    all_gains = []
    for n, p in model.named_parameters():
        if "norm" in n.lower() and "weight" in n:
            all_gains.extend(p.detach().float().flatten().cpu().numpy().tolist())
    arr = np.array(all_gains)
    return {
        "mean": float(arr.mean()), "cv": float(arr.std() / max(arr.mean(), 1e-12)),
        "max": float(arr.max()), "min": float(arr.min()),
        "n_above_10x_mean": int((arr > 10 * arr.mean()).sum()),
    }


def measure_embed_norms(model):
    for n, p in model.named_parameters():
        if "embed_tokens" in n and "weight" in n:
            rn = p.detach().float().norm(dim=-1).cpu().numpy()
            return {"mean": float(rn.mean()),
                    "cv": float(rn.std() / max(rn.mean(), 1e-12)),
                    "max": float(rn.max())}
    return None


print(f"device={device} dtype={dtype}")
print(f"Preconditioning: GAIN_CAP={GAIN_CAP}  EMBED_SCALE={EMBED_SCALE}  LMHEAD_TEMP_INIT={LMHEAD_TEMP_INIT}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)


# ─── Reference: base FP CE ───
print("\nMeasuring base FP CE (reference)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

print("Loading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)

T0 = lm_ce(ref_model, val_tokens)
fp_gains = measure_norm_gains(ref_model)
fp_embed = measure_embed_norms(ref_model)
print(f"T0 base FP: CE={T0:.4f}  ppl={math.exp(T0):.2f}")
print(f"  fp gains: mean={fp_gains['mean']:.3f} CV={fp_gains['cv']:.3f} max={fp_gains['max']:.1f} n>10×={fp_gains['n_above_10x_mean']}")
print(f"  fp embed: mean={fp_embed['mean']:.3f} max={fp_embed['max']:.3f}")
del ref_model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


# ─── Set up model ───
print("\nLoading model and applying full BitNet-shape preconditioning...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False


# 1. Body: unit-norm project, Bonsai-binary, install α-bridge
target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_mods.append((name, mod))

original_row_norms = {
    n: m.weight.data.float().norm(dim=-1, keepdim=True).clone()
    for n, m in target_mods
}
for name, mod in target_mods:
    rn = original_row_norms[name].clamp(min=1e-8).to(mod.weight.dtype)
    mod.weight.data = mod.weight.data / rn

for name, mod in target_mods:
    W_q = bonsai_style_quantize(mod.weight.data.float(), GROUP_SIZE)
    mod.weight.data = W_q.to(mod.weight.dtype)

parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

alphas = {}
for full_name, mod in target_mods:
    binary_rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mod.weight.data = (mod.weight.data.float() / binary_rn).to(mod.weight.dtype)
    new_layer = AlphaLinear(mod, original_row_norms[full_name])
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha
print(f"  body: {len(target_mods)} linears unit-projected, binary, α-bridge installed")


# 2. RMSNorm: hard-clamp |gain| ≤ GAIN_CAP at init
clamped_count = 0
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        with torch.no_grad():
            mask = p.abs() > GAIN_CAP
            clamped_count += int(mask.sum().item())
            p.data = torch.clamp(p.data, min=-GAIN_CAP, max=GAIN_CAP)
print(f"  norms: clamped {clamped_count} channels above |{GAIN_CAP}|")


# 3. Embeddings: scale weights by EMBED_SCALE
embed_scaled = False
for n, p in model.named_parameters():
    if "embed_tokens" in n and "weight" in n:
        with torch.no_grad():
            p.data = p.data * EMBED_SCALE
        embed_scaled = True
        break
print(f"  embeddings: scaled by {EMBED_SCALE}× {'✓' if embed_scaled else '(NOT FOUND — check naming)'}")


# 4. LM head temperature: trainable scalar that divides logits to neutralize embed scale
# Hook on the model's forward output to apply temperature.
class TemperatureWrapper(nn.Module):
    def __init__(self, base_model, init_temp):
        super().__init__()
        self.base_model = base_model
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))
    @property
    def config(self):
        return self.base_model.config
    def forward(self, *args, **kwargs):
        out = self.base_model(*args, **kwargs)
        if hasattr(out, "logits"):
            out.logits = out.logits / torch.exp(self.log_temperature).to(out.logits.dtype)
        return out


wrapped = TemperatureWrapper(model, LMHEAD_TEMP_INIT).to(device)
print(f"  lm_head temperature: init={LMHEAD_TEMP_INIT}, trainable")


# 5. Mark trainable: norms + α + temperature
trainable = []
norm_params = []
for n, p in wrapped.named_parameters():
    if "norm" in n.lower() and "weight" in n and "embed" not in n.lower():
        p.requires_grad = True
        trainable.append(p)
        norm_params.append(p)
for a in alphas.values():
    a.requires_grad = True
    trainable.append(a)
wrapped.log_temperature.requires_grad = True
trainable.append(wrapped.log_temperature)

print(f"  trainable params:")
print(f"    norms:       {sum(p.numel() for p in norm_params):,}")
print(f"    alpha:       {sum(a.numel() for a in alphas.values()):,}")
print(f"    log_temp:    1")
print(f"    total:       {sum(p.numel() for p in trainable):,}")


# ─── Initial measurements ───
init_ce = lm_ce(wrapped, val_tokens)
init_gains = measure_norm_gains(wrapped)
init_embed = measure_embed_norms(wrapped)
init_temp = float(torch.exp(wrapped.log_temperature).item())
print(f"\ninit (post-preconditioning, no train): CE={init_ce:.4f}  Δ={init_ce-T0:+.3f}")
print(f"  gains: mean={init_gains['mean']:.3f} CV={init_gains['cv']:.3f} max={init_gains['max']:.1f} n>10×={init_gains['n_above_10x_mean']}")
print(f"  embed: mean={init_embed['mean']:.3f}")
print(f"  temp:  {init_temp:.3f}")


# ─── Train ───
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


print(f"\nTraining {TRAIN_STEPS} steps with norm clamp at |gain| ≤ {GAIN_CAP}...")
it = iter_train()
trajectory = [{"step": 0, "ce": float(init_ce), "gains": init_gains,
               "embed": init_embed, "temp": init_temp}]
wrapped.train()
for step in range(TRAIN_STEPS):
    opt.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = wrapped(ids[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    opt.step()

    # Re-clamp norms after step (hard projection)
    with torch.no_grad():
        for p in norm_params:
            p.data = torch.clamp(p.data, min=-GAIN_CAP, max=GAIN_CAP)

    if (step + 1) % MEASURE_EVERY == 0:
        ce = lm_ce(wrapped, val_tokens)
        gains = measure_norm_gains(wrapped)
        embed = measure_embed_norms(wrapped)
        temp = float(torch.exp(wrapped.log_temperature).item())
        trajectory.append({"step": step + 1, "ce": float(ce), "gains": gains,
                           "embed": embed, "temp": temp})
        print(f"  step {step+1:>4}: CE={ce:.4f} Δ={ce-T0:+.3f}  "
              f"gain_max={gains['max']:.2f} CV={gains['cv']:.3f}  "
              f"embed_mean={embed['mean']:.2f}  temp={temp:.3f}",
              flush=True)


# ─── Summary ───
final_ce = trajectory[-1]["ce"]
print("\n" + "=" * 70)
print("SUMMARY: did BitNet-shape preconditioning unlock recovery?")
print("=" * 70)
print(f"  T0 (base FP):                         {T0:.4f}")
print(f"  Stage 184 plateau (body prep only):   +3.854 nats")
print(f"  Stage 187 plateau (weak outlier reg): +3.878 nats")
print(f"  Stage 188 plateau (full precondition):  Δ = {final_ce-T0:+.4f} nats")

drop_vs_184 = (final_ce - T0) - 3.854
print(f"\n  Δ vs Stage 184: {drop_vs_184:+.4f}")

if drop_vs_184 < -1.5:
    print(f"\n  ✓✓ MAJOR UNLOCK: preconditioning closes most of the gap.")
    print(f"     The recipe works — substrate sculpting toward BitNet-attractor lets compensation finish the job.")
elif drop_vs_184 < -0.3:
    print(f"\n  ✓ Significant improvement: {-drop_vs_184:.2f} nats below Stage 184 plateau.")
    print(f"    Each axis is contributing; consider sweep over GAIN_CAP / EMBED_SCALE for tuning.")
elif drop_vs_184 < 0.1:
    print(f"\n  - Plateau roughly matches Stage 184. Preconditioning didn't help by itself.")
    print(f"    Bottleneck may be elsewhere (per-group bias / QAT / capacity).")
else:
    print(f"\n  ✗ DISPLACED: plateau is HIGHER than Stage 184 by {drop_vs_184:.2f} nats.")
    print(f"    The outlier channels and embed scale were load-bearing; we displaced the system from")
    print(f"    its FP RG attractor without successfully relocating to BitNet's attractor.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "preconditioning": {
            "gain_cap": GAIN_CAP,
            "embed_scale": EMBED_SCALE,
            "lmhead_temp_init": LMHEAD_TEMP_INIT,
        },
        "T0_base_ce": float(T0),
        "stage_184_baseline_delta": 3.854,
        "stage_187_baseline_delta": 3.878,
        "trajectory": trajectory,
        "final_ce": float(final_ce),
        "final_delta": float(final_ce - T0),
        "delta_vs_stage_184": float(drop_vs_184),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
