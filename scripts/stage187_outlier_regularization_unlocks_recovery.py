"""Stage 187: does explicit RMSNorm-outlier regularization let
compensation training reach a lower plateau than Stage 184's +3.85?

Stage 185 finding: trained low-bit models flatten Qwen's 192× outlier
RMSNorm gains (BitNet max=1.01). Stage 186 (queued) tests whether
compensation training does this naturally.

If 186 shows outliers DON'T move on their own, this stage forces the
issue: add a soft cap on |gain| and rerun the same Stage 184 setup.

  Loss = CE + λ * Σ max(0, |gain| - cap)²

Procedure: single trajectory matching Stage 184 (binary forward, norms+α
trainable, weights frozen) but with the outlier penalty active.
Comparison is direct against Stage 184's +3.854 nat plateau.

Predictions:
  Plateau drops below +3.5 nats:  outliers WERE the bottleneck. Big win.
  Plateau ≈ +3.85, outliers shrink: regularization works on outliers
    but they weren't doing useful work — compensation has another floor.
  CE plateaus higher: outliers WERE doing useful work — flattening hurts.
    Tells us the outlier→bulk redistribution (BitNet's path) requires
    weight training, not just gain regularization.
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
GAIN_CAP = 10.0       # soft cap on |RMSNorm gain|
LAMBDA_REG = 1e-4     # outlier regularization strength
RESULTS_PATH = Path("results/stage187_outlier_regularization.json")
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
            v = p.detach().float().flatten().cpu().numpy()
            all_gains.extend(v.tolist())
    arr = np.array(all_gains)
    return {
        "mean": float(arr.mean()),
        "cv": float(arr.std() / max(arr.mean(), 1e-12)),
        "max": float(arr.max()),
        "p99_9": float(np.percentile(arr, 99.9)),
        "n_above_10x_mean": int((arr > 10 * arr.mean()).sum()),
    }


def gain_outlier_loss(norm_params, cap=GAIN_CAP):
    """Soft cap penalty on |gain| values exceeding `cap`."""
    total = 0.0
    for p in norm_params:
        overshoot = torch.clamp(p.abs() - cap, min=0)
        total = total + (overshoot ** 2).sum()
    return total


def setup():
    """Build the binary + α-bridge model with norms+α trainable."""
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

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

    norm_params = []
    alpha_params = list(alphas.values())
    for n, p in model.named_parameters():
        if "norm" in n.lower() and "weight" in n:
            p.requires_grad = True
            norm_params.append(p)
    for a in alpha_params:
        a.requires_grad = True

    return model, norm_params, alpha_params


print(f"device={device} dtype={dtype}")
print(f"GAIN_CAP={GAIN_CAP}  LAMBDA_REG={LAMBDA_REG}")
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
print(f"T0 base FP: CE={T0:.4f}  ppl={math.exp(T0):.2f}")
del ref_model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


# ─── Setup model with binary + α + norms trainable ───
print("\nSetting up binary + α-bridge + norms trainable...")
model, norm_params, alpha_params = setup()
trainable = norm_params + alpha_params
print(f"  trainable: {sum(p.numel() for p in norm_params):,} norm + "
      f"{sum(p.numel() for p in alpha_params):,} α = {sum(p.numel() for p in trainable):,}")

opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
init_ce = lm_ce(model, val_tokens)
init_gains = measure_norm_gains(model)
print(f"  init CE: {init_ce:.4f}  Δ={init_ce-T0:+.3f}")
print(f"  init gains: mean={init_gains['mean']:.3f}  CV={init_gains['cv']:.3f}  max={init_gains['max']:.1f}  n>10×: {init_gains['n_above_10x_mean']}")


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


print(f"\nTraining {TRAIN_STEPS} steps with outlier regularization (cap={GAIN_CAP}, λ={LAMBDA_REG})...")
it = iter_train()
trajectory = [{"step": 0, "ce": float(init_ce), "gains": init_gains, "loss_reg": None}]
model.train()
last_loss_reg = 0.0
for step in range(TRAIN_STEPS):
    opt.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = model(ids[:, :-1], use_cache=False)
        loss_ce = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss_reg = (LAMBDA_REG / GRAD_ACCUM) * gain_outlier_loss(norm_params, cap=GAIN_CAP)
        last_loss_reg = float(loss_reg.item() * GRAD_ACCUM)
        (loss_ce + loss_reg).backward()
    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    opt.step()

    if (step + 1) % MEASURE_EVERY == 0:
        ce = lm_ce(model, val_tokens)
        gains = measure_norm_gains(model)
        trajectory.append({"step": step + 1, "ce": float(ce), "gains": gains,
                           "loss_reg": last_loss_reg})
        print(f"  step {step+1:>4}: CE={ce:.4f} Δ={ce-T0:+.3f}  gain_max={gains['max']:.1f}  CV={gains['cv']:.3f}  n>10×={gains['n_above_10x_mean']}  L_reg={last_loss_reg:.3f}",
              flush=True)


# ─── Summary ───
final_ce = trajectory[-1]["ce"]
final_gains = trajectory[-1]["gains"]
print("\n" + "=" * 70)
print("SUMMARY: did outlier regularization unlock further recovery?")
print("=" * 70)
print(f"  T0 (base FP):           {T0:.4f}")
print(f"  Stage 184 plateau (no reg, recall): +3.854 nats")
print(f"  This run (with reg):    Δ = {final_ce-T0:+.4f} nats")
print(f"  Δ vs Stage 184:         {(final_ce-T0) - 3.854:+.4f}")
print(f"\n  init  gain max: {init_gains['max']:.1f}  CV: {init_gains['cv']:.3f}  n>10×: {init_gains['n_above_10x_mean']}")
print(f"  final gain max: {final_gains['max']:.1f}  CV: {final_gains['cv']:.3f}  n>10×: {final_gains['n_above_10x_mean']}")

drop_vs_184 = (final_ce - T0) - 3.854
if drop_vs_184 < -0.2:
    print(f"\n  ✓ OUTLIERS WERE A BOTTLENECK: regularization dropped plateau by {-drop_vs_184:.2f} nats below Stage 184.")
elif drop_vs_184 < -0.05:
    print(f"\n  ~ Outliers contributed modestly: regularization helped by {-drop_vs_184:.2f} nats.")
elif drop_vs_184 > 0.2:
    print(f"\n  ✗ Outliers were doing useful work: regularization HURT recovery by {drop_vs_184:.2f} nats.")
    print(f"    The 192× channels were carrying real information; flattening must come with weight training (BitNet's path).")
else:
    print(f"\n  - No meaningful change ({drop_vs_184:+.3f} nats). Outliers can be flattened cheaply but it doesn't")
    print(f"    affect the plateau either way. Compensation has a deeper floor than gain distribution.")


with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "gain_cap": GAIN_CAP,
        "lambda_reg": LAMBDA_REG,
        "T0_base_ce": float(T0),
        "trajectory": trajectory,
        "final_ce": float(final_ce),
        "final_delta": float(final_ce - T0),
        "stage_184_baseline_delta": 3.854,
        "delta_vs_stage_184": float((final_ce - T0) - 3.854),
        "final_gains": final_gains,
        "init_gains": init_gains,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
