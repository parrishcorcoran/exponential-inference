"""Stage 206: Full recipe + K=1 binary projection + post-K=1 refinement.

The production test for our recipe. End-to-end:

Phase 1: Magnitude anneal (Stage 169)
  Project body rows to unit norm, install α-bridge, train α briefly.

Phase 2: Sharpness anneal (Stage 204)
  PID-throttled cap descent on RMSNorm gains, body+α+norms train to absorb.

Phase 3: K=1 BINARY PROJECTION (the actual quantization)
  Apply Bonsai-style per-128-group binary to body (signs + per-group scale).
  This is where the recipe is tested: does the sharpened body absorb K=1?

Phase 4: Post-K=1 refinement
  Body frozen at binary; train α + RMSNorm + per-group scales (now stored)
  to refine compensation.

Phase 5: Coherency test
  Generate text from canonical prompts.

Headline metrics:
  - drift_post_k1: drift just after K=1 projection (before refinement)
  - drift_final: drift after refinement training (true production quality)
  - coherency: visual confirmation
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
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR_BODY = 1e-5
LR_ALPHA = 5e-4
LR_NORM = 5e-5
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage206_full_recipe_k1.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
TRAINABLE_BODY = ("o_proj", "down_proj")
GROUP_SIZE = 128

# Phase 1: magnitude anneal
PHASE1_TRAIN_STEPS = 200

# Phase 2: sharpness anneal
PHASE2_N_CYCLES = 40
TRAIN_STEPS_PER_CYCLE = 50
PID_SETPOINT = 0.1
RATE_MAX = 0.05
RATE_MIN = 0.001
T_TARGET = 1.0
QUALITY_LIMIT = 5.0

# Phase 4: post-K=1 refinement
PHASE4_TRAIN_STEPS = 300


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt_cached():
    return torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                      weights_only=True).long()


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
    return sum(losses) / max(len(losses), 1)


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


def bonsai_project(W, group_size=128):
    out_features, in_features = W.shape
    if in_features % group_size != 0:
        scale = W.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.sign(W) * scale
    n_groups = in_features // group_size
    grouped = W.reshape(out_features, n_groups, group_size)
    scales = grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)
    return (torch.sign(grouped) * scales).reshape(out_features, in_features)


def pid_rate(drift, setpoint):
    if drift <= 0: return RATE_MAX
    elif drift < setpoint:
        frac = drift / setpoint
        return RATE_MAX * (1 - frac) + RATE_MIN * frac
    elif drift < 2 * setpoint: return 0
    else: return -0.01


print(f"device={device} dtype={dtype}")
print(f"Stage 206: full recipe + K=1 projection + post-refinement")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("\nLoading val + train tokens...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 1024 : SEQ_LEN * 1024 + SEQ_LEN * 4096].tolist()


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

T0 = lm_ce(model, val_tokens)
print(f"\nT0 base FP CE: {T0:.4f}")


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


# ─── PHASE 1: Magnitude anneal ───
print("\n" + "=" * 70)
print("PHASE 1: Magnitude anneal")
print("=" * 70)

target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    target_mods.append((name, mod))

row_norms_init = {}
for name, mod in target_mods:
    rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clone()
    row_norms_init[name] = rn
    mod.weight.data = (mod.weight.data.float() / rn.clamp(min=1e-8)).to(mod.weight.dtype)

parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

alphas = {}
target_layers = {}
for full_name, mod in target_mods:
    parent, child_attr = parent_lookup[full_name]
    new_layer = AlphaLinear(mod, row_norms_init[full_name])
    setattr(parent, child_attr, new_layer)
    alphas[full_name] = new_layer.alpha
    target_layers[full_name] = new_layer

ce_post_project = lm_ce(model, val_tokens)
print(f"  After project + α-bridge: CE={ce_post_project:.4f}  Δ={ce_post_project-T0:+.4f}")

for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

opt_alpha = torch.optim.AdamW([a for a in alphas.values()], lr=LR_ALPHA, weight_decay=0.0)
it = iter_train()
print(f"  Training α for {PHASE1_TRAIN_STEPS} steps...")
model.train()
for step in range(PHASE1_TRAIN_STEPS):
    opt_alpha.zero_grad()
    for _ in range(GRAD_ACCUM):
        ids = next(it)
        out = model(ids[:, :-1], use_cache=False)
        loss = F.cross_entropy(
            out.logits.float().reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1)) / GRAD_ACCUM
        loss.backward()
    torch.nn.utils.clip_grad_norm_([a for a in alphas.values()], GRAD_CLIP)
    opt_alpha.step()

ce_phase1 = lm_ce(model, val_tokens)
print(f"  After α training: CE={ce_phase1:.4f}  Δ={ce_phase1-T0:+.4f}")


# ─── PHASE 2: Sharpness anneal ───
print("\n" + "=" * 70)
print("PHASE 2: Sharpness anneal")
print("=" * 70)

body_params = []
for name, layer in target_layers.items():
    if any(t in name for t in TRAINABLE_BODY):
        layer.weight.requires_grad = True
        body_params.append(layer.weight)

norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n:
        p.requires_grad = True
        norm_params.append(p)

opt_phase2 = torch.optim.AdamW([
    {"params": body_params, "lr": LR_BODY},
    {"params": list(alphas.values()), "lr": LR_ALPHA},
    {"params": norm_params, "lr": LR_NORM},
], weight_decay=0.0)


def train_steps_phase2(it, n_steps):
    model.train()
    for _ in range(n_steps):
        opt_phase2.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(body_params + list(alphas.values()) + norm_params, GRAD_CLIP)
        opt_phase2.step()


init_max = max(p.detach().float().abs().max().item() for p in norm_params)
phase2_traj = [{"cycle": 0, "T_cap": init_max, "ce": float(ce_phase1), "drift": float(ce_phase1 - T0)}]
current_T = init_max

for cycle in range(1, PHASE2_N_CYCLES + 1):
    prev_drift = phase2_traj[-1]["drift"]
    rate = pid_rate(prev_drift, PID_SETPOINT)
    new_T = current_T * (1 - rate)
    new_T = max(T_TARGET * 0.5, new_T)

    with torch.no_grad():
        for p in norm_params:
            sign = torch.sign(p.data)
            mag = p.data.abs()
            p.data = sign * torch.clamp(mag, max=new_T)

    train_steps_phase2(it, TRAIN_STEPS_PER_CYCLE)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    cur_max = max(p.detach().float().abs().max().item() for p in norm_params)
    phase2_traj.append({"cycle": cycle, "T_cap": float(new_T), "ce": float(ce),
                        "drift": float(drift), "norm_max": float(cur_max), "rate": float(rate)})

    if cycle <= 3 or cycle % 5 == 0 or cycle == PHASE2_N_CYCLES:
        print(f"  cycle {cycle:>3}/{PHASE2_N_CYCLES}  T={new_T:.2f}  CE={ce:.4f} drift={drift:+.4f}  norm_max={cur_max:.1f}",
              flush=True)
    if drift > QUALITY_LIMIT:
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
        break
    if drift > 10.0:
        print(f"  STOPPING")
        break
    if new_T <= T_TARGET:
        print(f"  Reached T={T_TARGET}")
        break
    current_T = new_T

ce_phase2 = phase2_traj[-1]["ce"]
print(f"  After sharpness anneal: CE={ce_phase2:.4f}  Δ={ce_phase2-T0:+.4f}")


# ─── PHASE 3: K=1 binary projection on body ───
print("\n" + "=" * 70)
print("PHASE 3: K=1 BINARY PROJECTION on body")
print("=" * 70)

with torch.no_grad():
    for name, layer in target_layers.items():
        W = layer.weight.data.float()
        W_q = bonsai_project(W, GROUP_SIZE)
        layer.weight.data = W_q.to(layer.weight.dtype)

ce_post_k1 = lm_ce(model, val_tokens)
drift_post_k1 = ce_post_k1 - T0
print(f"  After K=1 projection: CE={ce_post_k1:.4f}  Δ={drift_post_k1:+.4f}")


# ─── PHASE 4: Post-K=1 refinement ───
print("\n" + "=" * 70)
print("PHASE 4: Post-K=1 refinement (body frozen at binary, train α + norms)")
print("=" * 70)

# Freeze body, train only α + norms
for name, layer in target_layers.items():
    layer.weight.requires_grad = False

opt_phase4 = torch.optim.AdamW([
    {"params": list(alphas.values()), "lr": LR_ALPHA},
    {"params": norm_params, "lr": LR_NORM},
], weight_decay=0.0)


def train_steps_phase4(it, n_steps):
    model.train()
    for _ in range(n_steps):
        opt_phase4.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(alphas.values()) + norm_params, GRAD_CLIP)
        opt_phase4.step()


print(f"  Training α + norms for {PHASE4_TRAIN_STEPS} steps...")
train_steps_phase4(it, PHASE4_TRAIN_STEPS)

ce_phase4 = lm_ce(model, val_tokens)
drift_final = ce_phase4 - T0
print(f"  After refinement: CE={ce_phase4:.4f}  Δ={drift_final:+.4f}")


# ─── PHASE 5: Coherency ───
print("\n" + "=" * 70)
print("PHASE 5: Coherency test (post-K=1, post-refine)")
print("=" * 70)

test_prompts = [
    "The quick brown fox",
    "Once upon a time, there was",
    "The capital of France is",
    "To compute the sum of two numbers in Python, you can",
    "Step 1: First, we need to",
]
generations = []
model.eval()
for prompt in test_prompts:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            ids, max_new_tokens=50, do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0,
        )
    completion = tokenizer.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
    generations.append({"prompt": prompt, "completion": completion})
    print(f"\n  PROMPT: {repr(prompt)}")
    print(f"  CONT:   {repr(completion)}")


# ─── Summary ───
print("\n" + "=" * 70)
print("STAGE 206 — FULL RECIPE COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):           {T0:.4f}")
print(f"  Phase 1 (magnitude):    Δ={ce_phase1-T0:+.4f}")
print(f"  Phase 2 (sharpness):    Δ={ce_phase2-T0:+.4f}")
print(f"  Phase 3 (after K=1):    Δ={drift_post_k1:+.4f}")
print(f"  Phase 4 (post-refine):  Δ={drift_final:+.4f}  ← PRODUCTION QUALITY")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "ce_after_project": float(ce_post_project),
        "ce_after_phase1": float(ce_phase1),
        "ce_after_phase2": float(ce_phase2),
        "ce_post_k1": float(ce_post_k1),
        "ce_after_refinement": float(ce_phase4),
        "drift_post_k1": float(drift_post_k1),
        "drift_final": float(drift_final),
        "phase2_trajectory": phase2_traj,
        "coherency_generations": generations,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
