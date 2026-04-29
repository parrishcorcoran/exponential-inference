"""Stage 205: combined Magnitude + Sharpness anneal recipe.

Phase 1 — Magnitude anneal (Stage 169 protocol):
  - Project body rows to unit norm
  - Install per-row α-bridge (α = original row norm)
  - Train α briefly to refine (Stage 169 T3 mechanism)
  - Result: body on hypersphere, α holds magnitudes, Δ=0 lossless math identity

Phase 2 — Sharpness anneal (Stage 204 protocol):
  - PID-throttled cap descent on RMSNorm gains
  - Body + α + norms train to absorb each cap
  - Walk T from initial max down toward target

Phase 3 — Coherency test:
  - Generate text from canonical prompts
  - Verify capability is intact (not just val CE improved)

The bet: combining the two anneals stacks their improvements. Magnitude
anneal puts body on hypersphere; sharpness anneal removes outlier noise.
Together: better-shaped body, cleaner residual stream, room for both
training compensation and eventual K=1 binary projection.
"""
import gc
import json
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
RESULTS_PATH = Path("results/stage205_combined_anneal.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
TRAINABLE_BODY = ("o_proj", "down_proj")

# Phase 1: magnitude anneal
PHASE1_TRAIN_STEPS = 300

# Phase 2: sharpness anneal (Stage 204-style)
PHASE2_N_CYCLES = 60
TRAIN_STEPS_PER_CYCLE = 50
PID_SETPOINT = 0.1
RATE_MAX = 0.05
RATE_MIN = 0.001
QUALITY_LIMIT = 5.0
T_TARGET = 1.0


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


def pid_rate(drift, setpoint):
    if drift <= 0:
        return RATE_MAX
    elif drift < setpoint:
        frac = drift / setpoint
        return RATE_MAX * (1 - frac) + RATE_MIN * frac
    elif drift < 2 * setpoint:
        return 0
    else:
        return -0.01


print(f"device={device} dtype={dtype}")
print(f"Combined anneal recipe: magnitude + sharpness")

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


# ─── PHASE 1: Magnitude anneal (Stage 169) ───
print("\n" + "=" * 70)
print("PHASE 1: Magnitude anneal — project body to unit norm + α-bridge")
print("=" * 70)

target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in TARGET_NAMES): continue
    target_mods.append((name, mod))

# Save row norms then project
row_norms_init = {}
for name, mod in target_mods:
    rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clone()
    row_norms_init[name] = rn
    mod.weight.data = (mod.weight.data.float() / rn.clamp(min=1e-8)).to(mod.weight.dtype)

# Install AlphaLinear
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
print(f"  After project + α-bridge install: CE={ce_post_project:.4f}  Δ={ce_post_project-T0:+.4f}  (should be ~0)")


# Train α briefly (Phase 1 fine-tune)
for p in model.parameters():
    p.requires_grad = False
for a in alphas.values():
    a.requires_grad = True

opt_alpha = torch.optim.AdamW([a for a in alphas.values()], lr=LR_ALPHA, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


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


# ─── PHASE 2: Sharpness anneal (Stage 204) ───
print("\n" + "=" * 70)
print("PHASE 2: Sharpness anneal — cap RMSNorm gains, body absorbs")
print("=" * 70)

# Set up body trainable subset
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
print(f"  Initial norm max: {init_max:.1f}")
print(f"  Body trainable: {sum(p.numel() for p in body_params):,}")
print(f"  α trainable: {sum(a.numel() for a in alphas.values()):,}")
print(f"  Norms trainable: {sum(p.numel() for p in norm_params):,}")


phase2_trajectory = [{"cycle": 0, "T_cap": init_max, "ce": float(ce_phase1), "drift": float(ce_phase1 - T0),
                      "norm_max": init_max, "n_clipped": 0}]
current_T = init_max
broke_at = None

for cycle in range(1, PHASE2_N_CYCLES + 1):
    prev_drift = phase2_trajectory[-1]["drift"]
    rate = pid_rate(prev_drift, PID_SETPOINT)
    new_T = current_T * (1 - rate)
    new_T = max(T_TARGET * 0.5, new_T)

    n_clipped = 0
    with torch.no_grad():
        for p in norm_params:
            sign = torch.sign(p.data)
            mag = p.data.abs()
            n_clipped += int((mag > new_T).sum().item())
            p.data = sign * torch.clamp(mag, max=new_T)

    train_steps_phase2(it, TRAIN_STEPS_PER_CYCLE)

    ce = lm_ce(model, val_tokens)
    drift = ce - T0
    cur_max = max(p.detach().float().abs().max().item() for p in norm_params)

    phase2_trajectory.append({
        "cycle": cycle, "T_cap": float(new_T),
        "ce": float(ce), "drift": float(drift),
        "norm_max": float(cur_max), "n_clipped": n_clipped, "rate": float(rate),
    })

    if cycle <= 3 or cycle % 5 == 0:
        marker = "↓" if rate > 0 else ("↑" if rate < 0 else "·")
        print(f"  cycle {cycle:>3}/{PHASE2_N_CYCLES}  T={new_T:7.2f} {marker}  "
              f"CE={ce:.4f} drift={drift:+.4f}  clipped={n_clipped}  norm_max={cur_max:.1f}",
              flush=True)

    if drift > QUALITY_LIMIT and broke_at is None:
        broke_at = cycle
        print(f"  ⚠ broke past +{QUALITY_LIMIT} at cycle {cycle}")
    if drift > 10.0:
        print(f"  STOPPING at +10")
        break
    if new_T <= T_TARGET:
        print(f"  Reached T={T_TARGET}")
        break

    current_T = new_T


# ─── PHASE 3: Coherency test ───
print("\n" + "=" * 70)
print("PHASE 3: Coherency test")
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
final = phase2_trajectory[-1]
print("\n" + "=" * 70)
print("STAGE 205 COMPLETE")
print("=" * 70)
print(f"  T0 (base FP):                {T0:.4f}")
print(f"  After Phase 1 (magnitude):    Δ={ce_phase1-T0:+.4f}")
print(f"  After Phase 2 (sharpness):    Δ={final['drift']:+.4f}")
print(f"  Final cycle: {final['cycle']}  T={final['T_cap']:.2f}  norm_max={final['norm_max']:.2f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_base_ce": float(T0),
        "ce_after_project": float(ce_post_project),
        "ce_after_phase1": float(ce_phase1),
        "ce_after_phase2": float(final["ce"]),
        "phase1_train_steps": PHASE1_TRAIN_STEPS,
        "phase2_n_cycles": PHASE2_N_CYCLES,
        "phase2_trajectory": phase2_trajectory,
        "broke_at_cycle": broke_at,
        "coherency_generations": generations,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
