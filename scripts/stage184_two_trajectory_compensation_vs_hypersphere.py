"""Stage 184: Two-trajectory test — does the hypersphere prerequisite
shift the achievable plateau when only compensating mechanisms are
allowed to train?

Hypothesis (refined from Stage 183): RMSNorm gains and per-channel
α-bridges (the "compensating mechanisms") absorb the bulk of binary
damage cheaply. But there is a structural floor — the per-row noise
asymmetry — that no scalar compensation can fix. The hypersphere
removes that asymmetry, opening the last few percent of recovery.

Test: two parallel trajectories from a binary-quantized Qwen3-0.6B.
Same compensation budget in each (RMSNorm gains + α-bridge trainable).
The ONLY difference is whether rows were projected to unit norm before
binary quantization.

  Variant A:  base → Bonsai-binary → α=row_norm    → train (norms+α)
  Variant B:  base → unit-norm → Bonsai-binary → α=row_norm → train (norms+α)

Both start with α=original_row_norm so init CE matches Bonsai PTQ in
A and Stage 180-style lossless re-encoding in B (≈ same starting CE).
Training has equal compensation capacity. Plateau gap reveals what
hypersphere geometry is doing at the margin.

Predictions:
  A plateaus low (≈ Bonsai 89%-of-base):    hypersphere matters at margin
  A plateaus = B:                           hypersphere is irrelevant
  A plateaus higher than B:                 hypersphere actively hurts

Cheap: ~555K trainable params total. Long training is fine on Mac M4.
"""
import gc
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
LR = 5e-4
GRAD_CLIP = 1.0
TRAIN_STEPS = 600
MEASURE_EVERY = 50
GROUP_SIZE = 128
RESULTS_PATH = Path("results/stage184_two_trajectory.json")
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


def setup_variant(project_to_unit_norm: bool):
    """Returns (model, trainable_params, info_dict)."""
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

    # Save original row norms — these become α inits in BOTH variants.
    # In A: matches natural Bonsai output magnitude (≈ post-binary row norm).
    # In B: restores the magnitude lost to projection (Stage 180 identity).
    original_row_norms = {
        n: m.weight.data.float().norm(dim=-1, keepdim=True).clone()
        for n, m in target_mods
    }

    # Variant B: project rows to unit norm.
    if project_to_unit_norm:
        for name, mod in target_mods:
            rn = original_row_norms[name].clamp(min=1e-8).to(mod.weight.dtype)
            mod.weight.data = mod.weight.data / rn

    # Apply Bonsai-style per-128-group binary to all variants.
    for name, mod in target_mods:
        W = mod.weight.data.float()
        W_q = bonsai_style_quantize(W, GROUP_SIZE)
        mod.weight.data = W_q.to(mod.weight.dtype)

    # Renormalize binary to unit rows; install AlphaLinear with α = original_row_norms.
    parent_lookup = {}
    for name, m in model.named_modules():
        for child_name, child_mod in m.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (m, child_name)

    alphas = {}
    for full_name, mod in target_mods:
        binary_rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mod.weight.data = (mod.weight.data.float() / binary_rn).to(mod.weight.dtype)
        # α absorbs both pre-projection magnitude (lost in B) and the small
        # binary_rn correction (since binary doesn't perfectly preserve norm).
        alpha_init = (original_row_norms[full_name] / binary_rn) * binary_rn
        # ^ algebraically = original_row_norm; written this way so it's clear
        # in code that we're picking α to match the original FP output scale.
        alpha_init = original_row_norms[full_name]
        new_layer = AlphaLinear(mod, alpha_init)
        parent, child_attr = parent_lookup[full_name]
        setattr(parent, child_attr, new_layer)
        alphas[full_name] = new_layer.alpha

    # Trainable: RMSNorm gains + α-bridges. Embeddings, lm_head, binary
    # weights all frozen.
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    n_norm = 0; n_alpha = 0
    for n, p in model.named_parameters():
        if "norm" in n.lower() and "weight" in n:
            p.requires_grad = True
            trainable.append(p)
            n_norm += p.numel()
    for a in alphas.values():
        a.requires_grad = True
        trainable.append(a)
        n_alpha += a.numel()

    info = {"n_norm_params": n_norm, "n_alpha_params": n_alpha,
            "n_target_linears": len(target_mods)}
    print(f"  trainable: {n_norm:,} norm + {n_alpha:,} α = {n_norm + n_alpha:,}")
    return model, trainable, info


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("\nLoading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# ─── Reference: base FP CE ───
print("\nMeasuring base FP CE (reference)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(ref_model, val_tokens)
print(f"T0  base FP                            CE={T0:.4f}  ppl={math.exp(T0):.2f}")
del ref_model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


def iter_train(train_tokens):
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def run_variant(label, project_to_unit_norm):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    model, params, info = setup_variant(project_to_unit_norm)
    init_ce = lm_ce(model, val_tokens)
    print(f"  init CE (binary, no train): {init_ce:.4f}  Δ={init_ce-T0:+.3f}")

    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    it = iter_train(train_tokens)
    trajectory = [{"step": 0, "ce": float(init_ce)}]
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
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()

        if (step + 1) % MEASURE_EVERY == 0:
            ce = lm_ce(model, val_tokens)
            trajectory.append({"step": step + 1, "ce": float(ce)})
            print(f"  step {step+1:>4}: CE={ce:.4f}  Δ={ce-T0:+.3f}  ppl={math.exp(min(ce,30)):.2f}",
                  flush=True)

    final_ce = trajectory[-1]["ce"]
    print(f"  final: CE={final_ce:.4f}  Δ={final_ce-T0:+.4f}")
    del model, opt
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return trajectory, info


traj_A, info_A = run_variant("VARIANT A: binary + train (norms+α). NO hypersphere prereq.",
                             project_to_unit_norm=False)
traj_B, info_B = run_variant("VARIANT B: unit-norm + binary + train (norms+α). WITH hypersphere prereq.",
                             project_to_unit_norm=True)


# ─── Compare ───
print(f"\n{'='*70}\nSUMMARY: does the hypersphere prerequisite shift the plateau?\n{'='*70}")
print(f"  T0 (base FP):   CE={T0:.4f}")
print(f"\n  {'step':>5}  {'A (no hyper)':>14}  {'B (hyper)':>14}  {'A − B':>10}")
n_steps = max(len(traj_A), len(traj_B))
for i in range(n_steps):
    a = traj_A[i] if i < len(traj_A) else None
    b = traj_B[i] if i < len(traj_B) else None
    step = a["step"] if a else b["step"]
    ca = f"{a['ce']:.4f}" if a else "—"
    cb = f"{b['ce']:.4f}" if b else "—"
    gap = f"{a['ce']-b['ce']:+.4f}" if (a and b) else "—"
    print(f"  {step:>5}  {ca:>14}  {cb:>14}  {gap:>10}")

final_A = traj_A[-1]["ce"]
final_B = traj_B[-1]["ce"]
gap = final_A - final_B
print(f"\nFinal plateau gap (A − B):   Δ = {gap:+.4f} nats")
print(f"  A reached:  Δ_vs_T0 = {final_A-T0:+.4f}  (no hypersphere prerequisite)")
print(f"  B reached:  Δ_vs_T0 = {final_B-T0:+.4f}  (with hypersphere prerequisite)")

if gap > 0.05:
    print(f"\n  ✓ Hypersphere prerequisite OPENS RECOVERY: B plateau is {gap:.3f} nats lower than A.")
    print(f"    The structural floor is real; unit-norm geometry is the way past it.")
elif gap < -0.05:
    print(f"\n  ✗ Hypersphere prerequisite HURTS: A plateau is lower than B by {-gap:.3f} nats.")
    print(f"    Compensation alone does better when substrate is unprepped.")
else:
    print(f"\n  - No detectable plateau gap (|gap| < 0.05). Hypersphere may be irrelevant for compensation,")
    print(f"    or both runs need more steps to separate.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "group_size": GROUP_SIZE,
        "train_steps": TRAIN_STEPS,
        "lr": LR,
        "T0_base_ce": float(T0),
        "trajectory_A_no_hypersphere": traj_A,
        "trajectory_B_with_hypersphere": traj_B,
        "info_A": info_A,
        "info_B": info_B,
        "final_A": float(final_A),
        "final_B": float(final_B),
        "plateau_gap_A_minus_B": float(gap),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
