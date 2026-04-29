"""Stage 183: Binary QAT with NO geometric constraint — does master
weight drift toward hypersphere on its own?

The clean test. Apply per-128-group binary in forward (Bonsai-style),
master FP weights underneath with STE backward. Unfreeze ALL body
weights. Train. Track master row-norm CV across training.

Hypothesis (to test): the hypersphere is the natural attractor of
binary QAT. If true, master row-norm CV should monotonically decrease
during training, eventually approaching 0.

Falsifying outcome: CV stays at ~0.30 (the natural attractor) or
drifts toward something else entirely (some other geometry preferred
under binary).

Memory note: full-body training of Qwen3-0.6B requires ~5-6GB of
AdamW state on top of model. Tight on 16GB Mac but doable with small
batch + grad accumulation. Use 8-bit Adam if available to halve state.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 64                  # short to save memory
BATCH = 1
GRAD_ACCUM = 8                # effective batch 8
N_VAL_CHUNKS = 32
LR = 5e-5
GRAD_CLIP = 1.0
TRAIN_STEPS = 300
MEASURE_EVERY = 50
GROUP_SIZE = 128
RESULTS_PATH = Path("results/stage183_binary_qat_natural_drift.json")
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


def measure_row_norm_cv(target_modules):
    """Compute mean and CV of row L2 norms across all target master weights."""
    all_norms = []
    by_type = defaultdict(list)
    for name, mod in target_modules:
        W = mod.weight.detach().float()
        norms = W.norm(dim=-1).cpu().numpy()
        all_norms.extend(norms.tolist())
        for t in TARGET_NAMES:
            if t in name:
                by_type[t].extend(norms.tolist()); break
    arr = np.array(all_norms)
    overall = {"mean": float(arr.mean()), "std": float(arr.std()),
               "cv": float(arr.std() / arr.mean()), "min": float(arr.min()),
               "max": float(arr.max())}
    by_type_stats = {}
    for t, vals in by_type.items():
        a = np.array(vals)
        by_type_stats[t] = {"mean": float(a.mean()), "cv": float(a.std()/a.mean())}
    return overall, by_type_stats


def bonsai_style_quantize(W, group_size=128):
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

# Freeze embeddings + lm_head; unfreeze body
for n, p in model.named_parameters():
    if "embed_tokens" in n or "lm_head" in n:
        p.requires_grad = False
    else:
        p.requires_grad = True


# ─── Patch target linears with binary forward (STE through quantize) ───
target_modules = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(m in name for m in TARGET_NAMES): continue
    target_modules.append((name, mod))


def make_binary_forward(weight_param, bias_param):
    def forward(x):
        # STE: forward uses quantized weight, backward acts as identity through projection
        w = weight_param
        w_q = bonsai_style_quantize(w.float(), GROUP_SIZE).to(x.dtype)
        # STE: w_eff = w + (w_q - w).detach() — gradient flows to w as if no projection
        w_eff = w + (w_q - w).detach()
        return F.linear(x, w_eff, bias_param.to(x.dtype) if bias_param is not None else None)
    return forward


print(f"\nPatching {len(target_modules)} linears with binary STE forward...")
for name, mod in target_modules:
    mod.forward = make_binary_forward(mod.weight, mod.bias)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,}")


# ─── Tokens ───
print("\nLoading tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)


# ─── Initial measurement ───
print("\nInitial state:")
init_overall, init_by_type = measure_row_norm_cv(target_modules)
print(f"  master row-norm overall:  mean={init_overall['mean']:.4f}, CV={init_overall['cv']:.4f}")
for t, s in init_by_type.items():
    print(f"    {t:<12}  mean={s['mean']:.3f}, CV={s['cv']:.4f}")
init_ce = lm_ce(model, val_tokens)
print(f"  initial val CE (with binary forward): {init_ce:.4f}  ppl={math.exp(init_ce):.2f}")


# ─── Train with master gradients ───
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


it = iter_train()
trajectory = [{"step": 0, "ce": float(init_ce), "overall": init_overall, "by_type": init_by_type}]

print(f"\nTraining {TRAIN_STEPS} steps with binary forward, master weights trainable...")
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
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
    opt.step()

    if (step + 1) % MEASURE_EVERY == 0:
        overall, by_type = measure_row_norm_cv(target_modules)
        ce = lm_ce(model, val_tokens)
        trajectory.append({"step": step + 1, "ce": float(ce),
                           "overall": overall, "by_type": by_type})
        print(f"  step {step+1:>4}: CV={overall['cv']:.4f}  mean_norm={overall['mean']:.3f}  CE={ce:.4f}  ppl={math.exp(min(ce,30)):.2f}",
              flush=True)


# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY: did master weights drift toward hypersphere?\n{'='*70}")
print(f"  {'step':>5} {'CV':>8} {'mean':>8} {'CE':>8}")
for t in trajectory:
    print(f"  {t['step']:>5} {t['overall']['cv']:>8.4f} {t['overall']['mean']:>8.3f} {t['ce']:>8.3f}")

initial_cv = trajectory[0]["overall"]["cv"]
final_cv = trajectory[-1]["overall"]["cv"]
print(f"\nCV trajectory: {initial_cv:.4f} → {final_cv:.4f}  (Δ={final_cv-initial_cv:+.4f})")
if final_cv < initial_cv * 0.5:
    print("  ✓ DRIFTING TO HYPERSPHERE — CV halved or more")
elif final_cv < initial_cv * 0.9:
    print("  ~ trending toward hypersphere — modest CV reduction")
elif final_cv > initial_cv * 1.1:
    print("  ✗ DRIFTING AWAY from hypersphere — CV increased")
else:
    print("  - flat — no clear drift toward or away from hypersphere")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "group_size": GROUP_SIZE,
        "seq_len": SEQ_LEN,
        "batch": BATCH,
        "grad_accum": GRAD_ACCUM,
        "lr": LR,
        "train_steps": TRAIN_STEPS,
        "trajectory": trajectory,
        "initial_cv": initial_cv,
        "final_cv": final_cv,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
