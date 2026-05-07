"""Stage 234 — Extended Stage 232 training.

Stage 232 ended at +1.38 with cosine LR fully decayed to 0. We don't know if
that was a true optimum or just LR-limited (last 500 steps were finding new
bests, but LR was already too low to commit). Extending with a fresh cosine
schedule lets the optimizer take another descent phase.

Recipe (identical to Stage 232 except training horizon):
  1. Load Stage 232 best (per-128 + mag-perm, drift +1.3789, bpw 1.1354)
  2. Fresh cosine LR 1e-5 → 0 over 10000 steps (peak slightly lower than
     Stage 232's 1e-5 since we're refining a converged checkpoint)
  3. Same architecture: BinaryGroupLinearWithPerm at group_size=128
  4. Same distillation: top-K=64 KL, T=2.0
  5. Same trainable scope: o_proj + down_proj weights, SubLN, biases (147M)

Goal: break +1.38 plateau on Mac. If we hit ~+1.20 it's a clear win.
If we plateau again at +1.35 it confirms Stage 232 was at a real minimum.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 64
N_VAL_CHUNKS = 16
BATCH_SIZE = 1
N_TRAIN_STEPS = 10000
EVAL_EVERY = 100
CKPT_EVERY = 500

LR_PEAK = 1e-5
LR_FINAL = 0.0

ALPHA_CE = 1.0
BETA_KL = 1.0
TOP_K_KL = 64
TEMPERATURE = 2.0

GROUP_SIZE = 128

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAINABLE_SUFFIXES = ("o_proj", "down_proj")

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
STAGE232_BEST = CKPT_DIR / "stage232_magperm_best.pt"
CKPT_LATEST = CKPT_DIR / "stage234_ext232_latest.pt"
CKPT_BEST = CKPT_DIR / "stage234_ext232_best.pt"
RESULTS_PATH = Path("results/stage234_extended_232.json")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt_cached():
    return torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                      weights_only=True).long()


def lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS):
    losses = []
    model.eval()
    for i in range(n_chunks):
        s = i * SEQ_LEN
        window = val_tokens[s:s + SEQ_LEN + 1]
        if len(window) < SEQ_LEN + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            losses.append(F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)).item())
    return sum(losses) / max(len(losses), 1)


# ─── PerRowSphereLinear ───
class PerRowSphereLinear(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        W = original_linear.weight.data.clone()
        self.weight = nn.Parameter(W)
        with torch.no_grad():
            radius = W.norm(dim=-1, keepdim=True)
        self.register_buffer("row_radius", radius)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        W_unit = self.weight / self.weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W_eff = W_unit * self.row_radius
        return F.linear(x, W_eff, self.bias)


class SubLNWrappedLinear(nn.Module):
    def __init__(self, wrapped_linear, eps=1e-6):
        super().__init__()
        self.wrapped = wrapped_linear
        W = wrapped_linear.weight
        in_features = W.shape[1]
        self.subln_gain = nn.Parameter(torch.ones(in_features,
            device=W.device, dtype=W.dtype))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        x_normed = self.subln_gain * x / rms
        return self.wrapped(x_normed)


# ─── STE ───
class _BinarizeGroupSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, group_size):
        out_f, in_f = W.shape
        g = min(group_size, in_f)
        g_eff = g
        while g_eff > 1 and in_f % g_eff != 0:
            g_eff -= 1
        if g_eff < 1:
            g_eff = in_f
        n_groups = in_f // g_eff
        W_grouped = W.detach().view(out_f, n_groups, g_eff)
        alpha = W_grouped.abs().mean(dim=-1, keepdim=True)
        signs = torch.sign(W.view(out_f, n_groups, g_eff))
        W_q = (signs * alpha).view(out_f, in_f)
        return W_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def binarize_group_ste(W, group_size):
    return _BinarizeGroupSTE.apply(W, group_size)


class BinaryGroupLinear(nn.Module):
    def __init__(self, sphere_linear, group_size):
        super().__init__()
        self.weight = nn.Parameter(sphere_linear.weight.data.clone())
        if sphere_linear.bias is not None:
            self.bias = nn.Parameter(sphere_linear.bias.data.clone())
        else:
            self.bias = None
        self.group_size = group_size
        self.in_features = self.weight.shape[1]

    def set_group_size(self, group_size):
        self.group_size = min(group_size, self.in_features)

    def forward(self, x):
        W_q = binarize_group_ste(self.weight, self.group_size)
        return F.linear(x, W_q, self.bias)


class BinaryGroupLinearWithPerm(nn.Module):
    def __init__(self, src_binary, perm):
        super().__init__()
        self.weight = nn.Parameter(src_binary.weight.data[:, perm].clone())
        if src_binary.bias is not None:
            self.bias = nn.Parameter(src_binary.bias.data.clone())
        else:
            self.bias = None
        self.group_size = src_binary.group_size
        self.in_features = self.weight.shape[1]
        self.register_buffer("perm", perm.to(torch.long))

    def set_group_size(self, group_size):
        self.group_size = min(group_size, self.in_features)

    def forward(self, x):
        x_perm = torch.index_select(x, -1, self.perm)
        W_q = binarize_group_ste(self.weight, self.group_size)
        return F.linear(x_perm, W_q, self.bias)


def replace_linears_with_sphere(model, target_names):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in target_names): continue
        new_layer = PerRowSphereLinear(mod)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def insert_subln_before(model, target_suffixes):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    for name, mod in list(model.named_modules()):
        if not any(name.endswith(s) for s in target_suffixes): continue
        new_layer = SubLNWrappedLinear(mod)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def replace_sphere_with_binary(model, group_size):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, PerRowSphereLinear): continue
        new_layer = BinaryGroupLinear(mod, group_size=group_size)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def replace_binary_with_perm_placeholder(model):
    """Walk model; convert each BinaryGroupLinear → BinaryGroupLinearWithPerm
    with identity permutation (will be overwritten by state_dict load)."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, BinaryGroupLinear): continue
        in_f = mod.weight.shape[1]
        identity_perm = torch.arange(in_f, dtype=torch.long, device=mod.weight.device)
        new_layer = BinaryGroupLinearWithPerm(mod, identity_perm)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def compute_bits_per_weight_with_perm(model):
    total_alpha = 0
    total_weights = 0
    perm_bits = 0
    for mod in model.modules():
        if isinstance(mod, BinaryGroupLinearWithPerm):
            out_f, in_f = mod.weight.shape
            g = min(mod.group_size, in_f)
            g_eff = g
            while g_eff > 1 and in_f % g_eff != 0:
                g_eff -= 1
            n_groups = in_f // max(g_eff, 1)
            total_alpha += out_f * n_groups
            total_weights += out_f * in_f
            perm_bits += in_f * 16
    return (total_weights + total_alpha * 16 + perm_bits) / max(total_weights, 1)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def top_k_kl_loss(student_logits, teacher_logits, k=TOP_K_KL, T=TEMPERATURE):
    s = student_logits / T
    t = teacher_logits / T
    top_t, top_idx = t.topk(k, dim=-1)
    top_s = s.gather(-1, top_idx)
    t_logp = F.log_softmax(top_t, dim=-1)
    s_logp = F.log_softmax(top_s, dim=-1)
    return F.kl_div(s_logp.reshape(-1, k), t_logp.reshape(-1, k),
                    reduction='batchmean', log_target=True) * (T * T)


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 64:SEQ_LEN * 64 + 1_000_000].tolist()
print(f"  val={len(val_tokens)}  train={len(train_tokens)}", flush=True)


# ─── TEACHER ───
print("\nBuilding TEACHER (frozen FP Qwen3-0.6B)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False
T0 = lm_ce(teacher, val_tokens)
print(f"  Teacher T0 = {T0:.4f}", flush=True)


# ─── STUDENT: build Stage 232 arch (perm), load Stage 232 best ───
print("\nBuilding STUDENT (Stage 232 arch + load best)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
n_binary = replace_sphere_with_binary(student, GROUP_SIZE)
n_perm = replace_binary_with_perm_placeholder(student)
print(f"  Built arch: {n_perm} BinaryGroupLinearWithPerm, {n_subln} SubLN", flush=True)

print(f"  Loading Stage 232 best from {STAGE232_BEST} ...", flush=True)
ckpt = torch.load(STAGE232_BEST, map_location=device, weights_only=False)
load_result = student.load_state_dict(ckpt["model"], strict=False)
print(f"    loaded; missing keys: {len(load_result.missing_keys)}", flush=True)
print(f"    Stage 232 step={ckpt.get('step')}  drift_at_save={ckpt.get('drift'):+.4f}",
      flush=True)

bpw = compute_bits_per_weight_with_perm(student)
print(f"  bpw = {bpw:.4f}", flush=True)

ce_post_load = lm_ce(student, val_tokens)
drift_post_load = ce_post_load - T0
print(f"  Drift after load: {drift_post_load:+.4f}  (should match ≈ +1.38)",
      flush=True)


# ─── Training setup with cosine LR over 10000 steps ───
n_frozen = 0
for name, p in student.named_parameters():
    is_body_master = "weight" in name and any(s in name for s in BODY_TRAINABLE_SUFFIXES)
    is_subln = "subln_gain" in name
    is_bias = "bias" in name and "norm" not in name
    if not (is_body_master or is_subln or is_bias):
        p.requires_grad_(False)
        n_frozen += p.numel()
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
print(f"\nFrozen params:    {n_frozen:,}", flush=True)
print(f"Trainable params: {n_trainable:,}", flush=True)
optimizer = torch.optim.Adam(trainable_params, lr=LR_PEAK)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=N_TRAIN_STEPS, eta_min=LR_FINAL)
rng = np.random.default_rng(42)


def step_fn(batch):
    student.train()
    with torch.no_grad():
        teacher_logits = teacher(batch[:, :-1], use_cache=False).logits
    student_logits = student(batch[:, :-1], use_cache=False).logits

    L_ce = F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.size(-1)),
        batch[:, 1:].reshape(-1))

    L_kl = top_k_kl_loss(student_logits.float(), teacher_logits.float(),
                         k=TOP_K_KL, T=TEMPERATURE)

    L_total = ALPHA_CE * L_ce + BETA_KL * L_kl
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    scheduler.step()
    return float(L_ce.item()), float(L_kl.item()), float(L_total.item())


# ─── Training loop ───
t_start = time.time()
history = [{"event": "init", "drift": float(drift_post_load), "bpw": float(bpw)}]
print(f"\n{'─'*60}")
print(f"Stage 234 — Extended Stage 232 (10000 more steps)")
print(f"  Cosine LR {LR_PEAK:.0e} → {LR_FINAL:.0e}  Top-K={TOP_K_KL}  T={TEMPERATURE}")
print(f"  bpw = {bpw:.4f}")
print('─'*60, flush=True)

best_drift = drift_post_load
best_step = 0

for step in range(1, N_TRAIN_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_total = step_fn(batch)

    if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = drift < best_drift
        marker = " ⭐" if is_best else ""
        print(f"  step {step:>5}  L_ce={L_ce:.3f}  L_kl={L_kl:.4f}  "
              f"val_ce={val_ce:.4f}  drift={drift:+.4f}  lr={cur_lr:.2e}  "
              f"{elapsed:.0f}s{marker}", flush=True)
        history.append({"step": step, "L_ce": L_ce, "L_kl": L_kl,
                        "val_ce": float(val_ce), "drift": float(drift),
                        "lr": float(cur_lr)})
        if is_best:
            best_drift = drift
            best_step = step
            torch.save({
                "step": step,
                "val_ce": val_ce,
                "drift": drift,
                "bpw": float(bpw),
                "model": student.state_dict(),
            }, CKPT_BEST)
            print(f"    → saved BEST to {CKPT_BEST.name}  (drift={drift:+.4f})",
                  flush=True)

    if step % CKPT_EVERY == 0:
        torch.save({
            "step": step,
            "val_ce": val_ce,
            "drift": drift,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, CKPT_LATEST)
        print(f"    → saved checkpoint to {CKPT_LATEST.name}", flush=True)


# Final
final_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
final_drift = final_ce - T0
print(f"\n{'─'*60}")
print(f"STAGE 234 RESULT (extended Stage 232):")
print('─'*60)
print(f"  Teacher T0:           {T0:.4f}")
print(f"  Init drift (Stage 232): {drift_post_load:+.4f}")
print(f"  Drift final:          {final_drift:+.4f}")
print(f"  Drift best:           {best_drift:+.4f}  (step {best_step})")
print(f"  bpw:                  {bpw:.4f}")
print(f"  Δ vs Stage 232 best:  {best_drift - 1.3789:+.4f}")
print(f"\n  Comparison points:")
print(f"    Stage 227:               drift=+1.6968  bpw=1.0125")
print(f"    Stage 231 (per-128):     drift=+1.3992  bpw=1.125")
print(f"    Stage 232 (mag-perm):    drift=+1.3789  bpw=1.135")
print(f"    Stage 233 v2 (factored): drift=+1.5814  bpw=1.023")
print(f"    Stage 234 (ext 232):     drift={best_drift:+.4f}  bpw={bpw:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_init": float(drift_post_load),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "bpw": float(bpw),
        "delta_vs_stage232": float(best_drift - 1.3789),
        "n_train_steps": int(N_TRAIN_STEPS),
        "lr_peak": float(LR_PEAK),
        "lr_final": float(LR_FINAL),
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
