"""Stage 233 — Factored α: per-row × per-column.

User direction (2026-05-06): "let's do option F" — after Stage 232 confirmed
magnitude-permutation gives only marginal lift and the carry-flip pair test
showed weights have no exploitable fine-grained pair structure.

Recipe:
  W_quantized[r, c] = sign(W[r, c]) * α_row[r] * α_col[c]

  α_row and α_col are LEARNABLE PARAMETERS (not derived statistics).
  Storage at deployment: 1 sign-bit/weight + (out + in) fp16 per Linear.

Bit-rate per Linear with shape [out, in]:
  bpw = 1 + 16 * (out + in) / (out * in)

For Qwen3-0.6B's 196 Linears:
  in=1024, out=1024 (q/k/v/o):       1 + 32/1024  = 1.0312 bpw
  in=1024, out=3072 (gate/up):        1 + 16*4096/(3072*1024) = 1.021 bpw
  in=3072, out=1024 (down):           1 + 16*4096/(1024*3072) = 1.021 bpw
  Weighted average:                   ~1.025 bpw

vs comparison points:
  Stage 227 per-row α:        1.0128 bpw  drift +1.6968
  Stage 230 Bonsai PTQ-mean:  1.1250 bpw  drift +10.82
  Stage 231 per-128 random:   1.1250 bpw  drift +1.3992
  Stage 232 per-128 mag-perm: 1.1354 bpw  drift +1.3789
  Stage 233 factored α:       ~1.025 bpw  drift ???

Target: beat Stage 227's +1.70 at near-identical bit-rate (~1.025 vs 1.013).
Even matching Stage 231 (+1.40) would be a major efficiency win since we'd
use 0.10 fewer bits/weight than Bonsai's grouping.

Init from Stage 227 best (per-row α already trained). Set α_col = 1 so the
initial forward equals Stage 227. α_col can then diverge to add column-aware
structure.

Distillation: same as Stage 231 (top-K=64 KL, T=2.0, cosine LR 2e-5 → 0).
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
N_TRAIN_STEPS = 5000
EVAL_EVERY = 100
CKPT_EVERY = 500

LR_PEAK = 2e-5
LR_FINAL = 0.0

ALPHA_CE = 1.0
BETA_KL = 1.0
TOP_K_KL = 64
TEMPERATURE = 2.0

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAINABLE_SUFFIXES = ("o_proj", "down_proj")

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
STAGE227_BEST = CKPT_DIR / "binary_perrow_ste_best.pt"
CKPT_LATEST = CKPT_DIR / "stage233_factored_alpha_latest.pt"
CKPT_BEST = CKPT_DIR / "stage233_factored_alpha_best.pt"
RESULTS_PATH = Path("results/stage233_factored_alpha.json")


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


# ─── PerRowSphereLinear (placeholder for arch reconstruction) ───
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


# ─── STE on sign(W); α_row and α_col flow through normal autograd ───
class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        return torch.sign(W)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes unchanged
        return grad_output


def sign_ste(W):
    return _SignSTE.apply(W)


# ─── BinaryFactoredLinear: forward = sign(W) * α_row * α_col ───
class BinaryFactoredLinear(nn.Module):
    """Forward: sign(W) * α_row[r] * α_col[c], STE backward on W.
    α_row and α_col are learnable parameters (separate from W).
    Initialization sets α_row = mean(|W[r,:]|), α_col = ones — so initial
    forward equals per-row α (Stage 227 baseline).
    """
    def __init__(self, src_linear):
        super().__init__()
        self.weight = nn.Parameter(src_linear.weight.data.clone())
        out_f, in_f = self.weight.shape
        with torch.no_grad():
            row_mag = self.weight.detach().abs().mean(dim=1, keepdim=True)  # [out, 1]
        self.alpha_row = nn.Parameter(row_mag.clone())                       # [out, 1]
        self.alpha_col = nn.Parameter(torch.ones(1, in_f,
            dtype=self.weight.dtype, device=self.weight.device))             # [1, in]
        if src_linear.bias is not None:
            self.bias = nn.Parameter(src_linear.bias.data.clone())
        else:
            self.bias = None
        self.in_features = in_f
        self.out_features = out_f

    def forward(self, x):
        signs = sign_ste(self.weight)                # [out, in], STE through W
        W_q = signs * self.alpha_row * self.alpha_col  # broadcasts to [out, in]
        return F.linear(x, W_q, self.bias)

    @torch.no_grad()
    def deployment_artifact(self):
        signs = (self.weight > 0)
        return signs, self.alpha_row.to(torch.float16), self.alpha_col.to(torch.float16)


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


def replace_sphere_with_factored(model):
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    total_alpha_fp16 = 0
    total_weights = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, PerRowSphereLinear): continue
        out_f, in_f = mod.weight.shape
        total_alpha_fp16 += out_f + in_f
        total_weights += out_f * in_f
        new_layer = BinaryFactoredLinear(mod)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    bpw = (total_weights + total_alpha_fp16 * 16) / max(total_weights, 1)
    return n, bpw


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


# ─── STUDENT: build sphere arch, load Stage 227 best, swap to factored α ───
print("\nBuilding STUDENT (sphere arch + load Stage 227 best)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
print(f"  Built sphere arch: {n_replaced} sphere, {n_subln} SubLN", flush=True)

print(f"  Loading Stage 227 state_dict from {STAGE227_BEST} ...", flush=True)
ckpt = torch.load(STAGE227_BEST, map_location=device, weights_only=False)
load_result = student.load_state_dict(ckpt["model"], strict=False)
print(f"    loaded; missing keys: {len(load_result.missing_keys)} (expected: row_radius)",
      flush=True)
print(f"    Stage 227 step={ckpt.get('step')}  drift_at_save={ckpt.get('drift'):+.4f}",
      flush=True)

ce_post_load = lm_ce(student, val_tokens)
drift_post_load = ce_post_load - T0
print(f"  After load (sphere forward): drift={drift_post_load:+.4f}", flush=True)

# Swap to factored α
n_factored, bpw = replace_sphere_with_factored(student)
print(f"  Swapped {n_factored} sphere → BinaryFactoredLinear", flush=True)
print(f"  bpw with factored α: {bpw:.4f}  (vs Bonsai 1.125, Stage 227 1.0125)",
      flush=True)

ce_post_swap = lm_ce(student, val_tokens)
drift_post_swap = ce_post_swap - T0
print(f"  After swap to factored α (α_col=1, init): drift={drift_post_swap:+.4f}",
      flush=True)
print(f"  (should be ≈ Stage 227 +1.6968 since α_col=1 makes it identical)", flush=True)


# ─── Training setup with cosine LR ───
# Mac memory: only train alphas + SubLN gains + biases (~640K params).
# Body W weights frozen — Stage 227 already trained them under per-row α,
# so they're well-fit; alphas can refine on top. This avoids the 147M-param
# autograd graph that swapped Mac to disk in v1.
n_frozen = 0
for name, p in student.named_parameters():
    is_subln = "subln_gain" in name
    is_bias = "bias" in name and "norm" not in name
    is_alpha = "alpha_row" in name or "alpha_col" in name
    if not (is_subln or is_bias or is_alpha):
        p.requires_grad_(False)
        n_frozen += p.numel()
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
n_alpha_row = sum(p.numel() for n, p in student.named_parameters() if "alpha_row" in n)
n_alpha_col = sum(p.numel() for n, p in student.named_parameters() if "alpha_col" in n)
print(f"\nFrozen params:       {n_frozen:,}", flush=True)
print(f"Trainable params:    {n_trainable:,}", flush=True)
print(f"  alpha_row params:  {n_alpha_row:,}", flush=True)
print(f"  alpha_col params:  {n_alpha_col:,}", flush=True)
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
history = [
    {"event": "post_load_sphere", "drift": float(drift_post_load)},
    {"event": "post_swap_factored", "drift": float(drift_post_swap), "bpw": float(bpw)},
]
print(f"\n{'─'*60}")
print(f"Stage 233 — Factored α (per-row × per-col)")
print(f"  N_steps = {N_TRAIN_STEPS}  Cosine LR {LR_PEAK:.0e} → {LR_FINAL:.0e}")
print(f"  Top-K={TOP_K_KL} KL, T={TEMPERATURE}")
print(f"  bpw = {bpw:.4f}")
print('─'*60, flush=True)

best_drift = drift_post_swap
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
print(f"STAGE 233 RESULT (factored α per-row × per-col):")
print('─'*60)
print(f"  Teacher T0:                       {T0:.4f}")
print(f"  Init drift (Stage 227 + α_col=1): {drift_post_swap:+.4f}")
print(f"  Drift final:                      {final_drift:+.4f}")
print(f"  Drift best:                       {best_drift:+.4f}  (step {best_step})")
print(f"  bpw:                              {bpw:.4f}")
print(f"\n  Comparison points:")
print(f"    Stage 227 (per-row α):       drift=+1.6968  bpw=1.0125")
print(f"    Stage 230 (Bonsai PTQ-mean): drift=+10.82   bpw=1.125")
print(f"    Stage 231 (per-128 random):  drift=+1.3992  bpw=1.125")
print(f"    Stage 232 (per-128 mag-perm):drift=+1.3789  bpw=1.135")
print(f"    Stage 233 (factored α):      drift={best_drift:+.4f}  bpw={bpw:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_post_load_sphere": float(drift_post_load),
        "drift_post_swap_factored": float(drift_post_swap),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "bpw": float(bpw),
        "n_factored_linears": int(n_factored),
        "n_subln_inserted": int(n_subln),
        "n_alpha_row_params": int(n_alpha_row),
        "n_alpha_col_params": int(n_alpha_col),
        "n_train_steps": int(N_TRAIN_STEPS),
        "lr_peak": float(LR_PEAK),
        "lr_final": float(LR_FINAL),
        "top_k_kl": int(TOP_K_KL),
        "temperature": float(TEMPERATURE),
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
