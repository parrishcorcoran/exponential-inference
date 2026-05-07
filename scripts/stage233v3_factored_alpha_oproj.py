"""Stage 233 v3 — Factored α + o_proj W training.

v2 took factored α as far as it could with frozen W: drift +1.58 at 1.023 bpw.
The plateau came from W signs being optimized for per-row α, not per-row × per-col.
This v3 unfreezes o_proj body weights only (~28M params, fits in 16GB Mac)
to let signs migrate under the new α structure.

Recipe:
  1. Load Stage 233 v2 best (factored α already trained, frozen W).
  2. Unfreeze o_proj body weights (28 layers × 1024×1024 = 28.7M params).
  3. Train α_row + α_col + o_proj W + SubLN gains + biases.
  4. Cosine LR 1e-5 → 0 over 3000 steps. ~50 min.

Total trainable: ~29.5M (5x v2, but 5x less than v1's 147M).

If v3 lands below Stage 232's +1.38 at 1.023 bpw, factored α wins per-bit
efficiency. If it lands at +1.40-1.50, factored α is close but per-128 wins.
If it lands at +1.55+, the factorization is fundamentally limited (rank-1
magnitude approximation can't capture the structure that per-128 captures).
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
N_TRAIN_STEPS = 3000
EVAL_EVERY = 100
CKPT_EVERY = 500

LR_PEAK = 1e-5  # gentler than Stage 231 (we're refining a converged checkpoint)
LR_FINAL = 0.0

ALPHA_CE = 1.0
BETA_KL = 1.0
TOP_K_KL = 64
TEMPERATURE = 2.0

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
# v3: only o_proj W unfrozen (down_proj stays frozen — too much memory)
W_TRAINABLE_SUFFIXES = ("o_proj",)

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
STAGE233_V2_BEST = CKPT_DIR / "stage233_factored_alpha_best.pt"
CKPT_LATEST = CKPT_DIR / "stage233v3_factored_oproj_latest.pt"
CKPT_BEST = CKPT_DIR / "stage233v3_factored_oproj_best.pt"
RESULTS_PATH = Path("results/stage233v3_factored_alpha_oproj.json")


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


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        return torch.sign(W)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign_ste(W):
    return _SignSTE.apply(W)


class BinaryFactoredLinear(nn.Module):
    def __init__(self, src_linear):
        super().__init__()
        self.weight = nn.Parameter(src_linear.weight.data.clone())
        out_f, in_f = self.weight.shape
        with torch.no_grad():
            row_mag = self.weight.detach().abs().mean(dim=1, keepdim=True)
        self.alpha_row = nn.Parameter(row_mag.clone())
        self.alpha_col = nn.Parameter(torch.ones(1, in_f,
            dtype=self.weight.dtype, device=self.weight.device))
        if src_linear.bias is not None:
            self.bias = nn.Parameter(src_linear.bias.data.clone())
        else:
            self.bias = None
        self.in_features = in_f
        self.out_features = out_f

    def forward(self, x):
        signs = sign_ste(self.weight)
        W_q = signs * self.alpha_row * self.alpha_col
        return F.linear(x, W_q, self.bias)


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


# ─── STUDENT: rebuild factored arch + load Stage 233 v2 best ───
print("\nBuilding STUDENT (factored arch + load v2 best)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
n_factored, bpw = replace_sphere_with_factored(student)
print(f"  Built arch: {n_replaced} sphere → factored, {n_subln} SubLN", flush=True)
print(f"  bpw = {bpw:.4f}", flush=True)

print(f"  Loading Stage 233 v2 best from {STAGE233_V2_BEST} ...", flush=True)
ckpt = torch.load(STAGE233_V2_BEST, map_location=device, weights_only=False)
load_result = student.load_state_dict(ckpt["model"], strict=False)
print(f"    loaded; missing keys: {len(load_result.missing_keys)}", flush=True)
print(f"    v2 step={ckpt.get('step')}  drift_at_save={ckpt.get('drift'):+.4f}",
      flush=True)

ce_post_load = lm_ce(student, val_tokens)
drift_post_load = ce_post_load - T0
print(f"  Drift after load: {drift_post_load:+.4f}  "
      f"(should match v2 best ≈ +1.58)", flush=True)


# ─── Training setup: o_proj W + alphas + SubLN + biases ───
n_frozen = 0
n_oproj_w = 0
for name, p in student.named_parameters():
    is_oproj_weight = (any(s in name for s in W_TRAINABLE_SUFFIXES)
                       and "weight" in name and "alpha" not in name and "subln" not in name)
    is_subln = "subln_gain" in name
    is_bias = "bias" in name and "norm" not in name
    is_alpha = "alpha_row" in name or "alpha_col" in name
    if is_oproj_weight:
        n_oproj_w += p.numel()
    if not (is_oproj_weight or is_subln or is_bias or is_alpha):
        p.requires_grad_(False)
        n_frozen += p.numel()
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
n_alpha = sum(p.numel() for n, p in student.named_parameters()
              if "alpha_row" in n or "alpha_col" in n)
print(f"\nFrozen params:       {n_frozen:,}", flush=True)
print(f"Trainable params:    {n_trainable:,}", flush=True)
print(f"  o_proj W params:   {n_oproj_w:,}", flush=True)
print(f"  alpha params:      {n_alpha:,}", flush=True)
print(f"  (rest: SubLN gains + biases)", flush=True)
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
print(f"Stage 233 v3 — Factored α + o_proj W training")
print(f"  N_steps = {N_TRAIN_STEPS}  Cosine LR {LR_PEAK:.0e} → {LR_FINAL:.0e}")
print(f"  Top-K={TOP_K_KL} KL, T={TEMPERATURE}")
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
print(f"STAGE 233 v3 RESULT (factored α + o_proj W training):")
print('─'*60)
print(f"  Teacher T0:           {T0:.4f}")
print(f"  Init drift (v2 load): {drift_post_load:+.4f}")
print(f"  Drift final:          {final_drift:+.4f}")
print(f"  Drift best:           {best_drift:+.4f}  (step {best_step})")
print(f"  bpw:                  {bpw:.4f}")
print(f"\n  Comparison points:")
print(f"    Stage 227 (per-row α):       drift=+1.6968  bpw=1.0125")
print(f"    Stage 231 (per-128 random):  drift=+1.3992  bpw=1.125")
print(f"    Stage 232 (per-128 mag-perm):drift=+1.3789  bpw=1.135")
print(f"    Stage 233 v2 (factored α):   drift=+1.5814  bpw=1.0229")
print(f"    Stage 233 v3 (+o_proj W):    drift={best_drift:+.4f}  bpw={bpw:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_init_v2_load": float(drift_post_load),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "bpw": float(bpw),
        "n_oproj_w_trainable": int(n_oproj_w),
        "n_alpha_trainable": int(n_alpha),
        "n_train_steps": int(N_TRAIN_STEPS),
        "lr_peak": float(LR_PEAK),
        "lr_final": float(LR_FINAL),
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
