"""Stage 227 — BitDistill 1-bit-aware fine-tune from per-row-sphere bake.

Goal (user, 2026-05-04):
  Beat Bonsai at 1-bit. α scope = (layer, type, row) — one fp16 scale
  per output neuron, no sub-row groups. The per-row-sphere bake already
  put each row on its own shell at its natural radius; we now binarize
  the within-row directions and let STE training drive drift to zero.

α-tensor layout:
  Each Linear has α of shape [out_features, 1].
  Storage: 1 sign bit/weight + 1 fp16/row.
    in=1024 Linears: 1.0156 bits/weight
    in=3072 Linears: 1.0052 bits/weight
    weighted avg:    1.0128 bits/weight
  vs. Bonsai's per-128 = 1.125 bits/weight at 89% retention.

Recipe:
  Load Stage 226 best per-row-sphere checkpoint. Replace each
  PerRowSphereLinear with BinaryPerRowLinear: forward computes
  sign(W) * α_row with straight-through estimator on backward.
  Continue KL-distillation against the frozen FP teacher.

  The forward pass IS the quantized math. FP weights remain as a continuous
  gradient proxy that the optimizer moves to compensate for quant error.
  At deployment, only sign bits + α_row are needed.

Why STE not PTQ:
  - Pure post-training sign·α leaves residual within-row magnitude variance
    as irreducible quant error
  - STE drives the FP weights toward a configuration where sign·α IS
    optimal — the network learns to live with the binarization, so the
    forward pass at deployment matches training exactly
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

LR = 2e-5  # gentler than bake; we're fine-tuning under harder constraint

ALPHA_CE = 1.0
BETA_KL = 1.0

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAINABLE_SUFFIXES = ("o_proj", "down_proj")

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
BAKE_BEST = CKPT_DIR / "perrow_sphere_bake_best.pt"
BAKE_LATEST = CKPT_DIR / "perrow_sphere_bake_latest.pt"
CKPT_LATEST = CKPT_DIR / "binary_perrow_ste_latest.pt"
CKPT_BEST = CKPT_DIR / "binary_perrow_ste_best.pt"
RESULTS_PATH = Path("results/stage227_binary_perrow_ste.json")


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


# ─── PerRowSphereLinear (must match Stage 226 for state_dict load) ───
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


# ─── SubLN wrapper (must match Stage 226) ───
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


# ─── STE quant op: forward = sign(W)*α_row, backward = identity wrt W ───
class _BinarizePerRowSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W):
        # α_row = mean(|w|) per row — optimal scalar for 1-bit MSE under sign
        alpha = W.detach().abs().mean(dim=-1, keepdim=True)   # [out, 1]
        return torch.sign(W) * alpha

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes through the quantizer unchanged.
        return grad_output


def binarize_per_row_ste(W):
    return _BinarizePerRowSTE.apply(W)


# ─── BinaryPerRowLinear: replaces PerRowSphereLinear at start of Stage 227 ───
class BinaryPerRowLinear(nn.Module):
    """Forward computes sign(W) * α_row, STE backward.
    α tensor shape: [out_features, 1] = mean(|W_row|) — recomputed every
    forward from current FP weights so α tracks during STE training.
    At deployment, freeze α from the final FP weights and store sign bits.
    """
    def __init__(self, sphere_linear: PerRowSphereLinear):
        super().__init__()
        self.weight = nn.Parameter(sphere_linear.weight.data.clone())
        if sphere_linear.bias is not None:
            self.bias = nn.Parameter(sphere_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        W_q = binarize_per_row_ste(self.weight)
        return F.linear(x, W_q, self.bias)

    @torch.no_grad()
    def deployment_artifact(self):
        """Returns (signs_bool, alpha_fp16):
            signs: [out, in] bool — 1 bit each at storage
            alpha: [out, 1]      fp16
        """
        signs = (self.weight > 0)
        alpha = self.weight.abs().mean(dim=-1, keepdim=True).to(torch.float16)
        return signs, alpha


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


def replace_sphere_with_binary(model):
    """Walk the model and swap every PerRowSphereLinear → BinaryPerRowLinear.
    Done AFTER state_dict load so we inherit the baked FP master weights."""
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
        total_alpha_fp16 += out_f
        total_weights += out_f * in_f
        new_layer = BinaryPerRowLinear(mod)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    bits_per_weight = (total_weights * 1 + total_alpha_fp16 * 16) / max(total_weights, 1)
    return n, bits_per_weight


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def pick_bake_checkpoint():
    if BAKE_BEST.exists():
        return BAKE_BEST
    if BAKE_LATEST.exists():
        print(f"  ! best not found, falling back to {BAKE_LATEST.name}", flush=True)
        return BAKE_LATEST
    raise FileNotFoundError(
        f"Need either {BAKE_BEST} or {BAKE_LATEST} from Stage 226 bake.")


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 64:SEQ_LEN * 64 + 1_000_000].tolist()
print(f"  val={len(val_tokens)}  train={len(train_tokens)}", flush=True)


# ─── TEACHER (frozen FP) ───
print("\nBuilding TEACHER (frozen FP Qwen3-0.6B)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False
T0 = lm_ce(teacher, val_tokens)
print(f"  Teacher T0 = {T0:.4f}", flush=True)


# ─── STUDENT: rebuild sphere architecture, load bake state_dict, swap to binary ───
print("\nBuilding STUDENT (sphere arch, load bake, swap to binary)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
print(f"  Built sphere arch: {n_replaced} PerRowSphereLinear, {n_subln} SubLN", flush=True)

bake_path = pick_bake_checkpoint()
print(f"  Loading bake state_dict from {bake_path} ...", flush=True)
ckpt = torch.load(bake_path, map_location=device, weights_only=False)
student.load_state_dict(ckpt["model"])
print(f"    bake step={ckpt.get('step')}  drift={ckpt.get('drift'):+.4f}",
      flush=True)

ce_post_load = lm_ce(student, val_tokens)
drift_post_load = ce_post_load - T0
print(f"  After load (sphere forward): drift={drift_post_load:+.4f}",
      flush=True)

# Now swap PerRowSphereLinear → BinaryPerRowLinear (forward = sign·α_row, STE)
n_binary, bits_per_weight = replace_sphere_with_binary(student)
print(f"  Swapped {n_binary} PerRowSphereLinear → BinaryPerRowLinear", flush=True)
print(f"  α scope: (layer, type, row)  "
      f"→ {bits_per_weight:.4f} bits/weight at deployment "
      f"(vs Bonsai's 1.125)", flush=True)

ce_post_quant = lm_ce(student, val_tokens)
drift_post_quant = ce_post_quant - T0
print(f"  After binarization (PTQ pre-fine-tune): drift={drift_post_quant:+.4f}",
      flush=True)


# ─── Training setup — Stage 189 scope ───
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
optimizer = torch.optim.Adam(trainable_params, lr=LR)
rng = np.random.default_rng(42)


def step_fn(batch):
    student.train()
    with torch.no_grad():
        teacher_logits = teacher(batch[:, :-1], use_cache=False).logits
    student_logits = student(batch[:, :-1], use_cache=False).logits

    L_ce = F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.size(-1)),
        batch[:, 1:].reshape(-1))

    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    L_kl = F.kl_div(
        student_log_probs.reshape(-1, student_log_probs.size(-1)),
        teacher_log_probs.reshape(-1, teacher_log_probs.size(-1)),
        reduction='batchmean', log_target=True)

    L_total = ALPHA_CE * L_ce + BETA_KL * L_kl
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    return float(L_ce.item()), float(L_kl.item()), float(L_total.item())


# ─── Training loop ───
t_start = time.time()
history = [
    {"event": "post_load_sphere", "drift": float(drift_post_load)},
    {"event": "post_binarize_PTQ", "drift": float(drift_post_quant)},
]
print(f"\n{'─'*60}")
print(f"Stage 227 — 1-bit STE fine-tune from per-row-sphere bake")
print(f"  N_steps = {N_TRAIN_STEPS}  LR = {LR}  α/β = {ALPHA_CE}/{BETA_KL}")
print('─'*60, flush=True)

best_drift = drift_post_quant
best_step = 0
for step in range(1, N_TRAIN_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_total = step_fn(batch)

    if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        is_best = drift < best_drift
        marker = " ⭐" if is_best else ""
        print(f"  step {step:>5}  L_ce={L_ce:.3f}  L_kl={L_kl:.4f}  "
              f"val_ce={val_ce:.4f}  drift={drift:+.4f}  {elapsed:.0f}s{marker}",
              flush=True)
        history.append({"step": step, "L_ce": L_ce, "L_kl": L_kl,
                        "val_ce": float(val_ce), "drift": float(drift)})
        if is_best:
            best_drift = drift
            best_step = step
            torch.save({
                "step": step,
                "val_ce": val_ce,
                "drift": drift,
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
        }, CKPT_LATEST)
        print(f"    → saved checkpoint to {CKPT_LATEST.name}", flush=True)


# Final summary
final_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
final_drift = final_ce - T0
print(f"\n{'─'*60}")
print(f"STAGE 227 RESULT:")
print('─'*60)
print(f"  Teacher T0:           {T0:.4f}")
print(f"  Drift after bake load:{drift_post_load:+.4f}")
print(f"  Drift after binarize: {drift_post_quant:+.4f}  (PTQ pre-STE)")
print(f"  Drift final:          {final_drift:+.4f}  (after {N_TRAIN_STEPS} STE steps)")
print(f"  Drift best:           {best_drift:+.4f}  (step {best_step})")
print(f"\nForward IS exact 1-bit math (sign·α_row).")
print(f"Deployment artifact: 1 sign-bit/weight + fp16 α/row "
      f"= {bits_per_weight:.4f} bits/weight  (vs Bonsai 1.125)")

torch.save({
    "step": N_TRAIN_STEPS,
    "val_ce": final_ce,
    "drift": final_drift,
    "model": student.state_dict(),
    "optimizer": optimizer.state_dict(),
}, CKPT_LATEST)

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_post_load_sphere": float(drift_post_load),
        "drift_post_binarize_PTQ": float(drift_post_quant),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "n_replaced_linears": int(n_replaced),
        "n_subln_inserted": int(n_subln),
        "n_binary_swapped": int(n_binary),
        "n_trainable_params": int(n_trainable),
        "alpha_scope": "per (layer, type, row)",
        "bits_per_weight": float(bits_per_weight),
        "loss_weights": {"alpha_ce": ALPHA_CE, "beta_kl": BETA_KL},
        "n_train_steps": N_TRAIN_STEPS,
        "lr": LR,
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
