"""Stage 231 — Group-size annealing + richer distillation, 0.6B teacher.

Built on user's "baby steps with training awareness" framing (2026-05-05):
  Bonsai's per-128 PTQ alone is +10.82 on 0.6B (catastrophic). Their 89%
  must come from training. We approach the same per-128 grouping but as
  a gradual training-aware anneal from our Stage 227 per-row baseline.

  The model already knows binarization (Stage 227 baked + STE'd to +1.70).
  Now we let each row's single α progressively split into more αs as the
  group_size shrinks toward 128. At each split, new αs init equal to the
  parent so the forward is identical at the moment of split. STE training
  then lets each new α diverge to fit its own 128-group's distribution.

Schedule (all phases × 1500 steps = 7500 total, ~80 min on Mac):
  Phase 1 (    0-1500): group_size = in_features         (per-row, == Stage 227)
  Phase 2 (1500-3000):  group_size = in_features/2
  Phase 3 (3000-4500):  group_size = in_features/4
  Phase 4 (4500-6000):  group_size = in_features/8
  Phase 5 (6000-7500):  group_size = 128                 (Bonsai-format target)

For in=1024 Linears: 1024 → 512 → 256 → 128 → 128 (last two equal)
For in=3072 Linears: 3072 → 1536 → 768 → 384 → 128 (final snap)

Distillation upgrades over Stage 227 (still 0.6B teacher):
  - Top-K KL on top-64 tokens (vs full 151K vocab) — denser signal, less tail noise
  - Temperature T=2.0 softening — amplifies tail probability mass
  - Cosine LR 2e-5 → 0 over 7500 steps — let signs commit at end
  - Hidden-state distill still off (Mac memory)

Expected: drift below +1.70 (improving over Stage 227) at end-of-anneal,
landing at Bonsai's bit-rate (~1.125 bpw) but with our QAT recipe.
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

PHASE_STEPS = 1500
N_PHASES = 5
N_TRAIN_STEPS = PHASE_STEPS * N_PHASES   # 7500
EVAL_EVERY = 100
CKPT_EVERY = 500

LR_PEAK = 2e-5
LR_FINAL = 0.0

ALPHA_CE = 1.0
BETA_KL = 1.0

TOP_K_KL = 64       # KL only over top-64 teacher tokens (per position)
TEMPERATURE = 2.0   # softens both teacher and student before KL

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAINABLE_SUFFIXES = ("o_proj", "down_proj")

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
STAGE227_BEST = CKPT_DIR / "binary_perrow_ste_best.pt"
CKPT_LATEST = CKPT_DIR / "stage231_group_anneal_latest.pt"
CKPT_BEST = CKPT_DIR / "stage231_group_anneal_best.pt"
RESULTS_PATH = Path("results/stage231_group_anneal_richer_distill.json")


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


# ─── PerRowSphereLinear (must match Stage 226 to load checkpoint cleanly) ───
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


# ─── SubLN wrapper ───
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


# ─── Group-aware STE binarization ───
class _BinarizeGroupSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, group_size):
        out_f, in_f = W.shape
        # Snap to in_f if group_size >= in_f (per-row)
        g = min(group_size, in_f)
        # Use the largest divisor of in_f that's ≤ g (to avoid padding)
        # In practice in_f = 1024 or 3072, both have many divisors
        # Find a g' ≤ g such that in_f % g' == 0
        g_eff = g
        while g_eff > 1 and in_f % g_eff != 0:
            g_eff -= 1
        if g_eff < 1:
            g_eff = in_f  # fallback to per-row
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


# ─── BinaryGroupLinear: group_size is mutable, recomputed each forward ───
class BinaryGroupLinear(nn.Module):
    """Forward: sign(W) * α[row, group_within_row], STE backward.
    group_size is an attribute on the module, updated externally between phases.
    α is recomputed every forward from current FP weights at the current group_size.
    """
    def __init__(self, sphere_linear, group_size: int):
        super().__init__()
        # Inherit FP master weights from baked sphere linear
        self.weight = nn.Parameter(sphere_linear.weight.data.clone())
        if sphere_linear.bias is not None:
            self.bias = nn.Parameter(sphere_linear.bias.data.clone())
        else:
            self.bias = None
        self.group_size = group_size
        self.in_features = self.weight.shape[1]

    def set_group_size(self, group_size: int):
        self.group_size = min(group_size, self.in_features)

    def forward(self, x):
        W_q = binarize_group_ste(self.weight, self.group_size)
        return F.linear(x, W_q, self.bias)

    def effective_group_size(self):
        """Largest divisor of in_features that's ≤ self.group_size."""
        g = min(self.group_size, self.in_features)
        while g > 1 and self.in_features % g != 0:
            g -= 1
        return max(g, 1)

    @torch.no_grad()
    def deployment_artifact(self):
        out_f, in_f = self.weight.shape
        g_eff = self.effective_group_size()
        n_groups = in_f // g_eff
        signs = (self.weight > 0)
        alpha = self.weight.view(out_f, n_groups, g_eff).abs().mean(dim=-1).to(torch.float16)
        return signs, alpha, g_eff


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


def replace_sphere_with_binary_group(model, initial_group_size_fn):
    """Walk model, swap PerRowSphereLinear → BinaryGroupLinear with given initial group_size.
    initial_group_size_fn(in_features) returns the initial group size for that Linear."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, PerRowSphereLinear): continue
        in_f = mod.weight.shape[1]
        gs = initial_group_size_fn(in_f)
        new_layer = BinaryGroupLinear(mod, group_size=gs)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def update_all_group_sizes(model, group_size_fn):
    """Set group_size on every BinaryGroupLinear via group_size_fn(in_features)."""
    n = 0
    for mod in model.modules():
        if isinstance(mod, BinaryGroupLinear):
            mod.set_group_size(group_size_fn(mod.in_features))
            n += 1
    return n


def compute_bits_per_weight(model):
    """Effective bit-rate at deployment given current group sizes."""
    total_alpha = 0
    total_weights = 0
    for mod in model.modules():
        if isinstance(mod, BinaryGroupLinear):
            out_f, in_f = mod.weight.shape
            g_eff = mod.effective_group_size()
            n_groups = in_f // g_eff
            total_alpha += out_f * n_groups
            total_weights += out_f * in_f
    return (total_weights + total_alpha * 16) / max(total_weights, 1)


# ─── Phase schedule: returns group_size for given Linear's in_features at given phase ───
def phase_group_size(phase_idx, in_features):
    """Phase 1..5; phase_idx is 0-indexed.
    Phase 1 (0): in_features (per-row)
    Phase 2 (1): in_features/2
    Phase 3 (2): in_features/4
    Phase 4 (3): in_features/8
    Phase 5 (4): 128 (Bonsai)
    Always at least 128."""
    if phase_idx >= 4:  # final phase: snap to 128
        return 128
    divisor = 2 ** phase_idx
    return max(in_features // divisor, 128)


def step_to_phase(step):
    return min(step // PHASE_STEPS, N_PHASES - 1)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


def top_k_kl_loss(student_logits, teacher_logits, k=TOP_K_KL, T=TEMPERATURE):
    """Top-K KL divergence with temperature softening.
    Selects top-k tokens by teacher logits, computes KL only on those,
    scaled by T^2 to preserve gradient magnitude."""
    s = student_logits / T
    t = teacher_logits / T
    top_t, top_idx = t.topk(k, dim=-1)              # [..., k]
    top_s = s.gather(-1, top_idx)                    # [..., k]
    # log-softmax over the top-k slice
    t_logp = F.log_softmax(top_t, dim=-1)
    s_logp = F.log_softmax(top_s, dim=-1)
    # KL(s || t) over top-k, then T² scaling per Hinton
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


# ─── STUDENT: rebuild sphere arch, load Stage 227 best, swap to binary-group ───
print("\nBuilding STUDENT (sphere arch, load Stage 227 best, swap to binary-group)...",
      flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
print(f"  Built sphere arch: {n_replaced} PerRowSphereLinear, {n_subln} SubLN", flush=True)

# Stage 227 saved BinaryPerRowLinear state_dict, but Stage 226 used
# PerRowSphereLinear. Stage 227 swapped sphere → binary. The state_dict from
# Stage 227 has parameter names matching that model (binary linears).
# To load cleanly we need to swap to binary first, THEN load.
# But binary needs sphere as input. Workaround: load Stage 227 onto sphere arch
# (param names are the same: weight, bias, subln_gain — sphere's row_radius
# is a buffer, so missing key warning is ok)
print(f"  Loading Stage 227 state_dict from {STAGE227_BEST} ...", flush=True)
ckpt = torch.load(STAGE227_BEST, map_location=device, weights_only=False)
# Stage 227 model stored `weight` for each binary layer (no row_radius). The
# sphere arch will accept those weights but its row_radius buffer stays at the
# original FP teacher's values — which is fine, we're swapping to binary next.
load_result = student.load_state_dict(ckpt["model"], strict=False)
print(f"    loaded; missing keys: {len(load_result.missing_keys)} "
      f"(expected: row_radius buffers)", flush=True)
print(f"    bake step={ckpt.get('step')}  drift_at_save={ckpt.get('drift'):+.4f}",
      flush=True)

# Swap PerRowSphereLinear → BinaryGroupLinear with initial group_size = in_features (per-row)
def initial_group_size_fn(in_f):
    return phase_group_size(0, in_f)   # phase 0 = per-row

n_binary = replace_sphere_with_binary_group(student, initial_group_size_fn)
print(f"  Swapped {n_binary} PerRowSphereLinear → BinaryGroupLinear", flush=True)

bpw_init = compute_bits_per_weight(student)
ce_post_swap = lm_ce(student, val_tokens)
drift_post_swap = ce_post_swap - T0
print(f"  After load + swap (per-row): bpw={bpw_init:.4f}  "
      f"drift={drift_post_swap:+.4f}", flush=True)
print(f"  (should match Stage 227 best ≈ +1.70)", flush=True)


# ─── Training setup with cosine LR ───
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


# ─── Training loop with phase transitions ───
t_start = time.time()
history = [{"event": "init", "drift": float(drift_post_swap), "bpw": float(bpw_init)}]
print(f"\n{'─'*60}")
print(f"Stage 231 — Group-anneal + richer distillation (0.6B teacher)")
print(f"  Schedule: 5 phases × {PHASE_STEPS} steps = {N_TRAIN_STEPS} total")
print(f"  Cosine LR {LR_PEAK:.0e} → {LR_FINAL:.0e}")
print(f"  Top-K={TOP_K_KL} KL, temperature T={TEMPERATURE}")
print('─'*60, flush=True)

best_drift = drift_post_swap
best_step = 0
current_phase = -1

for step in range(1, N_TRAIN_STEPS + 1):
    # Phase transition?
    new_phase = step_to_phase(step - 1)  # 0-indexed; step 1 starts phase 0
    if new_phase != current_phase:
        current_phase = new_phase
        # Update all BinaryGroupLinear group_sizes
        update_all_group_sizes(student, lambda in_f: phase_group_size(current_phase, in_f))
        bpw_now = compute_bits_per_weight(student)
        print(f"\n  ── Phase {current_phase + 1}/{N_PHASES} starts at step {step} ──",
              flush=True)
        print(f"     in=1024 group→{phase_group_size(current_phase, 1024)}  "
              f"in=3072 group→{phase_group_size(current_phase, 3072)}  "
              f"bpw={bpw_now:.4f}", flush=True)
        # Evaluate at phase boundary
        val_ce_t = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift_t = val_ce_t - T0
        print(f"     drift at phase start: {drift_t:+.4f}", flush=True)
        history.append({"event": f"phase_{current_phase}_start", "step": step,
                        "drift": float(drift_t), "bpw": float(bpw_now)})

    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_total = step_fn(batch)

    if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = drift < best_drift
        marker = " ⭐" if is_best else ""
        print(f"  step {step:>5} P{current_phase + 1}  L_ce={L_ce:.3f}  "
              f"L_kl={L_kl:.4f}  val_ce={val_ce:.4f}  drift={drift:+.4f}  "
              f"lr={cur_lr:.2e}  {elapsed:.0f}s{marker}", flush=True)
        history.append({"step": step, "phase": current_phase + 1,
                        "L_ce": L_ce, "L_kl": L_kl,
                        "val_ce": float(val_ce), "drift": float(drift),
                        "lr": float(cur_lr)})
        if is_best:
            best_drift = drift
            best_step = step
            torch.save({
                "step": step,
                "phase": current_phase + 1,
                "val_ce": val_ce,
                "drift": drift,
                "bpw": float(compute_bits_per_weight(student)),
                "model": student.state_dict(),
            }, CKPT_BEST)
            print(f"    → saved BEST to {CKPT_BEST.name}  (drift={drift:+.4f})",
                  flush=True)

    if step % CKPT_EVERY == 0:
        torch.save({
            "step": step,
            "phase": current_phase + 1,
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
final_bpw = compute_bits_per_weight(student)
print(f"\n{'─'*60}")
print(f"STAGE 231 RESULT (group-anneal + richer distill, 0.6B teacher):")
print('─'*60)
print(f"  Teacher T0:           {T0:.4f}")
print(f"  Init drift (Stage 227 load + per-row): {drift_post_swap:+.4f}")
print(f"  Final drift:          {final_drift:+.4f}  bpw={final_bpw:.4f}")
print(f"  Best drift:           {best_drift:+.4f}  (step {best_step})")
print(f"\n  Comparison points:")
print(f"    Stage 227 (per-row, bake+STE):         drift=+1.6968  bpw=1.0125")
print(f"    Stage 230 Bonsai-PTQ-mean (no train):  drift=+10.8249 bpw=1.1250")
print(f"    Stage 231 final (per-128, full anneal):drift={final_drift:+.4f} bpw={final_bpw:.4f}")

torch.save({
    "step": N_TRAIN_STEPS,
    "phase": current_phase + 1,
    "val_ce": final_ce,
    "drift": final_drift,
    "bpw": final_bpw,
    "model": student.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}, CKPT_LATEST)

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_init_stage227_load": float(drift_post_swap),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "bpw_init": float(bpw_init),
        "bpw_final": float(final_bpw),
        "phase_steps": int(PHASE_STEPS),
        "n_phases": int(N_PHASES),
        "n_train_steps": int(N_TRAIN_STEPS),
        "lr_peak": float(LR_PEAK),
        "lr_final": float(LR_FINAL),
        "top_k_kl": int(TOP_K_KL),
        "temperature": float(TEMPERATURE),
        "loss_weights": {"alpha_ce": ALPHA_CE, "beta_kl": BETA_KL},
        "comparison": {
            "stage227_per_row_bake_ste_drift": 1.6968,
            "stage227_bpw": 1.0125,
            "stage230_bonsai_ptq_mean_drift": 10.8249,
            "stage230_bpw": 1.1250,
        },
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
