"""Stage 235 — Bake-cycle on Stage 234 best.

User direction (2026-05-06): "There must be more here, we are getting closer!"

Stage 234 reached +1.23 at 1.135 bpw — best so far. Stage 226→227's first
bake cycle gave +2.92 → +0.14 → +1.70 (huge structural gain). We've never
done a *second* bake cycle. This stage tests whether the bake's smooth
constraint can shake the model out of Stage 234's local minimum into a
better basin.

Recipe:
  Phase 1 (steps 1-3000): SMOOTH BAKE
    - Replace BinaryGroupLinearWithPerm → Per128GroupSphereLinearWithPerm
    - Forward: x_perm = x[perm]; W_unit = W_group / ||W_group||; W_eff = W_unit * locked_radii
    - Locked radii captured at init from Stage 234's current weights
    - Smooth differentiable forward (no STE)
    - Train body W under sphere constraint, distill from FP teacher
    - LR cosine 1e-5 → 1e-6 (don't fully decay, more learning in phase 2)

  Phase 2 (steps 3001-10000): STE refinement
    - Swap Per128GroupSphereLinearWithPerm → BinaryGroupLinearWithPerm
    - Forward: x_perm = x[perm]; W_q = sign(W) * mean(|W_group|)
    - STE backward
    - Fresh cosine LR 1e-5 → 0 over 7000 steps
    - Same distillation: top-K=64 KL, T=2.0

If Phase 1 ends with low drift (better than +1.23), Phase 2 should be
better than Stage 234 in absolute terms. If Phase 1 disrupts and doesn't
recover by end of Phase 2, we have Stage 234 best as fallback.
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

PHASE1_STEPS = 3000
PHASE2_STEPS = 7000
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS

EVAL_EVERY = 100
CKPT_EVERY = 500

PHASE1_LR_PEAK = 1e-5
PHASE1_LR_FINAL = 1e-6
PHASE2_LR_PEAK = 1e-5
PHASE2_LR_FINAL = 0.0

ALPHA_CE = 1.0
BETA_KL = 1.0
TOP_K_KL = 64
TEMPERATURE = 2.0

GROUP_SIZE = 128

TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
BODY_TRAINABLE_SUFFIXES = ("o_proj", "down_proj")

CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
STAGE234_BEST = CKPT_DIR / "stage234_ext232_best.pt"
CKPT_LATEST = CKPT_DIR / "stage235_bake_cycle_latest.pt"
CKPT_BEST = CKPT_DIR / "stage235_bake_cycle_best.pt"
RESULTS_PATH = Path("results/stage235_bake_cycle.json")


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


# ─── Architecture placeholders (sphere first to load Stage 234 weights) ───
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


# ─── STE binarization ───
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

    def forward(self, x):
        x_perm = torch.index_select(x, -1, self.perm)
        W_q = binarize_group_ste(self.weight, self.group_size)
        return F.linear(x_perm, W_q, self.bias)


# ─── Per128GroupSphereLinearWithPerm: smooth bake with permutation ───
class Per128GroupSphereLinearWithPerm(nn.Module):
    """Smooth normalization per 128-group + permutation gather.
    Each group's L2 norm is locked at init. Direction within group is
    free to move under gradient, magnitude (per-group radius) is fixed.
    """
    def __init__(self, src_binary_with_perm, group_size=128):
        super().__init__()
        self.weight = nn.Parameter(src_binary_with_perm.weight.data.clone())
        if src_binary_with_perm.bias is not None:
            self.bias = nn.Parameter(src_binary_with_perm.bias.data.clone())
        else:
            self.bias = None
        self.in_features = self.weight.shape[1]
        self.group_size = min(group_size, self.in_features)
        self.register_buffer("perm", src_binary_with_perm.perm.clone().to(torch.long))
        # Lock per-group radii from current weights
        out_f, in_f = self.weight.shape
        # Use largest divisor of in_f ≤ group_size
        g_eff = self.group_size
        while g_eff > 1 and in_f % g_eff != 0:
            g_eff -= 1
        self.g_eff = g_eff if g_eff >= 1 else in_f
        self.n_groups = in_f // self.g_eff
        with torch.no_grad():
            W_grouped = self.weight.view(out_f, self.n_groups, self.g_eff)
            radii = W_grouped.norm(dim=-1, keepdim=True)  # [out, n_groups, 1]
        self.register_buffer("group_radii", radii)

    def forward(self, x):
        x_perm = torch.index_select(x, -1, self.perm)
        out_f, in_f = self.weight.shape
        W_grouped = self.weight.view(out_f, self.n_groups, self.g_eff)
        W_norms = W_grouped.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W_unit = W_grouped / W_norms
        W_eff = (W_unit * self.group_radii).view(out_f, in_f)
        return F.linear(x_perm, W_eff, self.bias)


# ─── Architecture transitions ───
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


def replace_binary_perm_with_sphere_perm(model, group_size=128):
    """Phase 1 entry: BinaryGroupLinearWithPerm → Per128GroupSphereLinearWithPerm."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, BinaryGroupLinearWithPerm): continue
        new_layer = Per128GroupSphereLinearWithPerm(mod, group_size=group_size)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n


def replace_sphere_perm_with_binary_perm(model, group_size=128):
    """Phase 2 entry: Per128GroupSphereLinearWithPerm → BinaryGroupLinearWithPerm."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, Per128GroupSphereLinearWithPerm): continue
        # Build a BinaryGroupLinearWithPerm from sphere's weight + perm
        new_layer = BinaryGroupLinearWithPerm.__new__(BinaryGroupLinearWithPerm)
        nn.Module.__init__(new_layer)
        new_layer.weight = nn.Parameter(mod.weight.data.clone())
        if mod.bias is not None:
            new_layer.bias = nn.Parameter(mod.bias.data.clone())
        else:
            new_layer.bias = None
        new_layer.group_size = mod.group_size
        new_layer.in_features = mod.in_features
        new_layer.register_buffer("perm", mod.perm.clone().to(torch.long))
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


def setup_trainable(student):
    """Mark only body W (o/down) + SubLN gains + biases as trainable."""
    for name, p in student.named_parameters():
        is_body_master = "weight" in name and any(s in name for s in BODY_TRAINABLE_SUFFIXES)
        is_subln = "subln_gain" in name
        is_bias = "bias" in name and "norm" not in name
        if is_body_master or is_subln or is_bias:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


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


# ─── STUDENT: rebuild Stage 234 arch + load best ───
print("\nBuilding STUDENT (Stage 234 arch + load best)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

n_replaced = replace_linears_with_sphere(student, TARGET_NAMES)
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
n_binary = replace_sphere_with_binary(student, GROUP_SIZE)
n_perm = replace_binary_with_perm_placeholder(student)
print(f"  Built arch: {n_perm} BinaryGroupLinearWithPerm, {n_subln} SubLN", flush=True)

print(f"  Loading Stage 234 best from {STAGE234_BEST} ...", flush=True)
ckpt = torch.load(STAGE234_BEST, map_location=device, weights_only=False)
load_result = student.load_state_dict(ckpt["model"], strict=False)
print(f"    loaded; missing keys: {len(load_result.missing_keys)}", flush=True)
print(f"    Stage 234 step={ckpt.get('step')}  drift_at_save={ckpt.get('drift'):+.4f}",
      flush=True)

ce_post_load = lm_ce(student, val_tokens)
drift_post_load = ce_post_load - T0
print(f"  Drift after load: {drift_post_load:+.4f}  (should match ≈ +1.23)",
      flush=True)


# ─── PHASE 1: SMOOTH BAKE ───
print(f"\n{'─'*60}")
print(f"PHASE 1 — SMOOTH BAKE  ({PHASE1_STEPS} steps)")
print(f"  Replace BinaryGroupLinearWithPerm → Per128GroupSphereLinearWithPerm")
print(f"  Smooth per-group normalization, locked group radii")
print(f"  LR cosine {PHASE1_LR_PEAK:.0e} → {PHASE1_LR_FINAL:.0e}")
print('─'*60, flush=True)

n_sphere_perm = replace_binary_perm_with_sphere_perm(student, group_size=GROUP_SIZE)
print(f"  Swapped {n_sphere_perm} → Per128GroupSphereLinearWithPerm", flush=True)

ce_post_swap1 = lm_ce(student, val_tokens)
drift_post_swap1 = ce_post_swap1 - T0
print(f"  Drift after swap to sphere: {drift_post_swap1:+.4f}", flush=True)
print(f"  Δ from arch swap: {drift_post_swap1 - drift_post_load:+.4f}", flush=True)

setup_trainable(student)
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
print(f"  Trainable params: {n_trainable:,}", flush=True)

optimizer = torch.optim.Adam(trainable_params, lr=PHASE1_LR_PEAK)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=PHASE1_STEPS, eta_min=PHASE1_LR_FINAL)
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


t_start = time.time()
history = [
    {"event": "init_post_load", "drift": float(drift_post_load)},
    {"event": "phase1_init_post_swap", "drift": float(drift_post_swap1)},
]
best_drift = drift_post_swap1
best_step = 0

for step in range(1, PHASE1_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_total = step_fn(batch)

    if step % EVAL_EVERY == 0 or step == PHASE1_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = drift < best_drift
        marker = " ⭐" if is_best else ""
        print(f"  P1 step {step:>5}  L_ce={L_ce:.3f}  L_kl={L_kl:.4f}  "
              f"val_ce={val_ce:.4f}  drift={drift:+.4f}  lr={cur_lr:.2e}  "
              f"{elapsed:.0f}s{marker}", flush=True)
        history.append({"phase": 1, "step": step, "L_ce": L_ce, "L_kl": L_kl,
                        "val_ce": float(val_ce), "drift": float(drift),
                        "lr": float(cur_lr)})
        if is_best:
            best_drift = drift
            best_step = step
            torch.save({"step": step, "phase": 1, "drift": drift,
                        "model": student.state_dict()}, CKPT_BEST)
            print(f"    → saved BEST  (drift={drift:+.4f})", flush=True)

    if step % CKPT_EVERY == 0:
        torch.save({"step": step, "phase": 1, "drift": drift,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict()}, CKPT_LATEST)
        print(f"    → saved checkpoint", flush=True)

print(f"\nPhase 1 complete. Best drift: {best_drift:+.4f}", flush=True)


# ─── PHASE 2: STE refinement ───
print(f"\n{'─'*60}")
print(f"PHASE 2 — STE REFINEMENT  ({PHASE2_STEPS} steps)")
print(f"  Swap Per128GroupSphereLinearWithPerm → BinaryGroupLinearWithPerm")
print(f"  Fresh cosine LR {PHASE2_LR_PEAK:.0e} → {PHASE2_LR_FINAL:.0e}")
print('─'*60, flush=True)

n_swapped_back = replace_sphere_perm_with_binary_perm(student, group_size=GROUP_SIZE)
print(f"  Swapped {n_swapped_back} back to BinaryGroupLinearWithPerm", flush=True)
bpw = compute_bits_per_weight_with_perm(student)
print(f"  bpw = {bpw:.4f}", flush=True)

ce_post_swap2 = lm_ce(student, val_tokens)
drift_post_swap2 = ce_post_swap2 - T0
print(f"  Drift after swap to binary: {drift_post_swap2:+.4f}", flush=True)
print(f"  Δ from binary swap: {drift_post_swap2 - best_drift:+.4f}", flush=True)
history.append({"event": "phase2_init", "drift": float(drift_post_swap2),
                "bpw": float(bpw)})

setup_trainable(student)
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable_p2 = sum(p.numel() for p in trainable_params)
print(f"  Trainable params: {n_trainable_p2:,}", flush=True)

optimizer = torch.optim.Adam(trainable_params, lr=PHASE2_LR_PEAK)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=PHASE2_STEPS, eta_min=PHASE2_LR_FINAL)

if drift_post_swap2 < best_drift:
    best_drift = drift_post_swap2
    best_step = PHASE1_STEPS

for step in range(1, PHASE2_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_total = step_fn(batch)

    global_step = PHASE1_STEPS + step

    if step % EVAL_EVERY == 0 or step == PHASE2_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        is_best = drift < best_drift
        marker = " ⭐" if is_best else ""
        print(f"  P2 step {step:>5} (g{global_step})  L_ce={L_ce:.3f}  L_kl={L_kl:.4f}  "
              f"val_ce={val_ce:.4f}  drift={drift:+.4f}  lr={cur_lr:.2e}  "
              f"{elapsed:.0f}s{marker}", flush=True)
        history.append({"phase": 2, "step": step, "global_step": global_step,
                        "L_ce": L_ce, "L_kl": L_kl,
                        "val_ce": float(val_ce), "drift": float(drift),
                        "lr": float(cur_lr)})
        if is_best:
            best_drift = drift
            best_step = global_step
            torch.save({"step": step, "global_step": global_step, "phase": 2,
                        "drift": drift, "bpw": float(bpw),
                        "model": student.state_dict()}, CKPT_BEST)
            print(f"    → saved BEST  (drift={drift:+.4f})", flush=True)

    if step % CKPT_EVERY == 0:
        torch.save({"step": step, "global_step": global_step, "phase": 2,
                    "drift": drift, "model": student.state_dict(),
                    "optimizer": optimizer.state_dict()}, CKPT_LATEST)
        print(f"    → saved checkpoint", flush=True)


# Final
final_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
final_drift = final_ce - T0
print(f"\n{'─'*60}")
print(f"STAGE 235 RESULT (bake-cycle on Stage 234):")
print('─'*60)
print(f"  Teacher T0:            {T0:.4f}")
print(f"  Init (Stage 234 load): {drift_post_load:+.4f}")
print(f"  Phase 1 init (sphere): {drift_post_swap1:+.4f}")
print(f"  Phase 2 init (binary): {drift_post_swap2:+.4f}")
print(f"  Final drift:           {final_drift:+.4f}")
print(f"  Best drift:            {best_drift:+.4f}  (step {best_step})")
print(f"  bpw:                   {bpw:.4f}")
print(f"  Δ vs Stage 234 best:   {best_drift - 1.2313:+.4f}")
print(f"\n  Comparison:")
print(f"    Stage 234 best (no cycle):  drift=+1.2313  bpw=1.135")
print(f"    Stage 235 (bake-cycle):     drift={best_drift:+.4f}  bpw={bpw:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_init": float(drift_post_load),
        "drift_phase1_init_sphere": float(drift_post_swap1),
        "drift_phase2_init_binary": float(drift_post_swap2),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "best_step": int(best_step),
        "bpw": float(bpw),
        "delta_vs_stage234": float(best_drift - 1.2313),
        "phase1_steps": int(PHASE1_STEPS),
        "phase2_steps": int(PHASE2_STEPS),
        "phase1_lr": [float(PHASE1_LR_PEAK), float(PHASE1_LR_FINAL)],
        "phase2_lr": [float(PHASE2_LR_PEAK), float(PHASE2_LR_FINAL)],
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
