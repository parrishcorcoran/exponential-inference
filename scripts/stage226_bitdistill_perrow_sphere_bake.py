"""Stage 226 — BitDistill-style bake onto per-route, per-layer hyperspheres.

User's specification (2026-05-04, /remote-control mission):
  Take BitDistill technique. Apply it to anneal+distill weights onto
  per-row, per-layer hyperspheres. Each (layer, row) pair lives on its
  OWN shell — radius is the row's NATURAL L2 norm from FP, NOT uniform
  L2=1. Individualized per route.

Why this is better than uniform nGPT (Strix's existing magnitude_anneal_latest.pt):
  - FP teacher's natural per-row magnitude variation encodes useful info
    about which output channels contribute "loudly" vs "quietly"
  - Forcing all rows to L2=1 destroys this loudness pattern
  - Per-row individual radius preserves it while still constraining
    direction to its own hypersphere
  - More faithful starting point for downstream binary quantization

Protocol — pure BitDistill bake, applied to this constraint:

  1. Architecture surgery:
     - Replace targeted Linears with PerRowSphereLinear (forward applies
       row direction normalization, multiplies by saved per-row radius)
     - Insert SubLN before o_proj input and before down_proj input
       (the BitDistill pattern that absorbs constraint friction)

  2. Teacher-student setup:
     - Frozen FP teacher: vanilla Qwen3-0.6B
     - Trainable student: Qwen3-0.6B with PerRowSphereLinear + SubLN
     - Master weights initialized from teacher
     - Per-row radii saved at init from teacher (the locked targets)

  3. Loss = task CE + KL distillation + hidden-state distillation:
     L = α · CE(student_logits, target)
       + β · KL(student_logits || teacher_logits)
       + γ · sum_layers MSE(student_hidden[i], teacher_hidden[i])

  4. Training loop:
     - Standard backward through smooth normalization op (no STE needed)
     - Body weights move freely via gradient
     - Forward enforces the per-row radius constraint at every step
     - Train until weights "bake" into stable configuration

  5. End state:
     - Student matches teacher quality (within ε)
     - Each row of each Linear has L2 norm = its own initial radius
     - SubLN modules calibrated to absorb constraint friction
     - This is the foundation checkpoint for downstream binary work

Compute estimate:
  Mac MPS fp32: ~50-100K steps × ~1s/step = many hours, marginal
  Z8 (when SSH up): bf16 + larger batch = ~hours not days
"""
import json
import sys
import time
import types
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
N_CALIB_TOKENS = 128
BATCH_SIZE = 1
N_TRAIN_STEPS = 5000        # adjust per compute budget; Z8 can handle 50K+
EVAL_EVERY = 100
CKPT_EVERY = 500

LR = 5e-5

# Loss weights — Mac-trimmed: dropped hidden-state distillation to save memory
ALPHA_CE = 1.0
BETA_KL = 1.0
GAMMA_HIDDEN = 0.0   # 0 on Mac (storing both teacher+student hidden states OOMs); 0.5 on Z8

# Targeted Linears — start with all 7 (BitDistill quantizes all body Linears).
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")

RESULTS_PATH = Path("results/stage226_perrow_sphere_bake.json")
CKPT_DIR = Path("checkpoints/Qwen_Qwen3-0.6B")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_LATEST = CKPT_DIR / "perrow_sphere_bake_latest.pt"


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


# ─── PerRowSphereLinear: locks each row's L2 norm to its initial value ───
class PerRowSphereLinear(nn.Module):
    """Forward enforces:
       Each row of effective W has L2 norm equal to its initial L2 norm
       (the row's natural radius from the FP teacher). Direction can move.
       Magnitude (radius) is locked.

       Backward: standard autograd through the smooth normalization op.
                 No STE required (this is differentiable).
    """
    def __init__(self, original_linear):
        super().__init__()
        W = original_linear.weight.data.clone()
        self.weight = nn.Parameter(W)
        # Lock per-row radius (each row's own natural L2 norm)
        # Shape: [out_features, 1] — broadcasts correctly across in dimension
        with torch.no_grad():
            radius = W.norm(dim=-1, keepdim=True)
        self.register_buffer("row_radius", radius)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Unit-direction per row, then rescale to that row's locked radius
        W_unit = self.weight / self.weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        W_eff = W_unit * self.row_radius
        return F.linear(x, W_eff, self.bias)


# ─── SubLN insertion (BitDistill pattern) ───
class SubLNWrappedLinear(nn.Module):
    """Wraps a Linear-like module with SubLN normalization on its input.
    BitDistill insertion: SubLN BEFORE o_proj input, BEFORE down_proj input.

    Init γ=1 (per-channel) and ε for stability — at init this is approximately
    identity since input passes through normalization with γ=1.
    """
    def __init__(self, wrapped_linear, eps=1e-6):
        super().__init__()
        self.wrapped = wrapped_linear
        # Determine input dim from the wrapped Linear's weight shape
        W = wrapped_linear.weight if isinstance(wrapped_linear, nn.Linear) \
            else wrapped_linear.weight  # PerRowSphereLinear also has .weight
        in_features = W.shape[1]
        self.subln_gain = nn.Parameter(torch.ones(in_features,
            device=W.device, dtype=W.dtype))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        x_normed = self.subln_gain * x / rms
        return self.wrapped(x_normed)


def replace_linears_with_sphere(model, target_names):
    """Replace each Linear matching target_names with PerRowSphereLinear."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n = 0
    radii_summary = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in target_names): continue
        new_layer = PerRowSphereLinear(mod)
        radii_summary.append({
            "name": name,
            "radius_min": float(new_layer.row_radius.min().item()),
            "radius_max": float(new_layer.row_radius.max().item()),
            "radius_mean": float(new_layer.row_radius.mean().item()),
            "radius_std": float(new_layer.row_radius.std().item()),
        })
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n += 1
    return n, radii_summary


def insert_subln_before(model, target_suffixes):
    """Wrap target Linears with SubLNWrappedLinear (SubLN before their input).
    Targets: o_proj and down_proj per BitDistill convention."""
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


def get_hidden_states(model, ids):
    """Forward with output_hidden_states=True. Returns logits + tuple of hidden states."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    return out.logits, out.hidden_states


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    return torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
train_tokens = corpus[SEQ_LEN * 64:SEQ_LEN * 64 + 1_000_000].tolist()
print(f"  val={len(val_tokens)}  train={len(train_tokens)}", flush=True)


# ─── Build TEACHER (frozen FP) ───
print("\nBuilding TEACHER (frozen FP Qwen3-0.6B)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False
T0 = lm_ce(teacher, val_tokens)
print(f"  Teacher T0 = {T0:.4f}", flush=True)


# ─── Build STUDENT (PerRowSphereLinear + SubLN, trainable) ───
print("\nBuilding STUDENT (PerRowSphereLinear + SubLN)...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

# Replace targeted Linears with PerRowSphereLinear (locks per-row radius)
n_replaced, radii_summary = replace_linears_with_sphere(student, TARGET_NAMES)
print(f"  Replaced {n_replaced} Linears with PerRowSphereLinear", flush=True)
print(f"  Sample radii summary (first 3 layers):", flush=True)
for r in radii_summary[:3]:
    print(f"    {r['name']}: radius range [{r['radius_min']:.3f}, "
          f"{r['radius_max']:.3f}] mean {r['radius_mean']:.3f}", flush=True)

# Insert SubLN before o_proj and down_proj
n_subln = insert_subln_before(student, ("o_proj", "down_proj"))
print(f"  Inserted SubLN before {n_subln} Linears", flush=True)

# Verify: student forward still works and is approximately matching teacher at init
ce_init = lm_ce(student, val_tokens)
drift_init = ce_init - T0
print(f"\nStudent at init (after surgery, before bake):")
print(f"  CE = {ce_init:.4f}, drift vs teacher = {drift_init:+.4f}", flush=True)
print(f"  (constraint already enforced — rows on per-row hyperspheres)", flush=True)


# ─── Training setup — freeze embeddings + lm_head + ALL layer norms (Mac-friendly) ───
n_frozen = 0
for name, p in student.named_parameters():
    if any(t in name for t in ("embed_tokens", "lm_head", "input_layernorm",
                                 "post_attention_layernorm", "model.norm")):
        p.requires_grad_(False)
        n_frozen += p.numel()
trainable_params = [p for p in student.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable_params)
print(f"\nFrozen params:    {n_frozen:,}  (embeds, lm_head, layer norms)", flush=True)
print(f"Trainable params: {n_trainable:,}  (PerRowSphereLinear weights, biases, SubLN gains)",
      flush=True)
optimizer = torch.optim.Adam(trainable_params, lr=LR)
rng = np.random.default_rng(42)


def bake_step(batch):
    """One BitDistill-style bake step: CE + KL (no hidden distill on Mac)."""
    student.train()

    # Teacher forward (no grad), logits only — saves storing hidden states
    with torch.no_grad():
        teacher_logits = teacher(batch[:, :-1], use_cache=False).logits

    # Student forward, logits only
    student_logits = student(batch[:, :-1], use_cache=False).logits

    # Task CE
    L_ce = F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.size(-1)),
        batch[:, 1:].reshape(-1))

    # Distillation KL on logits
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    L_kl = F.kl_div(
        student_log_probs.reshape(-1, student_log_probs.size(-1)),
        teacher_log_probs.reshape(-1, teacher_log_probs.size(-1)),
        reduction='batchmean', log_target=True)

    L_total = ALPHA_CE * L_ce + BETA_KL * L_kl
    L_hidden = 0.0   # disabled on Mac

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

    return float(L_ce.item()), float(L_kl.item()), float(L_hidden), float(L_total.item())


# ─── Training loop ───
t_start = time.time()
history = [{"event": "init", "ce": ce_init, "drift": drift_init}]
print(f"\n{'─'*60}")
print(f"Stage 226 — BitDistill bake on per-route per-layer hyperspheres")
print(f"  N_steps = {N_TRAIN_STEPS}  LR = {LR}  α/β/γ = {ALPHA_CE}/{BETA_KL}/{GAMMA_HIDDEN}")
print('─'*60, flush=True)

best_drift = drift_init
for step in range(1, N_TRAIN_STEPS + 1):
    batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
    L_ce, L_kl, L_hidden, L_total = bake_step(batch)

    if step % EVAL_EVERY == 0 or step == N_TRAIN_STEPS:
        val_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
        drift = val_ce - T0
        elapsed = time.time() - t_start
        marker = " ⭐" if drift < best_drift else ""
        print(f"  step {step:>5}  L_ce={L_ce:.3f}  L_kl={L_kl:.4f}  "
              f"L_hidden={L_hidden:.4f}  val_ce={val_ce:.4f}  drift={drift:+.4f}  "
              f"{elapsed:.0f}s{marker}", flush=True)
        history.append({"step": step,
                        "L_ce": L_ce, "L_kl": L_kl, "L_hidden": L_hidden,
                        "val_ce": float(val_ce), "drift": float(drift)})
        if drift < best_drift:
            best_drift = drift

    if step % CKPT_EVERY == 0:
        torch.save({
            "step": step,
            "val_ce": val_ce,
            "drift": drift,
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, CKPT_LATEST)
        print(f"    → saved checkpoint to {CKPT_LATEST}", flush=True)


# Final
final_ce = lm_ce(student, val_tokens, n_chunks=N_VAL_CHUNKS)
final_drift = final_ce - T0
print(f"\n{'─'*60}")
print(f"STAGE 226 RESULT:")
print('─'*60)
print(f"  Teacher T0:     {T0:.4f}")
print(f"  Init drift:     {drift_init:+.4f}  (after surgery, before bake)")
print(f"  Final drift:    {final_drift:+.4f}  (after {N_TRAIN_STEPS} bake steps)")
print(f"  Best drift:     {best_drift:+.4f}")
print(f"  Architecture:   {n_replaced} PerRowSphereLinear + {n_subln} SubLN insertions")
print(f"\nThis baked checkpoint preserves per-row natural radii while constraining")
print(f"directions to per-route hyperspheres. Foundation for downstream binary work.")

# Save final
torch.save({
    "step": N_TRAIN_STEPS,
    "val_ce": final_ce,
    "drift": final_drift,
    "model": student.state_dict(),
    "optimizer": optimizer.state_dict(),
}, CKPT_LATEST)
print(f"\nSaved final checkpoint to {CKPT_LATEST}", flush=True)

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "drift_init_after_surgery": float(drift_init),
        "drift_final": float(final_drift),
        "drift_best": float(best_drift),
        "n_replaced_linears": int(n_replaced),
        "n_subln_inserted": int(n_subln),
        "n_trainable_params": int(n_trainable),
        "loss_weights": {"alpha_ce": ALPHA_CE, "beta_kl": BETA_KL, "gamma_hidden": GAMMA_HIDDEN},
        "n_train_steps": N_TRAIN_STEPS,
        "lr": LR,
        "radii_summary": radii_summary,
        "history": history,
    }, f, indent=2)
print(f"Saved {RESULTS_PATH}", flush=True)
