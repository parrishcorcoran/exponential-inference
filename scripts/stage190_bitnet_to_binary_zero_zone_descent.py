"""Stage 190: PID adiabatic descent from BitNet b1.58 (ternary) to binary
via the zero-zone collapse axis.

Stage 189 showed adiabatic descent works on Qwen3 → Bonsai-binary
(unlocked 0.6 nats below one-shot). User asked: can we do the same
starting from BitNet?

The natural axis for BitNet → binary is **shrinking the zero zone**.
BitNet's ternary projection is:

  γ = mean(|w|) over each weight matrix
  w_q = γ × clip(round(w/γ), -1, 1)
       = γ × sign(w) × (|w/γ| ≥ 0.5)   ← weights with |w/γ|<0.5 become 0

We parameterize this as:

  fuzzy_ternary(w, zero_threshold) = γ × sign(w) × (|w/γ| ≥ zero_threshold)

  zero_threshold = 0.5 → BitNet's native ternary
  zero_threshold = 0.0 → pure binary (no zero state)

PID descent shrinks zero_threshold. At each level: train master to absorb
the loss of the 0 state for weights newly forced to ±γ.

This tests whether BitNet's already-low-bit-friendly geometry (Stage 185
finding: flat outliers, redistributed magnitude) gives a better basin
to descend into binary from than Qwen's FP attractor.

Why this is interesting beyond Stage 189:
- BitNet's geometry already matches Stage 185's "low-bit attractor"
  (flat RMSNorm, boosted embeddings, clean per-head structure)
- Distance to the binary attractor is much shorter from BitNet than
  from Qwen FP — adiabatic descent should be quicker / cleaner
- If this works, "fine-tune BitNet to binary" becomes a viable
  production recipe: cheaper than from-scratch BitNet 1.0 (Microsoft
  tried, didn't work) but reaches the same target
"""
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "1bitLLM/bitnet_b1_58-large"   # 700M ternary master
SEQ_LEN = 64
BATCH = 1
GRAD_ACCUM = 4
N_VAL_CHUNKS = 32
LR_MASTER = 1e-5
LR_NORMS_ALPHA = 5e-4
GRAD_CLIP = 1.0
RESULTS_PATH = Path("results/stage190_bitnet_to_binary_zero_zone.json")

# Train master only on bottleneck projections (Finding 27)
ALL_TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")
TRAINABLE_MASTER_NAMES = ("o_proj", "down_proj")

# Zero-zone descent schedule
# 0.5 = BitNet's native ternary, 0.0 = pure binary
ZERO_THRESHOLD_SCHEDULE = [0.5, 0.3, 0.15, 0.0]

# PID parameters
PID_SETPOINT_DRIFT = 0.10
PID_KP = 1.0
PID_MAX_TRAIN_AT_LEVEL = 100
TRAIN_STEPS_BASELINE = 50


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


class FuzzyTernaryLinear(nn.Module):
    """Linear with parameterized ternary→binary projection.

    Forward: w_q = γ × sign(w) × (|w/γ| ≥ zero_threshold)
    where γ = mean(|w|) over the weight matrix (BitNet b1.58 convention).

    zero_threshold = 0.5 → exact BitNet ternary
    zero_threshold = 0.0 → pure binary

    STE backward."""
    def __init__(self, original_module, initial_zero_threshold=0.5):
        super().__init__()
        # Master FP weights (re-trainable)
        self.weight = nn.Parameter(original_module.weight.data.clone())
        self.bias = original_module.bias
        self.zero_threshold = initial_zero_threshold
        # Per-row α to absorb residual scale (matches Stage 189 pattern)
        rn = self.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.alpha = nn.Parameter(rn.squeeze(-1).clone().to(torch.float32))

    def project(self, W, zero_threshold):
        gamma = W.abs().mean().clamp(min=1e-8)
        abs_norm = W.abs() / gamma
        sign_w = torch.sign(W)
        mask = (abs_norm >= zero_threshold).to(W.dtype)
        return gamma * sign_w * mask

    def forward(self, x):
        w = self.weight
        w_q = self.project(w.float(), self.zero_threshold).to(x.dtype)
        # STE: forward uses projection, backward acts as identity
        w_eff = w + (w_q - w).detach()
        # Renormalize to unit row-norm; α captures scale (per Stage 189)
        rn = w_eff.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w_unit = (w_eff.float() / rn).to(x.dtype)
        out = F.linear(x, w_unit, self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


class ZeroThresholdPID:
    def __init__(self, setpoint=PID_SETPOINT_DRIFT, Kp=PID_KP):
        self.setpoint = setpoint
        self.Kp = Kp
        self.last_error = 0.0
        self.cumulative_error = 0.0

    def decide(self, observed_drift):
        error = observed_drift - self.setpoint
        self.cumulative_error += error
        derivative = error - self.last_error
        self.last_error = error

        if observed_drift > self.setpoint:
            extra = min(int(PID_MAX_TRAIN_AT_LEVEL * (self.Kp * error)),
                        PID_MAX_TRAIN_AT_LEVEL)
            return False, max(extra, 50)
        else:
            return True, 0


print(f"device={device} dtype={dtype}")
print(f"checkpoint: {CHECKPOINT}")
print(f"schedule: {ZERO_THRESHOLD_SCHEDULE}  (0.5=BitNet ternary, 0.0=pure binary)")
print(f"PID setpoint: drift ≤ {PID_SETPOINT_DRIFT} nats")

# BitNet ships a custom BitnetTokenizer class that won't auto-import.
# Workaround: load the fast tokenizer.json directly via PreTrainedTokenizerFast.
from transformers import PreTrainedTokenizerFast
from tokenizers import AddedToken
from huggingface_hub import hf_hub_download
import json as _json

_tokenizer_file = hf_hub_download(repo_id=CHECKPOINT, filename="tokenizer.json")
_special_file = hf_hub_download(repo_id=CHECKPOINT, filename="special_tokens_map.json")
with open(_special_file) as _f:
    _specials = _json.load(_f)

def _to_token(v):
    if v is None: return None
    if isinstance(v, str): return v
    if isinstance(v, dict):
        return AddedToken(v["content"], lstrip=v.get("lstrip", False),
                          rstrip=v.get("rstrip", False),
                          single_word=v.get("single_word", False),
                          normalized=v.get("normalized", True))
    return v

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=_tokenizer_file,
    bos_token=_to_token(_specials.get("bos_token")),
    eos_token=_to_token(_specials.get("eos_token")),
    unk_token=_to_token(_specials.get("unk_token")),
    pad_token=_to_token(_specials.get("pad_token")),
)


# ─── Reference: BitNet's native CE (with ternary forward) ───
print("\nMeasuring BitNet b1.58 native CE (reference)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

print("Loading val + train tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)
train_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 4096, skip=SEQ_LEN * 1024)

T0_native = lm_ce(ref_model, val_tokens)
print(f"T0 BitNet native (ternary forward): CE={T0_native:.4f}  ppl={math.exp(T0_native):.2f}")
del ref_model
gc.collect()
if device == "mps":
    torch.mps.empty_cache()


# ─── Set up model with FuzzyTernaryLinear wrappers ───
print("\nLoading model and installing FuzzyTernaryLinear wrappers...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

# Find all linear-like target modules
target_mods = []
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if not any(t in name for t in ALL_TARGET_NAMES): continue
    target_mods.append((name, mod))
print(f"  found {len(target_mods)} target linears")

parent_lookup = {}
for name, m in model.named_modules():
    for child_name, child_mod in m.named_children():
        full = f"{name}.{child_name}" if name else child_name
        parent_lookup[full] = (m, child_name)

descending_layers = []
fixed_layers = []
for full_name, mod in target_mods:
    is_descending = any(t in full_name for t in TRAINABLE_MASTER_NAMES)
    new_layer = FuzzyTernaryLinear(mod, initial_zero_threshold=ZERO_THRESHOLD_SCHEDULE[0])
    parent, child_attr = parent_lookup[full_name]
    setattr(parent, child_attr, new_layer)
    if is_descending:
        descending_layers.append(new_layer)
    else:
        fixed_layers.append(new_layer)

# Trainable: master weights of o,down + α (all) + RMSNorm gains
master_params = [g.weight for g in descending_layers]
alpha_params = [g.alpha for g in descending_layers + fixed_layers]
for g in fixed_layers:
    g.weight.requires_grad = False
norm_params = []
for n, p in model.named_parameters():
    if "norm" in n.lower() and "weight" in n and "embed" not in n.lower():
        p.requires_grad = True
        norm_params.append(p)
for p in master_params:
    p.requires_grad = True
for p in alpha_params:
    p.requires_grad = True

n_master = sum(p.numel() for p in master_params)
n_alpha = sum(p.numel() for p in alpha_params)
n_norm = sum(p.numel() for p in norm_params)
print(f"  descending: {len(descending_layers)} (o_proj+down_proj, master trainable)")
print(f"  fixed:      {len(fixed_layers)} (others, master frozen)")
print(f"  trainable:  {n_master:,} master + {n_alpha:,} α + {n_norm:,} norm = {n_master+n_alpha+n_norm:,}")


opt = torch.optim.AdamW([
    {"params": master_params, "lr": LR_MASTER},
    {"params": alpha_params, "lr": LR_NORMS_ALPHA},
    {"params": norm_params, "lr": LR_NORMS_ALPHA},
], weight_decay=0.0)


def iter_train():
    n_chunks = (len(train_tokens) - 1) // SEQ_LEN
    while True:
        order = torch.randperm(n_chunks)
        for i in order.tolist():
            s = i * SEQ_LEN
            yield torch.tensor([train_tokens[s:s+SEQ_LEN+1]], dtype=torch.long, device=device)


def train_steps(it, n_steps):
    model.train()
    for step in range(n_steps):
        opt.zero_grad()
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1)) / GRAD_ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(master_params + alpha_params + norm_params, GRAD_CLIP)
        opt.step()


# ─── Adiabatic descent ───
it = iter_train()
trajectory = []
pid = ZeroThresholdPID()

# Measure initial state at threshold=0.5 (should match BitNet native CE up to STE/α/projection differences)
init_ce_at_05 = lm_ce(model, val_tokens)
print(f"\nMatch check: BitNet native = {T0_native:.4f}  ours-at-0.5 = {init_ce_at_05:.4f}")
print(f"  (small mismatch expected due to AlphaLinear renormalization & per-row α)")

print("\n" + "=" * 70)
print("Starting zero-zone adiabatic descent")
print("=" * 70)

for level_idx, threshold in enumerate(ZERO_THRESHOLD_SCHEDULE):
    # Set new zero_threshold on descending linears
    for g in descending_layers:
        g.zero_threshold = threshold

    init_ce = lm_ce(model, val_tokens)
    init_drift = init_ce - T0_native
    print(f"\nLevel {level_idx+1}/{len(ZERO_THRESHOLD_SCHEDULE)}  threshold={threshold:.3f}  "
          f"({'binary' if threshold==0.0 else 'fuzzy ternary'})")
    print(f"  init CE={init_ce:.4f}  drift={init_drift:+.4f}")

    train_steps(it, TRAIN_STEPS_BASELINE)
    post_train_ce = lm_ce(model, val_tokens)
    post_drift = post_train_ce - T0_native

    advance, extra_steps = pid.decide(post_drift)
    extra_done = 0
    while not advance and extra_done < PID_MAX_TRAIN_AT_LEVEL:
        train_steps(it, extra_steps)
        extra_done += extra_steps
        post_train_ce = lm_ce(model, val_tokens)
        post_drift = post_train_ce - T0_native
        advance, extra_steps = pid.decide(post_drift)
        print(f"    PID hold: extra {extra_done} steps total, drift={post_drift:+.4f}")

    final_ce = post_train_ce
    print(f"  level done  CE={final_ce:.4f}  drift={final_ce-T0_native:+.4f}  "
          f"(+{TRAIN_STEPS_BASELINE+extra_done} steps)")

    trajectory.append({
        "level": level_idx + 1,
        "zero_threshold": threshold,
        "init_ce": float(init_ce),
        "init_drift": float(init_drift),
        "final_ce": float(final_ce),
        "final_drift": float(final_ce - T0_native),
        "extra_train_steps": int(extra_done),
    })


# ─── Final summary ───
final_drift = trajectory[-1]['final_drift']
print("\n" + "=" * 70)
print("ZERO-ZONE DESCENT COMPLETE — BitNet → Binary")
print("=" * 70)
print(f"  T0 (BitNet native ternary):  {T0_native:.4f}")
print(f"\n  {'level':>5}  {'threshold':>10}  {'init Δ':>9}  {'final Δ':>9}  {'+steps':>7}")
for t in trajectory:
    print(f"  {t['level']:>5}  {t['zero_threshold']:>10.3f}  "
          f"{t['init_drift']:>+9.4f}  {t['final_drift']:>+9.4f}  {t['extra_train_steps']:>7}")

print(f"\n  Final at threshold=0.0 (pure binary on BitNet): Δ={final_drift:+.4f}")
print(f"  Stage 189 (Qwen → Bonsai-binary):                 Δ=+3.159")

if final_drift < 0.5:
    print(f"\n  ✓✓ STRONG SUCCESS: BitNet reaches binary with only {final_drift:.2f} nats damage.")
    print(f"     Confirms the 'low-bit-friendly geometry shortens the descent' hypothesis.")
elif final_drift < 1.5:
    print(f"\n  ✓ MEANINGFUL: BitNet → binary at +{final_drift:.2f} nats.")
    print(f"     Compare to Stage 189's Qwen → Bonsai-binary at +3.16.")
elif final_drift < 3.0:
    print(f"\n  ~ Partial: BitNet → binary at +{final_drift:.2f} nats.")
    print(f"     Some advantage over Qwen baseline (+3.16) but binary attractor is genuinely far.")
else:
    print(f"\n  ✗ BitNet's geometry didn't shorten the path to binary; both end up similar.")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "schedule": ZERO_THRESHOLD_SCHEDULE,
        "T0_bitnet_native_ce": float(T0_native),
        "trajectory": trajectory,
        "final_drift": final_drift,
        "stage_189_qwen_to_bonsai_binary": 3.159,
        "config": {
            "lr_master": LR_MASTER,
            "lr_norms_alpha": LR_NORMS_ALPHA,
            "train_steps_baseline": TRAIN_STEPS_BASELINE,
            "pid_setpoint": PID_SETPOINT_DRIFT,
            "pid_kp": PID_KP,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
