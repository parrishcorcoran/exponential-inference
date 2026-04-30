"""Stage 214 — Round-robin PID across lever groups (spin-glass relaxation).

User's insight: joint Adam can't activate dormant levers because they
have no gradient signal once a "winner" lever (bias) hogs the work.
Round-robin gives each lever group its own turn, with others frozen.
This is the Glauber-dynamics analogue for solving glassy systems.

Stage 213-A confirmed: joint Adam with 20M trainable params at lr=5e-4
overshoots (drift +1.3 nats at γ=0). The capacity exists but joint
optimization can't allocate it usefully.

Round-robin protocol per γ-round:
  for each lever group (gate, h_scale, ..., lora_A, lora_B):
    save state
    eval ce_before (cheap, 4 chunks)
    train ONLY this group for K steps with that group's LR
    eval ce_after
    if ce_after < ce_before - TOLERANCE:
      keep state    (this lever helped)
    else:
      revert state  (lever didn't help; frozen for this round)
  PID: advance γ if drift in band, else hold/back off

Per-group LR is TUNED to each group's parameter count and dynamics:
  - tiny scalars (logit_tau, gate): lr=1e-2  (need big moves)
  - per-channel (gain, offset):     lr=5e-3
  - per-output (bias):              lr=5e-4
  - LoRA (large):                   lr=5e-5  (small per-param moves)

Order: dormant first (gate, h_scale, gains), then established (bias,
LoRA). Forces dormant levers to commit before easy levers grab work.

11 groups × 30 steps/group = 330 steps per round. ~15 rounds in 5000
steps. Each round, all 11 levers get their turn.

Hypothesis: round-robin pushes γ past Stage 212's 0.35 wall AND past
joint-Adam Stage 213's overshoot, by activating dormant levers under
no competition.
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
N_VAL_CHUNKS = 8
N_QUICK_CHUNKS = 4    # cheaper eval used for per-group accept/reject
N_CALIB_TOKENS = 64
BATCH_SIZE = 1
N_TOTAL_STEPS = 5000
STEPS_PER_GROUP = 30
TOLERANCE = 0.005     # ce must improve by at least this to keep group's update
LORA_RANK = 32

DRIFT_TARGET = 0.05
DRIFT_HIGH = 0.20
GAMMA_STEP_UP = 0.05
GAMMA_STEP_DOWN = 0.10
GAMMA_MIN = 0.0
GAMMA_MAX = 1.0

RESULTS_PATH = Path("results/stage214_round_robin.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
GROUP_SIZE = 128

# Order matters — dormant levers go first so they commit before easy ones grab work.
LEVER_GROUPS = [
    # (name,           name_check_fn,                                      lr,    note)
    ("subln_gate",    lambda n: "subln_gate" in n,                         1e-2),  # 56 scalars
    ("logit_tau",     lambda n: "logit_tau" in n,                          1e-2),  # 1 scalar
    ("h_scale",       lambda n: "h_scale" in n,                            5e-3),  # 28 × 16 scalars
    ("attn_gain",     lambda n: "attn_gain" in n,                          5e-3),  # 28 × 1024
    ("mlp_gain",      lambda n: "mlp_gain" in n,                           5e-3),
    ("subln_gain",    lambda n: "subln_gain" in n,                         5e-4),  # 56 × in_features
    ("attn_offset",   lambda n: "attn_offset" in n,                        5e-4),
    ("mlp_offset",    lambda n: "mlp_offset" in n,                         5e-4),
    ("bias",          lambda n: "bias" in n and "norm" not in n,           5e-4),
    ("lora_A",        lambda n: "lora_A" in n,                             5e-5),
    ("lora_B",        lambda n: "lora_B" in n,                             5e-5),
]


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


# ─── Reuse Stage 213's lever module classes ───
class AdiabaticQuantizedLinearLoRA(nn.Module):
    def __init__(self, original_linear, group_size=GROUP_SIZE, rank=LORA_RANK):
        super().__init__()
        W_fp = original_linear.weight.data.clone()
        self.weight_fp = nn.Parameter(W_fp, requires_grad=False)
        out, in_ = W_fp.shape
        self.has_groups = (in_ % group_size == 0)
        if self.has_groups:
            n_groups = in_ // group_size
            Wg = W_fp.float().reshape(out, n_groups, group_size)
            alpha = Wg.abs().mean(dim=-1, keepdim=True)
            self.register_buffer("alpha", alpha.to(W_fp.dtype))
        else:
            self.register_buffer("alpha",
                                 W_fp.abs().mean(dim=-1, keepdim=True).to(W_fp.dtype))
        self.group_size = group_size
        self.out_features, self.in_features = out, in_
        self.register_buffer("gamma", torch.tensor(0.0, dtype=W_fp.dtype))
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = nn.Parameter(torch.zeros(
                out, device=W_fp.device, dtype=W_fp.dtype))
        self.lora_A = nn.Parameter(torch.empty(
            rank, in_, device=W_fp.device, dtype=W_fp.dtype))
        nn.init.normal_(self.lora_A, std=1.0 / np.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(
            out, rank, device=W_fp.device, dtype=W_fp.dtype))

    def forward(self, x):
        γ = self.gamma
        if self.has_groups:
            Wg_fp = self.weight_fp.reshape(
                self.out_features, self.in_features // self.group_size, self.group_size)
            mag_eff = γ * self.alpha + (1 - γ) * Wg_fp.abs()
            W_eff = (torch.sign(Wg_fp) * mag_eff).reshape(
                self.out_features, self.in_features)
        else:
            W_eff = torch.sign(self.weight_fp) * (
                γ * self.alpha + (1 - γ) * self.weight_fp.abs())
        out = F.linear(x, W_eff, self.bias.to(x.dtype))
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out + lora_out


class SubLNLinear(nn.Module):
    def __init__(self, wrapped_linear, num_heads=None, head_dim=None, eps=1e-6):
        super().__init__()
        self.wrapped = wrapped_linear
        in_features = (wrapped_linear.weight_fp.shape[1] if hasattr(wrapped_linear, "weight_fp")
                       else wrapped_linear.weight.shape[1])
        device_ = (wrapped_linear.weight_fp if hasattr(wrapped_linear, "weight_fp")
                   else wrapped_linear.weight).device
        dtype_ = (wrapped_linear.weight_fp if hasattr(wrapped_linear, "weight_fp")
                  else wrapped_linear.weight).dtype
        self.subln_gain = nn.Parameter(torch.ones(in_features, device=device_, dtype=dtype_))
        self.subln_gate = nn.Parameter(torch.zeros((), device=device_, dtype=dtype_))
        self.eps = eps
        if num_heads is not None and head_dim is not None:
            assert in_features == num_heads * head_dim
            self.h_scale = nn.Parameter(torch.ones(num_heads, device=device_, dtype=dtype_))
            self.num_heads = num_heads
            self.head_dim = head_dim
        else:
            self.h_scale = None

    def forward(self, x):
        if self.h_scale is not None:
            shape = x.shape
            x = x.reshape(*shape[:-1], self.num_heads, self.head_dim)
            x = x * self.h_scale.view(*([1] * (len(shape) - 1)), self.num_heads, 1)
            x = x.reshape(*shape)
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        normed = self.subln_gain * x / rms
        x = (1.0 - self.subln_gate) * x + self.subln_gate * normed
        return self.wrapped(x)


class TemperedLMHead(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        device_ = wrapped.weight.device
        dtype_ = wrapped.weight.dtype
        self.logit_tau = nn.Parameter(torch.ones((), device=device_, dtype=dtype_))

    def forward(self, x):
        return self.wrapped(x) / self.logit_tau


def calibrate_input_rms(model, calib_ids, target_suffixes):
    rms_sums, counts, hooks = {}, {}, []
    def make_hook(name):
        def hook(mod, inp):
            x = inp[0].detach().float()
            mean_sq = x.pow(2).mean(dim=tuple(range(x.dim() - 1)))
            rms = mean_sq.sqrt()
            if name not in rms_sums:
                rms_sums[name] = rms.clone(); counts[name] = 1
            else:
                rms_sums[name] += rms; counts[name] += 1
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in target_suffixes):
            hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    with torch.no_grad():
        model(calib_ids, use_cache=False)
    for h in hooks: h.remove()
    return {name: (rms_sums[name] / counts[name]).cpu() for name in rms_sums}


def install_residual_gains_and_offsets(model):
    n_layers = 0
    for layer in model.model.layers:
        hidden_size = layer.input_layernorm.weight.shape[0]
        d, t = layer.input_layernorm.weight.device, layer.input_layernorm.weight.dtype
        layer.attn_gain = nn.Parameter(torch.ones(hidden_size, device=d, dtype=t))
        layer.mlp_gain = nn.Parameter(torch.ones(hidden_size, device=d, dtype=t))
        layer.attn_offset = nn.Parameter(torch.zeros(hidden_size, device=d, dtype=t))
        layer.mlp_offset = nn.Parameter(torch.zeros(hidden_size, device=d, dtype=t))

        def new_forward(self, hidden_states, **kwargs):
            residual = hidden_states
            x = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(hidden_states=x, **kwargs)
            x = residual + self.attn_gain * attn_out + self.attn_offset
            residual = x
            x = self.post_attention_layernorm(x)
            mlp_out = self.mlp(x)
            x = residual + self.mlp_gain * mlp_out + self.mlp_offset
            return x

        layer.forward = types.MethodType(new_forward, layer)
        n_layers += 1
    return n_layers


def build_full_architecture(num_heads, head_dim, calib_ids):
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    n_layers = install_residual_gains_and_offsets(m)
    rms_table = calibrate_input_rms(m, calib_ids, ("o_proj", "down_proj"))

    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n_quantized = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, nn.Linear): continue
        if not any(name.endswith(s) for s in TARGET_NAMES): continue
        new_layer = AdiabaticQuantizedLinearLoRA(mod, rank=LORA_RANK)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)
        n_quantized += 1

    parent_lookup2 = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup2[full] = (mod, child_name)

    n_subln = 0
    for name, mod in list(m.named_modules()):
        if not isinstance(mod, AdiabaticQuantizedLinearLoRA): continue
        is_o = name.endswith("o_proj"); is_d = name.endswith("down_proj")
        if not (is_o or is_d): continue
        if name not in rms_table: continue
        gain = rms_table[name].to(device=mod.weight_fp.device, dtype=mod.weight_fp.dtype)
        nh, hd = (num_heads, head_dim) if is_o else (None, None)
        new_layer = SubLNLinear(mod, num_heads=nh, head_dim=hd)
        with torch.no_grad():
            new_layer.subln_gain.data.copy_(gain)
        parent, child_attr = parent_lookup2[name]
        setattr(parent, child_attr, new_layer)
        n_subln += 1

    m.lm_head = TemperedLMHead(m.lm_head)
    return m, dict(n_quantized=n_quantized, n_residual_gain_layers=n_layers, n_subln=n_subln)


def set_gamma(model, gamma_value):
    for mod in model.modules():
        if isinstance(mod, AdiabaticQuantizedLinearLoRA):
            mod.gamma.fill_(gamma_value)


def sample_batch(tokens, batch_size, seq_len, rng):
    n = len(tokens)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    batch = torch.stack([
        torch.tensor(tokens[s:s + seq_len + 1], dtype=torch.long)
        for s in starts
    ]).to(device)
    return batch


def snapshot_group(model, filter_fn):
    """Detached clones of all parameters matching filter_fn."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if filter_fn(n)}


def restore_group(model, snapshot):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snapshot:
                p.data.copy_(snapshot[n])


def activate_only_group(model, filter_fn):
    """Set requires_grad=True only for params matching filter_fn."""
    actives = []
    for n, p in model.named_parameters():
        if filter_fn(n):
            p.requires_grad_(True)
            actives.append(p)
        else:
            p.requires_grad_(False)
    return actives


def is_any_lever(name):
    if "bias" in name and "norm" not in name: return True
    for kw in ("subln_gate","subln_gain","h_scale","attn_gain","mlp_gain",
               "attn_offset","mlp_offset","lora_A","lora_B","logit_tau"):
        if kw in name: return True
    return False


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 32].tolist()
train_tokens = corpus[SEQ_LEN * 32:SEQ_LEN * 32 + 1_000_000].tolist()
calib_ids = torch.tensor([corpus[:N_CALIB_TOKENS].tolist()], dtype=torch.long, device=device)

print("\nMeasuring T0 (base FP)...", flush=True)
m0 = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(m0, val_tokens)
cfg = m0.config
num_heads = cfg.num_attention_heads
head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
print(f"  T0 = {T0:.4f}", flush=True)
del m0
import gc; gc.collect()

print("\nBuilding enabled architecture...", flush=True)
model, install_stats = build_full_architecture(num_heads, head_dim, calib_ids)
print(f"  installed: {install_stats}", flush=True)
ce_g0 = lm_ce(model, val_tokens)
drift_g0 = ce_g0 - T0
print(f"  γ=0 verify: ce={ce_g0:.4f} Δ={drift_g0:+.6f}", flush=True)

# Count lever params per group at start
print("\nLever group sizes:", flush=True)
group_sizes = {}
for gname, filter_fn, lr in LEVER_GROUPS:
    sz = sum(p.numel() for n, p in model.named_parameters() if filter_fn(n))
    group_sizes[gname] = sz
    print(f"  {gname:18s} {sz:>12,}  lr={lr:.0e}", flush=True)

set_gamma(model, 0.0)
current_gamma = 0.0
rng = np.random.default_rng(42)
history = [{"step": 0, "round": 0, "gamma": 0.0, "ce": ce_g0, "drift": drift_g0}]
t_start = time.time()
total_steps = 0
round_idx = 0
group_keep_counts = {gname: [0, 0] for gname, _, _ in LEVER_GROUPS}  # [keeps, reverts]

print(f"\n{'─'*60}")
print(f"Round-robin PID: {N_TOTAL_STEPS} total steps, {STEPS_PER_GROUP}/group")
print('─'*60, flush=True)

while total_steps < N_TOTAL_STEPS:
    round_idx += 1
    print(f"\n[Round {round_idx}, γ={current_gamma:.2f}, total_steps={total_steps}]", flush=True)
    round_t0 = time.time()
    ce_round_start = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)

    # ─── Iterate lever groups ───
    for gname, filter_fn, group_lr in LEVER_GROUPS:
        if total_steps >= N_TOTAL_STEPS: break
        # Snapshot + cheap eval
        snap = snapshot_group(model, filter_fn)
        ce_before = lm_ce(model, val_tokens, n_chunks=N_QUICK_CHUNKS)

        # Activate this group, train it for K steps
        actives = activate_only_group(model, filter_fn)
        if not actives:
            continue
        opt = torch.optim.Adam(actives, lr=group_lr)
        model.train()
        for k in range(STEPS_PER_GROUP):
            batch = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN, rng)
            out = model(batch[:, :-1], use_cache=False)
            loss = F.cross_entropy(
                out.logits.float().reshape(-1, out.logits.size(-1)),
                batch[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_steps += 1
        ce_after = lm_ce(model, val_tokens, n_chunks=N_QUICK_CHUNKS)
        improvement = ce_before - ce_after

        if improvement > TOLERANCE:
            group_keep_counts[gname][0] += 1
            kept = "✓ kept"
        else:
            restore_group(model, snap)
            group_keep_counts[gname][1] += 1
            kept = "✗ reverted"
        print(f"  {gname:18s} ce {ce_before:.4f}→{ce_after:.4f} (Δ {-improvement:+.4f}) {kept}", flush=True)

    # ─── End of round: PID gamma advance ───
    ce_end = lm_ce(model, val_tokens, n_chunks=N_VAL_CHUNKS)
    drift = ce_end - T0
    if drift < DRIFT_TARGET:
        old_g = current_gamma
        current_gamma = min(current_gamma + GAMMA_STEP_UP, GAMMA_MAX)
        action = f"γ {old_g:.2f}→{current_gamma:.2f} +"
    elif drift > DRIFT_HIGH:
        old_g = current_gamma
        current_gamma = max(current_gamma - GAMMA_STEP_DOWN, GAMMA_MIN)
        action = f"γ {old_g:.2f}→{current_gamma:.2f} −"
    else:
        action = f"γ {current_gamma:.2f} hold"
    set_gamma(model, current_gamma)

    elapsed = time.time() - t_start
    print(f"  → round end: ce={ce_end:.4f} Δ={drift:+.4f} [{action}]  total_steps={total_steps}  {elapsed:.0f}s", flush=True)
    history.append({"step": total_steps, "round": round_idx, "gamma": float(current_gamma),
                    "ce": float(ce_end), "drift": float(drift),
                    "ce_round_start": float(ce_round_start), "action": action})

# Final
print(f"\nFinal γ = {current_gamma:.2f}, total_steps={total_steps}, rounds={round_idx}")
final_ce = lm_ce(model, val_tokens)
final_drift = final_ce - T0

set_gamma(model, 1.0)
ce_g1 = lm_ce(model, val_tokens)
set_gamma(model, current_gamma)

print(f"\n{'─'*60}")
print("STAGE 214 RESULT (round-robin PID):")
print('─'*60)
print(f"  T0:                {T0:.4f}")
print(f"  γ=0 (lossless):    {ce_g0:.4f}  Δ={drift_g0:+.6f}")
print(f"  γ=1 raw K=1:       {ce_g1:.4f}  Δ={ce_g1-T0:+.4f}")
print(f"  Final γ={current_gamma:.2f}  : {final_ce:.4f}  Δ={final_drift:+.4f}")
print(f"  Reference (S212):  γ=0.35, Δ=+0.161  (joint Adam, 545K params)")

print(f"\n  Lever group accept/reject (kept/reverted):")
for gname, (k, r) in group_keep_counts.items():
    total = k + r
    if total > 0:
        rate = 100.0 * k / total
        marker = "★" if rate >= 50 else ("·" if rate >= 20 else "✗")
        print(f"    {marker} {gname:18s} {k:>3}/{total:>3} kept ({rate:.0f}%)")

if current_gamma > 0.40:
    verdict = (f"BREAKTHROUGH: round-robin pushed γ past Stage 212's 0.35 ceiling. "
               f"Sequential commitment activated dormant levers — spin-glass framing wins.")
elif current_gamma >= 0.35:
    verdict = (f"MATCHED: γ reached same ceiling as joint Adam. Lever-only fundamentally "
               f"saturates at this scale; STE needed to push further.")
else:
    verdict = (f"REGRESSED: round-robin underperformed joint Adam. Either tolerance too tight, "
               f"per-group LRs miscalibrated, or the round-robin is fighting itself.")
print(f"\n  Verdict: {verdict}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "T0": float(T0),
        "ce_gamma_0": float(ce_g0),
        "ce_gamma_1_no_train": float(ce_g1),
        "final_gamma": float(current_gamma),
        "final_ce": float(final_ce),
        "final_drift": float(final_drift),
        "n_total_steps": int(total_steps),
        "n_rounds": int(round_idx),
        "steps_per_group": STEPS_PER_GROUP,
        "tolerance": TOLERANCE,
        "lora_rank": LORA_RANK,
        "lever_groups": [(g, lr) for g, _, lr in LEVER_GROUPS],
        "group_keep_counts": group_keep_counts,
        "group_sizes": group_sizes,
        "stage212_reference": {"final_gamma": 0.35, "final_drift": 0.161},
        "verdict": verdict,
        "history": history,
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
