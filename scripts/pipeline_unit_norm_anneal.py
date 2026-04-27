"""Magnitude-to-floor anneal: force each weight matrix's rows toward unit
norm gradually, with thermostat control.

Lever: replace every Linear's forward with W_eff = (1 - tau) * W + tau * Wn,
where Wn has each row normalized to unit L2. Master weights stay FP32.

tau anneals 0.0 -> 1.0 in 10 drops of 0.1. Each drop does 2000 fine-tune
steps, then checks val CE. If CE exceeds (base + THRESHOLD), the thermostat
HOLDS tau (re-runs another 2000 steps at the same level) until quality
recovers, or MAX_HOLDS exceeded -> abandon at this tau.

When tau=1.0, every projected matrix has unit-norm rows. The matmul becomes
y[i] = <W_row_i_unit, x>. Lost per-channel magnitude must be recovered by
RMSNorm gamma and activation magnitudes downstream.

Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
across all transformer layers. Embeddings + lm_head untouched.

Compensation channels (full precision): RMSNorm gamma per layer,
input/post-attention LayerNorms, q_norm/k_norm gammas, activations.
"""
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


SMOKE = os.environ.get("SMOKE", "0") == "1"


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "128"))
BATCH = int(os.environ.get("BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
STEPS_PER_DROP = int(os.environ.get("STEPS_PER_DROP", "2000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "250"))
N_VAL_TOK = SEQ_LEN * int(os.environ.get("N_VAL_CHUNKS", "64"))
TRAIN_SKIP_TOK = SEQ_LEN * 65536
LR = float(os.environ.get("LR", "2e-5"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
THRESHOLD_DELTA_CE_OVERRIDE = os.environ.get("THRESHOLD_DELTA_CE")
TAG = os.environ.get("RUN_TAG", "")  # appended to results filename for parallel runs
TAU_SCHEDULE = [round(0.1 * (i + 1), 2) for i in range(10)]   # 0.1 .. 1.0
THRESHOLD_DELTA_CE = float(THRESHOLD_DELTA_CE_OVERRIDE) if THRESHOLD_DELTA_CE_OVERRIDE else 0.5
MAX_HOLDS_PER_DROP = int(os.environ.get("MAX_HOLDS_PER_DROP", "3"))
_tag = f"_{TAG}" if TAG else ""
RESULTS_PATH = Path(f"results/pipeline_magnitude_anneal{_tag}.json")
CKPT_DIR = Path("checkpoints") / CHECKPOINT.replace("/", "_")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_LATEST = CKPT_DIR / f"magnitude_anneal{_tag}_latest.pt"

if SMOKE:
    print("=== SMOKE TEST MODE ===")
    STEPS_PER_DROP = 4
    EVAL_EVERY = 2
    N_VAL_TOK = SEQ_LEN * 4
    TRAIN_SKIP_TOK = SEQ_LEN * 32
    TAU_SCHEDULE = [0.5, 1.0]
    MAX_HOLDS_PER_DROP = 1
    THRESHOLD_DELTA_CE = 100.0  # don't trigger holds during smoke
    RESULTS_PATH = Path(f"results/pipeline_magnitude_anneal{_tag}_smoke.json")
    CKPT_LATEST = CKPT_DIR / f"magnitude_anneal{_tag}_smoke_latest.pt"


# ─── tau is global; closures in patched Linears read from it ───────────────
_TAU = 0.0
def set_tau(t): global _TAU; _TAU = t
def get_tau():     return _TAU


def project_rows(W, tau):
    """W: [out, in].  Returns W' with each row's norm interpolated toward 1.
    tau=0 -> W; tau=1 -> rows have unit L2 norm."""
    if tau <= 0.0:
        return W
    row_norms = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target_norms = (1.0 - tau) * row_norms + tau * torch.ones_like(row_norms)
    return W / row_norms * target_norms


def patch_linear_for_projection(module: nn.Linear):
    """Replace module.forward so it projects W toward unit-norm-rows by tau."""
    weight = module.weight
    bias = module.bias

    def projected_forward(x):
        W = project_rows(weight, get_tau())
        return F.linear(x, W, bias)

    module.forward = projected_forward


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    """Load a fixed slice of tokens (used for val only)."""
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def stream_train_sequences(tokenizer, seq_len, skip_tokens=0):
    """Endless generator of (seq_len + 1) token windows from streaming OWT.
    Each yielded sequence is fresh — no repeats — so the model can't memorize.
    """
    from datasets import load_dataset
    while True:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        buf = []
        skipped = 0
        for item in ds:
            t = item.get("text", "")
            if not t.strip(): continue
            e = tokenizer.encode(t, add_special_tokens=False)
            if skipped < skip_tokens:
                skipped += len(e); continue
            buf.extend(e)
            while len(buf) >= seq_len + 1:
                yield torch.tensor([buf[:seq_len + 1]], dtype=torch.long, device=device)
                buf = buf[seq_len:]  # advance with 1-token autoregressive overlap


def lm_ce(model, val_tokens, seq_len, n_chunks):
    losses = []
    model.eval()
    for i in range(n_chunks):
        start = i * seq_len
        window = val_tokens[start:start + seq_len + 1]
        if len(window) < seq_len + 1: break
        ids = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(ids[:, :-1], use_cache=False)
            logits = out.logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="mean")
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ─── Load model ─────────────────────────────────────────────────────────────
print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

# Freeze embeddings + lm_head; train rest (incl norms - they are the bridge)
for n, p in model.named_parameters():
    if "embed_tokens" in n or "lm_head" in n:
        p.requires_grad = False
    else:
        p.requires_grad = True

# Patch every Linear in the transformer body
TARGET_NAME_MARKERS = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")

n_patched = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and any(m in name for m in TARGET_NAME_MARKERS):
        patch_linear_for_projection(module)
        n_patched += 1

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"patched {n_patched} linear layers")
print(f"trainable: {trainable:,} / {total:,}")

# ─── Tokens ────────────────────────────────────────────────────────────────
print("Loading val tokens (fixed slice)...")
val_tokens = load_owt(tokenizer, max_tokens=N_VAL_TOK)
val_chunks = N_VAL_TOK // SEQ_LEN
print(f"  val tokens: {len(val_tokens):,}, val chunks: {val_chunks}")
print(f"Setting up training stream (skip first {TRAIN_SKIP_TOK:,} tokens to avoid val overlap)...")
train_stream = stream_train_sequences(tokenizer, SEQ_LEN, skip_tokens=TRAIN_SKIP_TOK)

# ─── Baseline (tau=0) ─────────────────────────────────────────────────────
set_tau(0.0)
base_ce = lm_ce(model, val_tokens, SEQ_LEN, val_chunks)
print(f"\nBASE val CE (tau=0): {base_ce:.4f}")
threshold_ce = base_ce + THRESHOLD_DELTA_CE
print(f"thermostat threshold: {threshold_ce:.4f}  (base + {THRESHOLD_DELTA_CE})")

# Sanity: tau=1 deployed CE without any fine-tune (cold-projection)
set_tau(1.0)
cold_ce = lm_ce(model, val_tokens, SEQ_LEN, val_chunks)
print(f"COLD tau=1 val CE (no adapt yet): {cold_ce:.4f}  (delta {cold_ce - base_ce:+.4f})")
set_tau(0.0)

# ─── Optimizer ────────────────────────────────────────────────────────────
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=LR, weight_decay=0.0)


it = train_stream


def fine_tune_chunk(tau_target, n_steps, log_prefix=""):
    """Run n_steps optimizer steps at tau=tau_target. Returns (final_val_ce, history)."""
    set_tau(tau_target)
    history = []
    model.train()
    for step in range(n_steps):
        opt.zero_grad()
        lm_loss_acc = 0.0
        for _ in range(GRAD_ACCUM):
            ids = next(it)
            out = model(ids[:, :-1], use_cache=False)
            logits = out.logits
            lm_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                reduction="mean") / GRAD_ACCUM
            lm_loss.backward()
            lm_loss_acc += lm_loss.item()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        opt.step()

        if (step + 1) % EVAL_EVERY == 0:
            val_ce = lm_ce(model, val_tokens, SEQ_LEN, val_chunks)
            rec = {"step": step + 1, "tau": tau_target,
                   "train_lm": round(lm_loss_acc, 4),
                   "val_ce": round(val_ce, 4),
                   "delta_vs_base": round(val_ce - base_ce, 4)}
            print(f"  {log_prefix}step {step+1:>4}/{n_steps}: tau={tau_target:.2f} "
                  f"train={lm_loss_acc:.3f} val={val_ce:.3f} "
                  f"d={val_ce - base_ce:+.3f}", flush=True)
            history.append(rec)

    final_val = lm_ce(model, val_tokens, SEQ_LEN, val_chunks)
    return final_val, history


# ─── Anneal loop ───────────────────────────────────────────────────────────
all_history = []
drops_completed = []
last_passed_tau = 0.0

def save_checkpoint(tau, holds, val_ce, outcome):
    """Save model + state after a 2000-step block. Writes a tagged file
    AND overwrites the 'latest' pointer."""
    payload = {"tau": tau, "holds": holds,
               "val_ce": round(val_ce, 4),
               "delta_vs_base": round(val_ce - base_ce, 4),
               "base_val_ce": round(base_ce, 4),
               "outcome": outcome,
               "model": model.state_dict(),
               "optimizer": opt.state_dict()}
    tag = f"tau{int(round(tau*100)):03d}_h{holds}"
    tagged = CKPT_DIR / f"magnitude_anneal{_tag}_{tag}.pt"
    torch.save(payload, tagged)
    torch.save(payload, CKPT_LATEST)
    print(f"  SAVED: {tagged.name}  (val_ce={val_ce:.4f}, "
          f"delta={val_ce - base_ce:+.4f}, outcome={outcome})")


for tau in TAU_SCHEDULE:
    print(f"\n{'='*70}")
    print(f"DROP tau={tau:.2f}")
    print(f"{'='*70}")
    holds = 0
    while True:
        prefix = f"[tau={tau:.2f}, hold={holds}] "
        final_val, hist = fine_tune_chunk(tau, STEPS_PER_DROP, log_prefix=prefix)
        all_history.extend(hist)
        delta = final_val - base_ce
        print(f"  end-of-block val CE: {final_val:.4f}  delta_vs_base: {delta:+.4f}")

        if delta <= THRESHOLD_DELTA_CE:
            outcome = "pass"
            print(f"  THERMOSTAT: pass (delta within +{THRESHOLD_DELTA_CE}) -> advance")
        else:
            holds += 1
            if holds >= MAX_HOLDS_PER_DROP:
                outcome = "abandoned"
                print(f"  THERMOSTAT: abandoned at tau={tau:.2f} after {holds} holds")
            else:
                outcome = "hold"
                print(f"  THERMOSTAT: hold (delta {delta:+.4f} > {THRESHOLD_DELTA_CE}) -> "
                      f"another {STEPS_PER_DROP} steps at same tau")

        # Save EVERY 2000-step block (pass, hold, abandon) with the val CE
        save_checkpoint(tau, holds, final_val, outcome)

        if outcome == "pass":
            drops_completed.append({"tau": tau, "holds": holds,
                                    "final_val_ce": round(final_val, 4),
                                    "delta_vs_base": round(delta, 4), "outcome": "pass"})
            last_passed_tau = tau
            break
        if outcome == "abandoned":
            drops_completed.append({"tau": tau, "holds": holds,
                                    "final_val_ce": round(final_val, 4),
                                    "delta_vs_base": round(delta, 4), "outcome": "abandoned"})
            break

        # Persist running results json after every block too, so we don't
        # lose the trace if the process is killed mid-anneal
        with open(RESULTS_PATH, "w") as f:
            json.dump({"checkpoint": CHECKPOINT,
                       "tau_schedule": TAU_SCHEDULE,
                       "base_val_ce": round(base_ce, 4),
                       "cold_tau1_ce": round(cold_ce, 4),
                       "highest_tau_passed": last_passed_tau,
                       "drops": drops_completed,
                       "history": all_history}, f, indent=2)

    # Stop if we abandoned
    if drops_completed[-1]["outcome"] == "abandoned":
        break

# ─── Final report ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("MAGNITUDE ANNEAL — FINAL")
print(f"{'='*70}")
print(f"  base val CE:           {base_ce:.4f}")
print(f"  cold tau=1 (no adapt): {cold_ce:.4f}  (delta {cold_ce - base_ce:+.4f})")
print(f"  highest tau passed:    {last_passed_tau:.2f}")
last_drop = drops_completed[-1] if drops_completed else None
if last_drop:
    print(f"  final val CE:          {last_drop['final_val_ce']}  "
          f"(delta {last_drop['delta_vs_base']:+.4f})")
print(f"\n  drop summary:")
for d in drops_completed:
    print(f"    tau={d['tau']:.2f}  holds={d['holds']}  "
          f"val_ce={d['final_val_ce']}  delta={d['delta_vs_base']:+.4f}  {d['outcome']}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT,
               "seq_len": SEQ_LEN, "batch": BATCH, "grad_accum": GRAD_ACCUM,
               "steps_per_drop": STEPS_PER_DROP, "lr": LR,
               "tau_schedule": TAU_SCHEDULE,
               "threshold_delta_ce": THRESHOLD_DELTA_CE,
               "max_holds_per_drop": MAX_HOLDS_PER_DROP,
               "base_val_ce": round(base_ce, 4),
               "cold_tau1_ce": round(cold_ce, 4),
               "highest_tau_passed": last_passed_tau,
               "drops": drops_completed,
               "history": all_history}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
print(f"Latest checkpoint: {CKPT_LATEST}")
