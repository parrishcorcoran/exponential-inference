"""Stage 230 — Bonsai Q1_0_g128 PTQ replica on Qwen3-0.6B.

Goal (user, 2026-05-05):
  Recreate Bonsai's disclosed format faithfully so we have a real baseline
  to improve from. Their training procedure is undisclosed; their format is
  Q1_0_g128: per-128-group along inner-dim, 1 sign-bit/weight + 1 fp16 scale
  per 128-group = 1.125 bpw. Applied to ALL matrix-heavy weights including
  embeddings and lm_head. No SubLN, no architectural prep, no training.

Two scale-rule variants (Bonsai doesn't disclose theirs):
  - 'max':  α = max(|w|) per 128-group. Naive interpretation of sign·α.
  - 'mean': α = mean(|w|) per 128-group. MSE-optimal for symmetric distrib.

Coverage:
  - All nn.Linear in attention (q/k/v/o) and MLP (gate/up/down)
  - lm_head (the final output projection — Bonsai quantizes this, we hadn't been)
  - embed_tokens (the input embedding — Bonsai quantizes this too)

Comparison points (drift vs FP teacher on OWT val):
  - Stage 227 (per-row, with bake, STE, body-only):     +1.6968
  - Stage 228 (per-row, no bake, STE, body-only):       +1.7800
  - Stage 230 (this — Bonsai per-128, PTQ-only, FULL):  ???

If Stage 230 drifts much more than +1.7, Bonsai's published 89% retention
must come from their undisclosed training. Confirms QAT direction.
If Stage 230 drifts close to or below +1.7, per-128 grouping is doing more
than we expected and we should adopt it in our STE pipeline.
"""
import json
import sys
import time
from pathlib import Path

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
GROUP_SIZE = 128

RESULTS_PATH = Path("results/stage230_bonsai_replica_ptq.json")


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


def quantize_q1_0_g128(W, group_size=128, scale_rule='max'):
    """Bonsai-style per-group binarization along the inner (last) dim.

    W shape: [out_or_vocab, in_or_hidden]
    Returns: same-shape tensor with sign(W) * α[row, group_within_row].
    """
    out_f, in_f = W.shape
    if in_f % group_size != 0:
        # Bonsai's spec assumes divisibility. Pad with zeros if needed.
        pad = group_size - (in_f % group_size)
        W = F.pad(W, (0, pad))
        in_f = W.shape[1]
    n_groups = in_f // group_size
    W_grouped = W.view(out_f, n_groups, group_size)
    if scale_rule == 'max':
        alpha = W_grouped.abs().amax(dim=-1, keepdim=True)
    elif scale_rule == 'mean':
        alpha = W_grouped.abs().mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unknown scale_rule={scale_rule}")
    W_q = (torch.sign(W_grouped) * alpha).view(out_f, in_f)
    return W_q


# ─── BonsaiQuantLinear: replaces nn.Linear with frozen Q1_0_g128 weights ───
class BonsaiQuantLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, group_size=128, scale_rule='max'):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.group_size = group_size
        with torch.no_grad():
            W_q = quantize_q1_0_g128(original_linear.weight.data,
                                     group_size=group_size, scale_rule=scale_rule)
            # If we padded, slice back to original in_features
            W_q = W_q[:, :self.in_features]
        self.register_buffer('weight_q', W_q)
        if original_linear.bias is not None:
            self.register_buffer('bias_q', original_linear.bias.data.clone())
        else:
            self.bias_q = None

    def forward(self, x):
        return F.linear(x, self.weight_q, self.bias_q)


# ─── BonsaiQuantEmbedding: replaces nn.Embedding with Q1_0_g128 weights ───
class BonsaiQuantEmbedding(nn.Module):
    def __init__(self, original_emb: nn.Embedding, group_size=128, scale_rule='max'):
        super().__init__()
        self.num_embeddings = original_emb.num_embeddings
        self.embedding_dim = original_emb.embedding_dim
        self.padding_idx = original_emb.padding_idx
        with torch.no_grad():
            W_q = quantize_q1_0_g128(original_emb.weight.data,
                                     group_size=group_size, scale_rule=scale_rule)
            W_q = W_q[:, :self.embedding_dim]
        self.register_buffer('weight_q', W_q)

    def forward(self, ids):
        return F.embedding(ids, self.weight_q, self.padding_idx)


def replace_all_with_bonsai(model, group_size=128, scale_rule='max'):
    """Walk the model, replace every nn.Linear + nn.Embedding with Bonsai-quantized
    versions. Tracks coverage stats."""
    parent_lookup = {}
    for name, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)

    n_linear = 0
    n_embed = 0
    total_alpha_fp16 = 0
    total_weights = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            out_f, in_f = mod.weight.shape
            n_groups = (in_f + group_size - 1) // group_size  # ceil for padding
            total_alpha_fp16 += out_f * n_groups
            total_weights += out_f * in_f
            new_layer = BonsaiQuantLinear(mod, group_size, scale_rule)
            parent, child_attr = parent_lookup[name]
            setattr(parent, child_attr, new_layer)
            n_linear += 1
        elif isinstance(mod, nn.Embedding):
            vocab, hidden = mod.weight.shape
            n_groups = (hidden + group_size - 1) // group_size
            total_alpha_fp16 += vocab * n_groups
            total_weights += vocab * hidden
            new_layer = BonsaiQuantEmbedding(mod, group_size, scale_rule)
            parent, child_attr = parent_lookup[name]
            setattr(parent, child_attr, new_layer)
            n_embed += 1
    bits_per_weight = (total_weights + total_alpha_fp16 * 16) / max(total_weights, 1)
    return n_linear, n_embed, bits_per_weight, total_weights


def run_one_config(scale_rule, val_tokens, T0):
    print(f"\n{'='*60}")
    print(f"Stage 230 — Bonsai Q1_0_g128 PTQ replica  (scale_rule={scale_rule!r})")
    print('='*60, flush=True)

    print(f"Building student (fresh FP Qwen3-0.6B)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    # Verify student matches teacher pre-quant
    ce_pre = lm_ce(student, val_tokens)
    print(f"  Pre-quantization CE: {ce_pre:.4f}  (drift={ce_pre - T0:+.4f})", flush=True)

    # Apply Bonsai PTQ
    t0 = time.time()
    n_linear, n_embed, bpw, total_weights = replace_all_with_bonsai(
        student, group_size=GROUP_SIZE, scale_rule=scale_rule)
    quant_time = time.time() - t0
    print(f"  Quantized {n_linear} Linears + {n_embed} Embeddings in {quant_time:.1f}s",
          flush=True)
    print(f"  Total quantized weights: {total_weights:,}", flush=True)
    print(f"  Bits/weight: {bpw:.4f}  (Bonsai disclosed: 1.125)", flush=True)

    # Measure post-PTQ drift
    ce_post = lm_ce(student, val_tokens)
    drift = ce_post - T0
    print(f"  Post-quantization CE: {ce_post:.4f}  drift={drift:+.4f}", flush=True)

    # Free student before next run
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "scale_rule": scale_rule,
        "group_size": GROUP_SIZE,
        "n_linear_quantized": n_linear,
        "n_embed_quantized": n_embed,
        "total_quantized_weights": int(total_weights),
        "bits_per_weight": float(bpw),
        "ce_pre_quant": float(ce_pre),
        "ce_post_quant": float(ce_post),
        "drift": float(drift),
        "quant_time_s": float(quant_time),
    }


print(f"device={device} dtype={dtype}")
print("Loading OWT corpus...", flush=True)
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()
print(f"  val_tokens={len(val_tokens)}", flush=True)


# ─── Build TEACHER (frozen FP — same one used in our other stages) ───
print("\nBuilding TEACHER (frozen FP Qwen3-0.6B)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
T0 = lm_ce(teacher, val_tokens)
print(f"  Teacher T0 = {T0:.4f}", flush=True)


# ─── Run both scale rule variants ───
results = []
for rule in ['max', 'mean']:
    r = run_one_config(rule, val_tokens, T0)
    results.append(r)


# ─── Final summary ───
print(f"\n{'─'*60}")
print(f"STAGE 230 SUMMARY (Bonsai Q1_0_g128 PTQ on Qwen3-0.6B):")
print('─'*60)
print(f"  Teacher T0:                    {T0:.4f}")
for r in results:
    print(f"  scale={r['scale_rule']:<5}  bpw={r['bits_per_weight']:.4f}  "
          f"drift={r['drift']:+.4f}")
print()
print(f"  Reference points:")
print(f"    Stage 227 (per-row, bake+STE, body-only):  drift=+1.6968  bpw=1.0125")
print(f"    Stage 228 (per-row, no-bake STE, body-only): drift=+1.7800 bpw=1.0125")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "T0_teacher": float(T0),
        "n_val_chunks": N_VAL_CHUNKS,
        "seq_len": SEQ_LEN,
        "configs": results,
        "comparison": {
            "stage227_per_row_bake_ste_body_drift": 1.6968,
            "stage227_bpw": 1.0125,
            "stage228_per_row_no_bake_ste_body_drift": 1.7800,
            "stage228_bpw": 1.0125,
        },
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}", flush=True)
