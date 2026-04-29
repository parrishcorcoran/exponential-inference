"""Stage 179: Head ablation — does our magnitude-based head importance
match actual ablation damage?

Hypothesis: per-head magnitude in o_proj rows reflects how much each
head matters. Test causally: zero out one head at a time, measure
CE damage.

Procedure:
  1. Pick a target layer (middle layer, say 14 of Qwen3-0.6B's 28)
  2. Reshape o_proj.weight to [out, n_heads, head_dim]
  3. For each head h, compute mean magnitude across rows
  4. Rank heads by mean magnitude (importance)
  5. For each head h: zero out the input slice
     [head_dim*h : head_dim*(h+1)] of o_proj's INPUT (equivalently,
     zero out that head's weights in o_proj rows). Measure val CE.
  6. Compare: do high-magnitude heads cause more CE damage?

Bonus: also do this for q_proj for comparison (which doesn't have
per-head structure). If q_proj head ablation damage is roughly uniform
regardless of magnitude, that confirms the structure is unique to
o_proj.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


CHECKPOINT = "Qwen/Qwen3-0.6B"
TARGET_LAYER = 14
SEQ_LEN = 128
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage179_head_ablation.json")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt(tokenizer, max_tokens):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
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
    return sum(losses) / len(losses)


print(f"device={device} dtype={dtype}")
cfg = AutoConfig.from_pretrained(CHECKPOINT, trust_remote_code=True)
n_heads = cfg.num_attention_heads
hidden = cfg.hidden_size
head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads)
n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
print(f"  hidden={hidden}, n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

print(f"\nLoading val tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * 64)


# ─── Find target o_proj and q_proj ───
o_proj_layer = None
q_proj_layer = None
for name, mod in model.named_modules():
    if not isinstance(mod, nn.Linear): continue
    if f"layers.{TARGET_LAYER}." not in name: continue
    if "o_proj" in name: o_proj_layer = mod
    elif "q_proj" in name: q_proj_layer = mod

assert o_proj_layer is not None and q_proj_layer is not None, "Couldn't find target layer projections"
print(f"\nTarget layer {TARGET_LAYER}: o_proj shape {tuple(o_proj_layer.weight.shape)}, q_proj shape {tuple(q_proj_layer.weight.shape)}")


# ─── Compute per-head magnitudes ───
W_o = o_proj_layer.weight.detach().float()  # [out, in] = [hidden, hidden]
W_o_reshaped = W_o.reshape(W_o.shape[0], n_heads, head_dim)
o_proj_per_head_mag = W_o_reshaped.norm(dim=-1).mean(dim=0).cpu().numpy()  # [n_heads]

W_q = q_proj_layer.weight.detach().float()  # [n_heads*head_dim, hidden] for non-MQA
# q_proj's OUTPUT dim is the heads, so reshape output dim
n_q_heads = cfg.num_attention_heads if W_q.shape[0] == n_heads * head_dim else (W_q.shape[0] // head_dim)
W_q_reshaped = W_q.reshape(n_q_heads, head_dim, W_q.shape[1])
q_proj_per_head_mag = W_q_reshaped.norm(dim=(-1, -2)).cpu().numpy()  # [n_q_heads]

print(f"\no_proj per-head importance (input-side groups):")
for h in range(n_heads):
    print(f"  head {h:>2}: magnitude = {o_proj_per_head_mag[h]:.4f}")

# ─── Baseline ───
T0 = lm_ce(model, val_tokens)
print(f"\nBaseline CE: {T0:.4f}")


# ─── Ablation test on o_proj ───
print(f"\n{'='*70}\nAblating o_proj heads one at a time\n{'='*70}")
o_results = []
W_o_orig = o_proj_layer.weight.data.clone()
for h in range(n_heads):
    # Zero out the input slice corresponding to head h
    W_modified = W_o_orig.clone()
    W_modified[:, h*head_dim:(h+1)*head_dim] = 0
    o_proj_layer.weight.data = W_modified
    ce = lm_ce(model, val_tokens)
    delta = ce - T0
    o_results.append({"head": h, "magnitude": float(o_proj_per_head_mag[h]),
                      "ce": float(ce), "delta": float(delta)})
    print(f"  head {h:>2} (mag={o_proj_per_head_mag[h]:.3f}): CE={ce:.4f}  Δ={delta:+.4f}")

o_proj_layer.weight.data = W_o_orig


# ─── Correlation: does damage match magnitude? ───
mags = np.array([r["magnitude"] for r in o_results])
deltas = np.array([r["delta"] for r in o_results])
corr = np.corrcoef(mags, deltas)[0, 1]
print(f"\n  Pearson correlation (magnitude vs CE damage): {corr:.4f}")
print(f"  Spearman rank correlation: {np.corrcoef(np.argsort(mags), np.argsort(deltas))[0,1]:.4f}")

if corr > 0.5:
    print(f"  ✓ STRONG positive correlation — high-magnitude heads cause more damage")
    print(f"    Magnitude-based head importance IS validated by ablation")
elif corr > 0.2:
    print(f"  ~ Moderate positive correlation — magnitude predicts damage somewhat")
else:
    print(f"  ✗ Weak/no correlation — magnitude doesn't predict ablation damage")
    print(f"    Need different mechanism to explain o_proj structure")


# ─── For comparison: ablate q_proj heads ───
print(f"\n{'='*70}\nAblating q_proj heads (control — should be uniform)\n{'='*70}")
q_results = []
W_q_orig = q_proj_layer.weight.data.clone()
for h in range(n_q_heads):
    W_modified = W_q_orig.clone()
    # Zero out the OUTPUT slice (the head's queries)
    W_modified[h*head_dim:(h+1)*head_dim, :] = 0
    q_proj_layer.weight.data = W_modified
    ce = lm_ce(model, val_tokens)
    delta = ce - T0
    q_results.append({"head": h, "magnitude": float(q_proj_per_head_mag[h]),
                      "ce": float(ce), "delta": float(delta)})
    print(f"  head {h:>2} (mag={q_proj_per_head_mag[h]:.3f}): CE={ce:.4f}  Δ={delta:+.4f}")

q_proj_layer.weight.data = W_q_orig

q_mags = np.array([r["magnitude"] for r in q_results])
q_deltas = np.array([r["delta"] for r in q_results])
q_corr = np.corrcoef(q_mags, q_deltas)[0, 1] if len(q_mags) > 1 else 0
print(f"\n  q_proj Pearson correlation: {q_corr:.4f}")


# ─── Save ───
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "target_layer": TARGET_LAYER,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "baseline_ce": float(T0),
        "o_proj_ablation": o_results,
        "o_proj_corr_pearson": float(corr),
        "q_proj_ablation": q_results,
        "q_proj_corr_pearson": float(q_corr),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
