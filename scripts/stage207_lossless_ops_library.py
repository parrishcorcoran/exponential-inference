"""Stage 207: build and validate a library of LOSSLESS preconditioning ops.

Each op is a math identity (Δ=0 by construction). We test each in
isolation, verify Δ=0, and characterize what it does to weight
distribution. Then in later stages we chain them before quantization.

Lossless ops to test:
1. Magnitude factoring (Stage 169 — reconfirm)
   W → α × unit_W
   α stores row magnitude

2. SmoothQuant-style RMSNorm-to-Linear migration
   For RMSNorm γ → Linear W chain:
     γ_new = γ / s
     W_new = W × diag(s)  (column-wise)
   Pulls outlier amplification from RMSNorm INTO body weight columns.

3. Output-row × input-column scaling absorption (within Linear)
   For Linear → RMSNorm → Linear chain:
     Out row of L1 scaled by C
     Gain of RMSNorm scaled by 1/C
     In col of L2 unchanged (RMSNorm-output is 1/C scaled, but normalization absorbs)
   Actually simpler: Linear1 row × C, Linear2 col × 1/C (between them)
   Net function unchanged.

4. Permutation matching (basic test only — full implementation is architecture-heavy)
   Permute rows of L1, columns of L2 (matching) — output unchanged.
   We test on a single isolated Linear pair.

Each test verifies Δ=0 in val CE. If yes, op is confirmed lossless.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128
N_VAL_CHUNKS = 32
RESULTS_PATH = Path("results/stage207_lossless_ops.json")
TARGET_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


def load_owt_cached():
    return torch.load("data/owt_tokens_50M.pt", map_location="cpu",
                      weights_only=True).long()


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
    return sum(losses) / max(len(losses), 1)


def measure_body_stats(model):
    """Body weight distribution stats for characterization."""
    row_norms = []
    intra_row_cv = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear): continue
        if not any(t in name for t in TARGET_NAMES): continue
        W = mod.weight.detach().float()
        rn = W.norm(dim=-1).cpu().numpy()
        row_norms.extend(rn.tolist())
        # Intra-row CV
        if W.shape[1] % 128 == 0:
            n_groups = W.shape[1] // 128
            grouped = W.reshape(W.shape[0], n_groups, 128)
            abs_w = grouped.abs()
            mean_abs = abs_w.mean(dim=-1, keepdim=True).clamp(min=1e-8)
            cv = (abs_w.std(dim=-1) / mean_abs.squeeze(-1)).cpu().numpy().flatten()
            intra_row_cv.extend(cv.tolist())
    return {
        "row_norm_mean": float(np.mean(row_norms)),
        "row_norm_cv": float(np.std(row_norms) / max(np.mean(row_norms), 1e-12)),
        "intra_row_cv_mean": float(np.mean(intra_row_cv)) if intra_row_cv else 0.0,
    }


def measure_norm_stats(model):
    all_g = []
    for n, p in model.named_parameters():
        if "norm" in n.lower() and "weight" in n:
            all_g.extend(p.detach().float().flatten().cpu().numpy().tolist())
    arr = np.array(all_g)
    return {"mean": float(arr.mean()), "max": float(np.abs(arr).max())}


# ─── Setup ───
print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading val tokens + base model (FP)...")
corpus = load_owt_cached()
val_tokens = corpus[:SEQ_LEN * 64].tolist()


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ─── Baseline ───
print("\nMeasuring T0 (base FP)...")
model = fresh_model()
T0 = lm_ce(model, val_tokens)
init_body = measure_body_stats(model)
init_norm = measure_norm_stats(model)
print(f"  T0 = {T0:.4f}")
print(f"  body row-norm mean={init_body['row_norm_mean']:.3f} CV={init_body['row_norm_cv']:.3f}")
print(f"  body intra-row CV={init_body['intra_row_cv_mean']:.3f}")
print(f"  norm: mean={init_norm['mean']:.3f}  max={init_norm['max']:.1f}")


results = {"T0_base_ce": float(T0), "init_body_stats": init_body, "init_norm_stats": init_norm,
           "ops_tested": []}


def test_op(op_name, op_func):
    """Apply op_func(model) in-place to a fresh model, measure Δ and stats."""
    print(f"\n{'─'*70}")
    print(f"OP: {op_name}")
    print('─'*70)
    m = fresh_model()
    op_func(m)
    ce = lm_ce(m, val_tokens)
    drift = ce - T0
    body = measure_body_stats(m)
    norm = measure_norm_stats(m)
    print(f"  Δ from T0: {drift:+.6f}  ({'✓ LOSSLESS' if abs(drift) < 1e-3 else '✗ LOSSY' if abs(drift) > 0.01 else '~ near-lossless'})")
    print(f"  body row-norm mean={body['row_norm_mean']:.3f}  CV={body['row_norm_cv']:.3f}")
    print(f"  body intra-row CV={body['intra_row_cv_mean']:.3f}")
    print(f"  norm: mean={norm['mean']:.3f}  max={norm['max']:.1f}")
    results["ops_tested"].append({
        "name": op_name,
        "drift": float(drift),
        "is_lossless": bool(abs(drift) < 1e-3),
        "body_stats": body,
        "norm_stats": norm,
    })
    del m
    import gc; gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


# ─── OP 1: Magnitude factoring (with α-bridge) ───
class AlphaLinear(nn.Module):
    def __init__(self, original_module, alpha_init):
        super().__init__()
        self.weight = original_module.weight
        self.bias = original_module.bias
        self.alpha = nn.Parameter(alpha_init.squeeze(-1).clone()
                                  .to(self.weight.device).to(torch.float32))
    def forward(self, x):
        out = F.linear(x, self.weight.to(x.dtype),
                       self.bias.to(x.dtype) if self.bias is not None else None)
        return out * self.alpha.to(out.dtype)


def op_magnitude_factoring(m):
    target_mods = [(n, mod) for n, mod in m.named_modules()
                   if isinstance(mod, nn.Linear) and any(t in n for t in TARGET_NAMES)]
    parent_lookup = {}
    for name, mod in m.named_modules():
        for child_name, child_mod in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            parent_lookup[full] = (mod, child_name)
    for name, mod in target_mods:
        rn = mod.weight.data.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
        mod.weight.data = (mod.weight.data.float() / rn).to(mod.weight.dtype)
        new_layer = AlphaLinear(mod, rn)
        parent, child_attr = parent_lookup[name]
        setattr(parent, child_attr, new_layer)


test_op("Magnitude factoring (Stage 169 reconfirm)", op_magnitude_factoring)


# ─── OP 2: SmoothQuant-style RMSNorm-to-Linear migration ───
# For RMSNorm before each body Linear: scale gain by 1/s, scale Linear cols by s.
# We use s = max(|gain|, 1.0) so we cap outliers DOWN by setting their s = gain
# (then 1/s makes them 1.0).
def op_smoothquant_migration(m):
    """Migrate RMSNorm gain outliers into Linear weight columns.

    For each (RMSNorm, Linear) pair:
      s_j = max(1, |gain_j| / TARGET_GAIN)  → s_j ≥ 1 only for outliers
      gain_new = gain / s
      W column j × s_j
    """
    TARGET_GAIN = 5.0   # outliers above this get migrated

    # Find each transformer block's (RMSNorm, body linears that follow)
    # Qwen3 structure: input_layernorm → q/k/v_proj; post_attention_layernorm → gate/up_proj
    norm_to_linears = []   # list of (RMSNorm module, [Linear modules that consume its output])

    # Walk model and pair norm+linears
    for layer_idx, layer in enumerate(m.model.layers):
        if hasattr(layer, "input_layernorm"):
            norm_to_linears.append((
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
            ))
        if hasattr(layer, "post_attention_layernorm"):
            norm_to_linears.append((
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj]
            ))

    for norm, linears in norm_to_linears:
        gain = norm.weight.data.float()
        sign = torch.sign(gain)
        mag = gain.abs()
        # Migration factor s: outliers get migrated, bulk untouched
        s = torch.clamp(mag / TARGET_GAIN, min=1.0)  # ≥ 1
        # Update gain: gain_new = gain / s (sign-preserving)
        norm.weight.data = (sign * (mag / s)).to(norm.weight.dtype)
        # Update Linear columns: W[:, j] *= s[j]  (input-channel-wise)
        for lin in linears:
            lin.weight.data = (lin.weight.data.float() * s.unsqueeze(0)).to(lin.weight.dtype)


test_op("SmoothQuant-style RMSNorm-to-body migration (cap=5)", op_smoothquant_migration)


# ─── OP 3: Linear-to-Linear scaling absorption ───
# For Linear1 → ... → Linear2, can scale row j of L1 by C_j and column j of L2 by 1/C_j.
# This works only if there's no nonlinearity between (and any RMSNorm absorbs the scaling).
# Skip for now — between linears in transformer there's always RMSNorm or attention.


# ─── OP 4: Permutation matching ───
# Within a single transformer block: permute output rows of q_proj. Then attention
# would produce permuted output. To compensate, permute o_proj's input columns
# by the inverse permutation.
def op_permutation_q_to_o(m):
    """For each layer: permute q_proj output rows + corresponding heads of o_proj input."""
    for layer in m.model.layers:
        n_heads = layer.self_attn.num_heads if hasattr(layer.self_attn, 'num_heads') else None
        if n_heads is None:
            continue
        head_dim = layer.self_attn.q_proj.weight.shape[0] // n_heads
        # Permute heads
        perm = torch.randperm(n_heads)
        q_w = layer.self_attn.q_proj.weight.data
        # q_proj.weight is [n_heads*head_dim, hidden]. Reshape to [n_heads, head_dim, hidden]
        q_w_reshape = q_w.view(n_heads, head_dim, q_w.shape[1])
        q_w_perm = q_w_reshape[perm]
        layer.self_attn.q_proj.weight.data = q_w_perm.view(q_w.shape)
        # Same for k_proj (number of heads might differ for GQA; need num_key_value_heads)
        # Skip k/v for now (GQA complications); just q + o.
        # For o_proj.weight [hidden, n_heads*head_dim], permute INPUT heads:
        o_w = layer.self_attn.o_proj.weight.data
        o_w_reshape = o_w.view(o_w.shape[0], n_heads, head_dim)
        # Inverse permutation: perm[i] is the new position of original head i
        # We need: reshuffle so head perm[i] in new q corresponds to head perm[i] in o
        # Since perm is the new arrangement of q heads, o needs same perm on its INPUT heads
        o_w_perm = o_w_reshape[:, perm, :]
        layer.self_attn.o_proj.weight.data = o_w_perm.view(o_w.shape)


test_op("Permutation matching (q_proj heads ↔ o_proj input heads)", op_permutation_q_to_o)


# ─── Save ───
print(f"\n{'='*70}")
print("LOSSLESS OPS LIBRARY SUMMARY")
print('='*70)
print(f"  {'op':<55} {'Δ':>10} {'verdict':>15}")
for op in results["ops_tested"]:
    verdict = "LOSSLESS ✓" if op["is_lossless"] else f"lossy ({op['drift']:+.4f})"
    print(f"  {op['name']:<55} {op['drift']:>+10.6f} {verdict:>15}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
