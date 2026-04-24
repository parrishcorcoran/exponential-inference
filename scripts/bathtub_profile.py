"""Bathtub profile — per-layer importance on Qwen3-14B.

Measure per-layer contribution to the output. Multiple signals:
  1. Per-layer probe CE (how well each layer can predict on its own)
  2. Per-layer weight norm (how much each layer "does")
  3. Per-layer activation norm (how much signal flows through)
  4. Per-layer ablation (zero out layer, measure damage)
  5. Per-layer KV compression sensitivity (SVD rank 64 per layer, measure damage)

If bathtub-shaped: middle layers are compressible, edges are not.
This gives us a per-layer compression budget map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
from pathlib import Path

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def iter_batches(tokens, seq_len, batch_size, device):
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


@torch.no_grad()
def eval_ce(model, val_tokens, seq_len, device, n_batches=10):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return total / max(n, 1)


FLOOR_PATH = "checkpoints/kv_floor_14b"
SEQ_LEN = 128

print("=" * 60)
print("BATHTUB PROFILE — per-layer importance")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(FLOOR_PATH, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 20, split="validation")

# Load model
print("Loading floor model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    FLOOR_PATH, dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size
print(f"  L={L} d={d}", flush=True)

# Baseline
baseline_ce = eval_ce(model, val_tokens, SEQ_LEN, device)
baseline_ppl = math.exp(baseline_ce)
print(f"  Baseline: ce={baseline_ce:.4f} ppl={baseline_ppl:.1f}\n", flush=True)

results = {"baseline_ce": baseline_ce, "baseline_ppl": baseline_ppl, "L": L}

# ═══════════════════════════════════════════════════════
# 1. Weight norms per layer
# ═══════════════════════════════════════════════════════
print("--- Weight norms per layer ---")
weight_norms = []
for i, layer in enumerate(model.model.layers):
    attn_norm = sum(getattr(layer.self_attn, n).weight.float().norm().item()
                    for n in ["q_proj", "k_proj", "v_proj", "o_proj"])
    mlp_norm = sum(getattr(layer.mlp, n).weight.float().norm().item()
                   for n in ["gate_proj", "up_proj", "down_proj"])
    total = attn_norm + mlp_norm
    weight_norms.append({"layer": i, "attn": attn_norm, "mlp": mlp_norm, "total": total})
    if i % 8 == 0 or i == L - 1:
        print(f"  L{i:>2}: attn={attn_norm:.1f}  mlp={mlp_norm:.1f}  total={total:.1f}")
results["weight_norms"] = weight_norms

# ═══════════════════════════════════════════════════════
# 2. Activation norms per layer (hidden state norms)
# ═══════════════════════════════════════════════════════
print("\n--- Activation norms per layer ---")
act_norms = []
with torch.no_grad():
    for inp, tgt in iter_batches(val_tokens, SEQ_LEN, 1, device):
        out = model(inp, use_cache=False, output_hidden_states=True)
        for i, h in enumerate(out.hidden_states):
            norm = h.float().norm(dim=-1).mean().item()
            if len(act_norms) <= i:
                act_norms.append({"layer": i, "norm": 0, "count": 0})
            act_norms[i]["norm"] += norm
            act_norms[i]["count"] += 1
        break  # one batch is enough for norms

for a in act_norms:
    a["norm"] /= max(a["count"], 1)
    del a["count"]
results["activation_norms"] = act_norms
for a in act_norms:
    if a["layer"] % 8 == 0 or a["layer"] == L:
        print(f"  L{a['layer']:>2}: norm={a['norm']:.1f}")

# ═══════════════════════════════════════════════════════
# 3. Per-layer residual contribution (how much each layer CHANGES the hidden state)
# ═══════════════════════════════════════════════════════
print("\n--- Residual contribution per layer ---")
residual_contrib = []
with torch.no_grad():
    for inp, tgt in iter_batches(val_tokens, SEQ_LEN, 1, device):
        out = model(inp, use_cache=False, output_hidden_states=True)
        for i in range(L):
            h_in = out.hidden_states[i].float()
            h_out = out.hidden_states[i + 1].float()
            delta = (h_out - h_in).norm(dim=-1).mean().item()
            ratio = delta / h_in.norm(dim=-1).mean().item()
            residual_contrib.append({"layer": i, "delta_norm": delta, "relative": ratio})
        break

results["residual_contribution"] = residual_contrib
for r in residual_contrib:
    if r["layer"] % 4 == 0 or r["layer"] == L - 1:
        print(f"  L{r['layer']:>2}: Δ={r['delta_norm']:.1f}  relative={r['relative']:.4f}")

# ═══════════════════════════════════════════════════════
# 4. Per-layer ablation (skip each layer, measure CE increase)
# ═══════════════════════════════════════════════════════
print("\n--- Per-layer ablation (skip layer, measure damage) ---")

# We'll use a hook to skip individual layers
ablation_results = []

for skip_layer in range(L):
    # Hook: make layer i an identity (output = input)
    original_forward = model.model.layers[skip_layer].forward

    def identity_forward(*args, **kwargs):
        # Return the hidden states unchanged
        # Layer forward returns (hidden_states, ...)
        hidden = args[0]
        # Return tuple matching original output format
        return (hidden,) + (None,) * (len(original_forward(*args, **kwargs)) - 1 if False else 0)

    # Simpler: just scale residual to zero by zeroing all weights temporarily
    # Even simpler: use output_hidden_states and reconstruct
    # Simplest: hook that replaces output
    handle = None
    skip_idx = skip_layer

    def make_hook(idx):
        def hook_fn(module, input, output):
            # output could be tuple or single tensor depending on version
            if isinstance(output, tuple):
                return (input[0],) + output[1:]
            else:
                return input[0]
        return hook_fn

    handle = model.model.layers[skip_layer].register_forward_hook(make_hook(skip_layer))

    ce = eval_ce(model, val_tokens, SEQ_LEN, device, n_batches=5)
    handle.remove()

    damage = ce - baseline_ce
    ppl = math.exp(ce)
    ablation_results.append({"layer": skip_layer, "ce": ce, "ppl": ppl, "damage": damage})
    if skip_layer % 4 == 0 or skip_layer == L - 1:
        print(f"  skip L{skip_layer:>2}: ce={ce:.3f} ppl={ppl:.1f} damage={damage:+.3f}")

results["ablation"] = ablation_results

# ═══════════════════════════════════════════════════════
# 5. Per-layer KV compression sensitivity
# ═══════════════════════════════════════════════════════
print("\n--- Per-layer KV compression sensitivity (rank 64) ---")
kv_sensitivity = []

for target_layer in range(L):
    # Reload fresh each time would be too slow. Instead, compress one layer,
    # measure, then restore it.
    layer = model.model.layers[target_layer]

    # Save original weights
    k_orig = layer.self_attn.k_proj.weight.data.clone()
    v_orig = layer.self_attn.v_proj.weight.data.clone()

    # Compress this layer's KV to rank 64
    for pname in ("k_proj", "v_proj"):
        proj = getattr(layer.self_attn, pname)
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(64, len(S))
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)

    ce = eval_ce(model, val_tokens, SEQ_LEN, device, n_batches=5)
    damage = ce - baseline_ce

    # Restore
    layer.self_attn.k_proj.weight.data = k_orig
    layer.self_attn.v_proj.weight.data = v_orig

    kv_sensitivity.append({"layer": target_layer, "ce": ce, "damage": damage})
    if target_layer % 4 == 0 or target_layer == L - 1:
        print(f"  KV compress L{target_layer:>2}: damage={damage:+.4f}")

results["kv_sensitivity"] = kv_sensitivity

# ═══════════════════════════════════════════════════════
# 6. Per-layer MLP compression sensitivity (90% keep)
# ═══════════════════════════════════════════════════════
print("\n--- Per-layer MLP compression sensitivity (90% keep) ---")
mlp_sensitivity = []

for target_layer in range(L):
    layer = model.model.layers[target_layer]

    # Save originals
    gate_orig = layer.mlp.gate_proj.weight.data.clone()
    up_orig = layer.mlp.up_proj.weight.data.clone()
    down_orig = layer.mlp.down_proj.weight.data.clone()

    # Zero out bottom 10% of MLP dims
    full = gate_orig.shape[0]
    keep = int(full * 0.9)
    layer.mlp.gate_proj.weight.data[keep:] = 0
    layer.mlp.up_proj.weight.data[keep:] = 0
    layer.mlp.down_proj.weight.data[:, keep:] = 0

    ce = eval_ce(model, val_tokens, SEQ_LEN, device, n_batches=5)
    damage = ce - baseline_ce

    # Restore
    layer.mlp.gate_proj.weight.data = gate_orig
    layer.mlp.up_proj.weight.data = up_orig
    layer.mlp.down_proj.weight.data = down_orig

    mlp_sensitivity.append({"layer": target_layer, "ce": ce, "damage": damage})
    if target_layer % 4 == 0 or target_layer == L - 1:
        print(f"  MLP 90% L{target_layer:>2}: damage={damage:+.4f}")

results["mlp_sensitivity"] = mlp_sensitivity
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Summary — is it bathtub shaped?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("BATHTUB ANALYSIS")
print(f"{'='*60}")

# Check ablation shape
abl = [r["damage"] for r in ablation_results]
third = L // 3
early = sum(abl[:third]) / third
middle = sum(abl[third:2*third]) / third
late = sum(abl[2*third:]) / (L - 2*third)
print(f"\n  Ablation damage by region:")
print(f"    Early (L0-{third-1}):   {early:.4f}")
print(f"    Middle (L{third}-{2*third-1}): {middle:.4f}")
print(f"    Late (L{2*third}-{L-1}):   {late:.4f}")
if early > middle and late > middle:
    print(f"    → BATHTUB CONFIRMED (early={early/middle:.1f}x, late={late/middle:.1f}x vs middle)")
elif early > middle:
    print(f"    → FRONT-LOADED (early matters most)")
elif late > middle:
    print(f"    → BACK-LOADED (late matters most)")

# Check KV sensitivity shape
kv = [r["damage"] for r in kv_sensitivity]
early_kv = sum(kv[:third]) / third
middle_kv = sum(kv[third:2*third]) / third
late_kv = sum(kv[2*third:]) / (L - 2*third)
print(f"\n  KV compression damage by region:")
print(f"    Early:  {early_kv:.4f}")
print(f"    Middle: {middle_kv:.4f}")
print(f"    Late:   {late_kv:.4f}")

# Check MLP sensitivity shape
mlp = [r["damage"] for r in mlp_sensitivity]
early_mlp = sum(mlp[:third]) / third
middle_mlp = sum(mlp[third:2*third]) / third
late_mlp = sum(mlp[2*third:]) / (L - 2*third)
print(f"\n  MLP compression damage by region:")
print(f"    Early:  {early_mlp:.4f}")
print(f"    Middle: {middle_mlp:.4f}")
print(f"    Late:   {late_mlp:.4f}")

# Full per-layer ablation curve (ASCII art)
print(f"\n  Ablation damage per layer (higher = more important):")
max_dmg = max(abs(d) for d in abl) if abl else 1
for i, d in enumerate(abl):
    bar = "█" * int(40 * abs(d) / max_dmg) if max_dmg > 0 else ""
    print(f"    L{i:>2} {bar} {d:+.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/bathtub_profile.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved results/bathtub_profile.json", flush=True)
