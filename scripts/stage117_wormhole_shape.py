"""Stage 117 — Wormhole shape measurement on 14B.

Measure the natural effective rank (r99 = rank capturing 99% of variance)
of the residual stream at each layer. This is what the residual stream
ACTUALLY uses, not what we force via compression.

Mac found on 0.6B:
  L0:  r99=153 (mouth)
  L2:  r99=1   (entering throat)
  L14: r99=1   (deep throat)
  L25: r99=35  (re-opening)
  L27: r99=130 (exit boundary)

If this is a wormhole (not just a bathtub), 14B should show the same
shape scaled by d_model ratio: 5120/1024 = 5x.

Then: the TOTAL ANNEAL uses this shape as the target compression
profile. Squeeze each layer to its natural rank. The wormhole throat
gets compressed to rank-1, the mouths stay high-rank.
"""
import torch
import torch.nn.functional as F
import math
import json
import numpy as np
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


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 256

print("=" * 60)
print("STAGE 117 — WORMHOLE SHAPE (14B)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 10, split="validation")

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size
print(f"  L={L} d={d}", flush=True)

# Run forward pass with hidden states
inp = torch.tensor([val_tokens[:SEQ_LEN]], dtype=torch.long, device=device)
with torch.no_grad():
    out = model(inp, use_cache=False, output_hidden_states=True)

hidden_states = out.hidden_states  # L+1 tensors of [1, seq_len, d]

print(f"\n  Per-layer residual stream analysis:")
print(f"  {'Layer':>5} {'r99':>5} {'r95':>5} {'r90':>5} {'PR':>6} {'||h||':>8} {'top_sv':>8} {'Interpretation'}")

results = []

for i, h in enumerate(hidden_states):
    # h: [1, seq_len, d] — use all positions
    H = h[0].float()  # [seq_len, d]

    # Center
    H_centered = H - H.mean(dim=0, keepdim=True)

    # SVD of the [seq_len × d] matrix
    U, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)

    # Effective ranks
    S_sq = S ** 2
    total_var = S_sq.sum().item()
    cumvar = torch.cumsum(S_sq, dim=0) / total_var

    r99 = (cumvar < 0.99).sum().item() + 1
    r95 = (cumvar < 0.95).sum().item() + 1
    r90 = (cumvar < 0.90).sum().item() + 1

    # Participation ratio
    S_norm = S_sq / total_var
    pr = 1.0 / (S_norm ** 2).sum().item() if total_var > 0 else 0

    # Norms
    h_norm = H.norm(dim=-1).mean().item()
    top_sv = S[0].item()

    # Interpret
    if r99 <= 3:
        interp = "THROAT (rank-1 wormhole)"
    elif r99 <= 20:
        interp = "narrow passage"
    elif r99 <= 100:
        interp = "re-opening"
    else:
        interp = "mouth (high-rank boundary)"

    results.append({
        "layer": i, "r99": r99, "r95": r95, "r90": r90,
        "pr": round(pr, 1), "h_norm": round(h_norm, 1),
        "top_sv": round(top_sv, 1), "interp": interp
    })

    print(f"  L{i:>3}  {r99:>5} {r95:>5} {r90:>5} {pr:>6.1f} {h_norm:>8.1f} {top_sv:>8.1f}  {interp}")

# Multi-sequence measurement for robustness
print(f"\n  Running 5 more sequences for stability...", flush=True)
all_r99 = {i: [] for i in range(L + 1)}

for seq_idx in range(5):
    start = (seq_idx + 1) * SEQ_LEN
    if start + SEQ_LEN >= len(val_tokens):
        break
    inp = torch.tensor([val_tokens[start:start + SEQ_LEN]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(inp, use_cache=False, output_hidden_states=True)
    for i, h in enumerate(out.hidden_states):
        H = h[0].float()
        H_c = H - H.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(H_c, full_matrices=False)
        S_sq = S ** 2
        cumvar = torch.cumsum(S_sq, dim=0) / S_sq.sum()
        r99 = (cumvar < 0.99).sum().item() + 1
        all_r99[i].append(r99)

# Add first measurement
for r in results:
    all_r99[r["layer"]].append(r["r99"])

print(f"\n  Stability check (mean ± std of r99 across sequences):")
stable_r99 = {}
for i in range(L + 1):
    vals = all_r99[i]
    if vals:
        mean_r = np.mean(vals)
        std_r = np.std(vals)
        stable_r99[i] = {"mean": round(mean_r, 1), "std": round(std_r, 1)}
        if i % 4 == 0 or i == L:
            print(f"    L{i:>2}: r99 = {mean_r:.0f} ± {std_r:.0f}")

# Wormhole identification
print(f"\n{'='*60}")
print("WORMHOLE ANALYSIS")
print(f"{'='*60}")

# Find throat (minimum r99 region)
r99_values = [r["r99"] for r in results]
min_r99 = min(r99_values)
throat_layers = [r["layer"] for r in results if r["r99"] <= min_r99 * 2]
mouth_entry = [r["layer"] for r in results if r["r99"] > 50 and r["layer"] < L // 2]
mouth_exit = [r["layer"] for r in results if r["r99"] > 50 and r["layer"] > L // 2]

print(f"  Minimum r99: {min_r99} (throat)")
print(f"  Throat layers (r99 ≤ {min_r99*2}): L{min(throat_layers)}-L{max(throat_layers)}")
if mouth_entry:
    print(f"  Entry mouth (r99 > 50): L{min(mouth_entry)}-L{max(mouth_entry)}")
if mouth_exit:
    print(f"  Exit mouth (r99 > 50): L{min(mouth_exit)}-L{max(mouth_exit)}")

# Comparison with 0.6B
print(f"\n  Scale comparison (0.6B → 14B):")
print(f"    d_model ratio: 5120/1024 = 5.0x")
print(f"    0.6B throat r99: 1")
print(f"    14B throat r99: {min_r99}")
print(f"    Ratio: {min_r99/1:.1f}x (expected ~5x if linear scaling)")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/stage117_wormhole_shape.json", "w") as f:
    json.dump({
        "model": MODEL, "L": L, "d": d, "seq_len": SEQ_LEN,
        "per_layer": results,
        "stability": {str(k): v for k, v in stable_r99.items()},
        "throat_min_r99": min_r99,
        "throat_layers": throat_layers,
    }, f, indent=2)
print(f"\nSaved results/stage117_wormhole_shape.json", flush=True)
