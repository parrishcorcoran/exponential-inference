"""Lever Matrix Part C — Fine-grained sweeps.

Loads the KV-128 floor model from SSD, then tests:
  1. KV heads UP (8→9→10→12→16) — can we ADD heads to compensate?
  2. MLP at 1% granularity (99%→98%→...→85%) — find exact inflection
  3. Head angle rotation — rotate Q/K subspaces, measure impact

All results append to results/full_lever_matrix.json.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import copy
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
def eval_ppl(model, val_tokens, seq_len, device):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= 10: break
    return total / max(n, 1)


def generate_sample(model, tokenizer, prompt, n=30):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════
# Load KV-128 floor model from SSD
# ═══════════════════════════════════════════════════════
FLOOR_PATH = "checkpoints/kv_floor_14b"
PROMPT = "The theory of general relativity describes gravity as"
SEQ_LEN = 128

print("=" * 60)
print("LEVER MATRIX PART C — Fine-grained sweeps")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(FLOOR_PATH, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 20, split="validation")

# Load existing results
results_path = Path("results/full_lever_matrix.json")
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
else:
    results = {}

floor_ppl = results.get("floor", {}).get("ppl", 15.7)
print(f"  Floor ppl (from Part A): {floor_ppl:.1f}")


# ═══════════════════════════════════════════════════════
# TEST 1: MLP at 1% granularity (99% → 85%)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 1: MLP intermediate at 1% granularity")
print(f"{'='*60}")

mlp_fine_results = []
for pct in range(99, 84, -1):
    print(f"\n  Loading fresh floor model for MLP {pct}%...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLOOR_PATH, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    # Shrink MLP intermediate dims
    for layer in model.model.layers:
        for name in ["gate_proj", "up_proj"]:
            w = getattr(layer.mlp, name).weight
            full = w.shape[0]
            keep = int(full * pct / 100)
            # Zero out the pruned rows (equivalent to removing them)
            w.data[keep:] = 0

        # down_proj: zero out pruned columns
        w = layer.mlp.down_proj.weight
        full = w.shape[1]
        keep = int(full * pct / 100)
        w.data[:, keep:] = 0

    ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
    ppl = math.exp(ce)
    text = generate_sample(model, tokenizer, PROMPT)
    delta = ppl - floor_ppl
    print(f"  MLP {pct}%: ppl={ppl:.1f} (Δ={delta:+.1f})  [{text[:60]}]")
    mlp_fine_results.append({
        "keep_pct": pct, "ppl": ppl, "delta": delta, "text": text[:80]
    })
    del model; torch.cuda.empty_cache()

    # Stop if way too broken
    if ppl > 1000:
        print(f"  ⚠ MLP floor reached at {pct}%")
        break

results["mlp_fine_sweep"] = mlp_fine_results

# Save intermediate
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved MLP fine sweep to {results_path}")


# ═══════════════════════════════════════════════════════
# TEST 2: KV heads UP (8 → 9 → 10 → 12 → 16)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 2: KV heads UP — adding heads beyond baseline 8")
print(f"{'='*60}")

# Qwen3-14B has 8 KV heads, 40 Q heads (GQA ratio 5:1)
# To ADD heads: duplicate existing KV heads and split Q groups
# This tests whether MORE KV heads can compensate for other compression

kv_up_results = []

# First: measure baseline 8 heads on floor model
print(f"\n  Loading floor model (8 KV heads baseline)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    FLOOR_PATH, dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
ppl = math.exp(ce)
text = generate_sample(model, tokenizer, PROMPT)
print(f"  KV 8 heads (baseline): ppl={ppl:.1f}  [{text[:60]}]")
kv_up_results.append({"heads": 8, "ppl": ppl, "delta": 0, "text": text[:80], "note": "baseline"})
del model; torch.cuda.empty_cache()

# Going UP: we can't truly add heads without reshaping the model architecture.
# But we CAN test: what if we DON'T share KV heads across Q groups?
# i.e., ungroup the GQA — give each Q head its own KV pair.
# This is equivalent to "more KV heads" without architecture change.
#
# Method: for each layer, expand k_proj/v_proj by duplicating rows
# to match q_proj dimensions (40 heads instead of 8).
# This increases KV cache but tests whether more angular resolution helps.

for target_heads in [10, 12, 16, 20, 40]:
    print(f"\n  Loading floor model for {target_heads} KV heads...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLOOR_PATH, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    num_kv_heads = model.config.num_key_value_heads  # 8
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 128

    if target_heads <= num_kv_heads:
        print(f"  Skipping {target_heads} (≤ baseline {num_kv_heads})")
        del model; torch.cuda.empty_cache()
        continue

    # Expand KV projections by interpolating between existing heads
    for layer in model.model.layers:
        for name in ("k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            W = proj.weight.data  # [num_kv_heads * head_dim, hidden_size]

            # Reshape to per-head: [num_kv_heads, head_dim, hidden_size]
            W_heads = W.view(num_kv_heads, head_dim, -1)

            # Interpolate to target_heads
            # Use repeat_interleave (same as GQA expansion) then average neighbors
            ratio = target_heads / num_kv_heads
            if target_heads % num_kv_heads == 0:
                # Clean multiple — just repeat
                W_new = W_heads.repeat_interleave(target_heads // num_kv_heads, dim=0)
            else:
                # Interpolate: create target_heads by lerping between neighbors
                W_new = torch.zeros(target_heads, head_dim, W_heads.shape[2],
                                    dtype=W_heads.dtype, device=W_heads.device)
                for i in range(target_heads):
                    src = i * num_kv_heads / target_heads
                    lo = int(src)
                    hi = min(lo + 1, num_kv_heads - 1)
                    frac = src - lo
                    W_new[i] = W_heads[lo] * (1 - frac) + W_heads[hi] * frac

            # Create new Linear with expanded size
            new_out = target_heads * head_dim
            new_proj = nn.Linear(W.shape[1], new_out, bias=proj.bias is not None,
                                 dtype=W.dtype, device=W.device)
            new_proj.weight.data = W_new.view(new_out, -1)
            if proj.bias is not None:
                # Expand bias similarly
                b = proj.bias.data.view(num_kv_heads, head_dim)
                if target_heads % num_kv_heads == 0:
                    b_new = b.repeat_interleave(target_heads // num_kv_heads, dim=0)
                else:
                    b_new = torch.zeros(target_heads, head_dim, dtype=b.dtype, device=b.device)
                    for i in range(target_heads):
                        src = i * num_kv_heads / target_heads
                        lo = int(src); hi = min(lo+1, num_kv_heads-1); frac = src - lo
                        b_new[i] = b[lo] * (1-frac) + b[hi] * frac
                new_proj.bias.data = b_new.view(-1)
            setattr(layer.self_attn, name, new_proj)

        # Update config for this layer's attention
        layer.self_attn.num_key_value_heads = target_heads
        # num_key_value_groups changes too
        if hasattr(layer.self_attn, 'num_key_value_groups'):
            layer.self_attn.num_key_value_groups = model.config.num_attention_heads // target_heads

    model.config.num_key_value_heads = target_heads
    torch.cuda.empty_cache()

    try:
        ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
        ppl = math.exp(ce)
        text = generate_sample(model, tokenizer, PROMPT)
        delta = ppl - floor_ppl
        print(f"  KV {target_heads} heads: ppl={ppl:.1f} (Δ={delta:+.1f})  [{text[:60]}]")
        kv_up_results.append({
            "heads": target_heads, "ppl": ppl, "delta": delta,
            "text": text[:80], "note": f"expanded from 8 via interpolation"
        })
    except Exception as e:
        print(f"  KV {target_heads} heads: FAILED — {e}")
        kv_up_results.append({"heads": target_heads, "ppl": None, "error": str(e)})

    del model; torch.cuda.empty_cache()

results["kv_head_sweep_up"] = kv_up_results

# Save intermediate
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved KV heads UP sweep to {results_path}")


# ═══════════════════════════════════════════════════════
# TEST 3: Head angle rotation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 3: Head angle rotation — rotate Q/K subspaces")
print(f"{'='*60}")

# Test: apply random orthogonal rotation to Q and K head subspaces.
# If heads care about ANGLE (not just rank), rotation should hurt.
# If heads are angle-invariant, rotation should be free.

angle_results = []

for angle_deg in [0, 5, 10, 15, 30, 45, 90]:
    print(f"\n  Loading floor model for {angle_deg}° rotation...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLOOR_PATH, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    if angle_deg > 0:
        angle_rad = math.radians(angle_deg)
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        for layer in model.model.layers:
            q_proj = layer.self_attn.q_proj
            k_proj = layer.self_attn.k_proj

            # For each head, apply a Givens rotation in the first 2 dims
            # This rotates the Q/K subspace by angle_deg
            Wq = q_proj.weight.data.float()
            Wk = k_proj.weight.data.float()

            num_q = model.config.num_attention_heads
            num_k = model.config.num_key_value_heads

            # Rotate Q heads
            for h in range(num_q):
                start = h * head_dim
                # Givens rotation on dims 0,1 of each head
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                row0 = Wq[start].clone()
                row1 = Wq[start + 1].clone()
                Wq[start] = cos_a * row0 - sin_a * row1
                Wq[start + 1] = sin_a * row0 + cos_a * row1

            # Rotate K heads with SAME rotation (preserves dot product if both rotated)
            # But we rotate K by a DIFFERENT angle to test sensitivity
            # Actually: rotate Q only. If Q·K^T matters, misaligned rotation hurts.
            # If only the projection subspace matters, rotation is free.

            q_proj.weight.data = Wq.to(q_proj.weight.dtype)
            # K stays the same — this creates an angular mismatch

    ce = eval_ppl(model, val_tokens, SEQ_LEN, device)
    ppl = math.exp(ce)
    text = generate_sample(model, tokenizer, PROMPT)
    delta = ppl - floor_ppl
    print(f"  Rotation {angle_deg}°: ppl={ppl:.1f} (Δ={delta:+.1f})  [{text[:60]}]")
    angle_results.append({
        "angle_deg": angle_deg, "ppl": ppl, "delta": delta,
        "text": text[:80],
        "note": "Givens rotation on Q only (dims 0,1 per head), K unchanged"
    })
    del model; torch.cuda.empty_cache()

results["head_angle_rotation"] = angle_results

# Save final
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)


# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART C SUMMARY")
print(f"{'='*60}")

print(f"\n  Floor ppl: {floor_ppl:.1f}")

print(f"\n  MLP fine sweep (1% steps):")
for r in mlp_fine_results:
    print(f"    {r['keep_pct']}%: ppl={r['ppl']:.1f} (Δ={r['delta']:+.1f})")

print(f"\n  KV heads UP:")
for r in kv_up_results:
    if r.get('ppl'):
        print(f"    {r['heads']} heads: ppl={r['ppl']:.1f} (Δ={r['delta']:+.1f})")

print(f"\n  Head angle rotation:")
for r in angle_results:
    print(f"    {r['angle_deg']}°: ppl={r['ppl']:.1f} (Δ={r['delta']:+.1f})")

print(f"\n  Saved to {results_path}", flush=True)
