"""Stage 115 — Bathtub-aware stacked compression on 14B.

Combine ALL compression levers with position-aware schedules:
  - Weight quant: Q6 edges, Q4 middle
  - KV rank: 512 edges, 64 middle (SVD)
  - MLP pruning: 100% edges, 90% middle
  - Layer skipping: test removing middle layers entirely

Each applied according to bathtub profile:
  Edge layers (L0-6, L33-39): 14 layers, full precision
  Middle layers (L7-32): 26 layers, aggressive compression

Test configs from least to most aggressive:
  A. Weight-only bathtub (Q6-edge + Q4-mid)
  B. + KV rank bathtub (512-edge + 64-mid)
  C. + MLP pruning bathtub (100%-edge + 90%-mid)
  D. Full stack A+B+C
  E. Full stack + skip every other middle layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import gc
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


def iter_batches(tokens, seq_len, device):
    n = (len(tokens) - 1) // seq_len
    for i in range(n):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        t = torch.tensor([window], dtype=torch.long, device=device)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    return math.exp(total / max(count, 1))


def generate_sample(model, tokenizer, prompt, n=30):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def quantize_layer(layer, bits):
    """Quantize all linear weights in a layer to n_bits. Per-channel (per output row)."""
    levels = 2 ** bits; half = levels // 2
    for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                          (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
        for name in names:
            w = getattr(parent, name).weight
            W = w.data.float()
            max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = max_abs / (half - 1)
            w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)


def compress_kv_layer(layer, rank):
    """SVD truncation on k_proj + v_proj of a single layer."""
    for name in ("k_proj", "v_proj"):
        proj = getattr(layer.self_attn, name)
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, len(S))
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)


def prune_mlp_layer(layer, keep_pct):
    """Zero out bottom rows of MLP intermediate."""
    for name in ["gate_proj", "up_proj"]:
        w = getattr(layer.mlp, name).weight
        keep = int(w.shape[0] * keep_pct / 100)
        w.data[keep:] = 0
    w = layer.mlp.down_proj.weight
    keep = int(w.shape[1] * keep_pct / 100)
    w.data[:, keep:] = 0


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"
EDGE_WIDTH = 7  # L0-6 and L33-39

print("=" * 60)
print("STAGE 115 — Bathtub-aware stacked compression (14B)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

# Teacher baseline
print("\nLoading teacher...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
L = model.config.num_hidden_layers
teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Teacher: ppl={teacher_ppl:.1f}  [{teacher_text[:60]}]")
del model; gc.collect(); torch.cuda.empty_cache()


def is_edge(layer_idx, L, edge_width):
    return layer_idx < edge_width or layer_idx >= L - edge_width


configs = [
    {
        "name": "A: Weight Q6-edge + Q4-mid",
        "weight_edge": 6, "weight_mid": 4,
        "kv_edge": None, "kv_mid": None,
        "mlp_edge": 100, "mlp_mid": 100,
        "skip_mid": False,
    },
    {
        "name": "B: KV rank 512-edge + 64-mid",
        "weight_edge": None, "weight_mid": None,
        "kv_edge": 512, "kv_mid": 64,
        "mlp_edge": 100, "mlp_mid": 100,
        "skip_mid": False,
    },
    {
        "name": "C: MLP 100%-edge + 90%-mid",
        "weight_edge": None, "weight_mid": None,
        "kv_edge": None, "kv_mid": None,
        "mlp_edge": 100, "mlp_mid": 90,
        "skip_mid": False,
    },
    {
        "name": "D: Full stack (Q6/Q4 + KV 512/64 + MLP 100/90)",
        "weight_edge": 6, "weight_mid": 4,
        "kv_edge": 512, "kv_mid": 64,
        "mlp_edge": 100, "mlp_mid": 90,
        "skip_mid": False,
    },
    {
        "name": "E: Full stack + skip every other middle layer",
        "weight_edge": 6, "weight_mid": 4,
        "kv_edge": 512, "kv_mid": 64,
        "mlp_edge": 100, "mlp_mid": 90,
        "skip_mid": True,
    },
    {
        "name": "F: Aggressive mid (Q4/Q3 + KV 512/32 + MLP 100/85)",
        "weight_edge": 4, "weight_mid": 3,
        "kv_edge": 512, "kv_mid": 32,
        "mlp_edge": 100, "mlp_mid": 85,
        "skip_mid": False,
    },
    {
        "name": "G: Q5 middle test (Q6/Q5)",
        "weight_edge": 6, "weight_mid": 5,
        "kv_edge": None, "kv_mid": None,
        "mlp_edge": 100, "mlp_mid": 100,
        "skip_mid": False,
    },
]

results = []

for cfg in configs:
    print(f"\n--- {cfg['name']} ---", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()

    skip_hooks = []

    for i in range(L):
        edge = is_edge(i, L, EDGE_WIDTH)
        layer = model.model.layers[i]

        # Weight quantization
        w_bits = cfg["weight_edge"] if edge else cfg["weight_mid"]
        if w_bits is not None:
            quantize_layer(layer, w_bits)

        # KV compression
        kv_rank = cfg["kv_edge"] if edge else cfg["kv_mid"]
        if kv_rank is not None:
            compress_kv_layer(layer, kv_rank)

        # MLP pruning
        mlp_pct = cfg["mlp_edge"] if edge else cfg["mlp_mid"]
        if mlp_pct < 100:
            prune_mlp_layer(layer, mlp_pct)

        # Layer skipping (every other middle layer)
        if cfg["skip_mid"] and not edge and i % 2 == 1:
            def make_skip_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (input[0],) + output[1:]
                    return input[0]
                return hook
            h = layer.register_forward_hook(make_skip_hook(i))
            skip_hooks.append(h)

    torch.cuda.empty_cache()

    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    delta = ppl - teacher_ppl

    # Count effective params
    n_skipped = len(skip_hooks)
    n_edge = sum(1 for i in range(L) if is_edge(i, L, EDGE_WIDTH))
    n_mid = L - n_edge

    print(f"  ppl={ppl:.1f} (Δ={delta:+.1f})  [{text[:60]}]")
    print(f"  edges={n_edge} mid={n_mid} skipped={n_skipped}")

    results.append({
        "name": cfg["name"],
        "ppl": ppl, "delta": delta,
        "text": text[:80],
        "n_edge": n_edge, "n_mid": n_mid, "n_skipped": n_skipped,
    })

    for h in skip_hooks:
        h.remove()
    del model; gc.collect(); torch.cuda.empty_cache()


# Summary
print(f"\n{'='*60}")
print("BATHTUB STACK SUMMARY")
print(f"{'='*60}")
print(f"  Teacher: ppl={teacher_ppl:.1f}")
for r in results:
    cost = ("free" if abs(r['delta']) < 0.5 else
            "cheap" if r['delta'] < 2 else
            "moderate" if r['delta'] < 10 else
            "expensive" if r['delta'] < 100 else
            "broken")
    print(f"  {r['name']}")
    print(f"    ppl={r['ppl']:.1f}  Δ={r['delta']:+.1f}  [{cost}]")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/stage115_bathtub_stack.json", "w") as f:
    json.dump({"teacher_ppl": teacher_ppl, "L": L, "edge_width": EDGE_WIDTH,
               "results": results}, f, indent=2)
print(f"\nSaved results/stage115_bathtub_stack.json", flush=True)
