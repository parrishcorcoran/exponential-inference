"""Tree-Medusa probe: top-K expansion of standard token-Medusa heads.

For each anchor t and each offset k:
  - Get head_k's top-K candidate tokens from h_t (sorted by logit).
  - Check whether the model's natural top-1 at position t+k is in this top-K set.

This is the *upper bound* on what tree-Medusa with branching factor K per offset
could accept: it assumes the verify pass picks the branch that contains the
right token. (A real tree-attention verify is what realizes this — we measure
the ceiling here without implementing the attention mask.)

Sweeps K = 1, 2, 4, 8, 16. Reports per-offset hit rate and chained tokens/step
under the assumption that all offsets are independently in the top-K.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
N_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
N_OFFSETS = 5
K_SWEEP = [1, 2, 4, 8, 16, 32, 64]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_tree.json")


class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])

    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
lm_head_weight = model.lm_head.weight.detach().float()

val_tokens = load_owt(tokenizer, SEQ_LEN * (N_SEQS + 5), skip_tokens=SEQ_LEN * 4000)

print(f"Loading {N_OFFSETS} token-Medusa heads...")
tok_heads = []
for k in range(1, N_OFFSETS + 1):
    h = MedusaHead(d_model, n_layers=1).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h.eval(); tok_heads.append(h)

# ─── Run probe ────────────────────────────────────────────────────────────
# For each (offset, K) measure: P(baseline_top1[t+k-1] in head_k's top-K from h_t)
hits = {(off, K): 0 for off in range(1, N_OFFSETS + 1) for K in K_SWEEP}
total = {off: 0 for off in range(1, N_OFFSETS + 1)}

n_done = 0
for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp, output_hidden_states=True, use_cache=False)
        baseline_top1 = out.logits.argmax(-1)  # [1, seq] — model's natural top-1
        h_final = out.hidden_states[-1].float()

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue
        h_t = h_final[:, t:t+1]  # [1, 1, d]

        for k in range(1, N_OFFSETS + 1):
            with torch.no_grad():
                logits_k = tok_heads[k - 1](h_t, lm_head_weight)[0, 0]  # [vocab]
                topk_max = max(K_SWEEP)
                topk_vals, topk_idx = torch.topk(logits_k, topk_max)

            target_tok = baseline_top1[0, t + k - 1].item()
            for K in K_SWEEP:
                if target_tok in topk_idx[:K].tolist():
                    hits[(k, K)] += 1
            total[k] += 1

    n_done += 1
    print(f"  seq {n_done}/{N_SEQS} done", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("TREE-MEDUSA PROBE — top-K expansion, Qwen3-0.6B")
print(f"{'='*70}")
header = f"  {'offset':<8}" + "".join(f"K={K:<5}" for K in K_SWEEP)
print(header)

per_offset_rates = {}
for off in range(1, N_OFFSETS + 1):
    n = total[off]
    rates = [hits[(off, K)] / n if n else 0 for K in K_SWEEP]
    per_offset_rates[off] = rates
    row = f"  t+{off:<6}" + "".join(f"{r:<7.3f}" for r in rates)
    print(row)

# Chained tokens/step under each K
print(f"\n  Chained tokens/step (1 + a1 + a1*a2 + ... assuming offsets independent):")
chained_per_K = {}
for K_idx, K in enumerate(K_SWEEP):
    rates = [per_offset_rates[off][K_idx] for off in range(1, N_OFFSETS + 1)]
    chained = 1.0; prod = 1.0
    for r in rates:
        prod *= r; chained += prod
    chained_per_K[K] = chained
    print(f"    K={K:>3}: {chained:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "n_seqs": N_SEQS,
        "anchors_per_seq": len(ANCHORS),
        "K_sweep": K_SWEEP,
        "per_offset_rates": {f"t+{off}": dict(zip([str(K) for K in K_SWEEP], rates))
                             for off, rates in per_offset_rates.items()},
        "chained_tokens_per_step": {str(K): round(v, 4) for K, v in chained_per_K.items()},
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
