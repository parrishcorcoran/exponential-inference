"""Placeholder-decode test: let the transformer's own softmax produce future tokens.

For each test sequence:
  1. Take prefix of length t.
  2. Append K placeholder tokens at positions t+1..t+K. Try several placeholder
     types: <pad>, <eos>, the previous token (replicated), and the prefix's
     last token shifted forward.
  3. Run ONE forward pass through the full model.
  4. At each placeholder position p, read argmax(LM_head(h_at_p)) as the
     predicted token at position p+1.
  5. Compare to baseline autoregressive top-1 at the same position.

This is the "softmax over futures" mechanism — let the model's native
attention coordinate all future positions in one forward, no training, no
guessing-then-verifying.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
N_SEQS = 10
ANCHORS = [40, 80, 120, 160, 200]
N_OFFSETS = 10
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_placeholder_decode.json")


def load_owt(tokenizer, max_tokens, skip_tokens=0):
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


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

print("Loading val tokens...")
val_tokens = load_owt(tokenizer, max_tokens=SEQ_LEN * (N_SEQS + 5),
                     skip_tokens=SEQ_LEN * 4000)

# Pick placeholder candidates
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
eos_id = tokenizer.eos_token_id
unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id

placeholders = {
    "pad": pad_id,
    "eos": eos_id,
    "unk": unk_id,
    "self": "self",   # replicate the last prefix token at all future positions
    "shift": "shift", # use baseline_top1 at t for placeholder at t+1, etc. (the model's own argmax fed-forward)
}

results = {name: {k: {"match": 0, "total": 0} for k in range(1, N_OFFSETS + 1)}
           for name in placeholders}

print(f"\nN={N_SEQS} sequences, {len(ANCHORS)} anchors per sequence, "
      f"{N_OFFSETS} offsets, {len(placeholders)} placeholder strategies")

for seq_idx in range(N_SEQS):
    start = seq_idx * SEQ_LEN
    window = val_tokens[start:start + SEQ_LEN + 1]
    if len(window) < SEQ_LEN + 1: break
    inp_full = torch.tensor([window[:-1]], dtype=torch.long, device=device)

    # Baseline: model's natural top-1 at every position
    with torch.no_grad():
        out_baseline = model(inp_full, use_cache=False)
        baseline_top1 = out_baseline.logits.argmax(-1)  # [1, S]

    for t in ANCHORS:
        if t + N_OFFSETS >= SEQ_LEN: continue

        # The "true" tokens we want to predict are the model's natural top-1
        # at positions (t+1-1, t+2-1, ..., t+N-1) — i.e. baseline_top1[t..t+N-1].
        # Equivalently, what the autoregressive model would emit at positions t+1..t+N.

        # Build prefix [0..t] and append N_OFFSETS placeholders
        for name, ph in placeholders.items():
            if ph == "self":
                ph_token = inp_full[0, t].item()
                ph_seq = torch.full((1, N_OFFSETS), ph_token, dtype=torch.long, device=device)
            elif ph == "shift":
                # The model's own predictions at positions t..t+N-1 fed as inputs at t+1..t+N
                ph_seq = baseline_top1[:, t:t + N_OFFSETS].clone()
            else:
                ph_seq = torch.full((1, N_OFFSETS), ph, dtype=torch.long, device=device)

            # Concatenate prefix [0..t] with placeholders
            prefix = inp_full[:, :t + 1]
            inp_test = torch.cat([prefix, ph_seq], dim=1)  # length t+1+N

            with torch.no_grad():
                out_test = model(inp_test, use_cache=False)
                test_top1 = out_test.logits.argmax(-1)  # [1, t+1+N]

            # Predicted token at position t+k (k=1..N) is test_top1[0, t+k-1]
            # Compare to baseline_top1[0, t+k-1] (model's autoregressive top-1 of
            # the token at position t+k, given the natural prefix).
            for k in range(1, N_OFFSETS + 1):
                pos = t + k
                if pos >= SEQ_LEN: continue
                pred = test_top1[0, pos - 1].item()
                target = baseline_top1[0, pos - 1].item()
                if pred == target:
                    results[name][k]["match"] += 1
                results[name][k]["total"] += 1

    print(f"  seq {seq_idx + 1}/{N_SEQS} done", flush=True)

# ─── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("PLACEHOLDER-DECODE — let the transformer's softmax produce futures")
print(f"{'='*70}")
print(f"  N seqs: {N_SEQS}, anchors per seq: {len(ANCHORS)}")
print(f"\n  {'placeholder':<10}", end="")
for k in range(1, N_OFFSETS + 1):
    print(f"t+{k:<3}", end="")
print()

summary = {}
for name in placeholders:
    rates = []
    print(f"  {name:<10}", end="")
    for k in range(1, N_OFFSETS + 1):
        n = results[name][k]["total"]
        r = results[name][k]["match"] / n if n else 0
        rates.append(r)
        print(f"{r:<5.2f}", end="")
    print()
    # Chained
    chained = 1.0; prod = 1.0
    for r in rates:
        prod *= r; chained += prod
    summary[name] = {"per_offset": rates, "chained": chained}
    print(f"  {'  → chain':<10}{chained:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_seqs": N_SEQS,
               "anchors_per_seq": len(ANCHORS),
               "summary": summary}, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
