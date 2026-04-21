"""
Stage 56 — How many tokens can the model itself not resolve?

For each held-out (context, true_next_token) pair:
  1. Teacher outputs a distribution over vocab (softmax of logits).
  2. Measure teacher's output entropy on that distribution.
  3. Is the true_next_token the teacher's top-1? (yes/no)

Bucket tokens by teacher entropy and report per-bucket:
  - Fraction of all held-out tokens in this bucket
  - Teacher's accuracy (top-1 == true) in this bucket

High-entropy buckets are "unresolved by the teacher itself" — multiple
tokens carry similar probability. A manifold-trained student might do
better or worse on these specifically; that's the next test layer. For
now, just quantify the unresolvability distribution.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def load_wikitext(tokenizer, max_tokens):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    all_ids = []
    for row in ds:
        text = row["text"].strip()
        if not text or text.startswith("="): continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        all_ids.append(ids)
        if sum(len(x) for x in all_ids) >= max_tokens:
            break
    return torch.cat(all_ids)[:max_tokens]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--context-len", type=int, default=32)
    p.add_argument("--heldout-tokens", type=int, default=5000)
    p.add_argument("--max-pairs", type=int, default=1000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)

    print(f"\n=== loading wikitext-2 held-out ===")
    ids = load_wikitext(tokenizer, args.heldout_tokens)
    print(f"  {len(ids)} tokens")

    # Build (ctx, true_id) pairs
    pairs = []
    for i in range(args.context_len, len(ids)):
        ctx = ids[i - args.context_len:i]
        pairs.append((ctx.to(device), int(ids[i].item())))
        if len(pairs) >= args.max_pairs: break
    print(f"  {len(pairs)} (context, next_token) pairs")

    print(f"\n=== measuring per-token teacher entropy and correctness ===")
    records = []
    with torch.inference_mode():
        for ctx, true_id in pairs:
            out = model(input_ids=ctx.unsqueeze(0), use_cache=False)
            logits = out.logits[0, -1].float()
            probs = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            entropy = float(-(probs * logp).sum().item())           # nats
            top_probs = probs.topk(5)
            top1_id = int(top_probs.indices[0].item())
            top1_p = float(top_probs.values[0].item())
            top2_p = float(top_probs.values[1].item())
            margin = top1_p - top2_p
            correct = (top1_id == true_id)
            true_prob = float(probs[true_id].item())
            records.append({
                "entropy": entropy, "margin": margin, "top1_p": top1_p,
                "correct": correct, "true_prob": true_prob,
            })
    n = len(records)
    print(f"  done ({n} pairs)")

    # Overall
    acc = sum(1 for r in records if r["correct"]) / n
    ppl = math.exp(-sum(math.log(max(r["true_prob"], 1e-12)) for r in records) / n)
    mean_ent = sum(r["entropy"] for r in records) / n
    print(f"\n=== teacher overall ===")
    print(f"  accuracy (top-1 == true): {acc:.3f}")
    print(f"  mean entropy (nats):      {mean_ent:.3f}")
    print(f"  perplexity:               {ppl:.2f}")

    # Bucket by entropy
    print(f"\n=== resolution buckets (by teacher entropy) ===")
    buckets = [
        ("very confident (ent < 0.3)",    lambda r: r["entropy"] < 0.3),
        ("confident (0.3–1.0)",           lambda r: 0.3 <= r["entropy"] < 1.0),
        ("moderate (1.0–2.0)",            lambda r: 1.0 <= r["entropy"] < 2.0),
        ("uncertain (2.0–3.0)",           lambda r: 2.0 <= r["entropy"] < 3.0),
        ("very uncertain (3.0–4.0)",      lambda r: 3.0 <= r["entropy"] < 4.0),
        ("unresolvable-ish (≥ 4.0)",      lambda r: r["entropy"] >= 4.0),
    ]
    print(f"  {'bucket':>34}  {'frac':>6}  {'count':>5}  {'acc':>6}  {'mean_true_p':>11}")
    for name, pred in buckets:
        bucket_records = [r for r in records if pred(r)]
        if not bucket_records:
            print(f"  {name:>34}  {'0.00':>6}  {'0':>5}  {'—':>6}  {'—':>11}")
            continue
        b_acc = sum(1 for r in bucket_records if r["correct"]) / len(bucket_records)
        b_true_p = sum(r["true_prob"] for r in bucket_records) / len(bucket_records)
        frac = len(bucket_records) / n
        print(f"  {name:>34}  {frac:>6.3f}  {len(bucket_records):>5}  "
              f"{b_acc:>6.3f}  {b_true_p:>11.4f}")

    # Interpretation notes
    print(f"\n=== interpretation ===")
    high_ent = sum(1 for r in records if r["entropy"] >= 2.0) / n
    print(f"  fraction with entropy ≥ 2.0 (multiple strong candidates): {high_ent:.3f}")
    very_high = sum(1 for r in records if r["entropy"] >= 4.0) / n
    print(f"  fraction with entropy ≥ 4.0 (effectively unresolved):      {very_high:.3f}")
    in_top1 = acc
    in_top5 = sum(1 for r in records if r["correct"] or r["true_prob"] > 0) / n
    # Actually just report top-5 coverage properly
    print(f"  ")
    print(f"  These are tokens the teacher itself CAN'T PIN DOWN.")
    print(f"  A manifold-trained student doesn't have to do worse on these —")
    print(f"  if the corpus's true next token is deterministic and the teacher")
    print(f"  is uncertain, the student trained against manifold structure might")
    print(f"  land on the true token while the teacher spreads its probability.")


if __name__ == "__main__":
    main()
