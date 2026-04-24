"""
Stage 114 — Is merge depth just token frequency in disguise?

Stage 113 showed per-token merge depth correlates with intuitive token
difficulty (punctuation early, rare words late). Literature (Timkey 2021,
Kovaleva 2021, Puccetti 2022) shows the rogue/outlier dimension tracks
token frequency. We need to check:

  - If merge_depth ∝ log(frequency) with r > 0.9: rediscovery, not novel
  - If r = 0.5-0.7: independent signal with added information
  - If r < 0.5: genuinely new signal, unpublished direction

Also test: correlation between merge depth and hidden-state norm (proxy
for massive-activation magnitude at that token).
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_wikitext_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--merge-json", default="results/stage113_token_bathtub_t05.json")
    p.add_argument("--corpus-tokens", type=int, default=500000,
                   help="Tokens from wikitext-2 train to build frequency table")
    p.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage114_merge_vs_freq.json")
    args = p.parse_args()

    # Load merge-depth data
    merge_data = json.load(open(args.merge_json))
    token_stats = merge_data["token_stats"]
    print(f"loaded {len(token_stats)} tokens from {args.merge_json}")

    # Build frequency table from wikitext train
    print("loading tokenizer + corpus...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    train_tokens = load_wikitext_tokens(tokenizer, args.corpus_tokens, split="train")
    print(f"corpus has {len(train_tokens)} tokens")

    freq_counter = Counter(train_tokens)
    corpus_size = len(train_tokens)
    # log frequency (unigram)
    def token_log_freq(tid):
        c = freq_counter.get(tid, 0)
        return math.log((c + 1) / corpus_size)   # +1 smoothing for OOV

    # Build aligned arrays for correlation
    # Only keep tokens with count >= 3 in stage 113 to reduce noise
    aligned = []
    for s in token_stats:
        if s["count"] < 3:
            continue
        tid = s["token_id"]
        log_f = token_log_freq(tid)
        aligned.append({
            "token_id": tid,
            "token_str": s["token_str"],
            "merge_depth": s["mean_merge"],
            "log_freq": log_f,
            "count_in_s113": s["count"],
            "corpus_count": freq_counter.get(tid, 0),
        })

    print(f"{len(aligned)} tokens with count ≥ 3 for correlation analysis")

    # Correlation: merge_depth vs log_freq
    md = np.array([a["merge_depth"] for a in aligned])
    lf = np.array([a["log_freq"] for a in aligned])
    r_pearson = float(np.corrcoef(md, lf)[0, 1])

    # Spearman via rank correlation (robust to nonlinearity)
    from scipy.stats import spearmanr
    r_spearman = float(spearmanr(md, lf).correlation)

    print(f"\n=== merge_depth vs log(frequency) ===")
    print(f"  Pearson r:  {r_pearson:+.4f}")
    print(f"  Spearman ρ: {r_spearman:+.4f}")

    # Verdict
    if abs(r_spearman) > 0.9:
        verdict = f"REDISCOVERY — merge depth is essentially log-frequency (|ρ|={abs(r_spearman):.2f})"
    elif abs(r_spearman) > 0.7:
        verdict = f"STRONG CORRELATION — merge depth is mostly frequency-driven (|ρ|={abs(r_spearman):.2f}). Some independent signal."
    elif abs(r_spearman) > 0.5:
        verdict = f"MODERATE CORRELATION — some shared information with frequency (|ρ|={abs(r_spearman):.2f}). Independent component exists."
    elif abs(r_spearman) > 0.3:
        verdict = f"WEAK CORRELATION — merge depth mostly independent of frequency (|ρ|={abs(r_spearman):.2f}). Likely NOVEL signal."
    else:
        verdict = f"INDEPENDENT — merge depth not explained by frequency (|ρ|={abs(r_spearman):.2f}). NOVEL signal."

    print(f"\n  verdict: {verdict}")

    # Show some examples of disagreement: high-freq but late-merge, low-freq but early-merge
    aligned.sort(key=lambda x: x["merge_depth"] - x["log_freq"] * 2)  # weight somewhat
    # Compute "residual" from best linear fit merge_depth = a*log_freq + b
    slope, intercept = np.polyfit(lf, md, 1)
    for a in aligned:
        pred = slope * a["log_freq"] + intercept
        a["residual"] = a["merge_depth"] - pred

    aligned.sort(key=lambda x: x["residual"])
    print(f"\n=== tokens that merge EARLIER than frequency predicts ===")
    for a in aligned[:10]:
        print(f"  resid={a['residual']:+6.1f}  merge@{a['merge_depth']:>5.1f}  "
              f"log_f={a['log_freq']:+6.2f}  count_corpus={a['corpus_count']:>5}  "
              f"{a['token_str'][:20]!r}")
    print(f"\n=== tokens that merge LATER than frequency predicts ===")
    for a in aligned[-10:]:
        print(f"  resid={a['residual']:+6.1f}  merge@{a['merge_depth']:>5.1f}  "
              f"log_f={a['log_freq']:+6.2f}  count_corpus={a['corpus_count']:>5}  "
              f"{a['token_str'][:20]!r}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "n_tokens": len(aligned),
                   "pearson_r": r_pearson,
                   "spearman_rho": r_spearman,
                   "linear_fit": {"slope": float(slope), "intercept": float(intercept)},
                   "verdict": verdict,
                   "aligned_tokens": aligned}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
