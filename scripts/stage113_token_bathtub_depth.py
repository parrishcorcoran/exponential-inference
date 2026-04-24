"""
Stage 113 — Per-token bathtub merge depth.

Finding 13 said the residual stream collapses to rank-1 in middle layers.
This stage asks: when does EACH INDIVIDUAL TOKEN merge into that rank-1
transit?

Hypothesis:
  - Common tokens (the, of, a): merge early (L2-5)
  - Content words: merge mid (L8-15)
  - Rare/technical: merge late (L18-25)
  - OOV-ish tokens: never merge cleanly

For each layer l:
  1. Compute dominant singular direction v_l from the [seq, d] matrix
     of hidden states at that layer
  2. For each token at each position, compute cos(h_token_at_l, v_l)
  3. Merge depth per token = first layer where cos > threshold

Output: per-token merge depth, correlated with token frequency, allowing
us to check the hypothesis and see the distribution.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


@torch.no_grad()
def measure_bathtub_merge(model, tokenizer, tokens, device, seq_len,
                          merge_threshold=0.9, n_windows=10):
    """For each token at each position in each window, measure the layer at which
       its hidden state aligns with that layer's dominant direction (cos > threshold)."""
    L = model.config.num_hidden_layers

    # Per-token aggregation
    token_merge_layers = {}  # token_id -> list of observed merge layers
    token_counts = Counter()

    n = min(n_windows, (len(tokens) - 1) // seq_len)
    for w in range(n):
        start = w * seq_len
        window = tokens[start:start + seq_len]
        if len(window) < 2: continue
        ids = torch.tensor([window], dtype=torch.long, device=device)

        out = model(ids, use_cache=False, output_hidden_states=True)
        # hidden_states: tuple of L+1 tensors, each [1, seq, d]
        seq = ids.shape[1]

        # For each layer, compute top singular direction across tokens in this window
        # and then cos per token to that direction
        per_layer_cos = np.zeros((L + 1, seq))   # [L+1, seq]
        for l, h_tuple in enumerate(out.hidden_states):
            H = h_tuple[0].float()  # [seq, d]
            # Dominant singular direction — via SVD
            try:
                U, S, V = torch.linalg.svd(H, full_matrices=False)
                v1 = V[0]  # [d] — top right singular vector = dominant direction
            except Exception:
                # Fallback: use mean direction
                v1 = H.mean(dim=0)
                v1 = v1 / (v1.norm() + 1e-8)
            # cos per token to v1
            H_norm = H / (H.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (H_norm @ v1).abs().cpu().numpy()
            per_layer_cos[l] = cos

        # For each token, find first layer where cos > threshold
        for pos in range(seq):
            token_id = ids[0, pos].item()
            token_counts[token_id] += 1
            # find merge
            merge_l = None
            for l in range(L + 1):
                if per_layer_cos[l, pos] > merge_threshold:
                    merge_l = l
                    break
            if merge_l is None:
                merge_l = L + 1   # never merged
            token_merge_layers.setdefault(token_id, []).append(merge_l)

    # Aggregate: mean merge layer per token
    token_stats = []
    for tid, layers in token_merge_layers.items():
        arr = np.array(layers)
        token_stats.append({
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "count": len(arr),
            "mean_merge": float(arr.mean()),
            "median_merge": float(np.median(arr)),
            "std_merge": float(arr.std()),
            "never_merged_frac": float((arr > L).mean()),
        })
    token_stats.sort(key=lambda x: x["mean_merge"])
    return {
        "L": L,
        "merge_threshold": merge_threshold,
        "n_windows": n,
        "total_positions": sum(len(v) for v in token_merge_layers.values()),
        "unique_tokens": len(token_merge_layers),
        "token_stats": token_stats,
    }


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--merge-threshold", type=float, default=0.9,
                   help="cos threshold to declare token merged into transit")
    p.add_argument("--out", default="results/stage113_token_bathtub.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokens = load_tokens(tokenizer, max_tokens=args.seq_len * args.n_windows + 100, split="validation")
    print(f"loaded {len(tokens)} tokens", flush=True)

    print("loading model...", flush=True)
    model = load_fresh(args.model, device)
    L = model.config.num_hidden_layers
    print(f"L={L}", flush=True)

    print(f"\nmeasuring per-token merge depth...", flush=True)
    t0 = time.time()
    r = measure_bathtub_merge(model, tokenizer, tokens, device,
                              args.seq_len, args.merge_threshold, args.n_windows)
    print(f"  {time.time()-t0:.0f}s  {r['total_positions']} positions, "
          f"{r['unique_tokens']} unique tokens", flush=True)

    # Distribution of mean_merge across token instances
    all_means = np.array([s["mean_merge"] for s in r["token_stats"]])
    all_counts = np.array([s["count"] for s in r["token_stats"]])
    weighted_merges = np.repeat(all_means, all_counts)
    print(f"\n=== merge depth distribution (weighted by occurrences) ===", flush=True)
    print(f"  total instances: {len(weighted_merges)}")
    print(f"  mean merge layer: {weighted_merges.mean():.2f}")
    print(f"  median:           {np.median(weighted_merges):.1f}")
    print(f"  p10 / p50 / p90:  {np.percentile(weighted_merges, [10, 50, 90])}")
    print(f"  early mergers (<L/4, before L{L//4}):     "
          f"{100 * (weighted_merges < L//4).mean():.1f}%")
    print(f"  middle mergers (L/4 - 3L/4):                  "
          f"{100 * ((weighted_merges >= L//4) & (weighted_merges < 3*L//4)).mean():.1f}%")
    print(f"  late mergers (>3L/4):                          "
          f"{100 * (weighted_merges >= 3*L//4).mean():.1f}%")

    # Top-10 earliest and latest mergers by high-count tokens
    high_count = [s for s in r["token_stats"] if s["count"] >= 5]
    high_count.sort(key=lambda x: x["mean_merge"])
    print(f"\n=== earliest-merging tokens (high count, count >= 5) ===")
    for s in high_count[:15]:
        tok_str = repr(s["token_str"][:20])
        print(f"  merge@{s['mean_merge']:>5.1f}  count={s['count']:>4}  {tok_str}")
    print(f"\n=== latest-merging tokens (high count, count >= 5) ===")
    for s in sorted(high_count, key=lambda x: -x["mean_merge"])[:15]:
        tok_str = repr(s["token_str"][:20])
        print(f"  merge@{s['mean_merge']:>5.1f}  count={s['count']:>4}  {tok_str}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, **r}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
