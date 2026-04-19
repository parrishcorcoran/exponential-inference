"""
Stage 37a — Layer-swap perturbation test.

Tests whether transformer layers are locally commutative under the rotation
framing. If each layer is a view-angle on the same manifold, then swapping
adjacent layers should barely change output (near-adjacent angles), while
swapping distant layers should degrade output (large angular gap).

Protocol:
  - Run baseline generation (no swaps) from teacher.
  - For gap j in {1, 2, 4, 8, 16}:
      For every valid i, swap layer[i] with layer[i+j], generate, measure
      token match vs baseline. Restore the swap. Aggregate.
  - Report mean/min/max match-ratio per gap.

Prediction (rotation framing): match_ratio decreases monotonically with gap.
Adjacent swaps preserve most output; distant swaps destroy it.
Falsification (sequential framing): any swap destroys output regardless of gap.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

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


def generate(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokens


def swap_layers(model, i, j):
    layers = model.model.layers
    layers[i], layers[j] = layers[j], layers[i]


def match_ratio(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    m = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return m / n


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gaps", default="1,2,4,8,16")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage37a_layer_swap.json")
    p.add_argument("--max-swaps-per-gap", type=int, default=0,
                   help="0 = all valid i for each gap")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    L = len(model.model.layers)
    print(f"  L = {L} layers")

    print(f"\n=== baseline ===")
    t0 = time.perf_counter()
    baseline_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    baseline_text = tokenizer.decode(baseline_tokens, skip_special_tokens=True)
    print(f"  generated in {time.perf_counter()-t0:.1f}s")
    print(f"  {baseline_text[:150]}")

    gaps = [int(x) for x in args.gaps.split(",")]
    results = []
    per_swap = []

    for gap in gaps:
        valid_i = list(range(0, L - gap))
        if args.max_swaps_per_gap > 0 and len(valid_i) > args.max_swaps_per_gap:
            stride = len(valid_i) // args.max_swaps_per_gap
            valid_i = valid_i[::max(stride, 1)][:args.max_swaps_per_gap]

        print(f"\n=== gap {gap}  ({len(valid_i)} swaps) ===")
        gap_results = []
        t0 = time.perf_counter()
        for i in valid_i:
            j = i + gap
            swap_layers(model, i, j)
            try:
                tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
            finally:
                swap_layers(model, i, j)  # restore
            r = match_ratio(baseline_tokens, tokens)
            fd = first_divergence(baseline_tokens, tokens)
            gap_results.append({"i": i, "j": j, "match_ratio": r, "first_divergence": fd})
            per_swap.append({"gap": gap, "i": i, "j": j, "match_ratio": r, "first_divergence": fd,
                             "sample": tokenizer.decode(tokens, skip_special_tokens=True)[:120]})
            print(f"  swap({i:2d},{j:2d})  match={r:.2f}  first_div={fd}")

        mean_r = sum(x["match_ratio"] for x in gap_results) / len(gap_results)
        min_r = min(x["match_ratio"] for x in gap_results)
        max_r = max(x["match_ratio"] for x in gap_results)
        mean_fd = sum(x["first_divergence"] for x in gap_results) / len(gap_results)
        print(f"  gap={gap}  mean_match={mean_r:.3f}  min={min_r:.3f}  max={max_r:.3f}  "
              f"mean_first_div={mean_fd:.1f}  ({time.perf_counter()-t0:.1f}s)")
        results.append({
            "gap": gap,
            "n_swaps": len(gap_results),
            "mean_match": mean_r,
            "min_match": min_r,
            "max_match": max_r,
            "mean_first_divergence": mean_fd,
        })

    print(f"\n=== summary ===")
    print(f"  {'gap':>4}  {'n':>3}  {'mean':>6}  {'min':>6}  {'max':>6}  {'mean_fd':>8}")
    for r in results:
        print(f"  {r['gap']:>4}  {r['n_swaps']:>3}  {r['mean_match']:.3f}  "
              f"{r['min_match']:.3f}  {r['max_match']:.3f}  {r['mean_first_divergence']:>8.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "baseline_sample": baseline_text[:400],
            "max_new_tokens": args.max_new_tokens,
            "gaps": results,
            "per_swap": per_swap,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
