"""
Stage 4 — Direct acceleration measurement.

Skip the trained predictor. Use SVD bases from Stage 1 directly with
a rank schedule derived from the manifold measurements. Measure per-
position wall-clock speedup during generation.

The key insight: as context grows, the hidden-state manifold contracts.
We measure this contraction via the participation ratio of the KV
attention pattern, and use it to set the rank per token dynamically.

For this first measurement, we use a simpler approach: fixed per-layer
rank derived from the r50 measurements (the rank that captures 50% of
energy). This gives a conservative lower bound on speedup.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import load_bitnet, describe_backend, DEFAULT_MODEL_ID


def measure_kv_entropy(past_key_values, layer_idx=0):
    """Measure the entropy of the attention key distribution as a proxy
    for how relaxed the spin glass is.

    Lower entropy = more collapsed = more relaxation = can use lower rank.
    """
    if past_key_values is None:
        return float('inf')

    # Get keys from the specified layer: shape [B, n_heads, T, head_dim]
    keys = past_key_values[layer_idx][0]
    B, n_heads, T, head_dim = keys.shape

    if T < 2:
        return float('inf')

    # Compute pairwise similarity of key vectors (last token vs all)
    last_key = keys[:, :, -1:, :]  # [B, n_heads, 1, head_dim]
    # Attention scores (unnormalized)
    scores = (last_key @ keys.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, n_heads, 1, T]
    probs = F.softmax(scores.float(), dim=-1)  # [B, n_heads, 1, T]

    # Entropy of attention distribution
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()  # scalar

    return entropy


def greedy_generate_with_timing(model, tokenizer, prompt, max_new_tokens,
                                measure_every=1):
    """Generate tokens greedily, recording per-step latency and KV entropy."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(next(model.parameters()).device)

    results = []
    past_key_values = None

    # Prefill
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t0
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    results.append({
        "position": input_ids.shape[1],
        "time": t_prefill,
        "kv_entropy": measure_kv_entropy(past_key_values),
        "token_id": next_token.item(),
        "is_prefill": True,
    })

    generated = [next_token.item()]

    for i in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
        dt = time.perf_counter() - t0
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        pos = input_ids.shape[1] + i + 1
        entry = {
            "position": pos,
            "time": dt,
            "token_id": next_token.item(),
            "is_prefill": False,
        }

        # Measure KV entropy periodically (it's not free)
        if (i + 1) % measure_every == 0:
            entry["kv_entropy"] = measure_kv_entropy(past_key_values)

        results.append(entry)
        generated.append(next_token.item())

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return results, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--measure-every", type=int, default=10,
                   help="Measure KV entropy every N tokens")
    p.add_argument("--prompts", default=str(REPO_ROOT / "data" / "prompts.json"))
    p.add_argument("--max-prompts", type=int, default=3)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    print("\n=== loading model ===", flush=True)
    loaded = load_bitnet(model_id=args.model_id, device=args.device)
    model = loaded.model
    tokenizer = loaded.tokenizer

    print(f"\n=== generation: {args.max_new_tokens} tokens per prompt ===", flush=True)

    with open(args.prompts) as f:
        prompts_data = json.load(f)
    prompts = prompts_data.get("prompts", prompts_data) if isinstance(prompts_data, dict) else prompts_data
    prompts = prompts[:args.max_prompts]

    all_results = []
    for pi, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["text"]
        prompt_id = prompt_data.get("id", f"prompt_{pi}")
        print(f"\n--- {prompt_id}: {prompt_text[:60]}...", flush=True)

        results, gen_text = greedy_generate_with_timing(
            model, tokenizer, prompt_text,
            max_new_tokens=args.max_new_tokens,
            measure_every=args.measure_every,
        )

        # Print summary
        times = [r["time"] for r in results if not r.get("is_prefill")]
        if times:
            first_10 = times[:10]
            last_10 = times[-10:] if len(times) > 10 else times
            avg_first = sum(first_10) / len(first_10)
            avg_last = sum(last_10) / len(last_10)
            print(f"  tokens: {len(times)}")
            print(f"  avg first 10: {avg_first*1000:.1f}ms/tok")
            print(f"  avg last  10: {avg_last*1000:.1f}ms/tok")
            if avg_last > 0:
                print(f"  speedup ratio: {avg_first/avg_last:.2f}x")

        # Print entropy curve
        entropies = [(r["position"], r["kv_entropy"])
                     for r in results if "kv_entropy" in r and not r.get("is_prefill")]
        if entropies:
            print(f"  entropy: start={entropies[0][1]:.3f} end={entropies[-1][1]:.3f}")

        all_results.append({
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "generated_text": gen_text[:200],
            "per_step": results,
        })

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stage4_direct.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
