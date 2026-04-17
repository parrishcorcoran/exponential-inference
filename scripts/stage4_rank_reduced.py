"""
Stage 4b — Generation with dynamic rank reduction.

Applies SVD-based rank projection during generation. For each token,
measures KV entropy and uses it to select the projection rank for
the next forward pass. Lower entropy = more relaxed = lower rank.

Compares base (full rank) vs dynamic (rank-reduced) generation,
measuring per-token latency for both.
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
from src.measurement.cache_hidden_states import load_layer, load_meta


def compute_svd_bases(cache_dir, target_layers, max_rank=256):
    """Load cached hidden states and compute SVD bases per layer."""
    bases = {}
    means = {}
    for li in target_layers:
        h = load_layer(cache_dir, li).to(torch.float32)
        mean = h.mean(dim=0)
        h_centered = h - mean
        _, _, vh = torch.linalg.svd(h_centered, full_matrices=False)
        bases[li] = vh[:max_rank].T.contiguous()  # (D, max_rank)
        means[li] = mean
    return bases, means


def project_hidden_state(h, basis, mean, rank):
    """Project hidden state onto top-rank SVD components.
    h' = mean + P_r P_r^T (h - mean)
    """
    orig_dtype = h.dtype
    h_f = h.float()
    mean_f = mean.float()
    centered = h_f - mean_f
    P_r = basis[:, :rank]  # (D, rank)
    projected = centered @ P_r @ P_r.T  # (*, D)
    return (mean_f + projected).to(orig_dtype)


def measure_kv_entropy(past_key_values, layer_idx=0):
    """Entropy of last token's attention over KV cache."""
    if past_key_values is None:
        return float('inf')
    keys = past_key_values[layer_idx][0]
    B, n_heads, T, head_dim = keys.shape
    if T < 2:
        return float('inf')
    last_key = keys[:, :, -1:, :]
    scores = (last_key @ keys.transpose(-2, -1)) / math.sqrt(head_dim)
    probs = F.softmax(scores.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1).mean().item()
    return entropy


def entropy_to_rank(entropy, max_rank=256, min_rank=8):
    """Map KV entropy to projection rank.

    Higher entropy = more frustrated = need more rank.
    Lower entropy = more relaxed = can use less rank.

    Uses a sigmoid-like mapping calibrated so:
    - entropy ~0.0 → min_rank
    - entropy ~1.0 → max_rank
    """
    # Sigmoid mapping: rank = min + (max - min) * sigmoid(scale * (entropy - midpoint))
    midpoint = 0.3
    scale = 8.0
    frac = 1.0 / (1.0 + math.exp(-scale * (entropy - midpoint)))
    rank = int(min_rank + (max_rank - min_rank) * frac)
    return max(min_rank, min(max_rank, rank))


def generate_with_rank_reduction(model, tokenizer, prompt, max_new_tokens,
                                  bases, means, target_layers,
                                  max_rank=256, min_rank=8,
                                  measure_every=5):
    """Generate with dynamic rank projection applied to hidden states."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Decoder layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    else:
        raise ValueError("Cannot find decoder layers")

    results = []
    past_key_values = None
    hooks = []
    current_rank = max_rank  # start full

    # Pre-compute bases on device
    bases_dev = {li: bases[li].to(device) for li in bases}
    means_dev = {li: means[li].to(device) for li in means}

    def make_pre_hook(layer_idx):
        def hook(module, args):
            nonlocal current_rank
            if layer_idx not in bases_dev:
                return None
            h = args[0]
            h_proj = project_hidden_state(h, bases_dev[layer_idx],
                                           means_dev[layer_idx],
                                           current_rank)
            return (h_proj,) + args[1:]
        return hook

    # Install hooks on target layers
    for li in target_layers:
        if li < len(decoder_layers):
            handle = decoder_layers[li].register_forward_pre_hook(make_pre_hook(li))
            hooks.append(handle)

    try:
        # Prefill at full rank
        current_rank = max_rank
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True)
        t_prefill = time.perf_counter() - t0
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        kv_ent = measure_kv_entropy(past_key_values)
        results.append({
            "position": input_ids.shape[1],
            "time": t_prefill,
            "rank": current_rank,
            "kv_entropy": kv_ent,
            "is_prefill": True,
        })

        generated = [next_token.item()]

        for i in range(max_new_tokens - 1):
            # Update rank based on KV entropy
            if i % measure_every == 0:
                kv_ent = measure_kv_entropy(past_key_values)
                current_rank = entropy_to_rank(kv_ent, max_rank, min_rank)

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

            entry = {
                "position": input_ids.shape[1] + i + 1,
                "time": dt,
                "rank": current_rank,
                "is_prefill": False,
            }
            if i % measure_every == 0:
                entry["kv_entropy"] = kv_ent

            results.append(entry)
            generated.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break
    finally:
        for h in hooks:
            h.remove()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return results, text


def generate_baseline(model, tokenizer, prompt, max_new_tokens):
    """Standard generation without rank reduction."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    results = []
    past_key_values = None

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    t_prefill = time.perf_counter() - t0
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    results.append({"position": input_ids.shape[1], "time": t_prefill, "is_prefill": True})
    generated = [next_token.item()]

    for i in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        dt = time.perf_counter() - t0
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        results.append({"position": input_ids.shape[1] + i + 1, "time": dt, "is_prefill": False})
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return results, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-rank", type=int, default=256)
    p.add_argument("--min-rank", type=int, default=8)
    p.add_argument("--target-layers", type=int, nargs="+",
                   default=[15, 20, 25, 29],
                   help="Decoder layers to apply rank reduction on")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "stage1_cache"))
    p.add_argument("--prompts", default=str(REPO_ROOT / "data" / "prompts.json"))
    p.add_argument("--max-prompts", type=int, default=2)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    print("\n=== loading model ===", flush=True)
    loaded = load_bitnet(model_id=args.model_id, device=args.device)
    model = loaded.model
    tokenizer = loaded.tokenizer

    print("\n=== computing SVD bases ===", flush=True)
    bases, means = compute_svd_bases(args.cache_dir, args.target_layers, args.max_rank)
    print(f"  target layers: {args.target_layers}")
    print(f"  max_rank={args.max_rank}  min_rank={args.min_rank}")

    with open(args.prompts) as f:
        prompts_data = json.load(f)
    prompts = prompts_data.get("prompts", prompts_data) if isinstance(prompts_data, dict) else prompts_data
    prompts = prompts[:args.max_prompts]

    all_results = []

    for pi, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["text"]
        prompt_id = prompt_data.get("id", f"prompt_{pi}")
        print(f"\n=== {prompt_id} ===", flush=True)

        # Baseline
        print("  baseline...", flush=True)
        base_results, base_text = generate_baseline(
            model, tokenizer, prompt_text, args.max_new_tokens)

        # Dynamic rank
        print("  dynamic rank...", flush=True)
        dyn_results, dyn_text = generate_with_rank_reduction(
            model, tokenizer, prompt_text, args.max_new_tokens,
            bases, means, args.target_layers,
            args.max_rank, args.min_rank)

        # Compare
        base_times = [r["time"] for r in base_results if not r.get("is_prefill")]
        dyn_times = [r["time"] for r in dyn_results if not r.get("is_prefill")]
        dyn_ranks = [r["rank"] for r in dyn_results if not r.get("is_prefill")]

        n = min(len(base_times), len(dyn_times))
        avg_base = sum(base_times[:n]) / n * 1000
        avg_dyn = sum(dyn_times[:n]) / n * 1000
        avg_rank = sum(dyn_ranks[:n]) / n

        # Check text match
        match = base_text == dyn_text
        print(f"  base avg: {avg_base:.1f}ms/tok")
        print(f"  dyn  avg: {avg_dyn:.1f}ms/tok  (avg rank: {avg_rank:.0f})")
        print(f"  speedup:  {avg_base/avg_dyn:.2f}x")
        print(f"  text match: {match}")
        if not match:
            # Find first divergence
            base_toks = tokenizer.encode(base_text)
            dyn_toks = tokenizer.encode(dyn_text)
            for j, (bt, dt) in enumerate(zip(base_toks, dyn_toks)):
                if bt != dt:
                    print(f"  first divergence at token {j}")
                    break

        all_results.append({
            "prompt_id": prompt_id,
            "base": base_results,
            "dynamic": dyn_results,
            "base_text": base_text[:300],
            "dyn_text": dyn_text[:300],
            "text_match": match,
        })

    out_path = Path(args.out_dir) / "stage4_rank_reduced.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
