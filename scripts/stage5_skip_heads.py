"""
Stage 5b — Skip heads before computation using previous step's sharpness.

Instead of computing all heads and zeroing diffuse ones (no savings),
this skips the Q/K/V projection for pruned heads entirely. Uses the
previous step's attention pattern to predict which heads to keep.

The attention pattern changes slowly between adjacent tokens — the
manifold position doesn't jump discontinuously. So step t's sharpness
is a good predictor of step t+1's useful heads.

Usage:
    python scripts/stage5_skip_heads.py \
        --model Qwen/Qwen3-0.6B \
        --max-new-tokens 200 \
        --threshold 0.3 \
        --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


def head_sharpness_from_attn(attn_weights):
    """[B, n_heads, T_q, T_kv] -> [n_heads] sharpness for last query."""
    last_attn = attn_weights[0, :, -1, :]  # [n_heads, T_kv]
    T = last_attn.shape[-1]
    if T <= 1:
        return torch.ones(last_attn.shape[0])
    entropy = -(last_attn * torch.log(last_attn + 1e-10)).sum(dim=-1)
    max_ent = math.log(T)
    return 1.0 - (entropy / max_ent) if max_ent > 0 else torch.ones_like(entropy)


class SkipHeadWrapper:
    """Wraps a model to skip attention heads based on previous step's sharpness.

    On each forward pass:
    1. Run with output_attentions=True
    2. Read sharpness from attention weights
    3. Build head_mask for NEXT step based on current sharpness
    4. Apply head_mask on next step to skip diffuse heads
    """

    def __init__(self, model, threshold=0.3, min_heads=4, recalibrate_every=20):
        self.model = model
        self.threshold = threshold
        self.min_heads = min_heads
        self.recalibrate_every = recalibrate_every
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

        # head_mask: [n_layers, n_heads] — 1 = keep, 0 = skip
        # Start with all heads active
        self.head_mask = torch.ones(self.n_layers, self.n_heads)
        self.step = 0
        self.stats = []

    def _update_mask(self, attentions):
        """Update head_mask from current step's attention weights."""
        n_kept_total = 0
        n_total = 0
        for layer_idx, attn_w in enumerate(attentions):
            if attn_w is None:
                continue
            sharpness = head_sharpness_from_attn(attn_w)  # [n_heads]

            # Keep heads above threshold
            keep = sharpness >= self.threshold

            # Ensure minimum heads
            if keep.sum() < self.min_heads:
                topk = sharpness.topk(min(self.min_heads, len(sharpness))).indices
                keep = torch.zeros_like(keep, dtype=torch.bool)
                keep[topk] = True

            self.head_mask[layer_idx] = keep.float()
            n_kept_total += keep.sum().item()
            n_total += len(keep)

        kept_ratio = n_kept_total / max(n_total, 1)
        self.stats.append({
            "step": self.step,
            "kept_ratio": kept_ratio,
            "n_kept": n_kept_total,
            "n_total": n_total,
        })

    def build_hf_head_mask(self, device):
        """Convert our mask to HuggingFace head_mask format.

        HF expects: [n_layers, n_heads] or [n_layers, 1, 1, n_heads, 1]
        depending on the model. We'll try the simple format first.
        """
        return self.head_mask.to(device)

    def generate_step(self, input_ids, past_key_values=None):
        """One generation step with head skipping."""
        device = input_ids.device
        self.step += 1

        # Every N steps, recalibrate (run all heads)
        use_mask = (self.step > 1) and (self.step % self.recalibrate_every != 0)

        if use_mask:
            head_mask = self.build_hf_head_mask(device)
        else:
            head_mask = None  # all heads

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                head_mask=head_mask,
            )

        # Update mask for next step
        if out.attentions is not None:
            self._update_mask(out.attentions)

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token, out.past_key_values

    def summary(self):
        if not self.stats:
            return {}
        ratios = [s["kept_ratio"] for s in self.stats]
        return {
            "avg_kept_ratio": sum(ratios) / len(ratios),
            "first_10_kept": sum(ratios[:10]) / min(10, len(ratios)),
            "last_10_kept": sum(ratios[-10:]) / min(10, len(ratios)) if len(ratios) >= 10 else sum(ratios) / len(ratios),
            "per_step_kept_ratio": ratios,
        }


def generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    """Standard generation with timing."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    generated = []
    past_key_values = None

    # Prefill
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    times.append(time.perf_counter() - t0)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())

    # Decode
    for i in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        times.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return times[1:], text, generated  # skip prefill time


def generate_with_skip(model, tokenizer, prompt, max_new_tokens, device,
                       threshold=0.3, min_heads=4, recalibrate_every=20):
    """Generation with dynamic head skipping."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    generated = []

    wrapper = SkipHeadWrapper(model, threshold=threshold,
                              min_heads=min_heads,
                              recalibrate_every=recalibrate_every)

    # Prefill (all heads)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, output_attentions=True)
    times.append(time.perf_counter() - t0)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())

    # Initialize mask from prefill attention
    if out.attentions is not None:
        wrapper._update_mask(out.attentions)

    # Decode with head skipping
    for i in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        next_token, past_key_values = wrapper.generate_step(
            next_token, past_key_values)
        times.append(time.perf_counter() - t0)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return times[1:], text, generated, wrapper.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--min-heads", type=int, default=4)
    p.add_argument("--recalibrate-every", type=int, default=20,
                   help="Run all heads every N steps to recalibrate")
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\ndevice: {device}")

    print(f"\n=== loading {args.model} ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()

    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {n_heads} heads/layer")
    print(f"  threshold={args.threshold}, min_heads={args.min_heads}")

    # === Baseline ===
    print(f"\n=== baseline ===", flush=True)
    base_times, base_text, base_tokens = generate_baseline(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    avg_base = sum(base_times) / len(base_times) * 1000
    print(f"  {len(base_times)} tokens, {avg_base:.1f}ms/tok")
    print(f"  {base_text[:120]}...")

    # === Skip heads ===
    print(f"\n=== skip heads ===", flush=True)
    skip_times, skip_text, skip_tokens, skip_stats = generate_with_skip(
        model, tokenizer, args.prompt, args.max_new_tokens, device,
        threshold=args.threshold, min_heads=args.min_heads,
        recalibrate_every=args.recalibrate_every)
    avg_skip = sum(skip_times) / len(skip_times) * 1000
    print(f"  {len(skip_times)} tokens, {avg_skip:.1f}ms/tok")
    print(f"  {skip_text[:120]}...")
    print(f"  avg heads kept: {skip_stats.get('avg_kept_ratio', 0):.1%}")

    # === Comparison ===
    print(f"\n=== results ===")
    speedup = avg_base / avg_skip if avg_skip > 0 else 0
    print(f"  baseline: {avg_base:.1f}ms/tok")
    print(f"  skip:     {avg_skip:.1f}ms/tok")
    print(f"  speedup:  {speedup:.2f}x")

    min_len = min(len(base_tokens), len(skip_tokens))
    match = sum(1 for a, b in zip(base_tokens[:min_len], skip_tokens[:min_len]) if a == b)
    print(f"  token match: {match}/{min_len} ({match/max(min_len,1):.1%})")

    first_10 = skip_stats.get("first_10_kept", 0)
    last_10 = skip_stats.get("last_10_kept", 0)
    print(f"  heads kept first 10: {first_10:.1%}")
    print(f"  heads kept last 10:  {last_10:.1%}")
    if last_10 < first_10:
        print(f"  → manifold narrowing confirmed: fewer heads needed later")

    # Save
    out_path = Path(args.out_dir) / "stage5_skip_heads.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "threshold": args.threshold,
            "min_heads": args.min_heads,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "baseline_ms": avg_base,
            "skip_ms": avg_skip,
            "speedup": speedup,
            "token_match": f"{match}/{min_len}",
            "avg_kept_ratio": skip_stats.get("avg_kept_ratio", 0),
            "first_10_kept": first_10,
            "last_10_kept": last_10,
            "base_text": base_text[:500],
            "skip_text": skip_text[:500],
            "per_step_kept_ratio": skip_stats.get("per_step_kept_ratio", []),
            "base_times_ms": [t * 1000 for t in base_times],
            "skip_times_ms": [t * 1000 for t in skip_times],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
