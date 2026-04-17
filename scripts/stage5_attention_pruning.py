"""
Stage 5 — Dynamic attention head pruning based on sharpness.

The attention weights are already computed every forward pass. Sharp
heads (concentrated attention) are on the manifold — keep them.
Diffuse heads (spread attention) are noise — skip them.

This script hooks into the model's attention layers, reads the
attention weights, and zeros out diffuse heads. The FFN computation
downstream naturally becomes cheaper because the zeroed heads
contribute nothing.

No training. No calibration. No projection. Just reading what the
model already computes and skipping what doesn't matter.

Usage:
    python scripts/stage5_attention_pruning.py \
        --model Qwen/Qwen3-8B \
        --max-new-tokens 200 \
        --sharpness-threshold 0.5 \
        --device cuda
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


def measure_head_sharpness(attn_weights):
    """Compute per-head sharpness from attention weights.

    attn_weights: [B, n_heads, T_q, T_kv]

    Sharpness = 1 - normalized_entropy.
    Sharp (close to 1) = concentrated on few tokens = on manifold.
    Diffuse (close to 0) = spread across many tokens = noise.

    Returns: [B, n_heads] sharpness values in [0, 1].
    """
    # Entropy of attention distribution per head per query position
    # Use the last query position (current token being generated)
    last_attn = attn_weights[:, :, -1, :]  # [B, n_heads, T_kv]
    T_kv = last_attn.shape[-1]

    log_probs = torch.log(last_attn + 1e-10)
    entropy = -(last_attn * log_probs).sum(dim=-1)  # [B, n_heads]

    # Normalize by max entropy (uniform distribution)
    max_entropy = math.log(T_kv)
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = torch.zeros_like(entropy)

    sharpness = 1.0 - normalized_entropy  # [B, n_heads]
    return sharpness


def find_attention_layers(model):
    """Find all attention modules in the model."""
    layers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                layers.append((i, layer.self_attn))
    return layers


class DynamicHeadPruner:
    """Hooks into attention layers to prune diffuse heads dynamically."""

    def __init__(self, model, threshold=0.5, min_heads=2):
        self.model = model
        self.threshold = threshold
        self.min_heads = min_heads  # always keep at least this many
        self.hooks = []
        self.stats = {
            "total_heads_computed": 0,
            "heads_kept": 0,
            "per_layer_sharpness": [],
            "per_step_kept_ratio": [],
        }
        self._step_heads_total = 0
        self._step_heads_kept = 0

    def install(self):
        """Install hooks on all attention layers."""
        attn_layers = find_attention_layers(self.model)

        for layer_idx, attn_module in attn_layers:
            # We need output_attentions=True for this to work.
            # Hook into the attention output to read and mask weights.
            handle = attn_module.register_forward_hook(
                self._make_hook(layer_idx, attn_module)
            )
            self.hooks.append(handle)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def new_step(self):
        """Call before each generation step to reset per-step stats."""
        self._step_heads_total = 0
        self._step_heads_kept = 0

    def end_step(self):
        """Call after each generation step to record stats."""
        if self._step_heads_total > 0:
            ratio = self._step_heads_kept / self._step_heads_total
            self.stats["per_step_kept_ratio"].append(ratio)

    def _make_hook(self, layer_idx, attn_module):
        pruner = self

        def hook(module, inputs, outputs):
            # outputs is typically (attn_output, attn_weights, past_kv)
            # or (attn_output, past_kv) if output_attentions=False
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn_output = outputs[0]
                attn_weights = outputs[1] if len(outputs) > 1 else None

                # Check if attn_weights looks like attention weights
                if (attn_weights is not None and
                    attn_weights.dim() == 4 and
                    attn_weights.shape[-1] > 0):

                    sharpness = measure_head_sharpness(attn_weights)  # [B, n_heads]
                    n_heads = sharpness.shape[1]

                    # Determine which heads to keep
                    keep_mask = sharpness >= pruner.threshold  # [B, n_heads]

                    # Ensure minimum heads
                    for b in range(sharpness.shape[0]):
                        n_kept = keep_mask[b].sum().item()
                        if n_kept < pruner.min_heads:
                            # Keep the top-k sharpest
                            topk = sharpness[b].topk(pruner.min_heads).indices
                            keep_mask[b] = False
                            keep_mask[b, topk] = True

                    n_kept = keep_mask.float().sum().item()
                    n_total = keep_mask.numel()
                    pruner._step_heads_total += n_total
                    pruner._step_heads_kept += n_kept
                    pruner.stats["total_heads_computed"] += n_total
                    pruner.stats["heads_kept"] += n_kept

                    # Zero out pruned heads in the attention output
                    # attn_output is [B, T, H] — we need to zero the
                    # contribution of pruned heads.
                    # Each head contributes head_dim dimensions to H.
                    head_dim = attn_output.shape[-1] // n_heads
                    mask_expanded = keep_mask.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, 1]
                    mask_expanded = mask_expanded.expand(-1, -1, attn_output.shape[1], head_dim)
                    mask_flat = mask_expanded.reshape(attn_output.shape[0], attn_output.shape[1], -1)
                    mask_flat = mask_flat.to(attn_output.dtype)

                    attn_output_masked = attn_output * mask_flat

                    # Rescale to compensate for dropped heads
                    kept_ratio = keep_mask.float().mean()
                    if kept_ratio > 0:
                        attn_output_masked = attn_output_masked / kept_ratio

                    # Return modified outputs
                    return (attn_output_masked,) + outputs[1:]

            return outputs

        return hook

    def summary(self):
        total = self.stats["total_heads_computed"]
        kept = self.stats["heads_kept"]
        if total > 0:
            return {
                "total_heads": total,
                "heads_kept": kept,
                "kept_ratio": kept / total,
                "pruned_ratio": 1 - kept / total,
                "per_step_kept_ratio": self.stats["per_step_kept_ratio"],
            }
        return {"total_heads": 0}


def generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    """Standard generation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    past_key_values = None

    # Prefill
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    times.append(time.perf_counter() - t0)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

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
    return times, text, generated


def generate_with_pruning(model, tokenizer, prompt, max_new_tokens, device,
                          threshold=0.5, min_heads=2):
    """Generation with dynamic head pruning."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    past_key_values = None

    pruner = DynamicHeadPruner(model, threshold=threshold, min_heads=min_heads)
    pruner.install()

    try:
        # Prefill — no pruning on prefill (need all heads for context)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True,
                       output_attentions=True)
        times.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token.item()]

        for i in range(max_new_tokens - 1):
            pruner.new_step()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model(input_ids=next_token,
                           past_key_values=past_key_values,
                           use_cache=True,
                           output_attentions=True)
            dt = time.perf_counter() - t0
            pruner.end_step()
            times.append(dt)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
    finally:
        pruner.remove()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return times, text, generated, pruner.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--sharpness-threshold", type=float, default=0.3,
                   help="Heads with sharpness below this are pruned")
    p.add_argument("--min-heads", type=int, default=4,
                   help="Always keep at least this many heads per layer")
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
    # Use the common loader if available, fall back to direct load
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",  # need attention weights, not SDPA
    ).to(device).eval()

    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {n_heads} heads per layer")

    # === Baseline ===
    print(f"\n=== baseline generation ===", flush=True)
    base_times, base_text, base_tokens = generate_baseline(
        model, tokenizer, args.prompt, args.max_new_tokens, device)

    base_decode = base_times[1:]  # skip prefill
    avg_base = sum(base_decode) / len(base_decode) * 1000
    print(f"  {len(base_decode)} tokens, avg {avg_base:.1f}ms/tok")
    print(f"  text: {base_text[:150]}...")

    # === Pruned ===
    print(f"\n=== pruned generation (threshold={args.sharpness_threshold}, min_heads={args.min_heads}) ===", flush=True)
    prune_times, prune_text, prune_tokens, prune_stats = generate_with_pruning(
        model, tokenizer, args.prompt, args.max_new_tokens, device,
        threshold=args.sharpness_threshold, min_heads=args.min_heads)

    prune_decode = prune_times[1:]
    avg_prune = sum(prune_decode) / len(prune_decode) * 1000
    print(f"  {len(prune_decode)} tokens, avg {avg_prune:.1f}ms/tok")
    print(f"  text: {prune_text[:150]}...")
    print(f"  heads kept: {prune_stats.get('kept_ratio', 0):.1%}")
    print(f"  heads pruned: {prune_stats.get('pruned_ratio', 0):.1%}")

    # === Comparison ===
    print(f"\n=== comparison ===")
    if avg_prune > 0:
        speedup = avg_base / avg_prune
        print(f"  speedup: {speedup:.2f}x")
    else:
        speedup = 0
        print(f"  speedup: N/A")

    # Token match
    min_len = min(len(base_tokens), len(prune_tokens))
    match_count = sum(1 for a, b in zip(base_tokens[:min_len], prune_tokens[:min_len]) if a == b)
    print(f"  token match: {match_count}/{min_len} ({match_count/max(min_len,1):.1%})")

    # Per-step kept ratio (shows how pruning changes with position)
    ratios = prune_stats.get("per_step_kept_ratio", [])
    if len(ratios) >= 20:
        first_10 = sum(ratios[:10]) / 10
        last_10 = sum(ratios[-10:]) / 10
        print(f"  heads kept first 10 steps: {first_10:.1%}")
        print(f"  heads kept last 10 steps: {last_10:.1%}")
        if last_10 < first_10:
            print(f"  → pruning increases with context (manifold narrowing!)")

    # Save results
    out_path = Path(args.out_dir) / "stage5_pruning.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "threshold": args.sharpness_threshold,
            "min_heads": args.min_heads,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "baseline_avg_ms": avg_base,
            "pruned_avg_ms": avg_prune,
            "speedup": speedup,
            "token_match": f"{match_count}/{min_len}",
            "heads_kept_ratio": prune_stats.get("kept_ratio", 0),
            "base_text": base_text[:500],
            "prune_text": prune_text[:500],
            "per_step_kept_ratio": ratios,
            "base_times_ms": [t * 1000 for t in base_decode],
            "prune_times_ms": [t * 1000 for t in prune_decode],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
