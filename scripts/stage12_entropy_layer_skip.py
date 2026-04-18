"""
Stage 12 — Entropy-driven dynamic layer skip.

Stage 11 showed projection-and-back in the ambient basis can't beat the
baseline on MPS because it adds kernel launches for no FLOP reduction.
The right way to cash in the physics signal is to *not run* layers the
system doesn't need.

Recipe:
    1. Eager attention module's post-hook captures the last query's
       normalized attention entropy per layer (free — attn_weights
       already materialized).
    2. Decoder layer's forward is wrapped: if the entropy at this layer
       on the PREVIOUS step was below a threshold, return the input
       unchanged (layer skipped). Otherwise run normally.

No training. No projection. No basis. Just: "if the system is relaxed
here, don't do work here." The entropy signal drives which layers run
per token.

At a threshold of, say, normalized entropy 0.3, early-stack layers
(high-frustration) run every step while late-stack layers
(low-frustration once context is built) skip often.

Usage:
    python scripts/stage12_entropy_layer_skip.py \\
        --model Qwen/Qwen3-0.6B --threshold 0.3 --device mps
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


class EntropyLayerSkip:
    """Captures per-layer entropy via attention post-hook; decides whether
    to skip each decoder layer on the next step via pre-hook."""

    def __init__(self, n_layers, threshold, skip_from_layer=0,
                 always_run_first=2, always_run_last=2):
        self.n_layers = n_layers
        self.threshold = threshold       # normalized entropy threshold below which to skip
        self.skip_from_layer = skip_from_layer  # don't consider skipping layers before this
        self.always_run_first = always_run_first
        self.always_run_last = always_run_last
        self.entropy_per_layer = {}
        self.skip_count = 0
        self.total_count = 0
        self.per_layer_skip = {i: 0 for i in range(n_layers)}
        self.per_layer_total = {i: 0 for i in range(n_layers)}

    def entropy_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None:
                return
            last = w[0, :, -1, :]            # [H, T_kv]
            T = last.shape[-1]
            if T <= 1:
                self.entropy_per_layer[layer_idx] = 1.0
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
            norm = math.log(T)
            self.entropy_per_layer[layer_idx] = float(ent.mean().item() / norm)
        return hook

    def should_skip(self, layer_idx):
        if layer_idx < self.skip_from_layer:
            return False
        if layer_idx < self.always_run_first:
            return False
        if layer_idx >= self.n_layers - self.always_run_last:
            return False
        ent = self.entropy_per_layer.get(layer_idx, None)
        if ent is None:
            return False
        return ent < self.threshold

    def skip_pre_hook(self, layer_idx):
        def hook(module, args, kwargs):
            self.total_count += 1
            self.per_layer_total[layer_idx] += 1
            if self.should_skip(layer_idx):
                self.skip_count += 1
                self.per_layer_skip[layer_idx] += 1
                # Return the hidden state unchanged wrapped in a tuple
                # (decoder layer is expected to return a tuple of len 1)
                h = args[0] if args else kwargs.get("hidden_states")
                # We can't easily short-circuit the layer via hook return value;
                # instead, set a flag the forward will check.
                module._skip_next = True
            else:
                module._skip_next = False
            return args, kwargs
        return hook


def install_layer_skip(model, skip_controller, device):
    """Replace each decoder layer's forward with a conditional wrapper."""
    handles = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        # Install attention entropy hook (captures entropy per step)
        h_ent = layer.self_attn.register_forward_hook(skip_controller.entropy_hook(i))
        handles.append(h_ent)

        # Install pre-hook that sets _skip_next flag
        h_pre = layer.register_forward_pre_hook(
            skip_controller.skip_pre_hook(i), with_kwargs=True)
        handles.append(h_pre)

        # Monkey-patch forward to early-exit when _skip_next is True
        orig_forward = layer.forward
        def make_new_forward(original_fwd, layer_idx):
            def new_forward(*args, **kwargs):
                if getattr(args[0] if hasattr(args, "__getitem__") else None, "__class__", None):
                    pass
                # args[0] is hidden_states for Qwen3DecoderLayer
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                # Check the skip flag set by the pre-hook
                # We stored it on the layer module, not self — use the closure
                mod = _layer_modules[layer_idx]
                if getattr(mod, "_skip_next", False):
                    # Return in the same shape the original forward would: a tuple
                    # with hidden_states first. No attention weights.
                    return (hidden_states,)
                return original_fwd(*args, **kwargs)
            return new_forward

        # Store originals so we can restore them
        if not hasattr(layer, "_orig_forward_stage12"):
            layer._orig_forward_stage12 = layer.forward
        layer.forward = make_new_forward(orig_forward, i)

    return handles


# Module references used by the monkey-patched forward
_layer_modules = {}


def install(model, skip_controller):
    layers = model.model.layers
    handles = []
    for i, layer in enumerate(layers):
        _layer_modules[i] = layer
        h_ent = layer.self_attn.register_forward_hook(skip_controller.entropy_hook(i))
        h_pre = layer.register_forward_pre_hook(
            skip_controller.skip_pre_hook(i), with_kwargs=True)
        handles.append(h_ent)
        handles.append(h_pre)

        if not hasattr(layer, "_orig_forward_stage12"):
            layer._orig_forward_stage12 = layer.forward
        # Create a closure over layer_idx and original forward
        orig_fwd = layer._orig_forward_stage12

        def make_new_forward(original_fwd, li):
            def new_forward(*args, **kwargs):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                mod = _layer_modules[li]
                if getattr(mod, "_skip_next", False):
                    # Qwen3DecoderLayer.forward returns hidden_states tensor directly.
                    return hidden_states
                return original_fwd(*args, **kwargs)
            return new_forward

        layer.forward = make_new_forward(orig_fwd, i)

    return handles


def uninstall(model, handles):
    for h in handles:
        h.remove()
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "_orig_forward_stage12"):
            layer.forward = layer._orig_forward_stage12
            del layer._orig_forward_stage12
        if hasattr(layer, "_skip_next"):
            del layer._skip_next
    _layer_modules.clear()


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    times = []
    for _ in range(max_new_tokens - 1):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return [t * 1000 for t in times], text, generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--thresholds", default="0.3,0.5,0.7",
                   help="Comma-separated entropy thresholds to sweep")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--always-run-first", type=int, default=2)
    p.add_argument("--always-run-last", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
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
    print(f"\ndevice={device}")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    # Teacher baseline
    print(f"\n=== teacher reference ===", flush=True)
    t_times, t_text, t_tokens = generate(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    t_ms = sum(t_times) / len(t_times)
    print(f"  {t_ms:.2f}ms/tok  {t_text[:120]}...")

    # Sweep thresholds
    results = []
    for th in thresholds:
        print(f"\n=== entropy threshold {th} ===", flush=True)
        ctrl = EntropyLayerSkip(n_layers, threshold=th,
                                always_run_first=args.always_run_first,
                                always_run_last=args.always_run_last)
        handles = install(model, ctrl)
        try:
            g_times, g_text, g_tokens = generate(
                model, tokenizer, args.prompt, args.max_new_tokens, device)
        finally:
            uninstall(model, handles)

        g_ms = sum(g_times) / len(g_times)
        min_len = min(len(t_tokens), len(g_tokens))
        match = sum(1 for a, b in zip(t_tokens[:min_len], g_tokens[:min_len]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, g_tokens)) if a != b), min_len)
        speedup = t_ms / g_ms if g_ms > 0 else 0
        skip_ratio = ctrl.skip_count / max(ctrl.total_count, 1)
        per_layer_skip_pct = {
            i: ctrl.per_layer_skip[i] / max(ctrl.per_layer_total[i], 1)
            for i in range(n_layers)
        }

        print(f"  {g_ms:.2f}ms/tok  {speedup:.2f}x  match {match}/{min_len}  "
              f"first div @ {first_div}")
        print(f"  skipped {ctrl.skip_count}/{ctrl.total_count} layer-calls "
              f"({skip_ratio:.1%})")
        # Show per-layer skip rate summary (first 5, last 5)
        brief = ", ".join(
            f"L{i}:{per_layer_skip_pct[i]:.0%}"
            for i in list(range(min(5, n_layers))) +
                    list(range(max(0, n_layers - 5), n_layers)))
        print(f"  per-layer skip rate: {brief}")
        print(f"  {g_text[:150]}...")

        results.append({
            "threshold": th,
            "ms_per_tok": g_ms,
            "speedup": speedup,
            "match": match,
            "total": min_len,
            "match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "skip_ratio": skip_ratio,
            "skip_count": ctrl.skip_count,
            "total_count": ctrl.total_count,
            "per_layer_skip_pct": per_layer_skip_pct,
            "sample": g_text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  teacher: {t_ms:.2f}ms/tok")
    print(f"  {'thresh':>7}  {'ms/tok':>8}  {'speedup':>8}  {'match':>10}  "
          f"{'skipped':>8}  {'first_div':>10}")
    for r in results:
        print(f"  {r['threshold']:>7.2f}  {r['ms_per_tok']:>8.2f}  "
              f"{r['speedup']:>7.2f}x  {r['match']}/{r['total']:<8}  "
              f"{r['skip_ratio']:>7.1%}  {r['first_divergence']:>10}")

    out_path = Path(args.out_dir) / f"stage12_entropy_layer_skip_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "teacher_ms_per_tok": t_ms,
            "teacher_sample": t_text[:400],
            "always_run_first": args.always_run_first,
            "always_run_last": args.always_run_last,
            "thresholds": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
