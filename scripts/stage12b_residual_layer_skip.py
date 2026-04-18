"""
Stage 12b — Residual-update-magnitude driven layer skip,
             with prompt-entropy compute tier.

Stage 12 showed attention entropy is the wrong signal for whole-layer
skipping: low attention entropy means "attention is peaked" which does
not imply "this layer does nothing." Layers 3-4 had 99% skip rate under
entropy-driven skipping, but they are doing MLP work — the output was
garbage.

The correct signal is the residual update magnitude:

    rel_update_i = ||h_{i+1} - h_i|| / ||h_i||

If rel_update_i is tiny, layer i is near-identity and can be skipped
without disturbing the downstream state. If it's large, the layer is
pushing the system along the manifold and must run.

Policy:
    1. At prompt end, measure attention entropy over the last prompt
       token averaged across heads and layers. This is the "prompt
       entropy" — a single scalar that ranks prompts by how many
       basins/saddles they'll need to traverse.
    2. Map prompt entropy to a compute tier (low / medium / high),
       which sets the layer-skip threshold:
         low  entropy -> aggressive (skip more)
         high entropy -> conservative (skip less)
    3. During generation, each layer runs unless its running-average
       rel_update over the last `window` tokens is below threshold.
       First `window` tokens always run all layers (need history).

No training. No projection overhead. One norm per layer per token.

Usage:
    python scripts/stage12b_residual_layer_skip.py \\
        --model Qwen/Qwen3-0.6B --device mps
"""

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


# Module-level registry so monkey-patched forwards can resolve layer indices.
_layer_modules = {}


class ResidualLayerSkip:
    def __init__(self, n_layers, threshold, window=8,
                 always_first=2, always_last=2):
        self.n_layers = n_layers
        self.threshold = threshold
        self.window = window
        self.always_first = always_first
        self.always_last = always_last
        self.history = {i: deque(maxlen=window) for i in range(n_layers)}
        self.skip_count = 0
        self.total_count = 0
        self.per_layer_skip = {i: 0 for i in range(n_layers)}
        self.per_layer_total = {i: 0 for i in range(n_layers)}
        self.per_layer_avg_update = {i: 0.0 for i in range(n_layers)}

    def record_update(self, layer_idx, h_in, h_out):
        # Compute relative update magnitude on last query only (decode step)
        # h_in, h_out shape: [B, T_q, d]
        last_in = h_in[0, -1]
        last_out = h_out[0, -1]
        delta = (last_out - last_in).float().norm().item()
        baseline = last_in.float().norm().clamp_min(1e-6).item()
        rel = delta / baseline
        self.history[layer_idx].append(rel)
        self.per_layer_avg_update[layer_idx] = (
            sum(self.history[layer_idx]) / len(self.history[layer_idx]))

    def should_skip(self, layer_idx):
        if layer_idx < self.always_first:
            return False
        if layer_idx >= self.n_layers - self.always_last:
            return False
        h = self.history[layer_idx]
        if len(h) < self.window:
            return False  # need history
        avg = sum(h) / len(h)
        return avg < self.threshold


def install(model, ctrl):
    handles = []
    layers = model.model.layers
    _layer_modules.clear()
    for i, layer in enumerate(layers):
        _layer_modules[i] = layer
        # Pre-hook decides skip
        def make_pre_hook(layer_idx):
            def hook(module, args, kwargs):
                ctrl.total_count += 1
                ctrl.per_layer_total[layer_idx] += 1
                if ctrl.should_skip(layer_idx):
                    ctrl.skip_count += 1
                    ctrl.per_layer_skip[layer_idx] += 1
                    module._skip_next = True
                    # Still need to record something to keep history fresh —
                    # for a skipped step we assume rel_update is 0 at that layer.
                    ctrl.history[layer_idx].append(0.0)
                    ctrl.per_layer_avg_update[layer_idx] = (
                        sum(ctrl.history[layer_idx]) / len(ctrl.history[layer_idx]))
                else:
                    module._skip_next = False
                return args, kwargs
            return hook

        handles.append(layer.register_forward_pre_hook(
            make_pre_hook(i), with_kwargs=True))

        # Post-hook records update magnitude (if the layer actually ran)
        def make_post_hook(layer_idx):
            def hook(module, inputs, output):
                if getattr(module, "_skip_next", False):
                    return  # we already appended 0 in the pre-hook
                h_in = inputs[0] if inputs else None
                h_out = output if isinstance(output, torch.Tensor) else (
                    output[0] if isinstance(output, tuple) else None)
                if h_in is None or h_out is None:
                    return
                ctrl.record_update(layer_idx, h_in, h_out)
            return hook

        handles.append(layer.register_forward_hook(make_post_hook(i)))

        # Monkey-patch forward to early-exit when _skip_next is True
        if not hasattr(layer, "_orig_forward_stage12b"):
            layer._orig_forward_stage12b = layer.forward
        orig_fwd = layer._orig_forward_stage12b

        def make_new_forward(original_fwd, li):
            def new_forward(*args, **kwargs):
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                mod = _layer_modules[li]
                if getattr(mod, "_skip_next", False):
                    return hidden_states
                return original_fwd(*args, **kwargs)
            return new_forward

        layer.forward = make_new_forward(orig_fwd, i)

    return handles


def uninstall(model, handles):
    for h in handles:
        h.remove()
    for layer in model.model.layers:
        if hasattr(layer, "_orig_forward_stage12b"):
            layer.forward = layer._orig_forward_stage12b
            del layer._orig_forward_stage12b
        if hasattr(layer, "_skip_next"):
            del layer._skip_next
    _layer_modules.clear()


def measure_prompt_entropy(model, tokenizer, prompt, device):
    """Run the prompt with output_attentions=True, return the mean normalized
    attention entropy at the last prompt token, averaged over heads and layers."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
    entropies = []
    for attn in out.attentions:
        if attn is None:
            continue
        last = attn[0, :, -1, :]  # [H, T_kv]
        T = last.shape[-1]
        if T <= 1:
            continue
        ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
        norm = math.log(T)
        entropies.append(float((ent / norm).mean().item()))
    return sum(entropies) / max(len(entropies), 1)


def tier_from_prompt_entropy(H):
    """Map prompt entropy to a compute tier. Thresholds derived from
    the entropy-profile zoo: H in [0, 1] normalized by log(T)."""
    if H < 0.30:
        return "low", 0.015    # aggressive skipping; system is committed
    elif H < 0.60:
        return "medium", 0.008  # moderate
    else:
        return "high", 0.003    # conservative; many basins to traverse


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
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--always-first", type=int, default=2)
    p.add_argument("--always-last", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--thresholds", default=None,
                   help="Override threshold list; if unset, use tier from prompt entropy")
    p.add_argument("--prompts-file", default=None,
                   help="JSON file with list of prompt strings; default = built-in 4 prompts")
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

    # Prompt set: one from each expected tier
    prompts = [
        ("arithmetic_easy", "What is 2 plus 2?"),
        ("factual_clear", "The discovery that inference accelerates with context is"),
        ("open_ended", "Tell me something interesting about"),
        ("multi_basin", "Write a poem about cheese and existentialism. Begin:"),
    ]

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers")

    all_results = []
    for pid, prompt in prompts:
        print(f"\n==================== prompt: {pid} ====================")
        print(f"  {prompt!r}")

        # Teacher reference
        t_times, t_text, t_tokens = generate(
            model, tokenizer, prompt, args.max_new_tokens, device)
        t_ms = sum(t_times) / len(t_times)
        print(f"  teacher: {t_ms:.2f}ms/tok   {t_text[:100]}...")

        # Prompt entropy
        H = measure_prompt_entropy(model, tokenizer, prompt, device)
        tier, auto_th = tier_from_prompt_entropy(H)
        print(f"  prompt_entropy={H:.3f}  tier={tier}  auto_threshold={auto_th}")

        # Thresholds to test
        if args.thresholds:
            ths = [float(x) for x in args.thresholds.split(",")]
        else:
            ths = [auto_th]  # use the tier's threshold

        prompt_results = []
        for th in ths:
            ctrl = ResidualLayerSkip(n_layers, threshold=th,
                                     window=args.window,
                                     always_first=args.always_first,
                                     always_last=args.always_last)
            handles = install(model, ctrl)
            try:
                g_times, g_text, g_tokens = generate(
                    model, tokenizer, prompt, args.max_new_tokens, device)
            finally:
                uninstall(model, handles)
            g_ms = sum(g_times) / len(g_times)
            min_len = min(len(t_tokens), len(g_tokens))
            match = sum(1 for a, b in zip(t_tokens[:min_len], g_tokens[:min_len]) if a == b)
            first_div = next((i for i, (a, b) in enumerate(
                zip(t_tokens, g_tokens)) if a != b), min_len)
            speedup = t_ms / g_ms if g_ms > 0 else 0
            skip_ratio = ctrl.skip_count / max(ctrl.total_count, 1)
            print(f"  th={th:.4f}  {g_ms:.2f}ms/tok  {speedup:.2f}x  "
                  f"match {match}/{min_len}  first_div @ {first_div}  "
                  f"skip {skip_ratio:.1%}")
            print(f"    {g_text[:120]}...")
            prompt_results.append({
                "threshold": th,
                "ms_per_tok": g_ms,
                "speedup": speedup,
                "match": match,
                "total": min_len,
                "match_ratio": match / max(min_len, 1),
                "first_divergence": first_div,
                "skip_ratio": skip_ratio,
                "sample": g_text[:300],
                "per_layer_skip_pct": {
                    i: ctrl.per_layer_skip[i] / max(ctrl.per_layer_total[i], 1)
                    for i in range(n_layers)
                },
                "per_layer_avg_update": ctrl.per_layer_avg_update,
            })

        all_results.append({
            "prompt_id": pid,
            "prompt": prompt,
            "prompt_entropy": H,
            "tier": tier,
            "auto_threshold": auto_th,
            "teacher_ms_per_tok": t_ms,
            "teacher_sample": t_text[:300],
            "runs": prompt_results,
        })

    # Summary
    print(f"\n=== summary ===")
    print(f"  {'prompt':>18} {'H':>6} {'tier':>8} {'th':>6} {'spd':>5} {'match':>9}")
    for r in all_results:
        for run in r["runs"]:
            print(f"  {r['prompt_id']:>18} {r['prompt_entropy']:>6.3f} "
                  f"{r['tier']:>8} {run['threshold']:>6.4f} "
                  f"{run['speedup']:>4.2f}x "
                  f"{run['match']}/{run['total']}")

    out_path = Path(args.out_dir) / f"stage12b_residual_skip_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "window": args.window,
            "always_first": args.always_first,
            "always_last": args.always_last,
            "prompts": all_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
