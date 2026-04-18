"""
Stage F — Saddle detection via ∂H/∂t.

Observational experiment: during generation from a set of prompts with
varying expected entropy profiles (bell, linear, plateau, spike),
capture per-step, per-layer attention entropy H_i(t). Compute derivatives
∂H_i/∂t. Track per-step distribution divergence from a reference.

Question: does ∂H/∂t > 0 (rising entropy = saddle approaching) predict
positions where aggressive pruning would break output?

We measure (no training, no pruning — just observation):
    1. H_i(t) per layer per step — normalized attention entropy.
    2. ∂H_i/∂t per layer — discrete difference from previous step.
    3. Aggregate statistics (mean H, max H, mean ∂H/∂t across layers).
    4. Per-step KL(teacher full || teacher with one-layer projected-rank-32)
       as a proxy for "compute mattered here."

Output: per-step trajectory of (H, ∂H/∂t, sensitivity) across prompts
from different expected profiles.

Usage:
    python scripts/stageF_saddle_detection.py --model Qwen/Qwen3-0.6B --device mps
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


PROMPT_SET = [
    ("arithmetic_easy", "What is 2 plus 2? The answer is"),
    ("factual_clear", "The capital of France is"),
    ("open_ended", "Tell me something interesting about"),
    ("multi_basin", "Write a short poem about cheese:"),
    ("reasoning_chain", "If all birds have feathers and penguins are birds, then"),
    ("ambiguous", "The meaning of life is"),
]


class PerStepEntropyCapture:
    """Capture per-layer entropy at each decode step."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.per_step = []  # list of {layer_idx: float}
        self.current = {}

    def reset(self):
        self.per_step = []
        self.current = {}

    def make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None:
                return
            # Last query position's attention over all cached keys
            last = w[0, :, -1, :]  # [H, T_kv]
            T = last.shape[-1]
            if T <= 1:
                ent_norm = 0.0
            else:
                ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
                ent_norm = float((ent.mean() / math.log(T)).item())
            self.current[layer_idx] = ent_norm
        return hook

    def commit_step(self):
        self.per_step.append(dict(self.current))
        self.current = {}


def install(model, cap):
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(cap.make_hook(i)))
    return handles


def uninstall(handles):
    for h in handles:
        h.remove()


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def trace_generation(model, tokenizer, prompt, max_new_tokens, device, cap):
    """Generate, capturing per-step per-layer entropy."""
    cap.reset()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    cap.commit_step()
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        cap.commit_step()
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated


def classify_profile(h_trace):
    """h_trace: list of length T, mean entropy across layers per step.
    Returns one of: bell, linear, plateau, spike, other."""
    if len(h_trace) < 5:
        return "short"
    t = list(range(len(h_trace)))
    mean = sum(h_trace) / len(h_trace)
    # Spike: any step's H exceeds mean + 2*std AND neighbors are lower
    import statistics
    try:
        std = statistics.stdev(h_trace)
    except statistics.StatisticsError:
        std = 0.0
    spikes = [i for i in range(1, len(h_trace) - 1)
              if h_trace[i] > mean + 2 * std
              and h_trace[i] > h_trace[i-1]
              and h_trace[i] > h_trace[i+1]]
    if spikes:
        return "spike"
    # Bell: rises then falls. Peak not at endpoints.
    peak_idx = h_trace.index(max(h_trace))
    if 1 < peak_idx < len(h_trace) - 2 and max(h_trace) > min(h_trace) * 1.3:
        return "bell"
    # Plateau: low variance
    if std / max(mean, 1e-6) < 0.1:
        return "plateau"
    # Linear: monotone decreasing or increasing
    deltas = [h_trace[i+1] - h_trace[i] for i in range(len(h_trace)-1)]
    if all(d < 0.02 for d in deltas) or all(d > -0.02 for d in deltas):
        return "linear"
    return "other"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--device", default=None)
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

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers

    cap = PerStepEntropyCapture(n_layers)
    handles = install(model, cap)

    try:
        all_results = []
        for pid, prompt in PROMPT_SET:
            print(f"\n--- {pid}: {prompt!r} ---")
            text, _ = trace_generation(model, tokenizer, prompt,
                                        args.max_new_tokens, device, cap)
            # Per-step mean entropy across layers (skip prefill = step 0 which has stale T)
            per_step_mean = []
            per_step_max = []
            per_step_min = []
            for step_ents in cap.per_step[1:]:  # drop prefill
                vals = list(step_ents.values())
                if not vals:
                    continue
                per_step_mean.append(sum(vals) / len(vals))
                per_step_max.append(max(vals))
                per_step_min.append(min(vals))
            # Derivatives
            dh_dt_mean = [per_step_mean[i+1] - per_step_mean[i]
                          for i in range(len(per_step_mean)-1)]
            dh_dt_max_layer = []
            for i in range(1, len(cap.per_step)-1):
                prev = cap.per_step[i]
                curr = cap.per_step[i+1]
                if not prev or not curr:
                    continue
                diffs = [curr[li] - prev.get(li, 0.0) for li in range(n_layers) if li in curr]
                dh_dt_max_layer.append(max(diffs) if diffs else 0.0)

            profile = classify_profile(per_step_mean)
            saddle_count = sum(1 for d in dh_dt_mean if d > 0.03)
            print(f"  text: {text[:80]}...")
            print(f"  profile: {profile}")
            print(f"  H range: {min(per_step_mean):.3f} - {max(per_step_mean):.3f}  mean: {sum(per_step_mean)/len(per_step_mean):.3f}")
            print(f"  saddle-like events (dH/dt_mean > 0.03): {saddle_count}/{len(dh_dt_mean)}")
            print(f"  max dH/dt (per-layer): {max(dh_dt_max_layer) if dh_dt_max_layer else 0:.3f}")

            all_results.append({
                "prompt_id": pid,
                "prompt": prompt,
                "text": text[:300],
                "profile": profile,
                "per_step_mean_H": per_step_mean,
                "per_step_max_H": per_step_max,
                "per_step_min_H": per_step_min,
                "dH_dt_mean": dh_dt_mean,
                "max_dH_dt_per_layer": dh_dt_max_layer,
                "saddle_events_count": saddle_count,
            })
    finally:
        uninstall(handles)

    out_path = Path(args.out_dir) / f"stageF_saddle_detection_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "prompts": all_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
