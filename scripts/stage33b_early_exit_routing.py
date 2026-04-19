"""
Stage 33b — Routing with a DESTRUCTIVE cheap path (early-exit).

Stage 33v1 found head masking is too lossless to stress the routing.
This stage replaces the cheap path with early-exit: run the first
(L - skip_layers) decoder layers, then final norm + lm_head on the
partial hidden state. Skipping late layers is genuinely destructive
(stage 12 established this).

Protocol:
  1. Fit 8-feature linear regressor on calibration prompts.
  2. On held-out prompts, generate 3 ways:
     (a) Teacher full-forward baseline.
     (b) Routed-with-predictor: if predicted entropy < threshold →
         early-exit forward, else full forward.
     (c) Always-cheap: early-exit every step.
  3. Measure token match (b) vs (a) and (c) vs (a).

If routing-predictor works:
  routed_match > always_cheap_match.

If the signal is useless for routing:
  routed_match ≈ always_cheap_match (dropping layers kills things
  regardless of when we drop them).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage29_structural_features import (
    CALIB_TEXTS, collect_calibration, collect,
)
from scripts.stage31_expanded_lopo import PROMPTS as ALL_PROMPTS
from scripts.stage33_deploy_routing import (
    ESSENTIAL_8, fit_linear_regressor, predict_with_regressor,
    generate_baseline,
)


def generate_with_early_exit_routing(
    model, tokenizer, prompt, max_new_tokens, device,
    calib_hidden, kde_sigma, beta, mu, sd, threshold,
    skip_layers, mode="routed"
):
    """mode: 'routed' uses predictor; 'always_cheap' always early-exits;
    'never_cheap' always full forward (baseline check)."""
    per_layer_H = {}
    per_layer_head_sharp = {}
    per_layer_last_attn = {}

    def make_hook(li):
        def hook(mod, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None:
                return
            last = w[0, :, -1, :]
            T = last.shape[-1]
            if T <= 1:
                per_layer_H[li] = 0.0
                per_layer_head_sharp[li] = [1.0] * last.shape[0]
                per_layer_last_attn[li] = last.detach().cpu()
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
            ent_norm = (ent / math.log(T)).cpu()
            per_layer_H[li] = float(ent_norm.mean().item())
            per_layer_head_sharp[li] = [float(1 - x) for x in ent_norm.tolist()]
            per_layer_last_attn[li] = last.detach().cpu()
        return hook

    n_layers = len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    # For early-exit we monkey-patch the last skip_layers to return identity
    orig_forwards = {}
    def enable_skip(n_skip):
        for i in range(n_layers - n_skip, n_layers):
            layer = model.model.layers[i]
            if i not in orig_forwards:
                orig_forwards[i] = layer.forward
            # identity: return hidden state unchanged
            def make_identity(li):
                def new_forward(*args, **kwargs):
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    return hidden
                return new_forward
            layer.forward = make_identity(i)

    def disable_skip():
        for i, f in orig_forwards.items():
            model.model.layers[i].forward = f
        orig_forwards.clear()

    records = []
    cal_mean = calib_hidden.mean(dim=0)
    try:
        # Start
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Prefill — always full
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prev_final = out.hidden_states[-1][0, -1].to(torch.float32).cpu()
        logits = out.logits[:, -1, :].float()
        next_token = logits.argmax(dim=-1, keepdim=True)

        # Decide first step mode (no prev signal for routed on first step; default cheap=False)
        mode_this_step = False  # cheap?

        for step in range(max_new_tokens - 1):
            if mode_this_step:
                enable_skip(skip_layers)
            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past,
                            use_cache=True, output_hidden_states=True)
            disable_skip()

            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            hidden_states = out.hidden_states
            per_layer_h = torch.stack([hs[0, -1].to(torch.float32).cpu()
                                        for hs in hidden_states[1:]])
            h_last = per_layer_h[-1]

            # Compute 8 features for NEXT routing decision (using the hooks from THIS step)
            layer_update_vecs = []; layer_update_mags = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                u = h_ip1 - h_i
                layer_update_vecs.append(u); layer_update_mags.append(u.norm().item())
            entropies = [per_layer_H.get(i, 0.0) for i in range(n_layers)]

            half = n_layers // 2
            late = per_layer_h[half:]
            v = late / late.norm(dim=1, keepdim=True).clamp_min(1e-8)
            G = (v @ v.T) / v.shape[0]
            eigvals = torch.linalg.eigvalsh(G.to(torch.float64)).clamp_min(0)
            s = eigvals.sum()
            if s > 0:
                eigvals = eigvals / s
            mask = eigvals > 1e-10
            bipartite_vn_late = float(-(eigvals[mask] * torch.log(eigvals[mask])).sum().item())

            hn_last = float(h_last.norm().item())
            prod_H_last_norm = entropies[-1] * hn_last
            centeredness = float((h_last - cal_mean).norm().item())
            knn_dist_min = float((calib_hidden - h_last).norm(dim=1).min().item())

            um = sum(layer_update_mags) / len(layer_update_mags)
            var = sum((u - um) ** 2 for u in layer_update_mags) / len(layer_update_mags)
            sdv = var ** 0.5
            kurt = 0.0 if sdv < 1e-10 else (
                sum((u - um) ** 4 for u in layer_update_mags) / (len(layer_update_mags) * sdv ** 4) - 3.0
            )
            upd_kurtosis = float(kurt)

            last_layer_attn = per_layer_last_attn.get(n_layers - 1)
            if last_layer_attn is not None and last_layer_attn.shape[-1] > 1:
                T_kv = last_layer_attn.shape[-1]
                last10_start = max(0, T_kv - 10)
                attn_peak_recency = float(last_layer_attn[:, last10_start:].sum(dim=-1).mean().item())
            else:
                attn_peak_recency = 0.0

            sq_d = ((calib_hidden - h_last) ** 2).sum(dim=1)
            kde_log_density = float(torch.logsumexp(-sq_d / (2 * kde_sigma * kde_sigma), dim=0).item())

            early_mean = sum(layer_update_vecs[:half]) / half
            late_mean = sum(layer_update_vecs[half:]) / (n_layers - half)
            denom = early_mean.norm() * late_mean.norm()
            layer_halves_align = float((early_mean @ late_mean) / denom.clamp_min(1e-8))

            features = [bipartite_vn_late, prod_H_last_norm, centeredness, knn_dist_min,
                         upd_kurtosis, attn_peak_recency, kde_log_density, layer_halves_align]
            predicted = predict_with_regressor(features, beta, mu, sd)

            # Decide NEXT step's mode
            if mode == "routed":
                mode_this_step = predicted < threshold
            elif mode == "always_cheap":
                mode_this_step = True
            else:  # never_cheap / baseline
                mode_this_step = False

            records.append({
                "step": step,
                "token_id": int(next_token.item()),
                "predicted_next": predicted,
                "cheap_next": mode_this_step,
            })

            prev_final = h_last
            next_token = logits.argmax(dim=-1, keepdim=True)
    finally:
        disable_skip()
        for h in handles:
            h.remove()
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--calib-prompt-count", type=int, default=15)
    p.add_argument("--skip-layers", type=int, default=6,
                   help="Number of late layers to skip on the cheap path")
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", default="results/stage33b_early_exit_routing.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}  skip_layers={args.skip_layers}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    calib_hidden = collect_calibration(model, tokenizer, CALIB_TEXTS, device)
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample); pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())

    random.seed(0)
    prompts = list(ALL_PROMPTS); random.shuffle(prompts)
    calib_prompts = prompts[:args.calib_prompt_count]
    test_prompts = prompts[args.calib_prompt_count:args.calib_prompt_count + 6]

    # Fit regressor
    print(f"\n=== fitting regressor on {len(calib_prompts)} prompts ===")
    calib_records = []
    for _, prompt in calib_prompts:
        recs = collect(model, tokenizer, prompt, 80, device, calib_hidden, kde_sigma, knn_k=10)
        calib_records.extend(recs)
    X = torch.tensor([[r[f] for f in ESSENTIAL_8] for r in calib_records], dtype=torch.float32)
    y = torch.tensor([r["output_entropy"] for r in calib_records], dtype=torch.float32)
    beta, mu, sd = fit_linear_regressor(X, y)

    # Threshold = median predicted
    preds = [predict_with_regressor(xi.tolist(), beta, mu, sd) for xi in X]
    threshold = args.threshold if args.threshold is not None else sorted(preds)[len(preds)//2]
    print(f"  threshold = {threshold:.3f}")

    # Deploy
    print(f"\n=== deploying on {len(test_prompts)} held-out prompts ===")
    all_results = []
    for cat, prompt in test_prompts:
        print(f"  ({cat}) {prompt[:40]!r}", flush=True)
        t_tokens = generate_baseline(model, tokenizer, prompt, args.max_new_tokens, device)
        # Routed
        r_recs = generate_with_early_exit_routing(
            model, tokenizer, prompt, args.max_new_tokens, device,
            calib_hidden, kde_sigma, beta, mu, sd, threshold,
            args.skip_layers, mode="routed")
        r_tokens = [rec["token_id"] for rec in r_recs]
        cheap_frac_routed = sum(1 for rec in r_recs if rec["cheap_next"]) / max(len(r_recs), 1)
        # Always cheap
        ac_recs = generate_with_early_exit_routing(
            model, tokenizer, prompt, args.max_new_tokens, device,
            calib_hidden, kde_sigma, beta, mu, sd, threshold,
            args.skip_layers, mode="always_cheap")
        ac_tokens = [rec["token_id"] for rec in ac_recs]

        ml = min(len(t_tokens), len(r_tokens))
        match_r = sum(1 for a, b in zip(t_tokens[:ml], r_tokens[:ml]) if a == b)
        ml_ac = min(len(t_tokens), len(ac_tokens))
        match_ac = sum(1 for a, b in zip(t_tokens[:ml_ac], ac_tokens[:ml_ac]) if a == b)

        print(f"    routed:       {match_r}/{ml} ({match_r/max(ml,1):.1%})  cheap={cheap_frac_routed:.1%}")
        print(f"    always-cheap: {match_ac}/{ml_ac} ({match_ac/max(ml_ac,1):.1%})")

        all_results.append({
            "category": cat, "prompt": prompt,
            "routed_match": match_r, "routed_total": ml,
            "always_cheap_match": match_ac, "always_cheap_total": ml_ac,
            "routed_match_ratio": match_r / max(ml, 1),
            "always_cheap_match_ratio": match_ac / max(ml_ac, 1),
            "cheap_frac": cheap_frac_routed,
        })

    avg_r = sum(r["routed_match_ratio"] for r in all_results) / len(all_results)
    avg_ac = sum(r["always_cheap_match_ratio"] for r in all_results) / len(all_results)
    avg_cheap = sum(r["cheap_frac"] for r in all_results) / len(all_results)
    print(f"\n=== aggregate ===")
    print(f"  skip_layers = {args.skip_layers} / {28} ({args.skip_layers / 28:.0%} of stack)")
    print(f"  routed match:       {avg_r:.1%}   cheap_frac = {avg_cheap:.1%}")
    print(f"  always-cheap match: {avg_ac:.1%}   cheap_frac = 100%")
    print(f"  routing gain:       {avg_r - avg_ac:+.1%}")
    print(f"  compute saved at routed: {avg_cheap:.1%} of steps used cheap path "
          f"-> ~{avg_cheap * args.skip_layers / 28:.1%} compute saved")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "skip_layers": args.skip_layers,
            "threshold": threshold,
            "avg_routed_match": avg_r, "avg_always_cheap_match": avg_ac,
            "avg_cheap_frac": avg_cheap,
            "routing_gain": avg_r - avg_ac,
            "per_prompt": all_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
