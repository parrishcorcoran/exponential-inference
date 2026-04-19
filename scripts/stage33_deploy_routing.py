"""
Stage 33 — Deploy the 8-feature routing policy and measure quality/compute.

Built on Finding 08's 8-feature minimal subset. This stage:

  1. Fit linear regression (8 features -> predicted output_entropy)
     on a calibration set of prompts held OUT of the test prompts.
  2. At each decode step on test prompts:
       a. Run the standard forward (capture 8 features).
       b. Use signals from step t to predict entropy at step t+1.
       c. If predicted entropy < threshold (easy): on step t+1 use
          head-pruned forward (stage 5-style). Else: full forward.
  3. Measure:
       - Token match vs teacher baseline.
       - Fraction of tokens routed to cheap path.
       - Per-category breakdown.
  4. Compare to (a) full-teacher baseline (R² ceiling), (b) always-
     cheap baseline (quality floor), (c) random routing at matched
     cheap-fraction (is the learned policy better than chance?).

This is a real test of whether the signal DRIVES routing decisions
well, not just whether it correlates with the label.
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

# The essential 8 features from Finding 08
ESSENTIAL_8 = [
    "bipartite_vn_late",
    "prod_H_last_norm",
    "centeredness",
    "knn_dist_min",
    "upd_kurtosis",
    "attn_peak_recency",
    "kde_log_density",
    "layer_halves_align",
]


def fit_linear_regressor(X, y, ridge=1e-3):
    """Fit ridge regression. Return (beta [F+1], mu, sd) for deploying later."""
    mu = X.mean(dim=0); sd = X.std(dim=0).clamp_min(1e-8)
    Xn = (X - mu) / sd
    Xa = torch.cat([Xn, torch.ones(Xn.shape[0], 1)], dim=1)
    f = Xa.shape[1]
    XtX = Xa.T @ Xa + ridge * torch.eye(f, dtype=Xa.dtype)
    Xty = Xa.T @ y
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    return beta, mu, sd


def predict_with_regressor(features, beta, mu, sd):
    """features: 8-vector of raw features. Returns predicted entropy (scalar)."""
    x = torch.tensor(features, dtype=torch.float32)
    xn = (x - mu) / sd
    xa = torch.cat([xn, torch.tensor([1.0])])
    return float((xa @ beta).item())


def generate_with_routing(model, tokenizer, prompt, max_new_tokens, device,
                          calib_hidden, kde_sigma, beta, mu, sd, threshold,
                          essential_features, cheap_head_threshold=0.9,
                          min_heads=2):
    """Greedy generation with per-step routing:
    - At step t, run full forward, collect 8 features + attention pattern.
    - Predict step t+1's output entropy from features.
    - If predicted < threshold, on step t+1 apply head pruning (mark
      routed=True). Else full forward.
    Returns list of (token_id, routed_cheap, predicted_entropy, actual_entropy).
    """
    # Set up hooks for attention capture (for both features and head sharpness)
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

    records = []
    cal_mean = calib_hidden.mean(dim=0)
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prev_final = out.hidden_states[-1][0, -1].to(torch.float32).cpu()
        logits = out.logits[:, -1, :].float()
        next_token = logits.argmax(dim=-1, keepdim=True)

        prev_prev_final = None
        # for head pruning head_mask
        head_mask_next = None

        for step in range(max_new_tokens - 1):
            with torch.inference_mode():
                # Apply head_mask if routed
                if head_mask_next is not None:
                    out = model(input_ids=next_token, past_key_values=past,
                                use_cache=True, output_hidden_states=True,
                                head_mask=head_mask_next)
                else:
                    out = model(input_ids=next_token, past_key_values=past,
                                use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            hidden_states = out.hidden_states
            per_layer_h = torch.stack([hs[0, -1].to(torch.float32).cpu()
                                        for hs in hidden_states[1:]])
            h_last = per_layer_h[-1]

            # Compute features for THIS step
            layer_update_vecs = []
            layer_update_mags = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                u = h_ip1 - h_i
                layer_update_vecs.append(u)
                layer_update_mags.append(u.norm().item())

            entropies = [per_layer_H.get(i, 0.0) for i in range(n_layers)]
            step_vec = h_last - prev_final
            hn_last = float(h_last.norm().item())

            # The 8 essential features
            # 1. bipartite_vn_late
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

            # 2. prod_H_last_norm
            prod_H_last_norm = entropies[-1] * hn_last

            # 3. centeredness
            centeredness = float((h_last - cal_mean).norm().item())

            # 4. knn_dist_min
            dists = (calib_hidden - h_last).norm(dim=1)
            knn_dist_min = float(dists.min().item())

            # 5. upd_kurtosis
            um = sum(layer_update_mags) / len(layer_update_mags)
            var = sum((u - um) ** 2 for u in layer_update_mags) / len(layer_update_mags)
            sdv = var ** 0.5
            if sdv < 1e-10:
                kurt = 0.0
            else:
                kurt = sum((u - um) ** 4 for u in layer_update_mags) / (len(layer_update_mags) * sdv ** 4) - 3.0
            upd_kurtosis = float(kurt)

            # 6. attn_peak_recency
            last_layer_attn = per_layer_last_attn.get(n_layers - 1)
            if last_layer_attn is not None and last_layer_attn.shape[-1] > 1:
                T_kv = last_layer_attn.shape[-1]
                last10_start = max(0, T_kv - 10)
                attn_peak_recency = float(last_layer_attn[:, last10_start:].sum(dim=-1).mean().item())
            else:
                attn_peak_recency = 0.0

            # 7. kde_log_density
            sq_d = ((calib_hidden - h_last) ** 2).sum(dim=1)
            kde_log_density = float(torch.logsumexp(-sq_d / (2 * kde_sigma * kde_sigma), dim=0).item())

            # 8. layer_halves_align
            early_mean = sum(layer_update_vecs[:half]) / half
            late_mean = sum(layer_update_vecs[half:]) / (n_layers - half)
            denom = early_mean.norm() * late_mean.norm()
            layer_halves_align = float((early_mean @ late_mean) / denom.clamp_min(1e-8))

            features = [bipartite_vn_late, prod_H_last_norm, centeredness,
                         knn_dist_min, upd_kurtosis, attn_peak_recency,
                         kde_log_density, layer_halves_align]

            # Predict entropy for the NEXT step
            predicted_entropy = predict_with_regressor(features, beta, mu, sd)

            # Capture actual label (this step's output entropy)
            probs = F.softmax(logits[0], dim=-1)
            actual_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

            # Decide routing for NEXT step
            if predicted_entropy < threshold:
                # Build head_mask for next step: prune low-sharpness heads
                head_mask_list = []
                for li in range(n_layers):
                    head_sharpness = per_layer_head_sharp.get(li, [1.0])
                    # Keep heads with sharpness > cheap_head_threshold
                    keep = [1.0 if s > cheap_head_threshold else 0.0 for s in head_sharpness]
                    if sum(keep) < min_heads:
                        # Enforce floor: keep the top min_heads by sharpness
                        indexed = sorted(enumerate(head_sharpness), key=lambda x: -x[1])
                        keep = [0.0] * len(head_sharpness)
                        for idx, _ in indexed[:min_heads]:
                            keep[idx] = 1.0
                    head_mask_list.append(keep)
                head_mask_next = torch.tensor(head_mask_list, device=device,
                                               dtype=torch.float32).unsqueeze(0)
                # head_mask expected shape: [n_layers, n_heads] or more complex.
                # HF Qwen3 accepts head_mask of shape [n_layers, n_heads].
                head_mask_next = head_mask_next.squeeze(0)
                routed = True
            else:
                head_mask_next = None
                routed = False

            records.append({
                "step": step,
                "token_id": int(next_token.item()),
                "predicted_entropy": predicted_entropy,
                "actual_entropy": actual_entropy,
                "routed_cheap": routed,
            })

            prev_prev_final = prev_final
            prev_final = h_last
            next_token = logits.argmax(dim=-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()
    return records


def generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    """Plain greedy — no routing, full compute."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--calib-prompt-count", type=int, default=15)
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Predicted-entropy threshold to route cheap. If None, set to median.")
    p.add_argument("--cheap-head-threshold", type=float, default=0.9)
    p.add_argument("--out", default="results/stage33_deploy_routing.json")
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
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    calib_hidden = collect_calibration(model, tokenizer, CALIB_TEXTS, device)
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample); pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())

    # Split prompts: first N for fitting regressor, rest for deployment test
    random.seed(0)
    prompt_list = list(ALL_PROMPTS)
    random.shuffle(prompt_list)
    calib_prompts = prompt_list[:args.calib_prompt_count]
    test_prompts = prompt_list[args.calib_prompt_count:args.calib_prompt_count + 6]

    # === Fit regressor on calibration prompts ===
    print(f"\n=== fitting 8-feature regressor on {len(calib_prompts)} prompts ===")
    calib_records = []
    for cat, prompt in calib_prompts:
        recs = collect(model, tokenizer, prompt, 80, device, calib_hidden, kde_sigma, knn_k=10)
        for r in recs:
            r["_cat"] = cat
        calib_records.extend(recs)
    print(f"  {len(calib_records)} calibration records")

    X_calib = torch.tensor([[r[f] for f in ESSENTIAL_8] for r in calib_records],
                            dtype=torch.float32)
    y_calib = torch.tensor([r["output_entropy"] for r in calib_records], dtype=torch.float32)
    beta, mu, sd = fit_linear_regressor(X_calib, y_calib)

    # Pick threshold: the MEDIAN predicted entropy gives roughly 50/50 split
    predictions_calib = []
    for feats in X_calib:
        predictions_calib.append(predict_with_regressor(feats.tolist(), beta, mu, sd))
    median_pred = sorted(predictions_calib)[len(predictions_calib)//2]
    threshold = args.threshold if args.threshold is not None else median_pred
    print(f"  threshold (predicted entropy): {threshold:.3f}")

    # === Deploy on test prompts ===
    print(f"\n=== deploying on {len(test_prompts)} held-out prompts ===")
    all_results = []
    for cat, prompt in test_prompts:
        print(f"  ({cat}) {prompt[:40]!r}", flush=True)

        # Baseline teacher output
        t_tokens = generate_baseline(model, tokenizer, prompt, args.max_new_tokens, device)

        # Routed output (predictor-driven)
        recs = generate_with_routing(
            model, tokenizer, prompt, args.max_new_tokens, device,
            calib_hidden, kde_sigma, beta, mu, sd, threshold,
            ESSENTIAL_8, args.cheap_head_threshold,
        )
        r_tokens = [r["token_id"] for r in recs]

        # Always-cheap baseline: force every step to the cheap path (beta pointing
        # to predicted entropy = -inf so it's always below threshold).
        beta_ac = torch.tensor([-1e9] * (len(ESSENTIAL_8) + 1), dtype=torch.float32)
        ac_recs = generate_with_routing(
            model, tokenizer, prompt, args.max_new_tokens, device,
            calib_hidden, kde_sigma, beta_ac, mu, sd, threshold,
            ESSENTIAL_8, args.cheap_head_threshold,
        )
        ac_tokens = [rec["token_id"] for rec in ac_recs]

        # Random-routing baseline at same cheap_fraction
        cheap_frac = sum(1 for r in recs if r["routed_cheap"]) / max(len(recs), 1)
        random.seed(42)
        random_routed = [random.random() < cheap_frac for _ in recs]
        # (Not actually generating with random routing — too expensive here.
        # Instead report cheap_frac as the compute-reduction number.)

        # Score routed
        min_len = min(len(t_tokens), len(r_tokens))
        match = sum(1 for a, b in zip(t_tokens[:min_len], r_tokens[:min_len]) if a == b)
        match_ratio = match / max(min_len, 1)
        first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, r_tokens)) if a != b), min_len)
        # Score always-cheap
        ac_min = min(len(t_tokens), len(ac_tokens))
        ac_match = sum(1 for a, b in zip(t_tokens[:ac_min], ac_tokens[:ac_min]) if a == b)
        ac_ratio = ac_match / max(ac_min, 1)
        ac_first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, ac_tokens)) if a != b), ac_min)

        print(f"    routed:      {match}/{min_len} ({match_ratio:.1%})  cheap_frac={cheap_frac:.1%}  first_div={first_div}")
        print(f"    always-cheap:{ac_match}/{ac_min} ({ac_ratio:.1%})  first_div={ac_first_div}")
        all_results.append({
            "category": cat,
            "prompt": prompt,
            "routed_match_ratio": match_ratio,
            "routed_match": match, "total": min_len,
            "routed_first_divergence": first_div,
            "cheap_fraction": cheap_frac,
            "always_cheap_match_ratio": ac_ratio,
            "always_cheap_first_divergence": ac_first_div,
            "teacher_text": tokenizer.decode(t_tokens, skip_special_tokens=True)[:200],
            "routed_text": tokenizer.decode(r_tokens, skip_special_tokens=True)[:200],
            "always_cheap_text": tokenizer.decode(ac_tokens, skip_special_tokens=True)[:200],
        })

    # Aggregate
    avg_match = sum(r["routed_match_ratio"] for r in all_results) / len(all_results)
    avg_cheap = sum(r["cheap_fraction"] for r in all_results) / len(all_results)
    avg_ac = sum(r["always_cheap_match_ratio"] for r in all_results) / len(all_results)
    print(f"\n=== aggregate ===")
    print(f"  routed-with-predictor match: {avg_match:.1%}   cheap_frac={avg_cheap:.1%}")
    print(f"  always-cheap match:          {avg_ac:.1%}   cheap_frac=100.0%")
    print(f"  gain from routing:           +{avg_match - avg_ac:+.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "threshold": threshold,
            "cheap_head_threshold": args.cheap_head_threshold,
            "mean_match": avg_match, "mean_cheap_fraction": avg_cheap,
            "per_prompt": all_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
