"""
Stage 34 — Logit-lens / view-stabilization features.

Per the 'layer as rotation' reframe: each layer is a different viewing
angle on the same invariant token point on the manifold. Testable
prediction: apply lm_head to each layer's hidden state (after final
norm) and see when the argmax stabilizes.

Easy token: argmax same across most layers (consistent view).
Hard token: argmax flips until late layers (views disagree).

Features computed per decode step (6 new):

  stabilization_depth     — latest layer index whose argmax !=
                            final-layer argmax, normalized by L.
                            Larger = harder.
  first_agreement_depth   — earliest layer whose argmax matches
                            final-layer argmax, normalized by L.
  agreement_fraction      — fraction of layers agreeing with final.
  argmax_entropy          — Shannon entropy of the per-layer argmax
                            distribution across L layers.
  top_token_frequency     — fraction of layers voting for the most
                            common token.
  logit_lens_avg_entropy  — mean output-entropy over all L per-layer
                            logit distributions.

Correlate these with output_entropy label under LOPO, and test whether
they CLOSE THE GAP beyond Finding 08's 8-feature set.
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
    CALIB_TEXTS, SUMMARY_FEATURES, CURVATURE_FEATURES, QUANTUM_FEATURES,
    STRUCTURAL_FEATURES, collect_calibration,
)
from scripts.stage31_expanded_lopo import PROMPTS as ALL_PROMPTS


LOGIT_LENS_FEATURES = [
    "stabilization_depth",
    "first_agreement_depth",
    "agreement_fraction",
    "argmax_entropy",
    "top_token_frequency",
    "logit_lens_avg_entropy",
]


def collect_with_logit_lens(model, tokenizer, prompt, max_new_tokens, device,
                              calib_hidden, kde_sigma):
    """Generate, capturing the full stage-29 feature set plus logit-lens
    features derived from applying lm_head to every per-layer hidden state."""
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
    final_norm = model.model.norm
    lm_head = model.lm_head

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
        next_token = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)

        prev_prev_final = None
        prev_H_mean = None
        prev_hidden_norm = None
        recent_step_energies = []

        for step in range(max_new_tokens - 1):
            entropies = [per_layer_H.get(i, 0.0) for i in range(n_layers)]
            head_sharp = [per_layer_head_sharp.get(i, []) for i in range(n_layers)]
            all_heads = [s for layer_h in head_sharp for s in layer_h]

            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            hidden_states = out.hidden_states
            per_layer_h = torch.stack([hs[0, -1].to(torch.float32).cpu()
                                        for hs in hidden_states[1:]])

            # === Logit lens: apply lm_head to each layer's hidden state ===
            with torch.inference_mode():
                per_layer_stack = torch.stack([hs[0, -1]
                                                for hs in hidden_states[1:]]).to(device)
                per_layer_normed = final_norm(per_layer_stack)
                per_layer_logits = lm_head(per_layer_normed)  # [L, V]
                per_layer_argmax = per_layer_logits.argmax(dim=-1).cpu().tolist()
                per_layer_probs = F.softmax(per_layer_logits, dim=-1)
                per_layer_entropy = -(per_layer_probs * torch.log(per_layer_probs.clamp_min(1e-12))).sum(dim=-1)
                logit_lens_avg_entropy = float(per_layer_entropy.mean().item())
            final_argmax = per_layer_argmax[-1]
            agreement = [1 if a == final_argmax else 0 for a in per_layer_argmax]
            agreement_fraction = sum(agreement) / len(agreement)
            # Latest layer disagreeing with final (0 = early, L-1 = just-before-final)
            last_disagree = -1
            for i in range(n_layers):
                if per_layer_argmax[i] != final_argmax:
                    last_disagree = i
            stabilization_depth = (last_disagree + 1) / max(n_layers, 1)
            # Earliest layer agreeing
            first_agree = n_layers
            for i in range(n_layers):
                if per_layer_argmax[i] == final_argmax:
                    first_agree = i; break
            first_agreement_depth = first_agree / max(n_layers, 1)
            # Argmax diversity: Shannon entropy of the L argmaxes
            argmax_counts = {}
            for a in per_layer_argmax:
                argmax_counts[a] = argmax_counts.get(a, 0) + 1
            total = sum(argmax_counts.values())
            ent = 0.0
            for c in argmax_counts.values():
                p = c / total
                ent -= p * math.log(p) if p > 0 else 0
            argmax_entropy = ent
            top_token_frequency = max(argmax_counts.values()) / total

            h_first = hidden_states[0][0, -1].to(torch.float32).cpu()
            h_last = per_layer_h[-1]
            h_mid = per_layer_h[n_layers // 2 - 1]

            # (Existing feature collection — condensed: features already known predictive)
            layer_update_vecs = []; layer_update_mags = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                u = h_ip1 - h_i
                layer_update_vecs.append(u); layer_update_mags.append(u.norm().item())
            hn_last = float(h_last.norm().item())
            hn_mid = float(h_mid.norm().item())
            hn_first = float(h_first.norm().item())
            cent = float((h_last - cal_mean).norm().item())
            total_upd = sum(layer_update_mags)
            max_upd = max(layer_update_mags)
            H_mean = sum(entropies) / len(entropies)
            H_var = sum((e - H_mean) ** 2 for e in entropies) / len(entropies)

            # bipartite_vn_late (best single from Finding 08)
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

            # knn_dist_min
            dists = (calib_hidden - h_last).norm(dim=1)
            knn_dist_min = float(dists.min().item())
            # kde_log_density
            sq_d = ((calib_hidden - h_last) ** 2).sum(dim=1)
            kde_log_density = float(torch.logsumexp(-sq_d / (2 * kde_sigma * kde_sigma), dim=0).item())
            # layer_halves_align
            early_mean = sum(layer_update_vecs[:half]) / half
            late_mean = sum(layer_update_vecs[half:]) / (n_layers - half)
            denom = early_mean.norm() * late_mean.norm()
            layer_halves_align = float((early_mean @ late_mean) / denom.clamp_min(1e-8))
            # upd_kurtosis
            um = sum(layer_update_mags) / len(layer_update_mags)
            var = sum((u - um) ** 2 for u in layer_update_mags) / len(layer_update_mags)
            sdv = var ** 0.5
            kurt = 0.0 if sdv < 1e-10 else (
                sum((u - um) ** 4 for u in layer_update_mags) / (len(layer_update_mags) * sdv ** 4) - 3.0
            )
            # attn_peak_recency
            last_layer_attn = per_layer_last_attn.get(n_layers - 1)
            if last_layer_attn is not None and last_layer_attn.shape[-1] > 1:
                T_kv = last_layer_attn.shape[-1]
                last10_start = max(0, T_kv - 10)
                attn_peak_recency = float(last_layer_attn[:, last10_start:].sum(dim=-1).mean().item())
            else:
                attn_peak_recency = 0.0
            # prod_H_last_norm
            prod_H_last_norm = entropies[-1] * hn_last

            probs = F.softmax(logits[0], dim=-1)
            output_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

            records.append({
                # essential 8 features (from Finding 08)
                "bipartite_vn_late": bipartite_vn_late,
                "prod_H_last_norm": prod_H_last_norm,
                "centeredness": cent,
                "knn_dist_min": knn_dist_min,
                "upd_kurtosis": float(kurt),
                "attn_peak_recency": attn_peak_recency,
                "kde_log_density": kde_log_density,
                "layer_halves_align": layer_halves_align,
                # logit lens features (NEW)
                "stabilization_depth": stabilization_depth,
                "first_agreement_depth": first_agreement_depth,
                "agreement_fraction": agreement_fraction,
                "argmax_entropy": argmax_entropy,
                "top_token_frequency": top_token_frequency,
                "logit_lens_avg_entropy": logit_lens_avg_entropy,
                # label
                "output_entropy": output_entropy,
            })

            prev_prev_final = prev_final
            prev_final = h_last
            next_token = logits.argmax(dim=-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()
    return records


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if vx <= 0 or vy <= 0: return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def linear_regression_r2(X_tr, y_tr, X_te, y_te, ridge=1e-3):
    f = X_tr.shape[1]
    XtX = X_tr.T @ X_tr + ridge * torch.eye(f, dtype=X_tr.dtype)
    Xty = X_tr.T @ y_tr
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_te @ beta
    ss_res = ((y_te - y_pred) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def lopo(X, y, prompt_ids, ridge=1e-3):
    scores = []
    for p in torch.unique(prompt_ids):
        tr_mask = prompt_ids != p
        te_mask = prompt_ids == p
        if te_mask.sum() < 10:
            continue
        X_tr = X[tr_mask]; X_te = X[te_mask]
        mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp_min(1e-8)
        X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd
        X_tr = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1)], dim=1)
        X_te = torch.cat([X_te, torch.ones(X_te.shape[0], 1)], dim=1)
        scores.append(linear_regression_r2(X_tr, y[tr_mask], X_te, y[te_mask], ridge))
    return sum(scores) / max(len(scores), 1)


ESSENTIAL_8 = [
    "bipartite_vn_late", "prod_H_last_norm", "centeredness", "knn_dist_min",
    "upd_kurtosis", "attn_peak_recency", "kde_log_density", "layer_halves_align",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage34_logit_lens.json")
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

    print(f"\n=== collecting with logit lens on {len(ALL_PROMPTS)} prompts ===")
    all_records = []; prompt_ids = []
    t0 = time.perf_counter()
    for pid, (cat, prompt) in enumerate(ALL_PROMPTS):
        recs = collect_with_logit_lens(model, tokenizer, prompt, args.max_new_tokens,
                                        device, calib_hidden, kde_sigma)
        all_records.extend(recs); prompt_ids.extend([pid] * len(recs))
        if (pid + 1) % 5 == 0:
            print(f"  {pid+1}/{len(ALL_PROMPTS)}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"  {len(all_records)} records in {time.perf_counter()-t0:.0f}s")

    y = torch.tensor([r["output_entropy"] for r in all_records], dtype=torch.float32)
    pids = torch.tensor(prompt_ids, dtype=torch.long)

    # Correlations
    print(f"\n=== logit-lens feature Pearson r with output_entropy ===")
    for f in LOGIT_LENS_FEATURES:
        xs = [r[f] for r in all_records]
        r = pearson(xs, y.tolist())
        print(f"  {f:>28}  r = {r:+.3f}")

    # LOPO R² — new features alone, essential-8 alone, combined
    X_new = torch.tensor([[r[f] for f in LOGIT_LENS_FEATURES] for r in all_records],
                          dtype=torch.float32)
    X_ess = torch.tensor([[r[f] for f in ESSENTIAL_8] for r in all_records],
                          dtype=torch.float32)
    X_all = torch.cat([X_ess, X_new], dim=1)

    print(f"\n=== LOPO linear R² ===")
    r_ess = lopo(X_ess, y, pids)
    r_new = lopo(X_new, y, pids)
    r_all = lopo(X_all, y, pids)
    print(f"  essential-8 alone:     {r_ess:.3f}")
    print(f"  logit-lens-6 alone:    {r_new:.3f}")
    print(f"  essential-8 + lens-6:  {r_all:.3f}")
    print(f"  gain from adding lens: +{r_all - r_ess:+.3f}")

    # Also how does logit-lens compare to the 47-feature set?
    print(f"\n  reference (from stage 31): full 47-feature set LOPO R² ≈ 0.341")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "n_records": len(all_records),
            "essential_8_lopo_r2": r_ess,
            "logit_lens_6_lopo_r2": r_new,
            "combined_14_lopo_r2": r_all,
            "gain_from_logit_lens": r_all - r_ess,
            "per_feature_pearson": {
                f: pearson([r[f] for r in all_records], y.tolist())
                for f in LOGIT_LENS_FEATURES
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
