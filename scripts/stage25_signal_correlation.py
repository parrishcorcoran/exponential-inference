"""
Stage 25 — Signal correlation matrix and independent-axis discovery.

Over the course of this project we've named many runtime signals for
per-token routing. Many of them are likely redundant: different
aggregations of the same underlying quantity. This stage computes
correlations between every pair of candidate signals on actual decode
data and identifies:

    - which signals are genuinely independent,
    - which are duplicates of the same underlying quantity,
    - which cluster together,
    - which has the highest correlation with the routing label
      (output entropy) per cluster.

Signals collected per decode step (21 features):

  Attention-entropy family (per-layer, normalized by log(T_kv)):
    H_mean, H_max, H_min, H_var (variance across layers),
    H_first_layer, H_mid_layer, H_last_layer,
    H_q1_layer (L/4), H_q3_layer (3L/4).

  Head-sharpness family:
    heads_above_0p5 (count across all layers),
    heads_above_0p9,
    mean_head_sharpness,
    max_head_sharpness.

  Hidden-state magnitude:
    hidden_norm_final,
    hidden_norm_mid,
    hidden_norm_first.

  Trajectory:
    step_size_abs       (||h_t - h_{t-1}||),
    step_size_rel       (normalized),
    cosine_prev         (cos sim with h_{t-1}),
    centeredness,
    total_layer_update  (sum of per-layer ||Δh||),
    max_layer_update    (max per-layer ||Δh||).

  Derivatives:
    dH_dt_mean          (change in H_mean),
    d_hidden_norm_dt.

Labels: output_entropy, logit_margin, log_p_top1.

Output: (a) full correlation matrix, (b) agglomerative clustering of
features, (c) representative signal per cluster.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROMPTS = [
    "The discovery that inference accelerates with context is",
    "The capital of France is",
    "To solve a quadratic equation we use the formula",
    "Tell me something interesting about the solar system",
    "Write a poem about cheese:",
    "If all birds have feathers and penguins are birds, then",
]


def collect(model, tokenizer, prompt, max_new_tokens, device, cal_mean):
    """Generate, collect all 21 signals + 3 labels per step."""
    # Per-layer storage accumulated via hooks
    per_layer_attn_entropy = {}
    per_layer_head_sharpness = {}  # layer_idx -> list of sharpness per head

    def make_hook(li):
        def hook(mod, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None:
                return
            last = w[0, :, -1, :]  # [H, T_kv]
            T = last.shape[-1]
            if T <= 1:
                per_layer_attn_entropy[li] = 0.0
                per_layer_head_sharpness[li] = [1.0] * last.shape[0]
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
            ent_norm = (ent / math.log(T)).cpu()
            per_layer_attn_entropy[li] = float(ent_norm.mean().item())
            # Per-head sharpness = 1 - ent_norm
            per_layer_head_sharpness[li] = [float(1 - x) for x in ent_norm.tolist()]
        return hook

    handles = []
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    records = []
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prev_final = out.hidden_states[-1][0, -1].to(torch.float32).cpu()
        prev_final_norm = float(prev_final.norm().item())
        logits = out.logits[:, -1, :].float()
        next_token = logits.argmax(dim=-1, keepdim=True)

        # Previous signals for derivatives
        prev_H_mean = None
        prev_hidden_norm = None

        for step in range(max_new_tokens - 1):
            # Snapshot per-layer attention signals from prev forward (they attach
            # to the PREVIOUS forward's attention computation)
            entropies = [per_layer_attn_entropy.get(i, 0.0) for i in range(n_layers)]
            head_sharpness = [per_layer_head_sharpness.get(i, []) for i in range(n_layers)]
            all_heads = [s for layer_h in head_sharpness for s in layer_h]

            # Run forward for this step
            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

            hidden_states = out.hidden_states  # tuple of L+1 tensors
            h_first = hidden_states[0][0, -1].to(torch.float32).cpu()
            h_last = hidden_states[-1][0, -1].to(torch.float32).cpu()
            h_mid = hidden_states[n_layers // 2][0, -1].to(torch.float32).cpu()

            # Per-layer updates
            layer_updates = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                upd = (h_ip1 - h_i).norm().item()
                layer_updates.append(upd)

            # Build feature dict
            H_mean = sum(entropies) / len(entropies)
            H_max = max(entropies)
            H_min = min(entropies)
            H_var = (sum((e - H_mean) ** 2 for e in entropies) / len(entropies))
            H_first = entropies[0]
            H_mid = entropies[n_layers // 2]
            H_last = entropies[-1]
            H_q1 = entropies[n_layers // 4]
            H_q3 = entropies[(3 * n_layers) // 4]

            heads_above_0p5 = sum(1 for s in all_heads if s > 0.5)
            heads_above_0p9 = sum(1 for s in all_heads if s > 0.9)
            mean_head_sharpness = sum(all_heads) / max(len(all_heads), 1)
            max_head_sharpness = max(all_heads) if all_heads else 0.0

            hn_last = float(h_last.norm().item())
            hn_mid = float(h_mid.norm().item())
            hn_first = float(h_first.norm().item())

            step_abs = float((h_last - prev_final).norm().item())
            step_rel = step_abs / max(prev_final_norm, 1e-8)
            cos_prev = float((h_last @ prev_final) / max(hn_last * prev_final_norm, 1e-8))
            cent = float((h_last - cal_mean).norm().item())
            total_layer_upd = sum(layer_updates)
            max_layer_upd = max(layer_updates)

            dH_dt_mean = (H_mean - prev_H_mean) if prev_H_mean is not None else 0.0
            d_hidden_norm_dt = (hn_last - prev_hidden_norm) if prev_hidden_norm is not None else 0.0

            # Labels
            top2 = logits.topk(2, dim=-1)
            margin = float((top2.values[0, 0] - top2.values[0, 1]).item())
            probs = F.softmax(logits[0], dim=-1)
            output_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
            log_p_top1 = float(torch.log(probs[int(top2.indices[0, 0].item())].clamp_min(1e-12)).item())

            records.append({
                # Attention-entropy family
                "H_mean": H_mean, "H_max": H_max, "H_min": H_min, "H_var": H_var,
                "H_first_layer": H_first, "H_mid_layer": H_mid, "H_last_layer": H_last,
                "H_q1_layer": H_q1, "H_q3_layer": H_q3,
                # Head-sharpness family
                "heads_above_0p5": float(heads_above_0p5),
                "heads_above_0p9": float(heads_above_0p9),
                "mean_head_sharpness": mean_head_sharpness,
                "max_head_sharpness": max_head_sharpness,
                # Hidden-norm family
                "hidden_norm_final": hn_last,
                "hidden_norm_mid": hn_mid,
                "hidden_norm_first": hn_first,
                # Trajectory
                "step_size_abs": step_abs,
                "step_size_rel": step_rel,
                "cosine_prev": cos_prev,
                "centeredness": cent,
                "total_layer_update": total_layer_upd,
                "max_layer_update": max_layer_upd,
                # Derivatives
                "dH_dt_mean": dH_dt_mean,
                "d_hidden_norm_dt": d_hidden_norm_dt,
                # Labels
                "label_output_entropy": output_entropy,
                "label_logit_margin": margin,
                "label_log_p_top1": log_p_top1,
            })

            prev_final = h_last
            prev_final_norm = hn_last
            prev_H_mean = H_mean
            prev_hidden_norm = hn_last
            next_token = logits.argmax(dim=-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()
    return records


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def cluster_features(features, corr_matrix, threshold=0.85):
    """Greedy: if feature i correlates > threshold with any feature already in
    a cluster, add it to that cluster; else start a new cluster."""
    clusters = []
    for i, f in enumerate(features):
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            ri = features.index(rep)
            if abs(corr_matrix[ri][i]) >= threshold:
                cluster.append(f)
                placed = True
                break
        if not placed:
            clusters.append([f])
    return clusters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--device", default=None)
    p.add_argument("--cluster-threshold", type=float, default=0.85)
    p.add_argument("--out", default="results/stage25_signal_correlation.json")
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
    print(f"device={device}  model={args.model}")

    print(f"\n=== loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    # Calibration mean (for centeredness feature)
    prefix = "The cell is the basic structural unit of life."
    with torch.inference_mode():
        ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        cal_mean = out.hidden_states[-1][0].to(torch.float32).cpu().mean(dim=0)

    all_records = []
    for prompt in PROMPTS:
        print(f"  collecting from {prompt!r}", flush=True)
        recs = collect(model, tokenizer, prompt, args.max_new_tokens, device, cal_mean)
        all_records.extend(recs)
    print(f"\n=== collected {len(all_records)} records with "
          f"{len(all_records[0]) - 3} features + 3 labels ===")

    # Feature names (exclude labels)
    label_names = ["label_output_entropy", "label_logit_margin", "label_log_p_top1"]
    feature_names = [k for k in all_records[0].keys() if k not in label_names]

    # Build matrices
    n = len(all_records)
    F_cols = len(feature_names)
    X = [[r[f] for f in feature_names] for r in all_records]

    # Feature-feature correlation matrix
    print(f"\n=== computing feature-feature correlations ===")
    corr = [[0.0] * F_cols for _ in range(F_cols)]
    for i, fi in enumerate(feature_names):
        xs_i = [row[i] for row in X]
        for j, fj in enumerate(feature_names):
            if j < i:
                corr[i][j] = corr[j][i]
                continue
            xs_j = [row[j] for row in X]
            corr[i][j] = pearson(xs_i, xs_j)

    # Feature-label correlations
    label_corr = {lab: {} for lab in label_names}
    for lab in label_names:
        ys = [r[lab] for r in all_records]
        for i, fi in enumerate(feature_names):
            xs_i = [row[i] for row in X]
            label_corr[lab][fi] = pearson(xs_i, ys)

    # Clustering
    clusters = cluster_features(feature_names, corr, threshold=args.cluster_threshold)
    print(f"\n=== {len(clusters)} clusters at |r| >= {args.cluster_threshold} ===")
    for c_id, cluster in enumerate(clusters):
        print(f"\n  Cluster {c_id+1} ({len(cluster)} feature(s)):")
        # Pick representative: highest |r| with output_entropy
        rep = max(cluster, key=lambda f: abs(label_corr["label_output_entropy"][f]))
        for f in cluster:
            marker = "  *" if f == rep else "   "
            r_oe = label_corr["label_output_entropy"][f]
            r_lm = label_corr["label_logit_margin"][f]
            r_lp = label_corr["label_log_p_top1"][f]
            print(f"  {marker}  {f:<26}  oe={r_oe:+.3f}  lm={r_lm:+.3f}  lp={r_lp:+.3f}")

    # Summary: one "best" signal per cluster
    print(f"\n=== independent axes (one representative per cluster) ===")
    for c_id, cluster in enumerate(clusters):
        rep = max(cluster, key=lambda f: abs(label_corr["label_output_entropy"][f]))
        r_oe = label_corr["label_output_entropy"][rep]
        print(f"  Axis {c_id+1}: {rep}  (r with output_entropy = {r_oe:+.3f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_records": n,
            "feature_names": feature_names,
            "label_names": label_names,
            "feature_feature_correlation": corr,
            "feature_label_correlation": label_corr,
            "clusters": [{"members": c} for c in clusters],
            "cluster_threshold": args.cluster_threshold,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
