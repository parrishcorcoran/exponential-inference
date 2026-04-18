"""
Stage 24 — Easy-token identification from manifold-position signals.

Question: can we predict which tokens are 'easy' (low compute needed)
vs 'hard' (full compute needed) from signals that are free from the
forward pass — WITHOUT running the full logit computation?

Definitions:
    Easy token:  teacher's top-1 logit >> top-2. The model is committed
                 to one answer. Low-margin is small, high-margin large.
    Hard token:  top-1 and top-2 close. Multiple plausible continuations.

Signal (label): logit_margin = top1_logit - top2_logit. Larger = easier.

Features (all cheap, available during decode):
    (a) attention_entropy_mean    — free from eager attn weights.
    (b) attention_entropy_max     — most uncertain layer.
    (c) dH_dt                     — change in entropy (stage F signal).
    (d) step_size                 — ||h_t - h_{t-1}|| / ||h_{t-1}||.
    (e) hidden_norm               — ||final hidden|| (energy proxy).
    (f) centeredness              — distance from calibration final mean.

Pipeline:
    1. Run teacher on several prompts (100+ tokens each).
    2. At each step collect all features + the margin label.
    3. Compute Pearson correlation between each feature and margin.
    4. Fit a simple linear regressor of margin on combined features.
    5. Report which signals actually predict difficulty.

If any single signal has r > 0.3 with margin, that's a usable routing
signal. If combined signals give r > 0.6, routing is genuinely
predictable at inference time.
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


def run_and_collect(model, tokenizer, prompt, max_new_tokens, device, cal_mean):
    """Generate max_new_tokens, per step collect:
       - mean/max attention entropy (normalized) across layers
       - final hidden norm
       - step_size (||h_t - h_{t-1}|| / ||h_{t-1}||)
       - centeredness (||h_t - cal_mean||)
       - teacher's top-1 logit, top-2 logit (logit margin)
       - (after: dH/dt is post-processing on collected entropy)
    """
    records = []
    # Track per-layer attention weights via post-hook
    last_entropy_per_layer = {}
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
                last_entropy_per_layer[li] = 0.0
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
            last_entropy_per_layer[li] = float((ent.mean() / math.log(T)).item())
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prev_hidden = out.hidden_states[-1][0, -1].to(torch.float32).cpu()
        logits = out.logits[:, -1, :].float()
        next_token = logits.argmax(dim=-1, keepdim=True)

        for step in range(max_new_tokens - 1):
            entropies = [last_entropy_per_layer.get(li, 0.0)
                         for li in range(len(model.model.layers))]
            # Pre-compute features BEFORE the forward for this step uses them:
            # actually we captured from the PREVIOUS forward's attention; that's the
            # signal available to route THIS step. So the record pairs (signal at t-1 → label at t).
            # But we record the LABEL from this step's new logits.

            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            cur_hidden = out.hidden_states[-1][0, -1].to(torch.float32).cpu()

            # Top-2 logits for margin
            top2 = logits.topk(2, dim=-1)
            margin = float((top2.values[0, 0] - top2.values[0, 1]).item())
            top1_id = int(top2.indices[0, 0].item())
            # Full-distribution uncertainty: output entropy and log-prob of top-1
            probs = F.softmax(logits[0], dim=-1)
            output_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())
            log_p_top1 = float(torch.log(probs[top1_id].clamp_min(1e-12)).item())

            # Features
            h_mean = sum(entropies) / len(entropies)
            h_max = max(entropies)
            h_min = min(entropies)
            h_norm = float(cur_hidden.norm().item())
            step_size = float((cur_hidden - prev_hidden).norm().item() /
                              max(prev_hidden.norm().item(), 1e-8))
            centeredness = float((cur_hidden - cal_mean).norm().item())

            # Per-layer entropy (store first and last for separate analysis)
            first_layer_h = entropies[0] if entropies else 0.0
            last_layer_h = entropies[-1] if entropies else 0.0
            # Count of "sharp" heads across last layer attention (proxy for attention concentration)
            records.append({
                "step": step,
                "top1_id": top1_id,
                # Labels
                "logit_margin": margin,
                "output_entropy": output_entropy,
                "log_p_top1": log_p_top1,
                # Features
                "attn_entropy_mean": h_mean,
                "attn_entropy_max": h_max,
                "attn_entropy_min": h_min,
                "attn_entropy_first_layer": first_layer_h,
                "attn_entropy_last_layer": last_layer_h,
                "hidden_norm": h_norm,
                "step_size": step_size,
                "centeredness": centeredness,
            })

            prev_hidden = cur_hidden
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


def linear_regression(X, y):
    """Simple closed-form least squares: y = X beta + residual.
    X: [N, F] includes a constant column for intercept.
    Returns (beta, r_squared)."""
    # beta = (X^T X)^-1 X^T y
    n, f = X.shape
    XTX = X.T @ X
    XTy = X.T @ y
    beta = torch.linalg.solve(XTX.to(torch.float64), XTy.to(torch.float64)).to(torch.float32)
    y_pred = X @ beta
    ss_res = ((y - y_pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r_squared = 1 - ss_res / max(ss_tot, 1e-12)
    return beta, r_squared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage24_easy_token_classifier.json")
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

    print(f"\n=== loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    # Calibration: collect final-hidden mean from a short prefix
    prefix = "The cell is the basic structural unit of life."
    with torch.inference_mode():
        ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        cal_mean = out.hidden_states[-1][0].to(torch.float32).cpu().mean(dim=0)

    all_records = []
    for prompt in PROMPTS:
        print(f"\n-- {prompt!r} --", flush=True)
        recs = run_and_collect(model, tokenizer, prompt, args.max_new_tokens,
                                device, cal_mean)
        # Add dH/dt as post-processing
        for i, r in enumerate(recs):
            r["prompt_id"] = prompt[:30]
            r["dH_dt"] = (recs[i]["attn_entropy_mean"] -
                          recs[i-1]["attn_entropy_mean"]) if i > 0 else 0.0
            all_records.append(r)

    print(f"\n=== collected {len(all_records)} (step, features, label) records ===")

    # Multiple labels: logit_margin, output_entropy, log_p_top1
    labels = ["logit_margin", "output_entropy", "log_p_top1"]
    features_to_test = [
        "attn_entropy_mean", "attn_entropy_max", "attn_entropy_min",
        "attn_entropy_first_layer", "attn_entropy_last_layer",
        "dH_dt", "hidden_norm", "step_size", "centeredness",
    ]

    print(f"\n=== Pearson r (feature, each label) ===")
    print(f"  {'feature':>28}  {'margin':>8}  {'out_ent':>8}  {'log_p':>8}")
    per_feature_per_label = {f: {} for f in features_to_test}
    for f in features_to_test:
        xs = [r[f] for r in all_records]
        rs = []
        for lab in labels:
            y = [r[lab] for r in all_records]
            rv = pearson(xs, y)
            per_feature_per_label[f][lab] = rv
            rs.append(rv)
        print(f"  {f:>28}  {rs[0]:+.3f}   {rs[1]:+.3f}   {rs[2]:+.3f}")

    # Use output_entropy as primary label — larger is harder
    y = [r["output_entropy"] for r in all_records]

    # Linear combination against each label
    print(f"\n=== linear regression R² (all features combined) ===")
    r2_per_label = {}
    for lab in labels:
        X = torch.tensor([[r[f] for f in features_to_test] + [1.0] for r in all_records],
                          dtype=torch.float32)
        y_t = torch.tensor([r[lab] for r in all_records], dtype=torch.float32)
        _, r2 = linear_regression(X, y_t)
        r2_per_label[lab] = r2
        print(f"  {lab:>24}  R² = {r2:.3f}")
    # Keep the classic margin-based classification output for legacy comparability
    y = [r["logit_margin"] for r in all_records]

    # Classification-style: can the signals split EASY (top 30% margin) from HARD (bottom 30%)?
    sorted_margins = sorted(y)
    easy_threshold = sorted_margins[int(0.7 * len(sorted_margins))]
    hard_threshold = sorted_margins[int(0.3 * len(sorted_margins))]
    easy_records = [r for r in all_records if r["logit_margin"] >= easy_threshold]
    hard_records = [r for r in all_records if r["logit_margin"] <= hard_threshold]
    print(f"\n=== classification: easy (margin >= {easy_threshold:.2f}, n={len(easy_records)}) vs hard (margin <= {hard_threshold:.2f}, n={len(hard_records)}) ===")
    print(f"  {'feature':>24}  {'easy_mean':>10}  {'hard_mean':>10}  {'ratio':>8}")
    for f in features_to_test:
        em = sum(r[f] for r in easy_records) / len(easy_records)
        hm = sum(r[f] for r in hard_records) / len(hard_records)
        ratio = em / hm if hm != 0 else float("nan")
        print(f"  {f:>24}  {em:>10.4f}  {hm:>10.4f}  {ratio:>8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_records": len(all_records),
            "labels": labels,
            "per_feature_per_label_pearson": per_feature_per_label,
            "linear_regression_r_squared_per_label": r2_per_label,
            "easy_threshold": easy_threshold,
            "hard_threshold": hard_threshold,
            "records": all_records,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
