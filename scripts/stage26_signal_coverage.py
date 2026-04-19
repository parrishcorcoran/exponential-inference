"""
Stage 26 — How much of the full picture are our summary signals capturing?

Output entropy is a deterministic function of the final hidden state
(via lm_head). So:

    R²_theoretical = 1.0  (if we could model the full nonlinear map)
    R²_linear_ceiling = best linear regression of output_entropy on h_final
                        (computed via PCA-truncated ridge to avoid overfit)
    R²_our_signals = best linear regression of output_entropy on our
                     14 summary signals (holdout-validated)

The ratio R²_our_signals / R²_linear_ceiling tells us what fraction of
the linearly-capturable uncertainty our summary signals already
express — i.e., what we're getting for free from cheap runtime signals
vs what the full hidden state (expensive to compute cheaply) contains.

For non-linear capture: we also fit a small MLP on the summary signals
to see how much linear regression is leaving on the table.

Output: three R² numbers, the ratio, and a conclusion about coverage.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROMPTS = [
    "The discovery that inference accelerates with context is",
    "The capital of France is",
    "To solve a quadratic equation we use the formula",
    "Tell me something interesting about the solar system",
    "Write a poem about cheese:",
    "If all birds have feathers and penguins are birds, then",
]


SUMMARY_FEATURES = [
    "H_last_layer", "H_first_layer", "H_q1_layer", "H_mid_layer", "H_q3_layer",
    "H_max", "H_var",
    "heads_above_0p9", "max_head_sharpness",
    "hidden_norm_final", "hidden_norm_mid", "hidden_norm_first",
    "centeredness", "total_layer_update", "max_layer_update",
    "dH_dt_mean", "d_hidden_norm_dt",
]


def collect(model, tokenizer, prompt, max_new_tokens, device, cal_mean):
    """Return list of dicts with summary signals + full final hidden state."""
    per_layer_H = {}
    per_layer_head_sharp = {}

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
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)
            ent_norm = (ent / math.log(T)).cpu()
            per_layer_H[li] = float(ent_norm.mean().item())
            per_layer_head_sharp[li] = [float(1 - x) for x in ent_norm.tolist()]
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
        next_token = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)

        prev_H_mean = None
        prev_hidden_norm = None

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
            h_first = hidden_states[0][0, -1].to(torch.float32).cpu()
            h_last = hidden_states[-1][0, -1].to(torch.float32).cpu()
            h_mid = hidden_states[n_layers // 2][0, -1].to(torch.float32).cpu()

            layer_updates = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                layer_updates.append((h_ip1 - h_i).norm().item())

            H_mean = sum(entropies) / len(entropies)
            H_var = (sum((e - H_mean) ** 2 for e in entropies) / len(entropies))
            heads_above_0p9 = sum(1 for s in all_heads if s > 0.9)
            max_head_sharp = max(all_heads) if all_heads else 0.0

            hn_last = float(h_last.norm().item())
            hn_mid = float(h_mid.norm().item())
            hn_first = float(h_first.norm().item())
            cent = float((h_last - cal_mean).norm().item())
            total_upd = sum(layer_updates)
            max_upd = max(layer_updates)
            dH = (H_mean - prev_H_mean) if prev_H_mean is not None else 0.0
            d_hn = (hn_last - prev_hidden_norm) if prev_hidden_norm is not None else 0.0

            probs = F.softmax(logits[0], dim=-1)
            output_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

            records.append({
                # Summary signals
                "H_last_layer": entropies[-1],
                "H_first_layer": entropies[0],
                "H_q1_layer": entropies[n_layers // 4],
                "H_mid_layer": entropies[n_layers // 2],
                "H_q3_layer": entropies[(3 * n_layers) // 4],
                "H_max": max(entropies),
                "H_var": H_var,
                "heads_above_0p9": float(heads_above_0p9),
                "max_head_sharpness": max_head_sharp,
                "hidden_norm_final": hn_last,
                "hidden_norm_mid": hn_mid,
                "hidden_norm_first": hn_first,
                "centeredness": cent,
                "total_layer_update": total_upd,
                "max_layer_update": max_upd,
                "dH_dt_mean": dH,
                "d_hidden_norm_dt": d_hn,
                # Full hidden state (for ceiling computation)
                "h_final": h_last.tolist(),
                # Label
                "output_entropy": output_entropy,
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


def linear_regression_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    """Fit linear regression on train, evaluate R² on test. Uses ridge for
    stability (especially when X is high-dim vs samples)."""
    n, f = X_train.shape
    XtX = X_train.T @ X_train + ridge * torch.eye(f, dtype=X_train.dtype)
    Xty = X_train.T @ y_train
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_test @ beta
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def mlp_fit_r2(X_train, y_train, X_test, y_test, hidden=64, epochs=500, lr=1e-2):
    """Small 2-layer MLP, train + eval."""
    n, f = X_train.shape
    net = nn.Sequential(
        nn.Linear(f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y_t = y_train.view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(X_train)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        y_pred = net(X_test).view(-1)
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32), mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--device", default=None)
    p.add_argument("--h-pca-k", type=int, default=64,
                   help="PCA rank for h_final baseline (to avoid overfit on 1024-dim)")
    p.add_argument("--out", default="results/stage26_signal_coverage.json")
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

    prefix = "The cell is the basic structural unit of life."
    with torch.inference_mode():
        ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        cal_mean = out.hidden_states[-1][0].to(torch.float32).cpu().mean(dim=0)

    all_records = []
    for prompt in PROMPTS:
        print(f"  collecting from {prompt!r}", flush=True)
        all_records.extend(collect(model, tokenizer, prompt, args.max_new_tokens,
                                    device, cal_mean))
    N = len(all_records)
    print(f"\n=== collected {N} records ===")

    # Build matrices
    X_summary = torch.tensor([[r[f] for f in SUMMARY_FEATURES] for r in all_records],
                              dtype=torch.float32)
    X_hfinal = torch.tensor([r["h_final"] for r in all_records], dtype=torch.float32)
    y = torch.tensor([r["output_entropy"] for r in all_records], dtype=torch.float32)

    # 80/20 split
    torch.manual_seed(0)
    perm = torch.randperm(N)
    n_train = int(0.8 * N)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    Xs_tr = X_summary[train_idx]; Xs_te = X_summary[test_idx]
    Xh_tr = X_hfinal[train_idx]; Xh_te = X_hfinal[test_idx]
    y_tr = y[train_idx]; y_te = y[test_idx]

    # Normalize summary features (z-score on train)
    mu_s = Xs_tr.mean(dim=0); sd_s = Xs_tr.std(dim=0).clamp_min(1e-8)
    Xs_tr_n = (Xs_tr - mu_s) / sd_s
    Xs_te_n = (Xs_te - mu_s) / sd_s
    # Add intercept column
    def add_intercept(X):
        return torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)
    Xs_tr_aug = add_intercept(Xs_tr_n)
    Xs_te_aug = add_intercept(Xs_te_n)

    # Reduce h_final to top-k PCA (fit on train only)
    P_hf, mean_hf = pca_basis(Xh_tr, args.h_pca_k)
    Xh_tr_pca = (Xh_tr - mean_hf) @ P_hf
    Xh_te_pca = (Xh_te - mean_hf) @ P_hf
    Xh_tr_aug = add_intercept(Xh_tr_pca)
    Xh_te_aug = add_intercept(Xh_te_pca)

    print(f"\n=== regression results (holdout R²) ===")
    r2_summary = linear_regression_r2(Xs_tr_aug, y_tr, Xs_te_aug, y_te)
    r2_hfinal = linear_regression_r2(Xh_tr_aug, y_tr, Xh_te_aug, y_te)
    r2_mlp_summary = mlp_fit_r2(Xs_tr_n, y_tr, Xs_te_n, y_te)
    r2_mlp_hfinal = mlp_fit_r2(Xh_tr_pca, y_tr, Xh_te_pca, y_te)

    print(f"  linear on summary signals   (n_feat={len(SUMMARY_FEATURES)}):  R² = {r2_summary:.3f}")
    print(f"  linear on h_final PCA-{args.h_pca_k}                         :  R² = {r2_hfinal:.3f}")
    print(f"  MLP on summary signals                                        :  R² = {r2_mlp_summary:.3f}")
    print(f"  MLP on h_final PCA-{args.h_pca_k}                            :  R² = {r2_mlp_hfinal:.3f}")

    ratio_linear = r2_summary / max(r2_hfinal, 1e-6)
    ratio_mlp = r2_mlp_summary / max(r2_mlp_hfinal, 1e-6)

    print(f"\n=== coverage ratio ===")
    print(f"  summary / h_final (linear):  {ratio_linear:.1%}")
    print(f"  summary / h_final (MLP):     {ratio_mlp:.1%}")
    print(f"\n  Interpretation: our 17 summary signals capture about")
    print(f"  {ratio_mlp:.0%} of what the (top-{args.h_pca_k} PCA of the) full hidden")
    print(f"  state linearly/nonlinearly says about output entropy.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_records": N,
            "n_summary_features": len(SUMMARY_FEATURES),
            "h_pca_k": args.h_pca_k,
            "r2_linear_summary": r2_summary,
            "r2_linear_hfinal_pca": r2_hfinal,
            "r2_mlp_summary": r2_mlp_summary,
            "r2_mlp_hfinal_pca": r2_mlp_hfinal,
            "coverage_ratio_linear": ratio_linear,
            "coverage_ratio_mlp": ratio_mlp,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
