"""
Stage 31 — Expanded LOPO validation + per-category analysis.

Stage 30 revealed LOPO R² is about half the random-split R² on 6
prompts. That test had only 6 folds, so LOPO estimates were noisy.
This stage uses 35 prompts across four categories:

  factual    — short, one-right-answer queries (9 prompts)
  reasoning  — multi-step logic / chains of inference (9 prompts)
  free_form  — open-ended creative / summary (9 prompts)
  ambiguous  — multiple plausible continuations (8 prompts)

On 35 × 150 = ~5250 records, we compute:

  (1) Random 80/20 R² (for comparison to stage-29 numbers)
  (2) Leave-one-prompt-out CV (all 35 folds)
  (3) Leave-one-CATEGORY-out CV (hold out all prompts of one category)
  (4) h_final PCA-64 LOPO ceiling
  (5) Per-category mean R² breakdown — does the reasoning-prompt
      failure from stage 30 hold up with more reasoning prompts?

Updates the honest baseline for the routing question.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage29_structural_features import (
    CALIB_TEXTS, SUMMARY_FEATURES, CURVATURE_FEATURES, QUANTUM_FEATURES,
    STRUCTURAL_FEATURES, collect_calibration, collect,
)

ALL_FEATURES = SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES + STRUCTURAL_FEATURES


# Expanded prompt set, categorized.
PROMPTS = [
    # --- factual (short, committed answers) ---
    ("factual", "The capital of France is"),
    ("factual", "The chemical symbol for gold is"),
    ("factual", "The largest planet in our solar system is"),
    ("factual", "The first president of the United States was"),
    ("factual", "The author of Romeo and Juliet is"),
    ("factual", "The speed of light in vacuum is approximately"),
    ("factual", "The tallest mountain on Earth is"),
    ("factual", "The capital of Japan is"),
    ("factual", "The chemical formula for water is"),

    # --- reasoning (multi-step logic) ---
    ("reasoning", "If all birds have feathers and penguins are birds, then"),
    ("reasoning", "To solve a quadratic equation we use the formula"),
    ("reasoning", "If a triangle has angles 30 and 60 degrees, the third angle is"),
    ("reasoning", "Given that x + 5 = 12, the value of x is"),
    ("reasoning", "If inflation is 3% and a loaf costs $4 today, next year it will cost approximately"),
    ("reasoning", "A train traveling 60 mph for 2 hours covers a distance of"),
    ("reasoning", "In a fair coin flip, the probability of getting two heads in a row is"),
    ("reasoning", "If a car accelerates from 0 to 60 mph in 6 seconds, its average acceleration is"),
    ("reasoning", "The derivative of x squared with respect to x is"),

    # --- free-form (open-ended generation) ---
    ("free_form", "Write a short poem about the ocean:"),
    ("free_form", "Tell me a story about a brave knight who"),
    ("free_form", "Describe a day in the life of"),
    ("free_form", "Write a letter to a friend about"),
    ("free_form", "The discovery that inference accelerates with context is"),
    ("free_form", "Explain the significance of the Renaissance in"),
    ("free_form", "Imagine a future where robots"),
    ("free_form", "Tell me something interesting about the solar system"),
    ("free_form", "Write an introduction to a research paper on"),

    # --- ambiguous (many plausible continuations) ---
    ("ambiguous", "She walked into the room and saw"),
    ("ambiguous", "The old man looked at his watch and"),
    ("ambiguous", "Water was pouring through the"),
    ("ambiguous", "The door creaked open as"),
    ("ambiguous", "It was the best of"),
    ("ambiguous", "Suddenly, a loud noise came from"),
    ("ambiguous", "In the middle of the forest, there stood"),
    ("ambiguous", "The meaning of life is"),
]


def linear_regression_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    f = X_train.shape[1]
    XtX = X_train.T @ X_train + ridge * torch.eye(f, dtype=X_train.dtype)
    Xty = X_train.T @ y_train
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_test @ beta
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def mlp_fit_r2(X_train, y_train, X_test, y_test, hidden=16, epochs=300, lr=1e-2):
    f = X_train.shape[1]
    net = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(), nn.Linear(hidden, 1))
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


def add_intercept(X):
    return torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)


def normalize(X_tr, X_te):
    mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp_min(1e-8)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def pca_basis(X, k):
    mean = X.mean(dim=0); Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32), mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--device", default=None)
    p.add_argument("--h-pca-k", type=int, default=64)
    p.add_argument("--out", default="results/stage31_expanded_lopo.json")
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
    print(f"device={device}  {len(PROMPTS)} prompts")

    print(f"\n=== loading ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()

    print(f"=== calibration ===")
    calib_hidden = collect_calibration(model, tokenizer, CALIB_TEXTS, device)
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample); pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())

    print(f"\n=== collecting generation records for {len(PROMPTS)} prompts ===")
    all_records = []
    prompt_ids = []
    categories = []
    t0 = time.perf_counter()
    for pid, (cat, prompt) in enumerate(PROMPTS):
        t_p = time.perf_counter()
        recs = collect(model, tokenizer, prompt, args.max_new_tokens,
                        device, calib_hidden, kde_sigma, knn_k=10)
        all_records.extend(recs)
        prompt_ids.extend([pid] * len(recs))
        categories.extend([cat] * len(recs))
        print(f"  [{pid:2d}/{len(PROMPTS)}] ({cat:>10}) {prompt[:40]!r:>45}  "
              f"({len(recs)} records, {time.perf_counter()-t_p:.1f}s)", flush=True)
    N = len(all_records)
    print(f"  total: {N} records in {time.perf_counter()-t0:.1f}s")

    # Build feature matrices
    X_summary = torch.tensor([[r[f] for f in SUMMARY_FEATURES] for r in all_records], dtype=torch.float32)
    X_sc = torch.tensor([[r[f] for f in SUMMARY_FEATURES + CURVATURE_FEATURES] for r in all_records], dtype=torch.float32)
    X_scq = torch.tensor([[r[f] for f in SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES] for r in all_records], dtype=torch.float32)
    X_all = torch.tensor([[r[f] for f in ALL_FEATURES] for r in all_records], dtype=torch.float32)
    X_hfinal = torch.tensor([r["h_final"] for r in all_records], dtype=torch.float32)
    y = torch.tensor([r["output_entropy"] for r in all_records], dtype=torch.float32)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)

    feature_sets = {
        "summary (17)": X_summary,
        "+curvature (28)": X_sc,
        "+quantum (36)": X_scq,
        "+structural (47)": X_all,
    }

    # ===== Random 80/20 baseline (for comparison) =====
    print(f"\n=== (1) random 80/20 baseline ===")
    torch.manual_seed(0)
    perm = torch.randperm(N)
    tr = perm[:int(0.8 * N)]; te = perm[int(0.8 * N):]
    random_r2 = {}
    for name, X_ in feature_sets.items():
        Xn_tr, Xn_te = normalize(X_[tr], X_[te])
        lin = linear_regression_r2(add_intercept(Xn_tr), y[tr], add_intercept(Xn_te), y[te])
        random_r2[name] = lin
        print(f"  {name:<22}  linear R² = {lin:.3f}")

    # h_final baseline
    P_hf, mu_hf = pca_basis(X_hfinal[tr], args.h_pca_k)
    Xh_tr = (X_hfinal[tr] - mu_hf) @ P_hf
    Xh_te = (X_hfinal[te] - mu_hf) @ P_hf
    lin_hf = linear_regression_r2(add_intercept(Xh_tr), y[tr], add_intercept(Xh_te), y[te])
    random_r2["h_final PCA-64"] = lin_hf
    print(f"  {'h_final PCA-64':<22}  linear R² = {lin_hf:.3f}")

    # ===== LOPO (leave-one-prompt-out) =====
    print(f"\n=== (2) LOPO linear R² (mean across {len(PROMPTS)} folds) ===")
    lopo_r2 = {name: [] for name in list(feature_sets.keys()) + ["h_final PCA-64"]}
    lopo_per_prompt_linear = []
    for held in range(len(PROMPTS)):
        tr_mask = prompt_ids_t != held
        te_mask = prompt_ids_t == held
        if te_mask.sum() < 10:
            continue
        fold_row = {"prompt_id": held, "category": PROMPTS[held][0],
                     "prompt": PROMPTS[held][1][:40], "r2_per_set": {}}
        for name, X_ in feature_sets.items():
            Xn_tr, Xn_te = normalize(X_[tr_mask], X_[te_mask])
            r2 = linear_regression_r2(add_intercept(Xn_tr), y[tr_mask],
                                       add_intercept(Xn_te), y[te_mask])
            lopo_r2[name].append(r2)
            fold_row["r2_per_set"][name] = r2
        # h_final ceiling too
        P_hf, mu_hf = pca_basis(X_hfinal[tr_mask], args.h_pca_k)
        Xh_tr_fold = (X_hfinal[tr_mask] - mu_hf) @ P_hf
        Xh_te_fold = (X_hfinal[te_mask] - mu_hf) @ P_hf
        r2_hf_fold = linear_regression_r2(add_intercept(Xh_tr_fold), y[tr_mask],
                                           add_intercept(Xh_te_fold), y[te_mask])
        lopo_r2["h_final PCA-64"].append(r2_hf_fold)
        fold_row["r2_per_set"]["h_final PCA-64"] = r2_hf_fold
        lopo_per_prompt_linear.append(fold_row)

    lopo_mean = {name: sum(v)/len(v) for name, v in lopo_r2.items()}
    for name in list(feature_sets.keys()) + ["h_final PCA-64"]:
        print(f"  {name:<22}  LOPO linear R² = {lopo_mean[name]:+.3f}")

    # ===== Per-category mean R² (using +structural) =====
    print(f"\n=== (3) per-category mean LOPO linear R² (+structural features) ===")
    cat_groups = {}
    for row in lopo_per_prompt_linear:
        cat_groups.setdefault(row["category"], []).append(row["r2_per_set"]["+structural (47)"])
    for cat, rs in cat_groups.items():
        mean_r = sum(rs) / len(rs)
        print(f"  {cat:>12}  mean R² = {mean_r:+.3f}  (n={len(rs)})")

    # ===== Leave-one-category-out (4 folds) =====
    print(f"\n=== (4) leave-one-CATEGORY-out linear R² (structural features) ===")
    categories_t = [row["category"] for row in lopo_per_prompt_linear[:0]]
    categories_t = [c for c in ["factual", "reasoning", "free_form", "ambiguous"]]
    # Remap using stored categories list aligned with records
    cat_per_record = torch.tensor([
        ["factual", "reasoning", "free_form", "ambiguous"].index(c)
        for c in categories
    ])
    for ci, cat in enumerate(categories_t):
        te_mask = cat_per_record == ci
        tr_mask = ~te_mask
        if te_mask.sum() < 20:
            continue
        Xn_tr, Xn_te = normalize(X_all[tr_mask], X_all[te_mask])
        r2 = linear_regression_r2(add_intercept(Xn_tr), y[tr_mask],
                                   add_intercept(Xn_te), y[te_mask])
        print(f"  held out {cat:>12}: R² = {r2:+.3f}")

    # ===== Interpretation =====
    print(f"\n=== interpretation ===")
    structural_random = random_r2["+structural (47)"]
    structural_lopo = lopo_mean["+structural (47)"]
    hf_random = random_r2["h_final PCA-64"]
    hf_lopo = lopo_mean["h_final PCA-64"]

    print(f"  +structural (47) features:")
    print(f"    random:  {structural_random:.3f}")
    print(f"    LOPO:    {structural_lopo:.3f}")
    print(f"    overfit gap: {structural_random - structural_lopo:.3f}")
    print()
    print(f"  h_final PCA-64 baseline:")
    print(f"    random:  {hf_random:.3f}")
    print(f"    LOPO:    {hf_lopo:.3f}")
    print(f"    overfit gap: {hf_random - hf_lopo:.3f}")
    print()
    print(f"  LOPO coverage: {structural_lopo / max(hf_lopo, 1e-6):.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "n_records": N, "n_prompts": len(PROMPTS),
            "random_r2": random_r2, "lopo_mean_r2": lopo_mean,
            "lopo_per_prompt": lopo_per_prompt_linear,
            "category_means": {c: sum(v)/len(v) for c, v in cat_groups.items()},
            "prompts": [{"id": i, "cat": c, "prompt": p} for i, (c, p) in enumerate(PROMPTS)],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
