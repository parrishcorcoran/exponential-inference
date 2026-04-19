"""
Stage 30 — Is our R² real or overfit?

Three validation tests:

(1) Label shuffle: shuffle output_entropy labels randomly, rerun
    regression. R² should drop to ~0. If it stays positive, the
    model is memorizing feature distributions rather than label
    relationships.

(2) Leave-one-prompt-out (LOPO) cross-validation: hold out one entire
    prompt, train on the other 5. Average R² across the 6 folds.
    Compares to random 80/20: if LOPO R² is close to random R², signal
    transfers to unseen prompts. If LOPO is substantially lower, we
    were fitting prompt-specific patterns.

(3) Small-MLP comparison: our stage-29 MLP had ~8500 params on ~715
    training samples. Also test with a tiny MLP (16-16-1, ~800 params)
    which can't memorize as easily. If small MLP gets similar R², we
    weren't overfitting.

Uses same 47 feature set from stage 29 (leakage-free).
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

# Reuse collect() and feature definitions from stage 29
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stage29_structural_features import (
    PROMPTS, CALIB_TEXTS,
    SUMMARY_FEATURES, CURVATURE_FEATURES, QUANTUM_FEATURES, STRUCTURAL_FEATURES,
    collect_calibration, collect,
)

ALL_FEATURES = SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES + STRUCTURAL_FEATURES


def linear_regression_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    f = X_train.shape[1]
    XtX = X_train.T @ X_train + ridge * torch.eye(f, dtype=X_train.dtype)
    Xty = X_train.T @ y_train
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_test @ beta
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12)


def mlp_fit_r2(X_train, y_train, X_test, y_test, hidden=64, epochs=500, lr=1e-2, layers=3):
    f = X_train.shape[1]
    if layers == 3:
        net = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(),
                            nn.Linear(hidden, hidden), nn.ReLU(),
                            nn.Linear(hidden, 1))
    else:
        net = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(),
                            nn.Linear(hidden, 1))
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


def normalize_train_test(X_tr, X_te):
    mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp_min(1e-8)
    return (X_tr - mu) / sd, (X_te - mu) / sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage30_validation.json")
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

    print(f"=== collecting with prompt labels ===")
    all_records = []
    prompt_ids = []
    for pid, prompt in enumerate(PROMPTS):
        recs = collect(model, tokenizer, prompt, args.max_new_tokens,
                        device, calib_hidden, kde_sigma, knn_k=10)
        all_records.extend(recs)
        prompt_ids.extend([pid] * len(recs))
    N = len(all_records)
    print(f"  {N} records, {len(PROMPTS)} prompts")

    X = torch.tensor([[r[f] for f in ALL_FEATURES] for r in all_records], dtype=torch.float32)
    y = torch.tensor([r["output_entropy"] for r in all_records], dtype=torch.float32)
    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)

    results = {"model": args.model, "n_records": N}

    # ===== Test 1: Random 80/20 (the "old" number) =====
    print(f"\n=== (1) Random 80/20 (baseline — the numbers we reported) ===")
    torch.manual_seed(0)
    perm = torch.randperm(N)
    tr = perm[:int(0.8 * N)]; te = perm[int(0.8 * N):]
    Xn_tr, Xn_te = normalize_train_test(X[tr], X[te])
    lin = linear_regression_r2(add_intercept(Xn_tr), y[tr], add_intercept(Xn_te), y[te])
    mlp_big = mlp_fit_r2(Xn_tr, y[tr], Xn_te, y[te], hidden=64)
    mlp_small = mlp_fit_r2(Xn_tr, y[tr], Xn_te, y[te], hidden=16, layers=2)
    print(f"  linear:             R² = {lin:.3f}")
    print(f"  MLP 64-64-1 (big):  R² = {mlp_big:.3f}")
    print(f"  MLP 16-1 (small):   R² = {mlp_small:.3f}")
    results["random_split"] = {"linear": lin, "mlp_big": mlp_big, "mlp_small": mlp_small}

    # ===== Test 2: Label shuffle =====
    print(f"\n=== (2) Label shuffle (sanity — should give R² ≈ 0) ===")
    torch.manual_seed(42)
    y_shuf = y[torch.randperm(N)]
    Xn_tr, Xn_te = normalize_train_test(X[tr], X[te])
    lin_sh = linear_regression_r2(add_intercept(Xn_tr), y_shuf[tr],
                                   add_intercept(Xn_te), y_shuf[te])
    mlp_sh = mlp_fit_r2(Xn_tr, y_shuf[tr], Xn_te, y_shuf[te], hidden=64)
    print(f"  linear:             R² = {lin_sh:.3f}")
    print(f"  MLP 64-64-1:        R² = {mlp_sh:.3f}")
    results["label_shuffle"] = {"linear": lin_sh, "mlp_big": mlp_sh}
    if lin_sh > 0.1 or mlp_sh > 0.1:
        print(f"  ! suspicious: random labels give R² > 0.1 - there may be leakage")

    # ===== Test 3: Leave-one-prompt-out =====
    print(f"\n=== (3) Leave-one-prompt-out CV (hold out entire prompt) ===")
    lopo_lin = []; lopo_mlp_big = []; lopo_mlp_small = []
    for held_out in range(len(PROMPTS)):
        tr_mask = prompt_ids_t != held_out
        te_mask = prompt_ids_t == held_out
        if te_mask.sum() < 10:
            continue
        Xn_tr, Xn_te = normalize_train_test(X[tr_mask], X[te_mask])
        lin_ = linear_regression_r2(add_intercept(Xn_tr), y[tr_mask],
                                     add_intercept(Xn_te), y[te_mask])
        mlp_b = mlp_fit_r2(Xn_tr, y[tr_mask], Xn_te, y[te_mask], hidden=64)
        mlp_s = mlp_fit_r2(Xn_tr, y[tr_mask], Xn_te, y[te_mask], hidden=16, layers=2)
        lopo_lin.append(lin_); lopo_mlp_big.append(mlp_b); lopo_mlp_small.append(mlp_s)
        print(f"  fold {held_out} ({PROMPTS[held_out][:40]!r:>45}): "
              f"lin={lin_:+.3f}  MLP-big={mlp_b:+.3f}  MLP-small={mlp_s:+.3f}")

    lopo_lin_mean = sum(lopo_lin) / len(lopo_lin)
    lopo_mlp_big_mean = sum(lopo_mlp_big) / len(lopo_mlp_big)
    lopo_mlp_small_mean = sum(lopo_mlp_small) / len(lopo_mlp_small)
    print(f"\n  LOPO mean R²:")
    print(f"    linear:            {lopo_lin_mean:+.3f}")
    print(f"    MLP 64-64-1 (big): {lopo_mlp_big_mean:+.3f}")
    print(f"    MLP 16-1 (small):  {lopo_mlp_small_mean:+.3f}")
    results["lopo_cv"] = {
        "linear_mean": lopo_lin_mean, "mlp_big_mean": lopo_mlp_big_mean,
        "mlp_small_mean": lopo_mlp_small_mean,
        "folds_linear": lopo_lin, "folds_mlp_big": lopo_mlp_big,
        "folds_mlp_small": lopo_mlp_small,
    }

    # ===== Interpretation =====
    print(f"\n=== interpretation ===")
    print(f"  Random 80/20 linear R²:    {lin:.3f}")
    print(f"  Label-shuffle linear R²:   {lin_sh:.3f}  ({'OK' if abs(lin_sh) < 0.1 else 'LEAK'})")
    print(f"  LOPO linear R² (true OOD): {lopo_lin_mean:.3f}")
    print(f"  Gap random→LOPO:           {lin - lopo_lin_mean:.3f}")
    print()
    print(f"  Random 80/20 big-MLP R²:   {mlp_big:.3f}")
    print(f"  Random big-MLP (shuffled): {mlp_sh:.3f}  ({'OK' if abs(mlp_sh) < 0.1 else 'LEAK'})")
    print(f"  LOPO big-MLP R²:           {lopo_mlp_big_mean:.3f}")
    print(f"  Gap random→LOPO:           {mlp_big - lopo_mlp_big_mean:.3f}")
    print()
    print(f"  MLP big vs small on random split: {mlp_big:.3f} vs {mlp_small:.3f}")
    print(f"    (if similar: no overfitting from capacity;")
    print(f"     if big >> small: big MLP is memorizing)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
