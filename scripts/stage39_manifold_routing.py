"""
Stage 39 — Manifold-routing diagnostic.

Claim: in an MoE, the router's decision (which expert to activate) is
predictable from the token's manifold coordinates at that layer. Routing is
geometric, not a learned function.

We don't have an MoE model to probe directly. We test the equivalent claim on
a dense model: the MLP's intermediate activation pattern (which "slices" of
the intermediate dim are active) is predictable from the token's manifold
coordinates. If yes, the same principle extends to MoE routing.

Protocol on Qwen3-0.6B:
  1. Forward on calibration texts.
  2. Per layer, collect: hidden states h (pre-MLP) and intermediate
     activations int_act = silu(gate) * up [..., d_intermediate].
  3. Build per-layer manifold basis P from hidden-state PCA (rank = ceil
     of TwoNN dim, then small multiples).
  4. Project h onto manifold: h_m = h @ P  [..., k].
  5. Fit linear regression h_m -> int_act.
  6. Report per-layer R². Also: prediction accuracy of "is dim i in top-K
     intermediates" (treating each dim as a binary classifier).

High R²  ==> geometric routing works. Routing is a linear function of
manifold coords.
Low R²   ==> routing is more than geometric. Manifold isn't enough.
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


CALIB_TEXTS = [
    "The cell is the basic structural unit of life, composed of cytoplasm enclosed within a membrane.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
    "Neural networks consist of parameterized layers trained by gradient descent to approximate functions.",
    "Plate tectonics describes the slow movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into complex three-dimensional structures determined by their amino acid sequences.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions.",
    "Evolution by natural selection operates on heritable variation in populations.",
    "Cryptography protects information using mathematical operations that are easy to compute.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges across many practical applications.",
    "Black holes are regions of spacetime from which nothing, not even light, can escape.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Bayesian inference updates a prior probability distribution using observed data.",
    "The immune system recognizes pathogens through pattern recognition receptors.",
    "The Riemann zeta function encodes deep information about the distribution of primes.",
]


def twonn_dim(X):
    X = X.to(torch.float32)
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    sorted_d, _ = D.sort(dim=1)
    r1 = sorted_d[:, 0]
    r2 = sorted_d[:, 1]
    mask = (r1 > 1e-8) & (r2 > r1 + 1e-10)
    if mask.sum() < 10:
        return float("nan")
    mu = r2[mask] / r1[mask]
    log_mu = torch.log(mu)
    return float(mask.sum().item() / log_mu.sum().item())


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_layer_data(model, tokenizer, texts, device, max_len=256,
                       max_samples_per_layer=400):
    """For each MLP layer, collect pre-MLP hidden state and intermediate act."""
    mlps = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp") and hasattr(mod, "gate_proj"):
            mlps.append((name, mod))
    L = len(mlps)

    # pre-MLP hidden state is the input to mlp (first arg) AFTER post-attn layernorm
    pre_hidden = [[] for _ in range(L)]
    gate_outs = [[] for _ in range(L)]
    up_outs = [[] for _ in range(L)]

    def make_mlp_hook(i):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            pre_hidden[i].append(x_flat)
        return hook

    def make_gate_hook(i):
        def hook(mod, inputs, output):
            y = output.detach().reshape(-1, output.shape[-1]).to(torch.float32).cpu()
            gate_outs[i].append(y)
        return hook

    def make_up_hook(i):
        def hook(mod, inputs, output):
            y = output.detach().reshape(-1, output.shape[-1]).to(torch.float32).cpu()
            up_outs[i].append(y)
        return hook

    handles = []
    for i, (_, mlp) in enumerate(mlps):
        handles.append(mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(mlp.gate_proj.register_forward_hook(make_gate_hook(i)))
        handles.append(mlp.up_proj.register_forward_hook(make_up_hook(i)))

    try:
        model.eval()
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # concatenate + subsample + compute int_act = silu(gate) * up
    layer_data = []
    for i in range(L):
        H = torch.cat(pre_hidden[i], dim=0)
        G = torch.cat(gate_outs[i], dim=0)
        U = torch.cat(up_outs[i], dim=0)
        N = H.shape[0]
        if N > max_samples_per_layer:
            idx = torch.randperm(N)[:max_samples_per_layer]
            H = H[idx]; G = G[idx]; U = U[idx]
        int_act = F.silu(G) * U
        layer_data.append({"H": H, "int_act": int_act})
    return layer_data


def pca_basis(H, k):
    """Top-k principal directions of H (mean-centered)."""
    Hc = H - H.mean(dim=0, keepdim=True)
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1])
    return P, H.mean(dim=0)


def linear_regression_r2(X, Y, train_frac=0.7):
    """Fit Y ≈ X W + b on train split, compute R² on test split."""
    N = X.shape[0]
    idx = torch.randperm(N)
    n_train = int(train_frac * N)
    tr, te = idx[:n_train], idx[n_train:]
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]

    # Solve closed form: W = (X^T X)^{-1} X^T Y
    Xa_tr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], dim=1)
    Xa_te = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], dim=1)
    # ridge for stability
    lam = 1e-3
    XtX = Xa_tr.T @ Xa_tr + lam * torch.eye(Xa_tr.shape[1])
    W = torch.linalg.solve(XtX, Xa_tr.T @ Ytr)
    Yhat = Xa_te @ W
    ss_res = ((Yte - Yhat) ** 2).sum()
    ss_tot = ((Yte - Yte.mean(dim=0, keepdim=True)) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot).item()
    return r2


def topk_prediction_accuracy(X, Y, k, train_frac=0.7):
    """Predict whether each output dim is in top-k of Y from X.
    Simpler variant: predict Y from X linearly, then check top-k overlap."""
    N = X.shape[0]
    idx = torch.randperm(N)
    n_train = int(train_frac * N)
    tr, te = idx[:n_train], idx[n_train:]
    Xa_tr = torch.cat([X[tr], torch.ones(n_train, 1)], dim=1)
    Xa_te = torch.cat([X[te], torch.ones(len(te), 1)], dim=1)
    lam = 1e-3
    XtX = Xa_tr.T @ Xa_tr + lam * torch.eye(Xa_tr.shape[1])
    W = torch.linalg.solve(XtX, Xa_tr.T @ Y[tr])
    Yhat = Xa_te @ W

    # Use |Y| ranking as target (which intermediate dims are most active)
    Y_abs = Y[te].abs()
    Yhat_abs = Yhat.abs()
    # Top-k overlap (Jaccard over top-k index sets, per-row)
    topk_true = Y_abs.topk(k, dim=1).indices
    topk_pred = Yhat_abs.topk(k, dim=1).indices
    overlaps = []
    for i in range(len(te)):
        true_set = set(topk_true[i].tolist())
        pred_set = set(topk_pred[i].tolist())
        overlap = len(true_set & pred_set) / k
        overlaps.append(overlap)
    return sum(overlaps) / len(overlaps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank-factors", default="1,2,4,8,16",
                   help="per-layer manifold rank = ceil(TwoNN_dim * factor)")
    p.add_argument("--topk-frac", type=float, default=0.10,
                   help="fraction of intermediate dim to treat as 'top-k active'")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage39_manifold_routing.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)

    print(f"\n=== calibrating (hidden + gate + up per layer) ===")
    t0 = time.perf_counter()
    layer_data = collect_layer_data(model, tokenizer, CALIB_TEXTS, device)
    L = len(layer_data)
    d_int = layer_data[0]["int_act"].shape[1]
    d_hidden = layer_data[0]["H"].shape[1]
    top_k_count = max(1, int(args.topk_frac * d_int))
    print(f"  L={L} layers, d_hidden={d_hidden}, d_intermediate={d_int}, "
          f"topk={top_k_count}, took {time.perf_counter()-t0:.1f}s")

    print(f"\n=== per-layer TwoNN dim (on pre-MLP hidden state) ===")
    twonn_dims = []
    for i in range(L):
        d = twonn_dim(layer_data[i]["H"])
        twonn_dims.append(d)
    mean_dim = sum(twonn_dims) / L
    print(f"  mean TwoNN dim: {mean_dim:.2f}")

    rank_factors = [float(x) for x in args.rank_factors.split(",")]
    results = []

    # Also compute full-hidden-state baseline: fit linear regression using
    # the full h as input (rank = d_hidden). This is the "ceiling" — how much
    # linear info is in h at all.
    print(f"\n=== baseline: full hidden state (d_hidden={d_hidden}) ===")
    full_r2 = []
    full_topk = []
    for i in range(L):
        H = layer_data[i]["H"]
        Y = layer_data[i]["int_act"]
        r2 = linear_regression_r2(H, Y)
        tk = topk_prediction_accuracy(H, Y, top_k_count)
        full_r2.append(r2)
        full_topk.append(tk)
    print(f"  mean R²={sum(full_r2)/L:.3f}   mean_topk_overlap={sum(full_topk)/L:.3f}")

    for factor in rank_factors:
        per_r2 = []
        per_topk = []
        per_rank = []
        for i in range(L):
            H = layer_data[i]["H"]
            Y = layer_data[i]["int_act"]
            k_manifold = max(1, int(math.ceil(twonn_dims[i] * factor)))
            per_rank.append(k_manifold)
            P, mu = pca_basis(H, k_manifold)
            Hm = (H - mu) @ P                    # [N, k_manifold]
            r2 = linear_regression_r2(Hm, Y)
            tk = topk_prediction_accuracy(Hm, Y, top_k_count)
            per_r2.append(r2)
            per_topk.append(tk)
        mean_rank = sum(per_rank) / L
        mean_r2 = sum(per_r2) / L
        mean_topk = sum(per_topk) / L
        print(f"  factor={factor:<4} mean_k={mean_rank:>5.1f}  "
              f"R²={mean_r2:>6.3f}  topk_overlap={mean_topk:>5.3f}  "
              f"(ceiling R²={sum(full_r2)/L:.3f}, ceiling_topk={sum(full_topk)/L:.3f})")
        results.append({
            "factor": factor,
            "mean_manifold_rank": mean_rank,
            "mean_r2": mean_r2,
            "mean_topk_overlap": mean_topk,
            "per_layer_r2": per_r2,
            "per_layer_topk": per_topk,
            "per_layer_rank": per_rank,
        })

    print(f"\n=== per-layer detail at factor=2 ===")
    f2 = next(r for r in results if r["factor"] == 2.0) if any(r["factor"] == 2.0 for r in results) else results[0]
    print(f"  {'layer':>5}  {'rank':>4}  {'R²':>6}  {'topk':>5}  {'R²_ceil':>8}")
    for i in range(L):
        print(f"  {i:>5}  {f2['per_layer_rank'][i]:>4}  "
              f"{f2['per_layer_r2'][i]:>6.3f}  {f2['per_layer_topk'][i]:>5.3f}  "
              f"{full_r2[i]:>8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "d_intermediate": d_int,
            "d_hidden": d_hidden,
            "top_k_count": top_k_count,
            "twonn_dims": twonn_dims,
            "mean_twonn_dim": mean_dim,
            "ceiling_full_hidden_r2_mean": sum(full_r2) / L,
            "ceiling_full_hidden_topk_mean": sum(full_topk) / L,
            "factor_results": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
