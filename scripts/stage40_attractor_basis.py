"""
Stage 40 — Attractor-basis routing test.

Stage 39 found per-layer routing R² varies: high at late layers (0.95 at
layer 27), low at early layers (0.09 at layer 0). That's a hybrid signal.

Unification candidate: there's one manifold — the final-layer attractor. Every
layer's hidden state is at a different rotation phase relative to it, but
routing should be geometric with respect to the *same* attractor at every
layer. Early-layer low R² was an artifact of using local PCA basis which
captures rotation dynamics, not attractor position.

Protocol:
  1. Reuse calibration: per-layer hidden states + intermediate activations.
  2. Build attractor bases from last layer hidden state PCA at several ranks.
  3. For each layer, project its hidden through the attractor basis, fit
     linear regression to its intermediate activation, measure R².
  4. Compare: per-layer local basis R² (stage 39) vs universal attractor R².

If R² is high and uniform across layers at some attractor rank: unification
is real, routing is geometric relative to the attractor.
If R² is still low at early layers: the attractor framing needs refinement.
"""

import argparse
import json
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
    """Collect per-MLP pre-input hidden + intermediate activation."""
    mlps = []
    for name, mod in model.named_modules():
        if name.endswith(".mlp") and hasattr(mod, "gate_proj"):
            mlps.append((name, mod))
    L = len(mlps)

    pre_hidden = [[] for _ in range(L)]
    gate_outs = [[] for _ in range(L)]
    up_outs = [[] for _ in range(L)]

    handles = []
    for i, (_, mlp) in enumerate(mlps):
        def make_mlp_hook(idx):
            def h(mod, inputs, output):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32).cpu()
                pre_hidden[idx].append(x)
            return h
        def make_g_hook(idx):
            def h(mod, inputs, output):
                y = output.detach().reshape(-1, output.shape[-1]).to(torch.float32).cpu()
                gate_outs[idx].append(y)
            return h
        def make_u_hook(idx):
            def h(mod, inputs, output):
                y = output.detach().reshape(-1, output.shape[-1]).to(torch.float32).cpu()
                up_outs[idx].append(y)
            return h
        handles.append(mlp.register_forward_hook(make_mlp_hook(i)))
        handles.append(mlp.gate_proj.register_forward_hook(make_g_hook(i)))
        handles.append(mlp.up_proj.register_forward_hook(make_u_hook(i)))

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

    # use a common random subsample index per layer so samples align across layers
    # (assumes same token count across texts, which holds since we feed the same prompts)
    # Simpler: subsample per-layer independently — fine since we evaluate per-layer.
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
    """Top-k principal directions of H (mean-centered). Returns (P, mean)."""
    mu = H.mean(dim=0, keepdim=True)
    Hc = H - mu
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1])
    return P, mu.squeeze(0)


def linear_regression_r2(X, Y, train_frac=0.7):
    """Ridge regression, return test-set R²."""
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
    ss_res = ((Y[te] - Yhat) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean(dim=0, keepdim=True)) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def topk_overlap(X, Y, k_overlap, train_frac=0.7):
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
    Y_abs = Y[te].abs()
    Yhat_abs = Yhat.abs()
    topk_true = Y_abs.topk(k_overlap, dim=1).indices
    topk_pred = Yhat_abs.topk(k_overlap, dim=1).indices
    ovs = []
    for i in range(len(te)):
        t = set(topk_true[i].tolist()); p = set(topk_pred[i].tolist())
        ovs.append(len(t & p) / k_overlap)
    return sum(ovs) / len(ovs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--attractor-ranks", default="16,32,64,128")
    p.add_argument("--topk-frac", type=float, default=0.10)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage40_attractor_basis.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)

    print(f"\n=== calibrating ===")
    t0 = time.perf_counter()
    layer_data = collect_layer_data(model, tokenizer, CALIB_TEXTS, device)
    L = len(layer_data)
    d_int = layer_data[0]["int_act"].shape[1]
    d_hidden = layer_data[0]["H"].shape[1]
    topk_count = max(1, int(args.topk_frac * d_int))
    print(f"  L={L}  d_hidden={d_hidden}  d_int={d_int}  topk={topk_count}  "
          f"({time.perf_counter()-t0:.1f}s)")

    # Attractor basis = PCA of LAST layer hidden state
    H_last = layer_data[-1]["H"]
    ranks = [int(x) for x in args.attractor_ranks.split(",")]

    results = []
    for k in ranks:
        P, mu = pca_basis(H_last, k)
        print(f"\n=== attractor basis rank {k} ===")
        r2_per_layer = []
        topk_per_layer = []
        for i in range(L):
            H = layer_data[i]["H"]
            Y = layer_data[i]["int_act"]
            X = (H - mu) @ P                # project through attractor basis
            r2 = linear_regression_r2(X, Y)
            tk = topk_overlap(X, Y, topk_count)
            r2_per_layer.append(r2)
            topk_per_layer.append(tk)
        mean_r2 = sum(r2_per_layer) / L
        mean_tk = sum(topk_per_layer) / L
        min_r2 = min(r2_per_layer)
        min_tk = min(topk_per_layer)
        print(f"  mean R²={mean_r2:.3f}  min R²={min_r2:.3f}  "
              f"mean topk={mean_tk:.3f}  min topk={min_tk:.3f}")

        # Print per-layer in compact 4-column layout
        print(f"  per-layer (layer | R² | topk):")
        for i in range(L):
            extra = "  <-- early" if i < 5 else ("  <-- late" if i >= L-3 else "")
            print(f"    {i:>2}  R²={r2_per_layer[i]:>6.3f}  topk={topk_per_layer[i]:>5.3f}{extra}")

        results.append({
            "attractor_rank": k,
            "mean_r2": mean_r2, "min_r2": min_r2,
            "mean_topk": mean_tk, "min_topk": min_tk,
            "per_layer_r2": r2_per_layer,
            "per_layer_topk": topk_per_layer,
        })

    print(f"\n=== summary ===")
    print(f"  d_hidden={d_hidden}, d_int={d_int}, L={L}")
    print(f"  {'rank':>5}  {'mean R²':>8}  {'min R²':>8}  {'mean topk':>9}  {'min topk':>8}")
    for r in results:
        print(f"  {r['attractor_rank']:>5}  "
              f"{r['mean_r2']:>8.3f}  {r['min_r2']:>8.3f}  "
              f"{r['mean_topk']:>9.3f}  {r['min_topk']:>8.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L, "d_hidden": d_hidden, "d_int": d_int,
            "topk_count": topk_count,
            "attractor": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
