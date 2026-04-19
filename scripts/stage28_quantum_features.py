"""
Stage 28 — Quantum-state / electron-cloud features on hidden-state ensembles.

Stage 27: summary (17) + curvature (11) = 28 features; 63% coverage
of h_final nonlinear capacity. Gap remaining ~32 pp.

This stage imports physics math we haven't used yet. Hidden states are
formally state-vector-like; per-layer ensembles have density-matrix
structure we can compute cheaply.

New features (all cheap, L² or smaller):

QUANTUM-ANALOG (per-token, over L per-layer hidden vectors):
  (a) purity            = Tr(ρ²), ρ = (1/L) Σ h_i h_i^T / ||h_i||²
                          — how pure vs mixed the per-layer state is.
  (b) von_neumann_entropy = -Σ λ log λ of ρ's eigenvalues.
                          — analog of occupation-number entropy.
  (c) effective_rank    = 1 / Tr(ρ²), inverse participation.
  (d) layer_gram_max_eig = largest eigenvalue of L×L Gram matrix
                          (h_i · h_j) — dominant mode strength.

DENSITY-ANALOG (over calibration cloud):
  (e) kde_log_density   = log Σ_j exp(-||h - c_j||² / 2σ²) with σ set
                          from median calibration-pair distance.
                          — how "typical" is this hidden state.

HIGHER MOMENTS of per-layer update magnitudes:
  (f) upd_skewness      = E[(u - ū)³] / σ³
  (g) upd_kurtosis      = E[(u - ū)⁴] / σ⁴

TRAJECTORY ACTION:
  (h) trajectory_action_5 = Σ ||Δh_t||² over last 5 steps (kinetic-
                            energy-like integral along trajectory).

Combine with stage 27's feature set and re-measure coverage.
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


# From stage 27
SUMMARY_FEATURES = [
    "H_last_layer", "H_first_layer", "H_q1_layer", "H_mid_layer", "H_q3_layer",
    "H_max", "H_var",
    "heads_above_0p9", "max_head_sharpness",
    "hidden_norm_final", "hidden_norm_mid", "hidden_norm_first",
    "centeredness", "total_layer_update", "max_layer_update",
    "dH_dt_mean", "d_hidden_norm_dt",
]
CURVATURE_FEATURES = [
    "knn_dist_mean", "knn_dist_min", "knn_dist_std",
    "trajectory_cos", "layer_update_var", "cross_layer_align",
    "prod_H_last_norm", "prod_maxupd_Hlast", "prod_cent_norm",
    "prod_totupd_Hlast", "prod_hiddennorm_centeredness",
]
QUANTUM_FEATURES = [
    "purity", "von_neumann_entropy", "effective_rank", "gram_max_eig",
    "kde_log_density",
    "upd_skewness", "upd_kurtosis",
    "trajectory_action_5",
]
ALL_FEATURES = SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES


def density_matrix_stats(layer_hiddens):
    """layer_hiddens: [L, d] per-layer final-position hidden states.
    Returns (purity, vn_entropy, effective_rank, max_eigenvalue_of_Gram).

    Trick: we need eigenvalues of ρ = (1/L) Σ v_i v_i^T where v_i are
    unit-normalized hidden states. Eigenvalues of that equal eigenvalues
    of the L×L Gram matrix G_ij = (1/L)(v_i · v_j). Much cheaper (L²
    rather than d² or d³).
    """
    L = layer_hiddens.shape[0]
    # Normalize to unit vectors per layer (treat each as a pure state)
    v = layer_hiddens / layer_hiddens.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = (v @ v.T) / L  # [L, L]
    eigvals = torch.linalg.eigvalsh(G.to(torch.float64)).clamp_min(0)
    # eigenvalues of ρ equal eigenvalues of G (same non-zero spectrum up to L≤d)
    eig = eigvals.sum()
    if eig > 0:
        eigvals = eigvals / eig  # normalize to sum 1
    purity = float((eigvals ** 2).sum().item())
    mask = eigvals > 1e-10
    vn = float(-(eigvals[mask] * torch.log(eigvals[mask])).sum().item())
    eff_rank = 1.0 / max(purity, 1e-10)
    gram_max = float(eigvals.max().item()) if len(eigvals) else 0.0
    return purity, vn, eff_rank, gram_max


def kde_log_density(h, calib_hidden, sigma):
    """Gaussian KDE log-density at h given calibration point cloud.
    log p(h) = logsumexp(-||h - c_j||² / 2σ²) - log N - (d/2) log(2πσ²)
    We drop the constant normalization (we only need a relative signal).
    """
    diffs = calib_hidden - h  # [N, d]
    sq_d = (diffs ** 2).sum(dim=1)  # [N]
    lse = torch.logsumexp(-sq_d / (2 * sigma * sigma), dim=0)
    return float(lse.item())


def moments_of_updates(update_mags):
    mu = sum(update_mags) / len(update_mags)
    var = sum((u - mu) ** 2 for u in update_mags) / len(update_mags)
    sd = var ** 0.5
    if sd < 1e-10:
        return 0.0, 0.0
    skew = sum((u - mu) ** 3 for u in update_mags) / (len(update_mags) * sd ** 3)
    kurt = sum((u - mu) ** 4 for u in update_mags) / (len(update_mags) * sd ** 4) - 3.0
    return float(skew), float(kurt)


def collect_calibration(model, tokenizer, texts, device, max_len=256):
    finals = []
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            finals.append(out.hidden_states[-1][0].to(torch.float32).cpu())
    return torch.cat(finals, dim=0)


def collect(model, tokenizer, prompt, max_new_tokens, device,
            calib_hidden, kde_sigma, knn_k=10):
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
            # Per-layer hidden vectors at last position (L+1 of them — include both pre and post)
            per_layer_h = torch.stack([hs[0, -1].to(torch.float32).cpu()
                                        for hs in hidden_states[1:]])  # [L, d]
            h_first = hidden_states[0][0, -1].to(torch.float32).cpu()
            h_last = per_layer_h[-1]
            h_mid = per_layer_h[n_layers // 2 - 1]

            layer_update_vecs = []
            layer_update_mags = []
            for i in range(n_layers):
                h_i = hidden_states[i][0, -1].to(torch.float32).cpu()
                h_ip1 = hidden_states[i+1][0, -1].to(torch.float32).cpu()
                u = h_ip1 - h_i
                layer_update_vecs.append(u)
                layer_update_mags.append(u.norm().item())

            # Summary
            H_mean = sum(entropies) / len(entropies)
            H_var = (sum((e - H_mean) ** 2 for e in entropies) / len(entropies))
            heads_above_0p9 = sum(1 for s in all_heads if s > 0.9)
            max_head_sharp = max(all_heads) if all_heads else 0.0
            hn_last = float(h_last.norm().item())
            hn_mid = float(h_mid.norm().item())
            hn_first = float(h_first.norm().item())
            cent = float((h_last - cal_mean).norm().item())
            total_upd = sum(layer_update_mags)
            max_upd = max(layer_update_mags)
            dH = (H_mean - prev_H_mean) if prev_H_mean is not None else 0.0
            d_hn = (hn_last - prev_hidden_norm) if prev_hidden_norm is not None else 0.0

            # Curvature
            diffs = calib_hidden - h_last
            dists = diffs.norm(dim=1)
            top_k, _ = dists.topk(knn_k, largest=False)
            knn_mean = float(top_k.mean().item())
            knn_min = float(top_k.min().item())
            knn_std = float(top_k.std().item())
            step_vec = h_last - prev_final
            if prev_prev_final is not None:
                prev_step = prev_final - prev_prev_final
                denom = step_vec.norm() * prev_step.norm()
                traj_cos = float((step_vec @ prev_step) / denom.clamp_min(1e-8))
            else:
                traj_cos = 0.0
            upd_mean = sum(layer_update_mags) / len(layer_update_mags)
            layer_update_var = sum((u - upd_mean) ** 2 for u in layer_update_mags) / len(layer_update_mags)
            cos_sum = 0.0; cos_count = 0
            for i in range(n_layers - 1):
                ua = layer_update_vecs[i]; ub = layer_update_vecs[i+1]
                denom = ua.norm() * ub.norm()
                if denom > 1e-8:
                    cos_sum += float((ua @ ub) / denom)
                    cos_count += 1
            cross_layer_align = cos_sum / max(cos_count, 1)
            prod_H_last_norm = entropies[-1] * hn_last
            prod_maxupd_Hlast = max_upd * entropies[-1]
            prod_cent_norm = cent * hn_last
            prod_totupd_Hlast = total_upd * entropies[-1]
            prod_hiddennorm_cent = hn_last * cent

            # Quantum features
            purity, vn_ent, eff_rank, gram_max = density_matrix_stats(per_layer_h)
            kde_ld = kde_log_density(h_last, calib_hidden, kde_sigma)
            skew, kurt = moments_of_updates(layer_update_mags)
            # Trajectory action over last up-to-5 steps
            step_energy = float((step_vec ** 2).sum().item())
            recent_step_energies.append(step_energy)
            if len(recent_step_energies) > 5:
                recent_step_energies.pop(0)
            trajectory_action_5 = sum(recent_step_energies)

            probs = F.softmax(logits[0], dim=-1)
            output_entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item())

            records.append({
                # summary
                "H_last_layer": entropies[-1], "H_first_layer": entropies[0],
                "H_q1_layer": entropies[n_layers // 4],
                "H_mid_layer": entropies[n_layers // 2],
                "H_q3_layer": entropies[(3 * n_layers) // 4],
                "H_max": max(entropies), "H_var": H_var,
                "heads_above_0p9": float(heads_above_0p9),
                "max_head_sharpness": max_head_sharp,
                "hidden_norm_final": hn_last, "hidden_norm_mid": hn_mid,
                "hidden_norm_first": hn_first,
                "centeredness": cent,
                "total_layer_update": total_upd, "max_layer_update": max_upd,
                "dH_dt_mean": dH, "d_hidden_norm_dt": d_hn,
                # curvature
                "knn_dist_mean": knn_mean, "knn_dist_min": knn_min, "knn_dist_std": knn_std,
                "trajectory_cos": traj_cos,
                "layer_update_var": layer_update_var,
                "cross_layer_align": cross_layer_align,
                "prod_H_last_norm": prod_H_last_norm,
                "prod_maxupd_Hlast": prod_maxupd_Hlast,
                "prod_cent_norm": prod_cent_norm,
                "prod_totupd_Hlast": prod_totupd_Hlast,
                "prod_hiddennorm_centeredness": prod_hiddennorm_cent,
                # quantum
                "purity": purity,
                "von_neumann_entropy": vn_ent,
                "effective_rank": eff_rank,
                "gram_max_eig": gram_max,
                "kde_log_density": kde_ld,
                "upd_skewness": skew,
                "upd_kurtosis": kurt,
                "trajectory_action_5": trajectory_action_5,
                # h_final for ceiling
                "h_final": h_last.tolist(),
                # label
                "output_entropy": output_entropy,
            })

            prev_prev_final = prev_final
            prev_final = h_last
            prev_H_mean = H_mean
            prev_hidden_norm = hn_last
            next_token = logits.argmax(dim=-1, keepdim=True)
    finally:
        for h in handles:
            h.remove()
    return records


def linear_regression_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    f = X_train.shape[1]
    XtX = X_train.T @ X_train + ridge * torch.eye(f, dtype=X_train.dtype)
    Xty = X_train.T @ y_train
    beta = torch.linalg.solve(XtX.to(torch.float64), Xty.to(torch.float64)).to(torch.float32)
    y_pred = X_test @ beta
    ss_res = ((y_test - y_pred) ** 2).sum().item()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    return 1 - ss_res / max(ss_tot, 1e-12), beta


def mlp_fit_r2(X_train, y_train, X_test, y_test, hidden=64, epochs=500, lr=1e-2):
    f = X_train.shape[1]
    net = nn.Sequential(nn.Linear(f, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
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


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32), mean


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if vx <= 0 or vy <= 0: return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--h-pca-k", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage28_quantum_features.json")
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

    print(f"\n=== calibration cloud ===")
    calib_hidden = collect_calibration(model, tokenizer, CALIB_TEXTS, device)
    # KDE sigma: median pairwise distance (robust)
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample)
    pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())
    print(f"  {calib_hidden.shape[0]} positions; KDE σ = {kde_sigma:.2f} (median pair dist)")

    print(f"\n=== collecting generation records ===")
    all_records = []
    for prompt in PROMPTS:
        print(f"  {prompt!r}", flush=True)
        all_records.extend(collect(model, tokenizer, prompt, args.max_new_tokens,
                                    device, calib_hidden, kde_sigma, args.knn_k))
    N = len(all_records)
    print(f"\n=== collected {N} records ({len(ALL_FEATURES)} features + 1 label) ===")

    y_list = [r["output_entropy"] for r in all_records]
    print(f"\n=== quantum features: Pearson r with output_entropy ===")
    for f in QUANTUM_FEATURES:
        xs = [r[f] for r in all_records]
        r = pearson(xs, y_list)
        print(f"  {f:>24}  r = {r:+.3f}")

    # Build matrices
    def mat(features):
        return torch.tensor([[r[f] for f in features] for r in all_records],
                             dtype=torch.float32)
    X_s = mat(SUMMARY_FEATURES)
    X_sc = mat(SUMMARY_FEATURES + CURVATURE_FEATURES)
    X_all = mat(ALL_FEATURES)
    X_hfinal = torch.tensor([r["h_final"] for r in all_records], dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)

    torch.manual_seed(0)
    perm = torch.randperm(N)
    tr = perm[:int(0.8 * N)]; te = perm[int(0.8 * N):]

    def norm(X_tr, X_te):
        mu = X_tr.mean(dim=0); sd = X_tr.std(dim=0).clamp_min(1e-8)
        return (X_tr - mu) / sd, (X_te - mu) / sd
    def add_int(X): return torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)

    Xs_tr, Xs_te = norm(X_s[tr], X_s[te])
    Xsc_tr, Xsc_te = norm(X_sc[tr], X_sc[te])
    Xall_tr, Xall_te = norm(X_all[tr], X_all[te])
    P_hf, mean_hf = pca_basis(X_hfinal[tr], args.h_pca_k)
    Xh_tr = (X_hfinal[tr] - mean_hf) @ P_hf
    Xh_te = (X_hfinal[te] - mean_hf) @ P_hf
    y_tr = y[tr]; y_te = y[te]

    print(f"\n=== regression R² (holdout) ===")
    r2_s_lin, _ = linear_regression_r2(add_int(Xs_tr), y_tr, add_int(Xs_te), y_te)
    r2_sc_lin, _ = linear_regression_r2(add_int(Xsc_tr), y_tr, add_int(Xsc_te), y_te)
    r2_all_lin, beta = linear_regression_r2(add_int(Xall_tr), y_tr, add_int(Xall_te), y_te)
    r2_hf_lin, _ = linear_regression_r2(add_int(Xh_tr), y_tr, add_int(Xh_te), y_te)
    r2_s_mlp = mlp_fit_r2(Xs_tr, y_tr, Xs_te, y_te)
    r2_sc_mlp = mlp_fit_r2(Xsc_tr, y_tr, Xsc_te, y_te)
    r2_all_mlp = mlp_fit_r2(Xall_tr, y_tr, Xall_te, y_te)
    r2_hf_mlp = mlp_fit_r2(Xh_tr, y_tr, Xh_te, y_te)

    print(f"  summary only                    linear  R² = {r2_s_lin:.3f}")
    print(f"  + curvature                     linear  R² = {r2_sc_lin:.3f}")
    print(f"  + curvature + quantum           linear  R² = {r2_all_lin:.3f}")
    print(f"  h_final PCA-64                  linear  R² = {r2_hf_lin:.3f}")
    print(f"  summary only                    MLP     R² = {r2_s_mlp:.3f}")
    print(f"  + curvature                     MLP     R² = {r2_sc_mlp:.3f}")
    print(f"  + curvature + quantum           MLP     R² = {r2_all_mlp:.3f}")
    print(f"  h_final PCA-64                  MLP     R² = {r2_hf_mlp:.3f}")

    print(f"\n=== quantum feature contribution ===")
    print(f"  linear: {r2_sc_lin:.3f} -> {r2_all_lin:.3f}  (+{r2_all_lin - r2_sc_lin:+.3f})")
    print(f"  MLP:    {r2_sc_mlp:.3f} -> {r2_all_mlp:.3f}  (+{r2_all_mlp - r2_sc_mlp:+.3f})")

    print(f"\n=== coverage (all features / h_final) ===")
    cov_l = r2_all_lin / max(r2_hf_lin, 1e-6)
    cov_m = r2_all_mlp / max(r2_hf_mlp, 1e-6)
    print(f"  linear: {cov_l:.1%}")
    print(f"  MLP:    {cov_m:.1%}")

    print(f"\n=== top-10 linear coefficients ===")
    named = list(zip(ALL_FEATURES, beta[:len(ALL_FEATURES)].tolist()))
    top = sorted(named, key=lambda t: abs(t[1]), reverse=True)[:10]
    for name, b in top:
        print(f"  {name:>32}  β = {b:+.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "n_records": N,
            "r2_summary_lin": r2_s_lin, "r2_summary_curv_lin": r2_sc_lin,
            "r2_all_lin": r2_all_lin, "r2_hfinal_lin": r2_hf_lin,
            "r2_summary_mlp": r2_s_mlp, "r2_summary_curv_mlp": r2_sc_mlp,
            "r2_all_mlp": r2_all_mlp, "r2_hfinal_mlp": r2_hf_mlp,
            "quantum_gain_linear": r2_all_lin - r2_sc_lin,
            "quantum_gain_mlp": r2_all_mlp - r2_sc_mlp,
            "coverage_linear": cov_l, "coverage_mlp": cov_m,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
