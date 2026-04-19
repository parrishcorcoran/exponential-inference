"""
Stage 29 — Structural features from prediction geometry, trajectory windows,
            and black-hole boundary analogs.

Stage 28 showed quantum/density-matrix features were redundant with
trajectory features. To push past R²=0.53 (MLP) we need GENUINELY
different geometric structures. This stage adds:

PREDICTION GEOMETRY (lm_head structure):
  lm_head_top10_frac    — fraction of logit mass in top-10 tokens
                          (how concentrated is the lookup).
  lm_head_top_var       — variance of top-10 logit values
                          (sharp peak vs soft plateau among candidates).

ATTENTION ANGULAR STRUCTURE (where does attention point):
  attn_peak_position_last_norm — normalized position of argmax in
                                  last-layer attention (0=start of cache,
                                  1=most recent).
  attn_peak_pos_var_heads      — variance of peak positions across heads
                                  in last layer.
  attn_peak_recency            — fraction of attention mass on last 10
                                  positions (locality signal).

TRAJECTORY WINDOWS (longer memory than adjacent pairs):
  traj_cos_window_5    — mean cos between consecutive step vectors
                         over last 5 steps (smoother trajectory = easier).
  traj_arc_length_5    — sum of step magnitudes over last 5 steps.

BLACK-HOLE BOUNDARY ANALOGS:
  scrambling_rate_5    — ||h_t - h_{t-5}|| / ||h_{t-5}||. Info
                         scrambling speed.
  layer_halves_align   — cos between mean early-half update vector and
                         mean late-half update vector. Measures how
                         aligned early vs late layers are.
  bipartite_vn_early   — von Neumann entropy of early-layers density
                         matrix (layers 0..L/2).
  bipartite_vn_late    — same for late layers (L/2..L).
  vn_asymmetry         — early - late (sign = which half is more mixed).
  boundary_attn_entropy — attention entropy restricted to LAST 10
                          positions of KV cache (boundary slice).

Combine with summary + curvature + quantum (43 features total) and
re-measure coverage vs h_final PCA-64 ceiling.
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

SUMMARY_FEATURES = [
    "H_last_layer", "H_first_layer", "H_q1_layer", "H_mid_layer", "H_q3_layer",
    "H_max", "H_var", "heads_above_0p9", "max_head_sharpness",
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
    "kde_log_density", "upd_skewness", "upd_kurtosis", "trajectory_action_5",
]
STRUCTURAL_FEATURES = [
    # prediction geometry — OMITTED (lm_head top-k stats are derived from the
    # same softmax as the label and cause leakage).
    # attention angular
    "attn_peak_position_last_norm", "attn_peak_pos_var_heads", "attn_peak_recency",
    # trajectory windows
    "traj_cos_window_5", "traj_arc_length_5",
    # black-hole analogs
    "scrambling_rate_5", "layer_halves_align",
    "bipartite_vn_early", "bipartite_vn_late", "vn_asymmetry",
    "boundary_attn_entropy",
]
ALL_FEATURES = SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES + STRUCTURAL_FEATURES


# --- Utility stats ---

def density_matrix_stats(layer_hiddens):
    L = layer_hiddens.shape[0]
    v = layer_hiddens / layer_hiddens.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = (v @ v.T) / L
    eigvals = torch.linalg.eigvalsh(G.to(torch.float64)).clamp_min(0)
    eig = eigvals.sum()
    if eig > 0:
        eigvals = eigvals / eig
    purity = float((eigvals ** 2).sum().item())
    mask = eigvals > 1e-10
    vn = float(-(eigvals[mask] * torch.log(eigvals[mask])).sum().item())
    eff_rank = 1.0 / max(purity, 1e-10)
    gram_max = float(eigvals.max().item()) if len(eigvals) else 0.0
    return purity, vn, eff_rank, gram_max


def density_matrix_vn_only(layer_hiddens):
    if layer_hiddens.shape[0] < 1:
        return 0.0
    v = layer_hiddens / layer_hiddens.norm(dim=1, keepdim=True).clamp_min(1e-8)
    L = v.shape[0]
    G = (v @ v.T) / L
    eigvals = torch.linalg.eigvalsh(G.to(torch.float64)).clamp_min(0)
    s = eigvals.sum()
    if s > 0:
        eigvals = eigvals / s
    mask = eigvals > 1e-10
    return float(-(eigvals[mask] * torch.log(eigvals[mask])).sum().item())


def kde_log_density(h, calib_hidden, sigma):
    sq_d = ((calib_hidden - h) ** 2).sum(dim=1)
    return float(torch.logsumexp(-sq_d / (2 * sigma * sigma), dim=0).item())


def moments_of_updates(updates):
    mu = sum(updates) / len(updates)
    var = sum((u - mu) ** 2 for u in updates) / len(updates)
    sd = var ** 0.5
    if sd < 1e-10:
        return 0.0, 0.0
    skew = sum((u - mu) ** 3 for u in updates) / (len(updates) * sd ** 3)
    kurt = sum((u - mu) ** 4 for u in updates) / (len(updates) * sd ** 4) - 3.0
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
    per_layer_last_attn = {}  # [H, T_kv] — store for attn angular features

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

    handles = []
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i)))

    records = []
    cal_mean = calib_hidden.mean(dim=0)
    # trajectory history for window features
    hidden_history = []  # list of final hidden states
    step_vec_history = []

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prev_final = out.hidden_states[-1][0, -1].to(torch.float32).cpu()
        hidden_history.append(prev_final)
        next_token = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)

        prev_prev_final = None
        prev_H_mean = None
        prev_hidden_norm = None
        recent_step_energies = []

        for step in range(max_new_tokens - 1):
            entropies = [per_layer_H.get(i, 0.0) for i in range(n_layers)]
            head_sharp = [per_layer_head_sharp.get(i, []) for i in range(n_layers)]
            all_heads = [s for layer_h in head_sharp for s in layer_h]
            last_layer_attn = per_layer_last_attn.get(n_layers - 1)  # [H, T_kv]

            with torch.inference_mode():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True,
                            output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            hidden_states = out.hidden_states
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

            # --- Summary ---
            H_mean = sum(entropies) / len(entropies)
            H_var = sum((e - H_mean) ** 2 for e in entropies) / len(entropies)
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

            # --- Curvature ---
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
                    cos_sum += float((ua @ ub) / denom); cos_count += 1
            cross_layer_align = cos_sum / max(cos_count, 1)
            prod_H_last_norm = entropies[-1] * hn_last
            prod_maxupd_Hlast = max_upd * entropies[-1]
            prod_cent_norm = cent * hn_last
            prod_totupd_Hlast = total_upd * entropies[-1]
            prod_hiddennorm_cent = hn_last * cent

            # --- Quantum ---
            purity, vn_ent, eff_rank, gram_max = density_matrix_stats(per_layer_h)
            kde_ld = kde_log_density(h_last, calib_hidden, kde_sigma)
            skew, kurt = moments_of_updates(layer_update_mags)
            step_energy = float((step_vec ** 2).sum().item())
            recent_step_energies.append(step_energy)
            if len(recent_step_energies) > 5:
                recent_step_energies.pop(0)
            trajectory_action_5 = sum(recent_step_energies)

            # --- Structural new: prediction geometry ---
            top10 = logits[0].topk(10).values  # [10]
            total_logit_mag = logits[0].softmax(dim=-1).sum().item()  # always 1.0; skip
            # Use softmax top-10 mass
            probs = F.softmax(logits[0], dim=-1)
            top10_frac = float(probs.topk(10).values.sum().item())
            top_var = float(top10.var().item())

            # Attention angular (last layer)
            if last_layer_attn is not None and last_layer_attn.shape[-1] > 1:
                T_kv = last_layer_attn.shape[-1]
                # argmax per head, normalized
                peak_idx = last_layer_attn.argmax(dim=-1).float()
                peak_pos_norm = float((peak_idx / max(T_kv - 1, 1)).mean().item())
                peak_pos_var = float(peak_idx.var().item() / max((T_kv - 1) ** 2, 1))
                # mass on last 10 positions
                last10_start = max(0, T_kv - 10)
                recency_mass = float(last_layer_attn[:, last10_start:].sum(dim=-1).mean().item())
            else:
                peak_pos_norm = 0.0; peak_pos_var = 0.0; recency_mass = 0.0

            # Trajectory window features
            hidden_history.append(h_last)
            if len(hidden_history) > 10:
                hidden_history.pop(0)
            step_vec_history.append(step_vec)
            if len(step_vec_history) > 5:
                step_vec_history.pop(0)
            if len(step_vec_history) >= 2:
                cos_sum_win = 0.0; cnt = 0
                for i in range(len(step_vec_history) - 1):
                    a = step_vec_history[i]; b = step_vec_history[i+1]
                    denom = a.norm() * b.norm()
                    if denom > 1e-8:
                        cos_sum_win += float((a @ b) / denom); cnt += 1
                traj_cos_window_5 = cos_sum_win / max(cnt, 1)
            else:
                traj_cos_window_5 = 0.0
            traj_arc_length_5 = sum(float(v.norm().item()) for v in step_vec_history)

            # Black-hole analogs
            # Scrambling rate — distance from hidden state 5 steps ago
            if len(hidden_history) >= 6:
                past_5 = hidden_history[-6]
                scrambling_rate_5 = float(((h_last - past_5).norm() /
                                            past_5.norm().clamp_min(1e-8)).item())
            else:
                scrambling_rate_5 = 0.0
            # Layer-halves alignment
            half = n_layers // 2
            early_mean = sum(layer_update_vecs[:half]) / half
            late_mean = sum(layer_update_vecs[half:]) / (n_layers - half)
            denom = early_mean.norm() * late_mean.norm()
            layer_halves_align = float((early_mean @ late_mean) / denom.clamp_min(1e-8))
            # Bipartite VN entropy
            bipartite_vn_early = density_matrix_vn_only(per_layer_h[:half])
            bipartite_vn_late = density_matrix_vn_only(per_layer_h[half:])
            vn_asymmetry = bipartite_vn_early - bipartite_vn_late
            # Boundary attention entropy (last 10 positions)
            if last_layer_attn is not None and last_layer_attn.shape[-1] > 1:
                T_kv = last_layer_attn.shape[-1]
                last10_start = max(0, T_kv - 10)
                boundary = last_layer_attn[:, last10_start:]
                boundary = boundary / boundary.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                w = boundary.shape[-1]
                if w > 1:
                    ent = -(boundary * torch.log(boundary + 1e-10)).sum(dim=-1)
                    boundary_attn_entropy = float((ent / math.log(w)).mean().item())
                else:
                    boundary_attn_entropy = 0.0
            else:
                boundary_attn_entropy = 0.0

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
                "hidden_norm_first": hn_first, "centeredness": cent,
                "total_layer_update": total_upd, "max_layer_update": max_upd,
                "dH_dt_mean": dH, "d_hidden_norm_dt": d_hn,
                # curvature
                "knn_dist_mean": knn_mean, "knn_dist_min": knn_min, "knn_dist_std": knn_std,
                "trajectory_cos": traj_cos, "layer_update_var": layer_update_var,
                "cross_layer_align": cross_layer_align,
                "prod_H_last_norm": prod_H_last_norm, "prod_maxupd_Hlast": prod_maxupd_Hlast,
                "prod_cent_norm": prod_cent_norm, "prod_totupd_Hlast": prod_totupd_Hlast,
                "prod_hiddennorm_centeredness": prod_hiddennorm_cent,
                # quantum
                "purity": purity, "von_neumann_entropy": vn_ent,
                "effective_rank": eff_rank, "gram_max_eig": gram_max,
                "kde_log_density": kde_ld,
                "upd_skewness": skew, "upd_kurtosis": kurt,
                "trajectory_action_5": trajectory_action_5,
                # structural new
                "lm_head_top10_frac": top10_frac, "lm_head_top_var": top_var,
                "attn_peak_position_last_norm": peak_pos_norm,
                "attn_peak_pos_var_heads": peak_pos_var,
                "attn_peak_recency": recency_mass,
                "traj_cos_window_5": traj_cos_window_5,
                "traj_arc_length_5": traj_arc_length_5,
                "scrambling_rate_5": scrambling_rate_5,
                "layer_halves_align": layer_halves_align,
                "bipartite_vn_early": bipartite_vn_early,
                "bipartite_vn_late": bipartite_vn_late,
                "vn_asymmetry": vn_asymmetry,
                "boundary_attn_entropy": boundary_attn_entropy,
                # h_final and label
                "h_final": h_last.tolist(),
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
    mean = X.mean(dim=0); Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32), mean


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
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
    p.add_argument("--out", default="results/stage29_structural_features.json")
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
    sample = calib_hidden[torch.randperm(len(calib_hidden))[:200]]
    pair = torch.cdist(sample, sample); pair = pair[pair > 0]
    kde_sigma = float(pair.median().item())
    print(f"  {calib_hidden.shape[0]} positions; σ = {kde_sigma:.2f}")

    print(f"\n=== collecting generation records ===")
    all_records = []
    for prompt in PROMPTS:
        print(f"  {prompt!r}", flush=True)
        all_records.extend(collect(model, tokenizer, prompt, args.max_new_tokens,
                                    device, calib_hidden, kde_sigma, args.knn_k))
    N = len(all_records)
    print(f"\n=== collected {N} records ({len(ALL_FEATURES)} features) ===")

    y_list = [r["output_entropy"] for r in all_records]
    print(f"\n=== NEW structural features: Pearson r with output_entropy ===")
    for f in STRUCTURAL_FEATURES:
        r = pearson([rr[f] for rr in all_records], y_list)
        print(f"  {f:>32}  r = {r:+.3f}")

    def mat(features):
        return torch.tensor([[r[f] for f in features] for r in all_records],
                             dtype=torch.float32)
    X_s = mat(SUMMARY_FEATURES)
    X_sc = mat(SUMMARY_FEATURES + CURVATURE_FEATURES)
    X_scq = mat(SUMMARY_FEATURES + CURVATURE_FEATURES + QUANTUM_FEATURES)
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
    Xscq_tr, Xscq_te = norm(X_scq[tr], X_scq[te])
    Xall_tr, Xall_te = norm(X_all[tr], X_all[te])
    P_hf, mean_hf = pca_basis(X_hfinal[tr], args.h_pca_k)
    Xh_tr = (X_hfinal[tr] - mean_hf) @ P_hf
    Xh_te = (X_hfinal[te] - mean_hf) @ P_hf
    y_tr = y[tr]; y_te = y[te]

    print(f"\n=== progressive R² (holdout) ===")
    r2s = {}
    for name, (tr_X, te_X) in [
        ("summary (17)", (Xs_tr, Xs_te)),
        ("+curvature (28)", (Xsc_tr, Xsc_te)),
        ("+quantum (36)", (Xscq_tr, Xscq_te)),
        ("+structural (49)", (Xall_tr, Xall_te)),
        ("h_final PCA-64", (Xh_tr, Xh_te)),
    ]:
        lin, beta_ = linear_regression_r2(add_int(tr_X), y_tr, add_int(te_X), y_te)
        mlp = mlp_fit_r2(tr_X, y_tr, te_X, y_te)
        r2s[name] = {"linear": lin, "mlp": mlp}
        print(f"  {name:<20}  linear R² = {lin:.3f}   MLP R² = {mlp:.3f}")

    print(f"\n=== structural gain (over +curv+quantum) ===")
    print(f"  linear: {r2s['+quantum (36)']['linear']:.3f} -> {r2s['+structural (49)']['linear']:.3f}"
          f"  (+{r2s['+structural (49)']['linear'] - r2s['+quantum (36)']['linear']:+.3f})")
    print(f"  MLP:    {r2s['+quantum (36)']['mlp']:.3f} -> {r2s['+structural (49)']['mlp']:.3f}"
          f"  (+{r2s['+structural (49)']['mlp'] - r2s['+quantum (36)']['mlp']:+.3f})")

    print(f"\n=== coverage (all / h_final) ===")
    cl = r2s['+structural (49)']['linear'] / max(r2s['h_final PCA-64']['linear'], 1e-6)
    cm = r2s['+structural (49)']['mlp'] / max(r2s['h_final PCA-64']['mlp'], 1e-6)
    print(f"  linear:  {cl:.1%}")
    print(f"  MLP:     {cm:.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "n_records": N,
                    "r2": r2s, "coverage_linear": cl, "coverage_mlp": cm}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
