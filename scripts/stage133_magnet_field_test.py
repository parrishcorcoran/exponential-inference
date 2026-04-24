"""
Stage 133 — The magnet field test on the KV cache.

Hypothesis (user): the model is a magnet. Wormhole = the magnet body
(residual stream through depth). KV cache = the magnetic field, spread
360° around the magnet, with field strength decaying from poles.

Three measurements:

  A. Angular spread within K subspace
     For each layer: PCA K to top-3 dims. Compute pairwise angles
     between K vectors at different positions. Compare distribution
     to uniform (which is what a true 360° field would produce) and
     to random vectors in same subspace.

  B. Attention weight decay with positional distance
     For each layer: extract attention weights (averaged across heads).
     For each Δ ∈ [1, 500]: mean of attn(t, t-Δ).
     Fit power-law (B ∝ 1/Δ^α) vs exponential (B ∝ e^{-λΔ}).
     Magnetic field should be power-law (1/r^3 in 3D, slower in lower-dim).

  C. Field continuity (information conservation)
     For each layer l: compute total "information" = sum over positions
     of ||residual_l||². Plot vs l.
     Magnetic field is conserved (∇·B=0). If info is conserved across
     layers, magnet metaphor holds.

Outputs three curves + verdicts on each.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


@torch.no_grad()
def collect_full(model, tokens, seq_len, device):
    """Run model, return:
       - K_per_layer: list[L] of [seq, d_kv]
       - V_per_layer: same
       - attentions_per_layer: list[L] of [num_heads, seq, seq]
       - residual_per_layer: list[L+1] of [seq, d_model]"""
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=True, output_hidden_states=True,
                 output_attentions=True)
    L = model.config.num_hidden_layers

    # KV cache extraction
    kv = out.past_key_values
    if hasattr(kv, "layers") and kv.layers:
        pairs = [(layer_cache.keys, layer_cache.values) for layer_cache in kv.layers]
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)

    K_all, V_all = [], []
    for K, V in pairs:
        K = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        V = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_all.append(K)
        V_all.append(V)

    # Attention weights per layer
    # out.attentions: tuple of L tensors, each [batch, num_heads, seq, seq]
    attns = [a[0].cpu().float() for a in out.attentions]  # [num_heads, seq, seq]

    # Residual stream
    res = [h[0].cpu().float() for h in out.hidden_states]  # [seq, d]
    return K_all, V_all, attns, res


def pca_reduce(X, k):
    """Center, PCA-reduce to top-k components."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    Z = Xc @ V[:, :k]
    return Z


def pairwise_angles(X):
    """X: [N, d]. Return pairwise angles in degrees [0, 180] for all pairs."""
    Xn = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    cos = Xn @ Xn.T  # [N, N]
    cos = cos.clamp(-1.0, 1.0)
    angles = torch.arccos(cos) * 180.0 / np.pi
    # Take upper triangle (i<j), exclude diagonal
    mask = torch.triu(torch.ones_like(angles, dtype=torch.bool), diagonal=1)
    return angles[mask]


def fit_power_law(deltas, weights):
    """Fit log(w) = -α log(Δ) + c. Return α (decay exponent) and R²."""
    deltas = np.array(deltas)
    weights = np.array(weights)
    mask = (deltas > 0) & (weights > 1e-12)
    if mask.sum() < 3:
        return None, None
    x = np.log(deltas[mask])
    y = np.log(weights[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = sol
    y_pred = A @ sol
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return float(-slope), float(r2)


def fit_exponential(deltas, weights):
    """Fit log(w) = -λΔ + c. Return λ and R²."""
    deltas = np.array(deltas)
    weights = np.array(weights)
    mask = (deltas > 0) & (weights > 1e-12)
    if mask.sum() < 3:
        return None, None
    x = deltas[mask]
    y = np.log(weights[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = sol
    y_pred = A @ sol
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    return float(-slope), float(r2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage133_magnet_field.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--pca-dims", type=int, default=3,
                   help="dims to keep in K subspace for angular test")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"L={L}  d={d}")

    print(f"loading WikiText-2 ({args.seq_len} tokens)...")
    tokens = load_tokens(tok, args.seq_len * 2, "train")[:args.seq_len]

    print("collecting K, V, attentions, residual...")
    t0 = time.time()
    K_all, V_all, attns, res = collect_full(model, tokens, args.seq_len, device)
    print(f"  done in {time.time()-t0:.0f}s")

    results = {"model": args.model, "seq_len": args.seq_len, "L": L, "d": d}

    # === A. Angular spread within K subspace ===
    print(f"\n{'=' * 60}\n=== A. Angular spread in K subspace (top-{args.pca_dims} PCA) ===\n{'=' * 60}")
    a_results = {}
    for l in range(L):
        K = K_all[l]
        Z = pca_reduce(K, args.pca_dims)
        angles = pairwise_angles(Z)
        # Compare to random isotropic vectors in same dim
        random_Z = torch.randn(Z.shape[0], args.pca_dims)
        random_angles = pairwise_angles(random_Z)

        mean_ang = angles.mean().item()
        std_ang = angles.std().item()
        rand_mean = random_angles.mean().item()
        rand_std = random_angles.std().item()
        # Coverage: spread / max
        coverage = std_ang / 90.0  # normalized — 1 ≈ uniform [0, 180]
        rand_coverage = rand_std / 90.0
        # Closer to random => more 360° spread
        ratio = std_ang / max(rand_std, 1e-6)
        a_results[l] = {
            "mean_angle": mean_ang,
            "std_angle": std_ang,
            "random_mean": rand_mean,
            "random_std": rand_std,
            "spread_ratio": ratio,
        }
        if l % 4 == 0 or l == L - 1:
            print(f"  L{l:>2d}: mean angle={mean_ang:>5.1f}°  std={std_ang:>5.1f}°  "
                  f"vs random mean={rand_mean:>5.1f}° std={rand_std:>5.1f}°  "
                  f"spread/random={ratio:.2f}")

    # Verdict A
    avg_spread = np.mean([a_results[l]["spread_ratio"] for l in range(L)])
    print(f"\n  average spread/random ratio across layers: {avg_spread:.2f}")
    if avg_spread > 0.85:
        verdict_A = "FIELD-LIKE — K vectors fill the subspace approximately uniformly"
    elif avg_spread > 0.5:
        verdict_A = "PARTIAL — some directional structure, but not fully clustered"
    else:
        verdict_A = "CLUSTERED — K vectors point in similar directions, not 360°"
    print(f"  verdict A: {verdict_A}")
    results["A_angular_spread"] = {"per_layer": a_results, "avg_spread_ratio": float(avg_spread),
                                     "verdict": verdict_A}

    # === B. Attention decay vs positional distance ===
    print(f"\n{'=' * 60}\n=== B. Attention decay with distance ===\n{'=' * 60}")
    b_results = {}
    for l in range(L):
        # attns[l]: [num_heads, seq, seq]
        A = attns[l].mean(0).numpy()  # average over heads → [seq, seq]
        # For each Δ ∈ [1, seq-1]: average attn[t, t-Δ] over valid t
        seq_len = A.shape[0]
        deltas = list(range(1, min(seq_len, 256)))
        avg_attn = []
        for delta in deltas:
            vals = [A[t, t - delta] for t in range(delta, seq_len)]
            avg_attn.append(float(np.mean(vals)))
        # Fit power-law and exponential
        alpha, r2_pow = fit_power_law(deltas, avg_attn)
        lam, r2_exp = fit_exponential(deltas, avg_attn)
        b_results[l] = {
            "deltas": deltas,
            "avg_attn": avg_attn,
            "power_alpha": alpha, "power_r2": r2_pow,
            "exp_lambda": lam, "exp_r2": r2_exp,
        }
        if l % 4 == 0 or l == L - 1:
            print(f"  L{l:>2d}: power-law α={alpha:.2f} R²={r2_pow:.3f} | "
                  f"exp λ={lam:.4f} R²={r2_exp:.3f}")

    # Verdict B
    avg_pow_r2 = np.mean([b_results[l]["power_r2"] for l in range(L) if b_results[l]["power_r2"] is not None])
    avg_exp_r2 = np.mean([b_results[l]["exp_r2"] for l in range(L) if b_results[l]["exp_r2"] is not None])
    avg_alpha = np.mean([b_results[l]["power_alpha"] for l in range(L) if b_results[l]["power_alpha"] is not None])
    print(f"\n  power-law avg R²: {avg_pow_r2:.3f}  (avg α={avg_alpha:.2f})")
    print(f"  exponential avg R²: {avg_exp_r2:.3f}")
    if avg_pow_r2 > avg_exp_r2 + 0.05:
        verdict_B = f"POWER-LAW (magnet-like) decay, average exponent α={avg_alpha:.2f}"
    elif avg_exp_r2 > avg_pow_r2 + 0.05:
        verdict_B = f"EXPONENTIAL decay (not magnet-like)"
    else:
        verdict_B = f"AMBIGUOUS — both fit similarly"
    print(f"  verdict B: {verdict_B}")
    results["B_decay"] = {"per_layer": b_results,
                           "avg_power_r2": float(avg_pow_r2),
                           "avg_exp_r2": float(avg_exp_r2),
                           "avg_alpha": float(avg_alpha),
                           "verdict": verdict_B}

    # === C. Field continuity (sum norm² per layer) ===
    print(f"\n{'=' * 60}\n=== C. Information conservation across layers ===\n{'=' * 60}")
    layer_total_norm2 = []
    for l, h in enumerate(res):
        total = h.float().pow(2).sum().item()
        layer_total_norm2.append(total)
        if l % 4 == 0 or l == len(res) - 1:
            print(f"  hidden_states[{l}]: total ||h||² = {total:.2e}")
    layer_total_norm2 = np.array(layer_total_norm2)
    rel_change = layer_total_norm2.max() / max(layer_total_norm2.min(), 1e-10)
    coeff_var = float(layer_total_norm2.std() / layer_total_norm2.mean())
    print(f"\n  max/min ratio: {rel_change:.1f}×")
    print(f"  coefficient of variation: {coeff_var:.3f}")
    if coeff_var < 0.1:
        verdict_C = "CONSERVED (magnet-like ∇·B=0)"
    elif coeff_var < 0.5:
        verdict_C = "ROUGHLY CONSERVED with bounded fluctuation"
    else:
        verdict_C = "NOT CONSERVED — total norm grows/shrinks dramatically through layers"
    print(f"  verdict C: {verdict_C}")
    results["C_continuity"] = {"layer_total_norm2": layer_total_norm2.tolist(),
                                "max_min_ratio": float(rel_change),
                                "coeff_var": coeff_var,
                                "verdict": verdict_C}

    # Final summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  A (angular spread):  {verdict_A}")
    print(f"  B (decay law):        {verdict_B}")
    print(f"  C (info conservation): {verdict_C}")

    confirms = sum(["FIELD-LIKE" in verdict_A,
                     "POWER-LAW" in verdict_B,
                     "CONSERVED" in verdict_C and "NOT" not in verdict_C])
    print(f"\n  → {confirms}/3 magnet-field hypotheses confirmed")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
