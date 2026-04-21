"""
Stage 53 — Q vs KV head specialization asymmetry.

Hypothesis: in GQA/MQA-style transformers, Q heads are specialist (each
carries distinct content) and KV heads are near-redundant (shared memory).
Scaling pattern supports: Qwen3 models go 16→32→40→64 Q heads as size
grows while KV stays at 8. LM heads (Q) scale, KV doesn't.

Two quantitative tests on Qwen3-0.6B:

  (A) Per-head operational rank.
      For each layer, compute the 95%-variance rank of each head's
      output (head_dim=128) across calibration tokens.
      If Q ranks > KV ranks consistently, Q heads are more specialist.

  (B) Cross-head similarity.
      For each layer, compute mean pairwise cosine similarity between
      heads (after averaging head outputs over tokens).
      High inter-head similarity in KV = redundancy. Low in Q = specialists.

Prediction if hypothesis holds:
  - Q per-head rank ≥ KV per-head rank
  - Q cross-head similarity < KV cross-head similarity
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


CALIB_TEXTS = [
    "The cell is the basic structural unit of life.",
    "Quantum mechanics describes matter and energy at atomic scales.",
    "Photosynthesis uses sunlight to convert carbon dioxide into glucose.",
    "Neural networks consist of parameterized layers trained by gradient descent.",
    "The immune system recognizes pathogens through pattern-recognition receptors.",
    "Plate tectonics describes the movement of lithospheric plates over the mantle.",
    "Proteins fold into three-dimensional structures determined by amino-acid sequences.",
    "Black holes are regions of spacetime from which nothing can escape.",
    "DNA encodes genetic information in a double-helix of paired nucleotide bases.",
    "Linear algebra provides the mathematical foundation for machine learning.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges across many applications.",
    "Cryptography protects information using mathematical operations.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Bayesian inference updates a prior using observed data.",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_head_outputs(model, tokenizer, texts, n_heads, n_kv_heads,
                          head_dim, device, max_len=256):
    """Capture Q and K output for each layer, reshape to per-head."""
    q_outs = {}  # layer_idx -> [N*T, n_heads, head_dim] tensor (accumulated)
    k_outs = {}
    v_outs = {}

    def make_q_hook(idx):
        def hook(m, inputs, output):
            y = output.detach().reshape(-1, n_heads, head_dim).to(torch.float32).cpu()
            q_outs.setdefault(idx, []).append(y)
        return hook

    def make_k_hook(idx):
        def hook(m, inputs, output):
            y = output.detach().reshape(-1, n_kv_heads, head_dim).to(torch.float32).cpu()
            k_outs.setdefault(idx, []).append(y)
        return hook

    def make_v_hook(idx):
        def hook(m, inputs, output):
            y = output.detach().reshape(-1, n_kv_heads, head_dim).to(torch.float32).cpu()
            v_outs.setdefault(idx, []).append(y)
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(make_q_hook(i)))
        handles.append(layer.self_attn.k_proj.register_forward_hook(make_k_hook(i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(i)))

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

    # Concatenate
    q_cat = {i: torch.cat(q_outs[i], dim=0) for i in q_outs}
    k_cat = {i: torch.cat(k_outs[i], dim=0) for i in k_outs}
    v_cat = {i: torch.cat(v_outs[i], dim=0) for i in v_outs}
    return q_cat, k_cat, v_cat


def effective_rank(X, threshold=0.95):
    """X: [N, d]. Returns number of singular values needed to capture `threshold`."""
    N, d = X.shape
    if N <= 1: return 1
    Xc = X - X.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return min(N, d)
    var = S ** 2
    total = var.sum()
    if total <= 0: return 1
    cum = torch.cumsum(var, dim=0) / total
    return max(1, int((cum < threshold).sum().item()) + 1)


def mean_pairwise_cosine_between_heads(X):
    """X: [N, n_heads, head_dim]. Average each head over tokens, then pairwise cos."""
    head_means = X.mean(dim=0)  # [n_heads, head_dim]
    normed = head_means / (head_means.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    sim = normed @ normed.T  # [n_heads, n_heads]
    n = sim.shape[0]
    # Extract upper triangle off-diagonal
    mask = torch.triu(torch.ones_like(sim), diagonal=1) > 0
    return float(sim[mask].mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage53_q_vs_kv_specialization.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    L = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    q_out_dim = model.model.layers[0].self_attn.q_proj.out_features
    head_dim = q_out_dim // n_heads
    print(f"  L={L}  n_Q={n_heads}  n_KV={n_kv_heads}  head_dim={head_dim}")

    print(f"\n=== collecting per-head Q/K/V outputs ===")
    t0 = time.perf_counter()
    q_cat, k_cat, v_cat = collect_head_outputs(
        model, tokenizer, CALIB_TEXTS,
        n_heads, n_kv_heads, head_dim, device)
    N = q_cat[0].shape[0]
    print(f"  {time.perf_counter()-t0:.1f}s  ({N} tokens)")

    # Per-layer, per-head analysis
    print(f"\n=== (A) per-head operational rank (95% variance) ===")
    q_rank_means = []; k_rank_means = []; v_rank_means = []
    q_rank_mins = []; k_rank_mins = []
    q_rank_maxs = []; k_rank_maxs = []
    for i in range(L):
        q = q_cat[i]; k = k_cat[i]; v = v_cat[i]
        q_ranks = [effective_rank(q[:, h, :]) for h in range(n_heads)]
        k_ranks = [effective_rank(k[:, h, :]) for h in range(n_kv_heads)]
        v_ranks = [effective_rank(v[:, h, :]) for h in range(n_kv_heads)]
        q_rank_means.append(sum(q_ranks) / len(q_ranks))
        k_rank_means.append(sum(k_ranks) / len(k_ranks))
        v_rank_means.append(sum(v_ranks) / len(v_ranks))
        q_rank_mins.append(min(q_ranks)); q_rank_maxs.append(max(q_ranks))
        k_rank_mins.append(min(k_ranks)); k_rank_maxs.append(max(k_ranks))

    q_grand = sum(q_rank_means) / L
    k_grand = sum(k_rank_means) / L
    v_grand = sum(v_rank_means) / L
    print(f"  grand means (across {L} layers):")
    print(f"    Q heads ({n_heads} per layer):  mean_rank = {q_grand:.2f}")
    print(f"    K heads ({n_kv_heads} per layer):  mean_rank = {k_grand:.2f}")
    print(f"    V heads ({n_kv_heads} per layer):  mean_rank = {v_grand:.2f}")
    print(f"  Q/K rank ratio = {q_grand/k_grand:.3f}")

    print(f"\n=== (B) cross-head cosine similarity (lower = more specialist) ===")
    q_sims = []; k_sims = []; v_sims = []
    for i in range(L):
        q_sims.append(mean_pairwise_cosine_between_heads(q_cat[i]))
        k_sims.append(mean_pairwise_cosine_between_heads(k_cat[i]))
        v_sims.append(mean_pairwise_cosine_between_heads(v_cat[i]))

    q_sim_mean = sum(q_sims) / L
    k_sim_mean = sum(k_sims) / L
    v_sim_mean = sum(v_sims) / L
    print(f"  mean pairwise cosine between heads (averaged over tokens):")
    print(f"    Q heads: {q_sim_mean:.3f}  (range [{min(q_sims):.3f}, {max(q_sims):.3f}])")
    print(f"    K heads: {k_sim_mean:.3f}  (range [{min(k_sims):.3f}, {max(k_sims):.3f}])")
    print(f"    V heads: {v_sim_mean:.3f}  (range [{min(v_sims):.3f}, {max(v_sims):.3f}])")

    # Per-layer breakdown at few points
    print(f"\n=== per-layer Q mean_rank vs K mean_rank (sampled layers) ===")
    print(f"  {'layer':>5}  {'Q_rank':>7}  {'K_rank':>7}  {'Q_sim':>7}  {'K_sim':>7}")
    for i in [0, 1, 5, 10, 15, 20, 25, L-1]:
        if i < L:
            print(f"  {i:>5}  {q_rank_means[i]:>7.1f}  {k_rank_means[i]:>7.1f}  "
                  f"{q_sims[i]:>7.3f}  {k_sims[i]:>7.3f}")

    # Verdict
    print(f"\n=== verdict ===")
    q_more_specialist = (q_sim_mean < k_sim_mean) and (q_grand >= k_grand)
    if q_more_specialist:
        print(f"  SUPPORTED: Q heads are more specialist than KV heads.")
        print(f"    Q rank > KV rank: {q_grand:.1f} > {k_grand:.1f}")
        print(f"    Q heads less similar than KV: {q_sim_mean:.3f} < {k_sim_mean:.3f}")
        print(f"  Architecturally: scaling Q heads adds specialists; KV is shared memory.")
    elif q_sim_mean < k_sim_mean:
        print(f"  PARTIAL: Q heads less similar, but rank comparison unclear.")
    else:
        print(f"  NOT SUPPORTED: KV heads are at least as specialist as Q heads.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "head_dim": head_dim, "n_tokens": N,
            "per_layer_q_rank": q_rank_means,
            "per_layer_k_rank": k_rank_means,
            "per_layer_v_rank": v_rank_means,
            "per_layer_q_sim": q_sims,
            "per_layer_k_sim": k_sims,
            "per_layer_v_sim": v_sims,
            "grand_means": {
                "Q_rank": q_grand, "K_rank": k_grand, "V_rank": v_grand,
                "Q_sim": q_sim_mean, "K_sim": k_sim_mean, "V_sim": v_sim_mean,
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
