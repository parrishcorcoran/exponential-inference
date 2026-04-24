"""
Stage 138 — Multi-axis compression topography of the KV cache.

Like stage 111's bathtub but extended to all the orthogonal compression
axes. Per-layer slack measurements for each axis. The shape of the
compression budget across layers tells us where each lever has room.

Axes measured per layer:
  1. Rank slack (K and V): PR and EVR-95 effective rank
  2. Quantization slack (K and V): reconstruction error at Q8/Q4/Q2/Q1
  3. Cluster redundancy (K): k-means recon error at various k
  4. Attention concentration (Gini coefficient — eviction tolerance)
  5. Position contribution (from stage 132, summarized)

Runs on CPU so doesn't interfere with stage 135b on MPS.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


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
def collect_forward(model, tokens, seq_len, device):
    """One forward pass — collect K, V, attentions, hidden states."""
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=True, output_hidden_states=False,
                 output_attentions=True)
    L = model.config.num_hidden_layers
    kv = out.past_key_values
    if hasattr(kv, "layers") and kv.layers:
        pairs = [(c.keys, c.values) for c in kv.layers]
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)
    K_list, V_list = [], []
    for K, V in pairs:
        K = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        V = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_list.append(K)
        V_list.append(V)
    attns = [a[0].cpu().float() for a in out.attentions]
    return K_list, V_list, attns


def participation_ratio(X):
    if X.shape[0] == 0: return 0.0
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    return float((s2.sum().pow(2) / s2.pow(2).sum().clamp(min=1e-20)).item())


def evr_rank(X, target_evr=0.95):
    """Min rank to capture target_evr of variance."""
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    evr = s2.cumsum(0) / s2.sum().clamp(min=1e-20)
    idx = (evr >= target_evr).nonzero(as_tuple=True)[0]
    return int(idx[0]) + 1 if len(idx) else len(s)


def quantize_uniform(x, bits):
    """Per-row scale-then-round symmetric quantization."""
    qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
    if bits == 1:
        # Sign quantization
        scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-10)
        return torch.sign(x) * scale
    scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def quant_err(x, bits):
    """Relative reconstruction error at given bit-width."""
    x_q = quantize_uniform(x, bits)
    return float(((x - x_q).norm() / x.norm().clamp(min=1e-10)).item())


def simple_kmeans(X, k, n_iters=15, seed=0):
    """Simple k-means in torch. Returns reconstruction error."""
    n = X.shape[0]
    if k >= n:
        return 0.0
    g = torch.Generator(); g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k]
    centers = X[idx].clone()
    for _ in range(n_iters):
        d = torch.cdist(X, centers)
        assign = d.argmin(dim=-1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = X[mask].mean(0)
        if (new_centers - centers).norm() < 1e-6:
            break
        centers = new_centers
    rec = centers[assign]
    err = float(((X - rec).norm() / X.norm().clamp(min=1e-10)).item())
    return err


def gini_coefficient(values):
    """Gini coefficient of a 1D non-negative array. 0=uniform, 1=concentrated."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    if v.sum() < 1e-10: return 0.0
    n = len(v)
    cumsum = v.cumsum()
    return float((n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n)


def attn_concentration(attn):
    """Mean Gini across attention rows.
       attn: [num_heads, seq, seq] (causal — only lower triangle nonzero)."""
    num_heads, seq, _ = attn.shape
    ginis = []
    for h in range(num_heads):
        for t in range(seq):
            row = attn[h, t, :t+1].numpy()
            if row.sum() > 1e-10:
                ginis.append(gini_coefficient(row))
    return float(np.mean(ginis)) if ginis else 0.0


def position_novelty_summary(K):
    """Return mean Δrank for first quartile, middle half, last quartile."""
    seq, d = K.shape
    # Stride sample
    stride = max(1, seq // 32)
    pr_curve = []
    for t in range(stride, seq + 1, stride):
        pr_curve.append((t, participation_ratio(K[:t])))
    if len(pr_curve) < 4: return None
    n = len(pr_curve)
    early_pr = pr_curve[n//4][1] - pr_curve[0][1]  # gain in first quarter
    mid_pr = pr_curve[3*n//4][1] - pr_curve[n//4][1]  # gain in middle half
    late_pr = pr_curve[-1][1] - pr_curve[3*n//4][1]  # gain in last quarter
    return {"early_gain": float(early_pr), "mid_gain": float(mid_pr),
             "late_gain": float(late_pr)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage138_compression_topography.json")
    p.add_argument("--device", default="cpu",
                   help="Use cpu so MPS stays free for stage 135b")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--bits-list", default="16,8,4,2,1")
    p.add_argument("--cluster-ks", default="5,10,50,100,500")
    args = p.parse_args()

    bits_list = [int(x) for x in args.bits_list.split(",")]
    cluster_ks = [int(x) for x in args.cluster_ks.split(",")]
    print(f"device={args.device}  bits={bits_list}  cluster_ks={cluster_ks}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    dtype = torch.float32  # CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(args.device).eval()
    L = model.config.num_hidden_layers
    print(f"L={L}")

    print(f"loading WikiText-2 ({args.seq_len} tokens)...")
    tokens = load_tokens(tok, args.seq_len * 2, "train")[:args.seq_len]

    print("forward pass collecting K, V, attentions...")
    t0 = time.time()
    K_all, V_all, attns = collect_forward(model, tokens, args.seq_len, args.device)
    d_kv = K_all[0].shape[1]
    print(f"  done in {time.time()-t0:.0f}s  d_kv={d_kv}")

    del model

    results = {"model": args.model, "seq_len": args.seq_len, "d_kv": d_kv,
                "L": L, "bits_list": bits_list, "cluster_ks": cluster_ks,
                "per_layer": {}}

    print(f"\n{'='*60}\n=== per-layer multi-axis topography ===\n{'='*60}")
    print(f"  L  | PR_K  EVR_K | PR_V  EVR_V | qK4  qV4 | clK10 clK100 |  Gini  | Δnov_E Δnov_M Δnov_L")
    print(f"  ---|-------------|-------------|---------|---------------|--------|---------------------")

    for l in range(L):
        K = K_all[l]
        V = V_all[l]
        A = attns[l]

        # Rank
        pr_k = participation_ratio(K)
        pr_v = participation_ratio(V)
        evr_k = evr_rank(K, 0.95)
        evr_v = evr_rank(V, 0.95)

        # Quantization
        quant_K = {b: quant_err(K, b) for b in bits_list}
        quant_V = {b: quant_err(V, b) for b in bits_list}

        # Clustering on K (V too expensive, K is the bottleneck)
        cluster_K = {k: simple_kmeans(K, k) for k in cluster_ks}

        # Attention concentration (Gini)
        # Subsample heads for speed
        gini = attn_concentration(A) if A.shape[0] <= 8 else attn_concentration(A[:4])

        # Position novelty
        pos = position_novelty_summary(K)

        per_layer = {
            "PR_K": pr_k, "PR_V": pr_v,
            "EVR_K": evr_k, "EVR_V": evr_v,
            "quant_err_K": quant_K, "quant_err_V": quant_V,
            "cluster_err_K": cluster_K,
            "attn_gini": gini,
            "novelty": pos,
        }
        results["per_layer"][str(l)] = per_layer

        print(f"  {l:>2d} | {pr_k:>4.1f}  {evr_k:>4d} | {pr_v:>4.1f}  {evr_v:>4d} | "
              f"{quant_K[4]:.3f} {quant_V[4]:.3f} | "
              f"{cluster_K[10]:.3f}  {cluster_K[100]:.3f} | "
              f"{gini:.3f}  | "
              + (f"{pos['early_gain']:>+.2f}  {pos['mid_gain']:>+.2f}  {pos['late_gain']:>+.2f}" if pos else "n/a"))

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Cross-axis verdicts
    print(f"\n{'='*60}\n=== cross-layer profiles per axis ===\n{'='*60}")

    print(f"\n  EVR_K rank profile (RANK FLOOR @95% EVR):")
    evr_k_profile = [results["per_layer"][str(l)]["EVR_K"] for l in range(L)]
    min_e, max_e = min(evr_k_profile), max(evr_k_profile)
    for l in range(L):
        v = evr_k_profile[l]
        bar = int(40 * (v - min_e) / max(1, max_e - min_e))
        print(f"    L{l:>2d}: {'█' * bar}{' ' * (40-bar)}  {v}")

    print(f"\n  K Q4 quant error profile:")
    qk4 = [results["per_layer"][str(l)]["quant_err_K"][4] for l in range(L)]
    min_q, max_q = min(qk4), max(qk4)
    for l in range(L):
        v = qk4[l]
        bar = int(40 * (v - min_q) / max(1e-10, max_q - min_q))
        print(f"    L{l:>2d}: {'█' * bar}{' ' * (40-bar)}  {v:.3f}")

    print(f"\n  Cluster-100 K reconstruction err profile (low = redundant):")
    ck100 = [results["per_layer"][str(l)]["cluster_err_K"][100] for l in range(L)]
    min_c, max_c = min(ck100), max(ck100)
    for l in range(L):
        v = ck100[l]
        bar = int(40 * (v - min_c) / max(1e-10, max_c - min_c))
        print(f"    L{l:>2d}: {'█' * bar}{' ' * (40-bar)}  {v:.3f}")

    print(f"\n  Attention Gini (high = sparse, eviction-friendly):")
    gini_profile = [results["per_layer"][str(l)]["attn_gini"] for l in range(L)]
    min_g, max_g = min(gini_profile), max(gini_profile)
    for l in range(L):
        v = gini_profile[l]
        bar = int(40 * (v - min_g) / max(1e-10, max_g - min_g))
        print(f"    L{l:>2d}: {'█' * bar}{' ' * (40-bar)}  {v:.3f}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
