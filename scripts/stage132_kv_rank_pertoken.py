"""
Stage 132 — Per-token rank contribution to the KV cache.

Hypothesis (user's): each token's contribution to the KV cache has
different intrinsic rank. Prediction: inverted bathtub shape across
position — low Δrank at sequence boundaries (constrained), high in
the middle (maximum variability). That profile would fit INSIDE the
per-layer wormhole.

Consequences if confirmed:
  - Current KV methods (uniform eviction, H2O heavy-hitters, MLA
    fixed latent dim) treat tokens uniformly — they're blunt proxies.
  - Adaptive per-token cache compression: high-Δrank tokens in full
    precision, low-Δrank evictable or compressible.
  - 40-60% of tokens might be redundant to cache rank → big gains
    beyond uniform Q4/Q2 quantization.

Measurements per layer, per position t in a long sequence:
  1. Participation ratio of K_l[:t+1]  (continuous "effective rank")
  2. Δ_PR_t = PR_after - PR_before
  3. Token novelty: fraction of K_l,t outside span of K_l[:t]

Also:
  - Cumulative PR curve per layer (where does cache saturate?)
  - Compare layers with each other — does the per-token shape match
    the wormhole (per-layer) shape or differ?
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def participation_ratio(X):
    """Continuous effective rank of X [N, d].
       PR = (Σ σ²)² / Σ σ⁴. Bounded by min(N, d)."""
    if X.shape[0] == 0:
        return 0.0
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    num = s2.sum().pow(2)
    den = s2.pow(2).sum().clamp(min=1e-20)
    return (num / den).item()


def token_novelty(K_cache, K_new, subspace_dim=None):
    """Fraction of K_new's L2 norm that lies OUTSIDE the column span
       of K_cache [N_cache, d]. 0 = fully in subspace, 1 = orthogonal."""
    if K_cache.shape[0] == 0:
        return 1.0
    # SVD of cache
    U, S, Vt = torch.linalg.svd(K_cache.float(), full_matrices=False)
    V = Vt.T
    # Use top components up to rank threshold (99% EVR)
    if subspace_dim is None:
        evr = S.pow(2).cumsum(0) / S.pow(2).sum().clamp(min=1e-20)
        k = int((evr < 0.99).sum().item()) + 1
        k = min(k, V.shape[1])
    else:
        k = min(subspace_dim, V.shape[1])
    sub = V[:, :k]  # [d, k]
    # Project K_new onto subspace
    K_new = K_new.float().view(-1)  # [d]
    proj = sub @ (sub.T @ K_new)  # [d]
    proj_norm = proj.norm().item()
    full_norm = K_new.norm().item()
    if full_norm < 1e-10:
        return 0.0
    # Novelty = 1 - (amount captured by subspace / total)
    return max(0.0, 1.0 - proj_norm / full_norm)


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
def collect_kv(model, tokens, seq_len, device):
    """Run model on one long sequence, return per-layer K tensors
       [L, seq, d_kv] and V tensors [L, seq, d_kv]."""
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=True, output_hidden_states=False)
    # past_key_values: tuple of (K, V) per layer
    # Each K has shape [1, num_kv_heads, seq, head_dim]
    kv = out.past_key_values
    # Normalize to list-of-(K, V). HF returns either tuple or Cache object.
    if hasattr(kv, "layers") and kv.layers:
        pairs = []
        for layer_cache in kv.layers:
            K = layer_cache.keys
            V = layer_cache.values
            pairs.append((K, V))
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)

    K_all, V_all = [], []
    for K, V in pairs:
        # K/V: [1, num_kv_heads, seq, head_dim] → flatten to [seq, d_kv]
        K = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        V = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_all.append(K)
        V_all.append(V)
    return K_all, V_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage132_kv_rank_pertoken.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--sample-stride", type=int, default=4,
                   help="Compute PR every N tokens (for speed)")
    p.add_argument("--novelty-stride", type=int, default=8,
                   help="Compute token novelty every N tokens")
    p.add_argument("--layers-to-plot", default="all",
                   help="comma-separated or 'all'")
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
    print(f"L={L}  d_model={d}")

    # Load tokens
    print(f"loading WikiText-2 (need {args.seq_len} tokens)...")
    tokens = load_tokens(tok, args.seq_len * 2, "train")
    tokens = tokens[:args.seq_len]
    print(f"  loaded {len(tokens)} tokens")

    # Collect K, V
    print(f"collecting K, V for single sequence of length {args.seq_len}...")
    t0 = time.time()
    K_all, V_all = collect_kv(model, tokens, args.seq_len, device)
    print(f"  done in {time.time()-t0:.0f}s")
    print(f"  per-layer K shape: {K_all[0].shape}")
    d_kv = K_all[0].shape[1]
    print(f"  d_kv = {d_kv}")

    # Measure per-layer per-position PR + novelty
    print(f"\ncomputing participation ratio curves (stride={args.sample_stride})...")
    positions = list(range(args.sample_stride, args.seq_len + 1, args.sample_stride))

    per_layer_pr_K = {}  # l -> list of (position, PR)
    per_layer_pr_V = {}
    per_layer_novelty = {}  # l -> list of (position, novelty)

    t0 = time.time()
    for l in range(L):
        K = K_all[l]
        V = V_all[l]
        # PR curves
        pr_K_curve = []
        pr_V_curve = []
        for t in positions:
            pr_K = participation_ratio(K[:t])
            pr_V = participation_ratio(V[:t])
            pr_K_curve.append((t, pr_K))
            pr_V_curve.append((t, pr_V))
        per_layer_pr_K[l] = pr_K_curve
        per_layer_pr_V[l] = pr_V_curve

        # Novelty curve (per token t, novelty of K_t relative to K[:t])
        novelty_curve = []
        for t in range(args.novelty_stride, args.seq_len, args.novelty_stride):
            nov = token_novelty(K[:t], K[t])
            novelty_curve.append((t, nov))
        per_layer_novelty[l] = novelty_curve

        if l % 5 == 0 or l == L - 1:
            print(f"  L{l:>2d}: PR_K @ seq={args.seq_len}: {pr_K_curve[-1][1]:.1f}  "
                  f"PR_V: {pr_V_curve[-1][1]:.1f}  "
                  f"mean novelty (middle half): "
                  f"{np.mean([n for _, n in novelty_curve[len(novelty_curve)//4:3*len(novelty_curve)//4]]):.3f}")
    print(f"  total compute: {time.time()-t0:.0f}s")

    # Summary metrics per layer
    print(f"\n{'='*60}\n=== per-layer summary ===\n{'='*60}")
    layer_summary = {}
    for l in range(L):
        pr_K_final = per_layer_pr_K[l][-1][1]
        pr_V_final = per_layer_pr_V[l][-1][1]
        # Δ PR first half vs second half
        nov_curve = per_layer_novelty[l]
        n_nov = len(nov_curve)
        early_nov = np.mean([n for _, n in nov_curve[:n_nov//4]])
        mid_nov = np.mean([n for _, n in nov_curve[n_nov//4:3*n_nov//4]])
        late_nov = np.mean([n for _, n in nov_curve[3*n_nov//4:]])
        shape = "inverted bathtub" if (mid_nov > early_nov and mid_nov > late_nov) else \
                "standard bathtub" if (mid_nov < early_nov and mid_nov < late_nov) else \
                "monotone decreasing" if (early_nov > mid_nov > late_nov) else \
                "monotone increasing" if (early_nov < mid_nov < late_nov) else \
                "mixed"
        layer_summary[l] = {
            "PR_K_final": pr_K_final,
            "PR_V_final": pr_V_final,
            "novelty_early": float(early_nov),
            "novelty_mid": float(mid_nov),
            "novelty_late": float(late_nov),
            "shape": shape,
        }
        print(f"  L{l:>2d}: PR_K={pr_K_final:>6.1f} / {d_kv}  "
              f"novelty E={early_nov:.3f} M={mid_nov:.3f} L={late_nov:.3f}  {shape}")

    # Hypothesis check: inverted bathtub
    invb_count = sum(1 for s in layer_summary.values() if s["shape"] == "inverted bathtub")
    std_count = sum(1 for s in layer_summary.values() if s["shape"] == "standard bathtub")
    print(f"\n=== hypothesis test: inverted bathtub? ===")
    print(f"  layers with inverted-bathtub novelty:  {invb_count}/{L}")
    print(f"  layers with standard-bathtub novelty:  {std_count}/{L}")
    print(f"  (user predicted: inverted-bathtub — low at edges, high in middle)")

    # Are the per-layer PR curves themselves wormhole-shaped?
    # (compares KV cache rank across layers to residual stream rank from finding 13)
    print(f"\n=== KV cache per-layer rank profile (vs wormhole) ===")
    pr_K_by_layer = [layer_summary[l]["PR_K_final"] for l in range(L)]
    pr_V_by_layer = [layer_summary[l]["PR_V_final"] for l in range(L)]
    # Normalize
    min_K, max_K = min(pr_K_by_layer), max(pr_K_by_layer)
    for l in range(L):
        bar_len = int(40 * (pr_K_by_layer[l] - min_K) / max(1, max_K - min_K))
        print(f"  L{l:>2d}: {'█' * bar_len}{' ' * (40 - bar_len)}  PR_K={pr_K_by_layer[l]:.1f}")

    results = {
        "model": args.model,
        "seq_len": args.seq_len,
        "d_kv": d_kv,
        "per_layer_pr_K": {str(l): per_layer_pr_K[l] for l in range(L)},
        "per_layer_pr_V": {str(l): per_layer_pr_V[l] for l in range(L)},
        "per_layer_novelty": {str(l): per_layer_novelty[l] for l in range(L)},
        "layer_summary": {str(l): s for l, s in layer_summary.items()},
        "hypothesis": {
            "inverted_bathtub_layers": invb_count,
            "standard_bathtub_layers": std_count,
            "total_layers": L,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
