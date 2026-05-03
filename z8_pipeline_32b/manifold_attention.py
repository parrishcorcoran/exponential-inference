"""
Manifold-Based Attention: use the ~10D manifold as a physics-based index.

Electron cloud model:
  - Each token = electron at a position on the ~10D manifold
  - Attention = interaction between nearby electrons
  - Tokens in same orbital (cluster) interact strongly
  - Tokens in different orbitals → skip (interaction ≈ 0)

Pipeline:
  1. Learn a small projection: hidden_dim → manifold_dim (e.g., 5120 → 10)
  2. Cluster tokens by manifold position (k-means or bucketing)
  3. Within-cluster: exact softmax attention
  4. Cross-cluster: skip
  5. Compare quality + speed vs full attention

No HRR. No FFT. Just project to manifold, find neighbors, attend locally.

Test on 0.6B first, then 32B.
"""

import gc
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class ManifoldProjector(nn.Module):
    """Projects hidden states to manifold coordinates.
    Small linear: hidden_dim → manifold_dim."""
    def __init__(self, hidden_dim, manifold_dim=10):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, manifold_dim, bias=False)
        # Initialize with PCA-like small random weights
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x):
        return self.proj(x)  # [B, seq, manifold_dim]


def manifold_clustered_attention(query, key, value, manifold_coords, n_clusters,
                                  n_heads, n_kv_heads, head_dim):
    """Clustered attention based on manifold proximity.

    1. Cluster tokens by manifold coordinates
    2. Within each cluster: exact softmax attention
    3. Cross-cluster: skip

    query: [B, n_heads, seq_q, head_dim]
    key:   [B, n_kv_heads, seq_k, head_dim]
    value: [B, n_kv_heads, seq_k, head_dim]
    manifold_coords: [B, seq, manifold_dim]
    """
    B, _, seq_q, d = query.shape
    seq_k = key.shape[2]
    kv_groups = n_heads // n_kv_heads

    # Expand K, V for GQA
    K_full = key.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_full = K_full.reshape(B, n_heads, seq_k, d)
    V_full = value.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    V_full = V_full.reshape(B, n_heads, seq_k, d)

    # Assign tokens to clusters based on manifold coordinates
    # Simple approach: quantize manifold coords into buckets
    coords = manifold_coords[0]  # [seq, manifold_dim] (batch=1 for now)

    # K-means style clustering (simple: use first PCA dim to bucket)
    # For speed: hash into n_clusters buckets
    # Use L2 norm of manifold coords as a simple 1D hash
    coord_hash = coords.norm(dim=-1)  # [seq]
    # Quantize into n_clusters buckets
    mn, mx = coord_hash.min(), coord_hash.max()
    bucket_size = (mx - mn + 1e-8) / n_clusters
    cluster_ids = ((coord_hash - mn) / bucket_size).long().clamp(0, n_clusters - 1)

    # Build output by attending within clusters only
    output = torch.zeros_like(query)  # [B, n_heads, seq_q, d]

    for c in range(n_clusters):
        # Find positions in this cluster
        mask = (cluster_ids == c)
        if mask.sum() == 0:
            continue

        indices = mask.nonzero(as_tuple=True)[0]  # positions in cluster
        n_c = len(indices)

        # Extract Q, K, V for this cluster
        Q_c = query[:, :, indices, :]   # [B, n_heads, n_c, d]
        K_c = K_full[:, :, indices, :]  # [B, n_heads, n_c, d]
        V_c = V_full[:, :, indices, :]  # [B, n_heads, n_c, d]

        # Standard attention within cluster
        scores = Q_c @ K_c.transpose(-2, -1) / math.sqrt(d)  # [B, n_heads, n_c, n_c]

        # Causal mask within cluster (based on original positions)
        pos_q = indices.unsqueeze(1)  # [n_c, 1]
        pos_k = indices.unsqueeze(0)  # [1, n_c]
        causal = (pos_q < pos_k)  # [n_c, n_c]
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out_c = attn @ V_c  # [B, n_heads, n_c, d]

        # Scatter back to output
        output[:, :, indices, :] = out_c

    return output


def manifold_topk_attention(query, key, value, manifold_coords, top_k,
                             n_heads, n_kv_heads, head_dim):
    """Top-k attention based on manifold distance.

    For each query, find the k nearest keys on the manifold,
    then run exact softmax on only those.

    More precise than clustering — per-token neighbor selection.
    """
    B, _, seq_q, d = query.shape
    seq_k = key.shape[2]
    kv_groups = n_heads // n_kv_heads
    k = min(top_k, seq_k)

    K_full = key.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_full = K_full.reshape(B, n_heads, seq_k, d)
    V_full = value.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    V_full = V_full.reshape(B, n_heads, seq_k, d)

    coords = manifold_coords[0]  # [seq, manifold_dim]

    # Pairwise distance on manifold: [seq_q, seq_k]
    # This IS O(n²) in manifold_dim, but manifold_dim=10 so it's tiny
    dist = torch.cdist(coords, coords)  # [seq, seq] — cheap in 10D

    # Causal mask: future positions get infinite distance
    causal = torch.triu(torch.ones(seq_q, seq_k, device=dist.device), diagonal=1).bool()
    dist = dist.masked_fill(causal, float('inf'))

    # Top-k nearest neighbors per position
    _, topk_idx = dist.topk(k, dim=-1, largest=False)  # [seq, k] — nearest

    # Gather K, V at nearest positions
    idx = topk_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, seq, k, 1]
    idx = idx.expand(B, n_heads, seq_q, k, d)

    K_selected = K_full.unsqueeze(2).expand(B, n_heads, seq_q, seq_k, d)
    K_selected = K_selected.gather(3, idx)  # [B, n_heads, seq, k, d]
    V_selected = V_full.unsqueeze(2).expand(B, n_heads, seq_q, seq_k, d)
    V_selected = V_selected.gather(3, idx)

    # Exact softmax on nearest neighbors
    Q_exp = query.unsqueeze(3)  # [B, n_heads, seq, 1, d]
    scores = (Q_exp * K_selected).sum(-1) / math.sqrt(d)  # [B, n_heads, seq, k]
    attn = torch.softmax(scores, dim=-1)

    output = (attn.unsqueeze(-1) * V_selected).sum(3)  # [B, n_heads, seq, d]
    return output


# -- Data + eval --

def load_data(seq_len=256, max_val=100_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    val_t = tokens[:max_val]
    def chunk(t):
        n = len(t) // (seq_len + 1)
        return t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return chunk(val_t)


@torch.inference_mode()
def eval_ppl(model, chunks, seq_len=256, n=20):
    model.eval()
    total = 0; c = 0
    for i in range(min(n, len(chunks))):
        inp = chunks[i:i+1, :seq_len]
        tgt = chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        c += 1
    return math.exp(total / max(c, 1))


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--manifold-dim", type=int, default=10)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"MANIFOLD ATTENTION: {cli.model}")
    print(f"  Project to {cli.manifold_dim}D manifold")
    print(f"  Find neighbors on manifold, attend locally")
    print(f"  Electron cloud model: orbital-based interaction")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', d_model // n_heads)

    print(f"  L={L}, d={d_model}, heads={n_heads}/{n_kv_heads}, head_dim={head_dim}")

    val_chunks = load_data()

    teacher_ppl = eval_ppl(model, val_chunks)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # First: measure actual manifold structure
    # Run a sample through and collect hidden states
    print(f"\nMeasuring manifold structure...", flush=True)
    with torch.inference_mode():
        inp = val_chunks[0:1, :256]
        out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
        hidden_states = [h.squeeze(0) for h in out.hidden_states]  # L+1 x [seq, d]

    # PCA on hidden states to find manifold directions
    print(f"  Computing PCA on hidden states...", flush=True)
    for li in [0, L//4, L//2, 3*L//4, L]:
        h = hidden_states[li].float()  # [seq, d]
        # Center
        h_centered = h - h.mean(dim=0)
        # SVD for PCA
        U, S, Vt = torch.linalg.svd(h_centered, full_matrices=False)
        # Explained variance ratio
        var_ratio = (S ** 2) / (S ** 2).sum()
        cum_var = var_ratio.cumsum(0)
        dim_90 = (cum_var < 0.90).sum().item() + 1
        dim_95 = (cum_var < 0.95).sum().item() + 1
        dim_99 = (cum_var < 0.99).sum().item() + 1
        print(f"    Layer {li:>3}: 90%={dim_90:>3}D  95%={dim_95:>3}D  99%={dim_99:>3}D  "
              f"top3={var_ratio[0]:.3f},{var_ratio[1]:.3f},{var_ratio[2]:.3f}")

    # Use middle layer PCA as manifold projection
    mid = L // 2
    h_mid = hidden_states[mid].float()
    h_centered = h_mid - h_mid.mean(dim=0)
    U, S, Vt = torch.linalg.svd(h_centered, full_matrices=False)
    pca_dirs = Vt[:cli.manifold_dim]  # [manifold_dim, d_model] — top PCA directions

    print(f"\n  Using top {cli.manifold_dim} PCA directions from layer {mid}")
    print(f"  Explained variance: {((S[:cli.manifold_dim]**2).sum() / (S**2).sum()):.3f}")

    # Benchmark: standard vs manifold attention at various top-k
    print(f"\n{'='*60}")
    print(f"CORE BENCHMARK: Manifold top-k attention (seq=256)")
    print(f"{'='*60}")

    seq = 256
    Q = torch.randn(1, n_heads, seq, head_dim)
    K = torch.randn(1, n_kv_heads, seq, head_dim)
    V = torch.randn(1, n_kv_heads, seq, head_dim)

    # Standard attention timing
    kv_groups = n_heads // n_kv_heads
    K_e = K.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
    V_e = V.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)

    for _ in range(3):
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        _ = a @ V_e

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        o = a @ V_e
        times.append(time.perf_counter() - t0)
    std_ms = sum(times) / len(times) * 1000

    # Standard output for quality comparison
    with torch.no_grad():
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        std_output = a @ V_e

    print(f"  Standard attention: {std_ms:.2f}ms")
    print()
    print(f"  {'Method':>20} | {'k':>4} | {'Time ms':>8} | {'Speedup':>8} | {'cos_sim':>8}")
    print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    # Random manifold coords for benchmark (real ones used in model test)
    fake_coords = torch.randn(1, seq, cli.manifold_dim)

    results = []
    for top_k in [4, 8, 16, 32, 64, 128]:
        if top_k > seq:
            continue

        # Manifold top-k
        for _ in range(3):
            _ = manifold_topk_attention(Q, K, V, fake_coords, top_k, n_heads, n_kv_heads, head_dim)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = manifold_topk_attention(Q, K, V, fake_coords, top_k, n_heads, n_kv_heads, head_dim)
            times.append(time.perf_counter() - t0)
        topk_ms = sum(times) / len(times) * 1000

        with torch.no_grad():
            out = manifold_topk_attention(Q, K, V, fake_coords, top_k, n_heads, n_kv_heads, head_dim)
            cos = F.cosine_similarity(out.reshape(-1), std_output.reshape(-1), dim=0).item()

        print(f"  {'manifold_topk':>20} | {top_k:>4} | {topk_ms:7.2f}ms | {std_ms/topk_ms:7.2f}x | {cos:7.4f}",
              flush=True)
        results.append({"method": "manifold_topk", "k": top_k,
                        "ms": round(topk_ms, 2), "speedup": round(std_ms/topk_ms, 2),
                        "cos": round(cos, 4)})

    # Clustered attention
    for n_clusters in [4, 8, 16, 32]:
        for _ in range(3):
            _ = manifold_clustered_attention(Q, K, V, fake_coords, n_clusters, n_heads, n_kv_heads, head_dim)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = manifold_clustered_attention(Q, K, V, fake_coords, n_clusters, n_heads, n_kv_heads, head_dim)
            times.append(time.perf_counter() - t0)
        cl_ms = sum(times) / len(times) * 1000

        with torch.no_grad():
            out = manifold_clustered_attention(Q, K, V, fake_coords, n_clusters, n_heads, n_kv_heads, head_dim)
            cos = F.cosine_similarity(out.reshape(-1), std_output.reshape(-1), dim=0).item()

        print(f"  {'manifold_cluster':>20} | {n_clusters:>4} | {cl_ms:7.2f}ms | {std_ms/cl_ms:7.2f}x | {cos:7.4f}",
              flush=True)
        results.append({"method": "manifold_cluster", "k": n_clusters,
                        "ms": round(cl_ms, 2), "speedup": round(std_ms/cl_ms, 2),
                        "cos": round(cos, 4)})

    # Model test with real manifold projection
    print(f"\n{'='*60}")
    print(f"MODEL TEST: Manifold attention on {cli.model}")
    print(f"  Using PCA from layer {mid} as manifold projection")
    print(f"{'='*60}")
    print(f"  {'Method':>20} | {'k':>4} | {'PPL':>8} | {'Ratio':>6}")
    print(f"  {'-'*20}-+-{'-'*4}-+-{'-'*8}-+-{'-'*6}")
    print(f"  {'standard':>20} | {'all':>4} | {teacher_ppl:8.2f} | {'1.00x':>6}", flush=True)

    model_results = []
    for top_k in [64, 32, 16, 8]:
        del model; gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            cli.model, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="eager").eval()

        hooks = []
        for li in range(L):
            attn_mod = model.model.layers[li].self_attn
            orig_fwd = attn_mod.forward

            def make_hook(orig, layer_i, k_val):
                def hooked(hidden_states, *args, **kwargs):
                    B, S, D = hidden_states.shape
                    hd = head_dim
                    am = model.model.layers[layer_i].self_attn

                    # Project to manifold
                    coords = hidden_states @ pca_dirs.T  # [B, S, manifold_dim]

                    q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                    k = am.k_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)
                    v = am.v_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)

                    out = manifold_topk_attention(q, k, v, coords, k_val,
                                                  n_heads, n_kv_heads, hd)
                    out = out.transpose(1, 2).reshape(B, S, n_heads * hd)
                    out = am.o_proj(out)
                    return (out, None)

                return hooked

            attn_mod.forward = make_hook(orig_fwd, li, top_k)
            hooks.append((attn_mod, orig_fwd))

        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl
        note = "within 5%" if ratio <= 1.05 else "within 10%" if ratio <= 1.10 else "degraded" if ratio <= 1.5 else "broken"

        print(f"  {'manifold_topk':>20} | {top_k:>4} | {ppl:8.2f} | {ratio:5.2f}x | {note}", flush=True)
        model_results.append({"method": "manifold_topk", "k": top_k,
                               "ppl": round(ppl, 2), "ratio": round(ratio, 4)})

        for attn_mod, orig_fwd in hooks:
            attn_mod.forward = orig_fwd

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher: {teacher_ppl:.2f}")
    print(f"  Manifold dim: {cli.manifold_dim}")
    print(f"  PCA variance explained: {((S[:cli.manifold_dim]**2).sum() / (S**2).sum()):.3f}")
    print(f"\n  Core operation:")
    for r in results:
        print(f"    {r['method']} k={r['k']}: {r['ms']:.1f}ms ({r['speedup']:.1f}x) cos={r['cos']:.4f}")
    print(f"\n  Model PPL:")
    for r in model_results:
        print(f"    k={r['k']}: PPL={r['ppl']:.2f} ({r['ratio']:.2f}x)")

    all_results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "manifold_dim": cli.manifold_dim,
        "core": results, "model": model_results,
    }
    with open(Path(save_dir) / "manifold_attention.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {save_dir}/manifold_attention.json")


if __name__ == "__main__":
    main()
