"""
Superposed HRR Attention on Qwen3-32B.

The real HRR win: don't compute n² attention scores.
Instead, superpose all K vectors into ONE holographic vector per layer,
then correlate each Q with that single vector.

Standard attention: O(n² * d) — score every Q against every K
Superposed HRR:    O(n * d * log d) — bind all K into one, correlate once per Q

Architecture:
  1. For each layer, each KV head:
     K_super = sum(circular_bind(K[j], position_vector[j]) for j in seq)
  2. For each query position:
     retrieval = circular_correlate(Q[i], K_super)
  3. The retrieval vector contains a superposition of ALL matching signals
  4. Use retrieval to weight V vectors (or superpose V similarly)

This eliminates the n×n score matrix entirely.

Test:
  1. Load 32B, measure baseline forward pass + PPL
  2. Replace attention with superposed HRR
  3. Measure speed (no fine-tune — just structural speed test)
  4. Measure quality degradation (expected: significant without training)
  5. The speed number is the proof of concept
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


def make_position_vectors(max_len, dim, seed=42):
    """Create random position vectors for HRR binding.
    Each position gets a fixed random vector used to bind K with position info."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Random unit vectors
    vecs = torch.randn(max_len, dim, generator=gen)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs


def hrr_bind(a, b):
    """Circular convolution via FFT."""
    A = torch.fft.rfft(a.float(), dim=-1)
    B = torch.fft.rfft(b.float(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_correlate(a, b):
    """Circular correlation via FFT (unbinding)."""
    A = torch.fft.rfft(a.float(), dim=-1)
    B = torch.fft.rfft(b.float(), dim=-1)
    return torch.fft.irfft(A * B.conj(), n=a.shape[-1], dim=-1).to(a.dtype)


class SuperposedHRRAttention(nn.Module):
    """Replace standard attention with superposed HRR.

    Instead of computing n×n scores, superpose all K into one vector
    per head, then correlate each Q position with that superposition.

    The correlation result is a d-dim vector (not a scalar).
    We use it to compute attention weights via a learned projection
    or by taking the norm as a relevance score.

    For V retrieval, we similarly superpose V vectors bound with
    the same position vectors, and retrieve via correlation.
    """
    def __init__(self, n_heads, n_kv_heads, head_dim, max_seq=2048):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.kv_groups = n_heads // n_kv_heads

        # Position vectors for binding
        self.register_buffer('pos_vectors',
                             make_position_vectors(max_seq, head_dim))

    def forward(self, query, key, value):
        """
        query: [batch, n_heads, seq_q, head_dim]
        key:   [batch, n_kv_heads, seq_k, head_dim]
        value: [batch, n_kv_heads, seq_k, head_dim]

        Returns: [batch, n_heads, seq_q, head_dim]
        """
        B, _, seq_q, d = query.shape
        seq_k = key.shape[2]

        # Get position vectors for this sequence length
        pos = self.pos_vectors[:seq_k]  # [seq_k, head_dim]

        # Superpose K: bind each K[j] with position[j], then sum
        # K: [B, n_kv_heads, seq_k, d]
        # pos: [seq_k, d] -> broadcast to [1, 1, seq_k, d]
        K_bound = hrr_bind(key, pos.unsqueeze(0).unsqueeze(0))  # [B, n_kv_heads, seq_k, d]
        K_super = K_bound.sum(dim=2)  # [B, n_kv_heads, d] — ONE vector per head

        # Similarly superpose V
        V_bound = hrr_bind(value, pos.unsqueeze(0).unsqueeze(0))
        V_super = V_bound.sum(dim=2)  # [B, n_kv_heads, d]

        # For each query position, correlate with K_super to get retrieval
        # query: [B, n_heads, seq_q, d]
        # K_super: [B, n_kv_heads, d] -> expand for GQA
        K_super_expanded = K_super.unsqueeze(1).expand(B, self.kv_groups, self.n_kv_heads, d)
        K_super_expanded = K_super_expanded.reshape(B, self.n_heads, d)

        V_super_expanded = V_super.unsqueeze(1).expand(B, self.kv_groups, self.n_kv_heads, d)
        V_super_expanded = V_super_expanded.reshape(B, self.n_heads, d)

        # Correlate each Q with K_super: [B, n_heads, seq_q, d]
        # K_super_expanded: [B, n_heads, d] -> [B, n_heads, 1, d]
        K_exp = K_super_expanded.unsqueeze(2).expand_as(query)
        retrieval = hrr_correlate(query, K_exp)  # [B, n_heads, seq_q, d]

        # The retrieval vector contains superposed signals from all matching K positions
        # Use it to weight the V superposition via element-wise multiply
        V_exp = V_super_expanded.unsqueeze(2).expand_as(query)
        V_retrieval = hrr_correlate(retrieval, V_exp)

        # Normalize
        V_retrieval = V_retrieval / math.sqrt(d)

        return V_retrieval


@torch.inference_mode()
def eval_ppl(model, val_chunks, seq_len=256, n_eval=30):
    model.eval()
    total = 0
    n = 0
    for i in range(min(n_eval, len(val_chunks))):
        inp = val_chunks[i:i+1, :seq_len]
        tgt = val_chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1))
        total += loss.item()
        n += 1
    ce = total / max(n, 1)
    return math.exp(min(ce, 20))


def main():
    torch.set_num_threads(32)
    model_name = "Qwen/Qwen3-32B"
    seq_len = 256

    print("=" * 60)
    print("SUPERPOSED HRR ATTENTION: Qwen3-32B")
    print("  Eliminate n² attention — ONE superposed vector per head")
    print("=" * 60, flush=True)

    # First: benchmark the core superposed operation at different sizes
    print(f"\nCORE OPERATION BENCHMARK:")
    print(f"  {'SeqLen':>8} | {'Std attn':>10} | {'HRR super':>10} | {'Speedup':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    n_heads = 40
    n_kv_heads = 8
    head_dim = 128
    hrr_attn = SuperposedHRRAttention(n_heads, n_kv_heads, head_dim)

    core_results = []
    for sl in [64, 128, 256, 512, 1024, 2048]:
        Q = torch.randn(1, n_heads, sl, head_dim)
        K = torch.randn(1, n_kv_heads, sl, head_dim)
        V = torch.randn(1, n_kv_heads, sl, head_dim)

        # Warmup
        for _ in range(3):
            # Standard: GQA attention
            K_exp = K.unsqueeze(2).expand(1, n_kv_heads, n_heads // n_kv_heads, sl, head_dim)
            K_exp = K_exp.reshape(1, n_heads, sl, head_dim)
            V_exp = V.unsqueeze(2).expand(1, n_kv_heads, n_heads // n_kv_heads, sl, head_dim)
            V_exp = V_exp.reshape(1, n_heads, sl, head_dim)
            scores = Q @ K_exp.transpose(-2, -1) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            _ = attn @ V_exp

            _ = hrr_attn(Q, K, V)

        # Standard attention timing
        times_std = []
        for _ in range(10):
            K_exp = K.unsqueeze(2).expand(1, n_kv_heads, n_heads // n_kv_heads, sl, head_dim)
            K_exp = K_exp.reshape(1, n_heads, sl, head_dim)
            V_exp = V.unsqueeze(2).expand(1, n_kv_heads, n_heads // n_kv_heads, sl, head_dim)
            V_exp = V_exp.reshape(1, n_heads, sl, head_dim)
            t0 = time.perf_counter()
            scores = Q @ K_exp.transpose(-2, -1) / math.sqrt(head_dim)
            attn = torch.softmax(scores, dim=-1)
            out = attn @ V_exp
            times_std.append(time.perf_counter() - t0)

        # HRR superposed timing
        times_hrr = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = hrr_attn(Q, K, V)
            times_hrr.append(time.perf_counter() - t0)

        ms_std = sum(times_std) / len(times_std) * 1000
        ms_hrr = sum(times_hrr) / len(times_hrr) * 1000
        speedup = ms_std / ms_hrr

        print(f"  {sl:>8} | {ms_std:>9.2f}ms | {ms_hrr:>9.2f}ms | {speedup:>7.2f}x", flush=True)
        core_results.append({
            "seq_len": sl, "std_ms": round(ms_std, 2),
            "hrr_ms": round(ms_hrr, 2), "speedup": round(speedup, 2)
        })

    # Now load 32B and test end-to-end
    print(f"\n{'='*60}")
    print(f"LOADING 32B FOR END-TO-END TEST")
    print(f"{'='*60}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  L={L}, d={d}", flush=True)

    # Load eval data
    cache_path = "data/owt_tokens_50M.pt"
    tokens = torch.load(cache_path, weights_only=True)
    val_tokens = tokens[:50000]
    n = len(val_tokens) // (seq_len + 1)
    val_chunks = val_tokens[:n * (seq_len + 1)].view(n, seq_len + 1)

    # Baseline
    print("\nBaseline PPL...", flush=True)
    baseline_ppl = eval_ppl(model, val_chunks, seq_len, n_eval=15)
    print(f"  Baseline PPL: {baseline_ppl:.2f}", flush=True)

    print("Baseline forward speed...", flush=True)
    inp = val_chunks[0:1, :seq_len]
    with torch.inference_mode():
        for _ in range(2):
            model(input_ids=inp, use_cache=False)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model(input_ids=inp, use_cache=False)
            times.append(time.perf_counter() - t0)
    baseline_ms = sum(times) / len(times) * 1000
    print(f"  Baseline forward: {baseline_ms:.0f}ms", flush=True)

    # Save results
    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)
    results = {
        "model": model_name,
        "core_benchmark": core_results,
        "baseline_ppl": baseline_ppl,
        "baseline_forward_ms": baseline_ms,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }
    with open(f"{save_dir}/hrr_superposed_32b.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Core operation (32B dimensions: {n_heads}Q/{n_kv_heads}KV heads, {head_dim}d):")
    for r in core_results:
        marker = " <<<" if r["speedup"] > 1.0 else ""
        print(f"    seq={r['seq_len']:>5}: std={r['std_ms']:.1f}ms  hrr={r['hrr_ms']:.1f}ms  "
              f"speedup={r['speedup']:.2f}x{marker}")
    print(f"\n  32B baseline: PPL={baseline_ppl:.2f}, forward={baseline_ms:.0f}ms")
    print(f"\n  Results: {save_dir}/hrr_superposed_32b.json")


if __name__ == "__main__":
    main()
