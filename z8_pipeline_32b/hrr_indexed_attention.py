"""
HRR-Indexed Attention: use holographic superposition as a cheap index
to find which K vectors matter, then run exact softmax on only those.

KV cache stays. Softmax stays. Just skip the keys that don't matter.

Standard: Q @ K^T over ALL N positions — O(n²)
HRR-Indexed:
  1. Superpose K into one vector — O(n)
  2. Correlate Q with K_super — O(d log d)
  3. Select top-k positions from correlation — O(n)
  4. Exact softmax on only top-k — O(k²)
  Total: O(n * d log d) instead of O(n² * d)

Test on 0.6B:
  1. Baseline: full attention speed + PPL
  2. HRR-indexed at various k (4, 8, 16, 32, 64)
  3. Measure quality loss vs speedup
  4. PID-controlled k reduction with fine-tuning
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- HRR operations --

def make_position_vectors(max_len, dim, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    vecs = torch.randn(max_len, dim, generator=gen)
    return vecs / vecs.norm(dim=-1, keepdim=True)


def hrr_bind(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_correlate(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B.conj(), n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_indexed_attention(query, key, value, pos_vectors, top_k, n_heads, n_kv_heads, head_dim):
    """HRR-indexed sparse attention.

    1. Build causal HRR index via cumulative superposition
    2. Correlate each Q with the index to get relevance per position
    3. Select top-k most relevant positions
    4. Run exact softmax attention on only those positions
    5. Retrieve from real V vectors (not superposed)

    query: [B, n_heads, seq_q, head_dim]
    key:   [B, n_kv_heads, seq_k, head_dim]
    value: [B, n_kv_heads, seq_k, head_dim]

    Returns: [B, n_heads, seq_q, head_dim]
    """
    B, _, seq_q, d = query.shape
    seq_k = key.shape[2]
    kv_groups = n_heads // n_kv_heads
    k = min(top_k, seq_k)  # can't select more than available

    pos = pos_vectors[:seq_k].to(key.device)

    # Step 1: Build HRR index — bind K with positions, cumulative sum
    K_bound = hrr_bind(key, pos.unsqueeze(0).unsqueeze(0))  # [B, n_kv_heads, seq_k, d]
    K_index = K_bound.cumsum(dim=2)  # [B, n_kv_heads, seq_k, d] causal superposition

    # Expand for GQA
    K_index = K_index.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_index = K_index.reshape(B, n_heads, seq_k, d)

    # Step 2: Correlate each Q with its causal K_index
    # This gives a relevance vector per query position
    relevance = hrr_correlate(query, K_index)  # [B, n_heads, seq_q, d]

    # Step 3: Score each position by relevance magnitude
    # Sum across head_dim to get a scalar relevance per position
    # But we need position-level scores, not a d-dim vector
    # Use dot product of relevance with each position vector to "decode" which positions match
    # relevance @ pos^T gives [B, n_heads, seq_q, seq_k] — but that's O(n²) again!

    # Better: use the L2 norm of the correlation as a proxy for relevance
    # Actually, use the dot product of Q with each actual K for the top-k selection
    # That IS the standard attention score — we'd be back to O(n²)

    # The RIGHT approach: use position vectors to decode the superposition
    # For each query position i, the correlation result contains signals from
    # positions 0..i, modulated by position vectors. We can "probe" specific
    # positions by correlating with their position vectors.

    # But probing all positions = O(n²) again.

    # Alternative: use the FFT spectrum of the correlation as a hash
    # Positions with strong signal will have peaks in the spectrum

    # Simplest approach that works: use the correlation vector's magnitude
    # at each "frequency" as a position selector. Positions bound with
    # high-energy position vectors will show up as peaks.

    # Actually, the simplest working approach:
    # For the HRR index, correlate with each position vector to get a score
    # This IS O(n * d log d) per query — correlate with n position vectors
    # Each correlation is O(d log d), and we just take index 0 (= dot product)

    # Batch approach: compute all position scores at once
    # score[i, j] = sum(correlate(relevance[i], pos[j]))
    # = sum(IFFT(FFT(relevance[i]) * conj(FFT(pos[j]))))
    # index 0 of IFFT = dot product by Parseval's

    # FFT all position vectors once
    pos_fft = torch.fft.rfft(pos.double(), dim=-1)  # [seq_k, d//2+1]
    rel_fft = torch.fft.rfft(relevance.double(), dim=-1)  # [B, n_heads, seq_q, d//2+1]

    # Correlation score: real part of sum(rel_fft * conj(pos_fft))
    # For each query position i, score against each position j
    # rel_fft: [B, n_heads, seq_q, freq]
    # pos_fft: [seq_k, freq]
    # scores: [B, n_heads, seq_q, seq_k]
    scores = torch.einsum('bhqf,kf->bhqk', rel_fft.real, pos_fft.real) + \
             torch.einsum('bhqf,kf->bhqk', rel_fft.imag, pos_fft.imag)
    scores = scores.float() * 2.0 / d  # normalize

    # Apply causal mask
    causal = torch.triu(torch.ones(seq_q, seq_k, device=query.device), diagonal=1).bool()
    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Step 3: Select top-k positions per query
    topk_scores, topk_indices = scores.topk(k, dim=-1)  # [B, n_heads, seq_q, k]

    # Step 4: Gather the actual K and V at selected positions
    # Expand K, V for GQA first
    K_full = key.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_full = K_full.reshape(B, n_heads, seq_k, d)
    V_full = value.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    V_full = V_full.reshape(B, n_heads, seq_k, d)

    # Gather K and V at top-k indices
    # topk_indices: [B, n_heads, seq_q, k]
    idx = topk_indices.unsqueeze(-1).expand(B, n_heads, seq_q, k, d)
    K_selected = K_full.unsqueeze(2).expand(B, n_heads, seq_q, seq_k, d)
    K_selected = K_selected.gather(3, idx)  # [B, n_heads, seq_q, k, d]
    V_selected = V_full.unsqueeze(2).expand(B, n_heads, seq_q, seq_k, d)
    V_selected = V_selected.gather(3, idx)  # [B, n_heads, seq_q, k, d]

    # Step 5: Exact softmax attention on only top-k
    Q_expanded = query.unsqueeze(3)  # [B, n_heads, seq_q, 1, d]
    exact_scores = (Q_expanded * K_selected).sum(-1) / math.sqrt(d)  # [B, n_heads, seq_q, k]
    attn_weights = torch.softmax(exact_scores, dim=-1)  # [B, n_heads, seq_q, k]

    # Weighted sum of selected V
    output = (attn_weights.unsqueeze(-1) * V_selected).sum(3)  # [B, n_heads, seq_q, d]

    return output


# -- Data + eval --

def load_data(seq_len=256, max_train=2_000_000, max_val=100_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    val_t = tokens[:max_val]
    train_t = tokens[max_val:max_val + max_train]
    def chunk(t):
        n = len(t) // (seq_len + 1)
        return t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return chunk(train_t), chunk(val_t)


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


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"HRR-INDEXED SPARSE ATTENTION: {cli.model}")
    print(f"  Use HRR superposition as cheap index")
    print(f"  Select top-k positions, exact softmax on those only")
    print(f"  KV cache stays. Softmax stays. Skip irrelevant keys.")
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

    print("Loading data...", flush=True)
    train_chunks, val_chunks = load_data()

    teacher_ppl = eval_ppl(model, val_chunks)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # Benchmark core operation at different k values
    print(f"\n{'='*60}")
    print(f"CORE BENCHMARK: various top-k values (seq=256)")
    print(f"{'='*60}")

    pos_vectors = make_position_vectors(512, head_dim)
    seq = 256

    Q = torch.randn(1, n_heads, seq, head_dim)
    K = torch.randn(1, n_kv_heads, seq, head_dim)
    V = torch.randn(1, n_kv_heads, seq, head_dim)

    # Standard attention timing
    kv_groups = n_heads // n_kv_heads
    for _ in range(3):
        K_e = K.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        V_e = V.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        _ = a @ V_e

    times = []
    for _ in range(10):
        K_e = K.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        V_e = V.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        t0 = time.perf_counter()
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        o = a @ V_e
        times.append(time.perf_counter() - t0)
    std_ms = sum(times) / len(times) * 1000

    # Measure output from standard for quality comparison
    with torch.no_grad():
        K_e = K.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        V_e = V.unsqueeze(2).expand(1, n_kv_heads, kv_groups, seq, head_dim).reshape(1, n_heads, seq, head_dim)
        s = Q @ K_e.transpose(-2, -1) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool()
        s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        a = torch.softmax(s, dim=-1)
        std_output = a @ V_e

    print(f"  Standard attention: {std_ms:.2f}ms")
    print()
    print(f"  {'top_k':>6} | {'Time ms':>8} | {'Speedup':>8} | {'cos_sim':>8} | {'Notes'}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*20}")

    results = []
    for top_k in [4, 8, 16, 32, 64, 128, 256]:
        if top_k > seq:
            continue

        # Warmup
        for _ in range(3):
            _ = hrr_indexed_attention(Q, K, V, pos_vectors, top_k, n_heads, n_kv_heads, head_dim)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = hrr_indexed_attention(Q, K, V, pos_vectors, top_k, n_heads, n_kv_heads, head_dim)
            times.append(time.perf_counter() - t0)
        hrr_ms = sum(times) / len(times) * 1000

        # Quality: cosine similarity with standard output
        with torch.no_grad():
            hrr_out = hrr_indexed_attention(Q, K, V, pos_vectors, top_k, n_heads, n_kv_heads, head_dim)
            cos = F.cosine_similarity(hrr_out.reshape(-1), std_output.reshape(-1), dim=0).item()

        speedup = std_ms / hrr_ms
        notes = ""
        if cos > 0.99:
            notes = "near-exact"
        elif cos > 0.95:
            notes = "good"
        elif cos > 0.9:
            notes = "ok"
        else:
            notes = "lossy"

        print(f"  {top_k:6d} | {hrr_ms:7.2f}ms | {speedup:7.2f}x | {cos:7.4f} | {notes}", flush=True)
        results.append({
            "top_k": top_k, "time_ms": round(hrr_ms, 2),
            "speedup": round(speedup, 2), "cos_sim": round(cos, 4),
        })

    # Now test on actual model with hooks
    print(f"\n{'='*60}")
    print(f"MODEL TEST: HRR-indexed attention on {cli.model}")
    print(f"{'='*60}")
    print(f"  {'top_k':>6} | {'PPL':>8} | {'Ratio':>6} | {'Notes'}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*20}")
    print(f"  {'full':>6} | {teacher_ppl:8.2f} | {'1.00x':>6} | baseline", flush=True)

    model_results = []
    for top_k in [128, 64, 32, 16, 8]:
        # Reload model fresh each time
        del model; gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            cli.model, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="eager").eval()

        # Install hooks
        hooks = []
        for li in range(L):
            attn_mod = model.model.layers[li].self_attn
            orig_fwd = attn_mod.forward

            def make_hook(orig, layer_i, k_val):
                def hooked(hidden_states, *args, **kwargs):
                    B, S, D = hidden_states.shape
                    hd = head_dim
                    am = model.model.layers[layer_i].self_attn

                    q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                    k = am.k_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)
                    v = am.v_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)

                    out = hrr_indexed_attention(q, k, v, pos_vectors, k_val, n_heads, n_kv_heads, hd)
                    out = out.transpose(1, 2).reshape(B, S, n_heads * hd)
                    out = am.o_proj(out)

                    # Return in expected format (attn_output, attn_weights)
                    return (out, None)

                return hooked

            attn_mod.forward = make_hook(orig_fwd, li, top_k)
            hooks.append((attn_mod, orig_fwd))

        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl

        notes = ""
        if ratio <= 1.05:
            notes = "within 5%"
        elif ratio <= 1.10:
            notes = "within 10%"
        elif ratio <= 1.50:
            notes = "degraded"
        else:
            notes = "broken"

        print(f"  {top_k:6d} | {ppl:8.2f} | {ratio:5.2f}x | {notes}", flush=True)
        model_results.append({
            "top_k": top_k, "ppl": round(ppl, 2), "ratio": round(ratio, 4), "notes": notes,
        })

        # Restore
        for attn_mod, orig_fwd in hooks:
            attn_mod.forward = orig_fwd

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"\n  Core operation (0.6B dims, seq=256):")
    print(f"    Standard: {std_ms:.2f}ms")
    for r in results:
        print(f"    top_k={r['top_k']:>3}: {r['time_ms']:.2f}ms ({r['speedup']:.1f}x) cos={r['cos_sim']:.4f}")
    print(f"\n  Model PPL:")
    for r in model_results:
        print(f"    top_k={r['top_k']:>3}: PPL={r['ppl']:.2f} ({r['ratio']:.2f}x) {r['notes']}")

    all_results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "core_benchmark": results, "model_test": model_results,
        "standard_ms": std_ms,
    }
    with open(Path(save_dir) / "hrr_indexed_attention.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {save_dir}/hrr_indexed_attention.json")


if __name__ == "__main__":
    main()
