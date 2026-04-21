"""Thin-Slice Generation Engine — the overnight build.

Dynamic width (sparse Q heads) × dynamic length (early exit) × full depth (MLP always full).
Full K/V always computed and cached (only 8 KV heads — cheap).
Sparse Q (10-20 of 40 query heads) saves the big compute.

The router reads manifold signals from the previous step:
- Width: logit entropy → low entropy = few heads needed, high = more heads
- Length: confidence threshold → high confidence at layer L = exit at L

Multiple thin slices per wall-clock step. Each one adds accurate KV depth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("THIN-SLICE GENERATION ENGINE — overnight build")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers      # 40
N_HEADS = model.config.num_attention_heads      # 40
N_KV = model.config.num_key_value_heads         # 8
HEAD_DIM = model.config.hidden_size // N_HEADS  # 128
H = model.config.hidden_size                    # 5120
GQA_RATIO = N_HEADS // N_KV                     # 5

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV} gqa={GQA_RATIO}:1")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Core: thin-slice layer with full KV cache, sparse Q
# ═══════════════════════════════════════════════════════

def thin_slice_layer(h, layer, active_q_heads, active_kv_heads, cos, sin, kv_cache, layer_idx):
    """One layer: sparse Q, sparse K/V, full MLP.

    All three projections are dynamic:
    - Q heads: determined by width (manifold spatial resolution)
    - KV heads: determined by depth need (how much new angle to add)
    - MLP: always full (holographic projection)

    Args:
        h: [B, 1, H] hidden state (single token)
        layer: transformer layer
        active_q_heads: list of query head indices to compute
        active_kv_heads: list of KV head indices to compute and cache
        cos, sin: rotary embeddings
        kv_cache: dict of {layer_idx: (k_cache, v_cache)}
            k_cache: [B, N_KV, cache_len, HD] — full width, sparse per token
            Positions where a KV head wasn't computed have zeros.
        layer_idx: which layer we're at

    Returns:
        h: [B, 1, H] output hidden state
        kv_cache: updated cache
    """
    B, T, D = h.shape
    n_active_q = len(active_q_heads)
    n_active_kv = len(active_kv_heads)
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    # SPARSE Q: only compute active query heads
    q_weight = attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
    q = (h_norm @ q_weight[active_q_heads].reshape(-1, H).T).view(B, T, n_active_q, HEAD_DIM)

    # SPARSE K/V: only compute active KV heads
    k_weight = attn.k_proj.weight.view(N_KV, HEAD_DIM, H)
    k = (h_norm @ k_weight[active_kv_heads].reshape(-1, H).T).view(B, T, n_active_kv, HEAD_DIM)
    v_weight = attn.v_proj.weight.view(N_KV, HEAD_DIM, H)
    v = (h_norm @ v_weight[active_kv_heads].reshape(-1, H).T).view(B, T, n_active_kv, HEAD_DIM)

    # QK norms
    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)  # [B, n_active_q, T, HD]
    k = k.transpose(1, 2)  # [B, n_active_kv, T, HD]
    v = v.transpose(1, 2)

    # Rotary
    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)

    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    # Update KV cache — store sparse KV in full-width cache with zeros for inactive
    if layer_idx in kv_cache:
        k_cached, v_cached = kv_cache[layer_idx]
        cache_len = k_cached.shape[2]
        # New entry: full width with zeros, fill active heads
        k_new = torch.zeros(B, N_KV, 1, HEAD_DIM, device=h.device, dtype=h.dtype)
        v_new = torch.zeros(B, N_KV, 1, HEAD_DIM, device=h.device, dtype=h.dtype)
        for idx, kv_h in enumerate(active_kv_heads):
            k_new[:, kv_h, :, :] = k[:, idx, :, :]
            v_new[:, kv_h, :, :] = v[:, idx, :, :]
        k_full = torch.cat([k_cached, k_new], dim=2)
        v_full = torch.cat([v_cached, v_new], dim=2)
    else:
        k_full = torch.zeros(B, N_KV, 1, HEAD_DIM, device=h.device, dtype=h.dtype)
        v_full = torch.zeros(B, N_KV, 1, HEAD_DIM, device=h.device, dtype=h.dtype)
        for idx, kv_h in enumerate(active_kv_heads):
            k_full[:, kv_h, :, :] = k[:, idx, :, :]
            v_full[:, kv_h, :, :] = v[:, idx, :, :]
    kv_cache[layer_idx] = (k_full, v_full)

    # GQA: map active Q heads to their KV heads in the full cache
    kv_indices = [qh // GQA_RATIO for qh in active_q_heads]
    k_exp = k_full[:, kv_indices]  # [B, n_active_q, cache_len, HD]
    v_exp = v_full[:, kv_indices]

    # Attention (only n_active_q heads)
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    # SPARSE O projection
    o_weight = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
    o_active = o_weight[:, active_q_heads, :].reshape(H, n_active_q * HEAD_DIM)

    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, n_active_q * HEAD_DIM)
    attn_proj = (attn_flat @ o_active.T) * (N_HEADS / n_active_q)

    if attn.o_proj.bias is not None:
        attn_proj = attn_proj + attn.o_proj.bias

    h = residual + attn_proj

    # FULL MLP (always — holographic projection)
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, kv_cache


def thin_slice_generate(model, input_ids, max_new_tokens=64,
                        default_heads_frac=0.5, confidence_threshold=0.95,
                        min_heads=4, min_layers=10):
    """Generate with thin slices: dynamic width × dynamic length × full depth.

    Router:
    - Width: starts at default_heads_frac, adjusts based on previous step's entropy
    - Length: exits when logit-lens confidence exceeds threshold, min min_layers

    Returns: generated token ids, telemetry
    """
    B = input_ids.shape[0]

    # Step 1: Prefill with full model (optimal for prompt processing)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    # Convert HF cache to our dict format
    kv_cache = {}
    for layer_idx in range(N_LAYERS):
        kv_cache[layer_idx] = (
            past.layers[layer_idx].keys.clone(),
            past.layers[layer_idx].values.clone(),
        )

    # First token from prefill
    logits = out.logits[0, -1]
    probs = F.softmax(logits.float(), dim=-1)
    next_tok = logits.argmax(-1).item()

    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]

    # Telemetry
    telem = {
        "heads_used": [],
        "kv_heads_used": [],
        "layers_used": [],
        "exit_reasons": [],
    }

    # Manifold state from previous step
    prev_entropy = -(probs * (probs + 1e-10).log()).sum().item()
    prev_top1 = probs.max().item()

    lm_head_weight = model.lm_head.weight
    final_norm_layer = model.model.norm

    for step in range(max_new_tokens - 1):
        seq_len += 1

        # ── ROUTER: manifold signals → (width, length, kv_depth) ──
        # Width: entropy-based. Low entropy = defined position = few heads.
        max_entropy = 12.0
        entropy_frac = min(prev_entropy / max_entropy, 1.0)
        n_heads = max(min_heads, int(N_HEADS * (0.2 + 0.8 * entropy_frac)))
        n_heads = min(n_heads, N_HEADS)

        # KV depth: dynamic. Early in generation = more KV heads (need angles).
        # Later = fewer (manifold reconstructed). Also scales with entropy.
        # step 0-5: all 8 KV heads (building reconstruction)
        # step 5-15: 4-8 based on entropy
        # step 15+: 2-4 (manifold saturated, fewer new angles needed)
        if step < 5:
            n_kv = N_KV  # all 8
        elif step < 15:
            n_kv = max(2, int(N_KV * (0.3 + 0.7 * entropy_frac)))
        else:
            n_kv = max(1, int(N_KV * (0.15 + 0.4 * entropy_frac)))
        n_kv = min(n_kv, N_KV)

        # Select which heads
        active_q_heads = list(range(n_heads))
        # KV heads: need to cover the Q heads' GQA groups
        needed_kv = sorted(set(qh // GQA_RATIO for qh in active_q_heads))
        # Add more KV heads up to n_kv for depth
        active_kv_heads = list(needed_kv)
        for kv_h in range(N_KV):
            if len(active_kv_heads) >= n_kv:
                break
            if kv_h not in active_kv_heads:
                active_kv_heads.append(kv_h)
        active_kv_heads = sorted(active_kv_heads)

        # Embed the token
        h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))

        # Position
        pos_ids = torch.tensor([[seq_len - 1]], device=device)
        cos, sin = model.model.rotary_emb(h, pos_ids)

        # ── THIN SLICE: run layers with dynamic exit ──
        exit_layer = N_LAYERS
        exit_reason = "full"

        with torch.no_grad():
            for i in range(N_LAYERS):
                h, kv_cache = thin_slice_layer(
                    h, model.model.layers[i], active_q_heads, active_kv_heads,
                    cos, sin, kv_cache, i
                )

                # Length check: apply logit lens every few layers after min_layers
                if i >= min_layers and (i % 5 == 4 or i == N_LAYERS - 1):
                    h_check = final_norm_layer(h)
                    logits_check = F.linear(h_check, lm_head_weight)[0, 0]
                    probs_check = F.softmax(logits_check.float(), dim=-1)
                    conf = probs_check.max().item()

                    if conf >= confidence_threshold:
                        exit_layer = i + 1
                        exit_reason = f"conf={conf:.3f}"
                        break

        # Final projection
        h = final_norm_layer(h)
        logits = F.linear(h, lm_head_weight)[0, 0]
        probs = F.softmax(logits.float(), dim=-1)
        next_tok = logits.argmax(-1).item()

        gen_tokens.append(next_tok)

        # Update manifold state for next step's router
        prev_entropy = -(probs * (probs + 1e-10).log()).sum().item()
        prev_top1 = probs.max().item()

        # Telemetry
        telem["heads_used"].append(n_heads)
        telem["kv_heads_used"].append(len(active_kv_heads))
        telem["layers_used"].append(exit_layer)
        telem["exit_reasons"].append(exit_reason)

        # EOS check
        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "In computer science, the most fundamental data structure is",
    "The history of mathematics spans thousands of years and includes",
]

N_GEN = 64

print(f"\n{'='*60}")
print("BASELINE: full model generation")
print(f"{'='*60}")

# Warmup
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

baseline_results = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    tps = N_GEN / elapsed
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    baseline_results.append({"tps": tps, "text": text[:60]})
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

avg_baseline = sum(r["tps"] for r in baseline_results) / len(baseline_results)
print(f"  Average: {avg_baseline:.1f} tok/s", flush=True)

print(f"\n{'='*60}")
print("THIN SLICES: dynamic width × dynamic length")
print(f"{'='*60}")

# Test different router settings
configs = [
    {"name": "aggressive", "heads_frac": 0.3, "conf": 0.99, "min_heads": 4, "min_layers": 10},
    {"name": "moderate",   "heads_frac": 0.5, "conf": 0.95, "min_heads": 6, "min_layers": 15},
    {"name": "conservative", "heads_frac": 0.7, "conf": 0.90, "min_heads": 8, "min_layers": 20},
    {"name": "full_width_early_exit", "heads_frac": 1.0, "conf": 0.95, "min_heads": 40, "min_layers": 10},
    {"name": "half_heads_full_length", "heads_frac": 0.5, "conf": 0.99999, "min_heads": 20, "min_layers": 40},
]

all_results = {}

for cfg in configs:
    print(f"\n  Config: {cfg['name']}")
    print(f"  {'Prompt':>50} {'tok/s':>7} {'speedup':>8} {'avg_h':>6} {'avg_L':>6}")

    cfg_results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            tokens, telem = thin_slice_generate(
                model, ids, max_new_tokens=N_GEN,
                default_heads_frac=cfg["heads_frac"],
                confidence_threshold=cfg["conf"],
                min_heads=cfg["min_heads"],
                min_layers=cfg["min_layers"],
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        n_gen = len(tokens)
        tps = n_gen / elapsed
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        avg_heads = sum(telem["heads_used"]) / max(len(telem["heads_used"]), 1)
        avg_kv = sum(telem["kv_heads_used"]) / max(len(telem["kv_heads_used"]), 1)
        avg_layers = sum(telem["layers_used"]) / max(len(telem["layers_used"]), 1)
        speedup = tps / avg_baseline

        cfg_results.append({
            "tps": tps, "speedup": speedup, "text": text[:60],
            "avg_heads": avg_heads, "avg_kv": avg_kv, "avg_layers": avg_layers, "n_gen": n_gen,
        })
        print(f"  {prompt[:48]:>50} {tps:>6.1f} {speedup:>7.2f}x Q={avg_heads:>4.1f} KV={avg_kv:>3.1f} L={avg_layers:>4.1f}")

    avg_tps = sum(r["tps"] for r in cfg_results) / len(cfg_results)
    avg_speedup = avg_tps / avg_baseline
    print(f"  Average: {avg_tps:.1f} tok/s ({avg_speedup:.2f}x)")

    all_results[cfg["name"]] = {
        "avg_tps": avg_tps, "avg_speedup": avg_speedup,
        "config": cfg, "per_prompt": cfg_results,
    }

# ═══════════════════════════════════════════════════════
# Text quality comparison
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEXT QUALITY COMPARISON")
print(f"{'='*60}")

for prompt in prompts[:2]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt}'")

    # Baseline
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    base_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  Full:       '{base_text[:80]}'")

    # Moderate thin slice
    with torch.no_grad():
        tokens, telem = thin_slice_generate(model, ids, max_new_tokens=N_GEN,
                                            default_heads_frac=0.5,
                                            confidence_threshold=0.95,
                                            min_heads=6, min_layers=15)
    thin_text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"  Thin:       '{thin_text[:80]}'")

    # Token match
    with torch.no_grad():
        out_ids = out[0][ids.shape[1]:ids.shape[1]+len(tokens)].tolist()
    match = sum(1 for a, b in zip(out_ids, tokens) if a == b)
    total = min(len(out_ids), len(tokens))
    print(f"  Token match: {match}/{total} = {match/max(total,1)*100:.0f}%")
    print(f"  Avg Q heads: {sum(telem['heads_used'])/max(len(telem['heads_used']),1):.1f}/{N_HEADS}")
    print(f"  Avg KV heads: {sum(telem['kv_heads_used'])/max(len(telem['kv_heads_used']),1):.1f}/{N_KV}")
    print(f"  Avg layers: {sum(telem['layers_used'])/max(len(telem['layers_used']),1):.1f}/{N_LAYERS}")

# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Baseline: {avg_baseline:.1f} tok/s")
for name, res in all_results.items():
    print(f"  {name:>25}: {res['avg_tps']:.1f} tok/s ({res['avg_speedup']:.2f}x)")

with open("machines/strix_halo/results/thin_slice_engine.json", "w") as f:
    json.dump({
        "baseline_tps": avg_baseline,
        "results": {k: {"avg_tps": v["avg_tps"], "avg_speedup": v["avg_speedup"]}
                    for k, v in all_results.items()},
    }, f, indent=2)
print(f"\nSaved results.", flush=True)
