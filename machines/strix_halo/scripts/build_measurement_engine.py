"""Measurement-based thin-slice engine.

No signals (entropy, sharpness). Only measurements:
- Width: measured from hidden state geometry after coarse layers
- Length: measured from layer-to-layer cosine (rotation flattening)
- KV depth: measured from KV rank saturation

First 5 layers: full width (coarse projection, biggest rotations).
Measure geometry. Adapt for remaining layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("MEASUREMENT-BASED ENGINE — no signals, only geometry")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.config.hidden_size // N_HEADS
H = model.config.hidden_size
GQA_RATIO = N_HEADS // N_KV

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

# ═══════════════════════════════════════════════════════
# Measurement functions
# ═══════════════════════════════════════════════════════

def measure_head_geometry(h_norm, layer):
    """Measure which Q heads produce geometrically distinct outputs.

    Project h through each Q head separately. Heads whose projections
    are nearly collinear with others are redundant — the manifold
    doesn't need them at this position.

    Returns: sorted list of head indices by geometric distinctness.
    """
    q_weight = layer.self_attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
    # Project through each head: [N_HEADS, HEAD_DIM]
    per_head = (h_norm[0, 0] @ q_weight.reshape(N_HEADS, -1).T)  # [N_HEADS * HD] nope
    # Actually: per_head[i] = h @ q_weight[i].T → [HEAD_DIM]
    projections = torch.stack([h_norm[0, 0] @ q_weight[i].T for i in range(N_HEADS)])  # [N_HEADS, HEAD_DIM]

    # Measure pairwise cosine — heads with unique directions are important
    proj_norm = F.normalize(projections.float(), dim=-1)
    cos_matrix = proj_norm @ proj_norm.T  # [N_HEADS, N_HEADS]

    # Each head's "distinctness": 1 - max cosine with any other head
    cos_matrix.fill_diagonal_(-1)  # ignore self
    max_cos = cos_matrix.max(dim=-1).values  # [N_HEADS]
    distinctness = 1.0 - max_cos  # higher = more unique

    # Return heads sorted by distinctness (most unique first)
    sorted_heads = distinctness.argsort(descending=True).tolist()
    return sorted_heads, distinctness


def measure_rotation(h_prev, h_curr):
    """Measure rotation between consecutive layers.

    Returns cosine similarity — high cos = small rotation = stabilizing.
    """
    return F.cosine_similarity(
        h_prev.view(1, -1).float(),
        h_curr.view(1, -1).float()
    ).item()


def sparse_layer(h, layer, active_q_heads, active_kv_heads, cos, sin, kv_cache, layer_idx):
    """One layer with sparse Q, sparse KV, full MLP."""
    B, T, D = h.shape
    n_active_q = len(active_q_heads)
    n_active_kv = len(active_kv_heads)
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

    # Sparse Q
    q_weight = attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
    q = (h_norm @ q_weight[active_q_heads].reshape(-1, H).T).view(B, T, n_active_q, HEAD_DIM)

    # Sparse KV
    k_weight = attn.k_proj.weight.view(N_KV, HEAD_DIM, H)
    k = (h_norm @ k_weight[active_kv_heads].reshape(-1, H).T).view(B, T, n_active_kv, HEAD_DIM)
    v_weight = attn.v_proj.weight.view(N_KV, HEAD_DIM, H)
    v = (h_norm @ v_weight[active_kv_heads].reshape(-1, H).T).view(B, T, n_active_kv, HEAD_DIM)

    # QK norms
    if attn.q_norm is not None:
        q = attn.q_norm(q)
    if attn.k_norm is not None:
        k = attn.k_norm(k)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
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

    # Update KV cache with sparse entries
    if layer_idx in kv_cache:
        k_cached, v_cached = kv_cache[layer_idx]
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

    # GQA expand
    kv_indices = [qh // GQA_RATIO for qh in active_q_heads]
    k_exp = k_full[:, kv_indices]
    v_exp = v_full[:, kv_indices]

    # Attention
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

    # Sparse O
    o_weight = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
    o_active = o_weight[:, active_q_heads, :].reshape(H, n_active_q * HEAD_DIM)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, n_active_q * HEAD_DIM)
    attn_proj = (attn_flat @ o_active.T) * (N_HEADS / n_active_q)

    if attn.o_proj.bias is not None:
        attn_proj = attn_proj + attn.o_proj.bias

    h = residual + attn_proj

    # Full MLP
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, kv_cache


def measurement_generate(model, input_ids, max_new_tokens=64,
                         coarse_layers=5, rotation_threshold=0.995,
                         min_heads=4, min_kv=1):
    """Generate with measurement-based routing.

    Phase 1 (layers 0-coarse_layers): full width. Measure geometry.
    Phase 2 (remaining layers): adapted width/kv based on measurements.
    Exit when rotation between layers exceeds threshold (projection stabilized).
    """
    B = input_ids.shape[0]

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    kv_cache = {}
    for layer_idx in range(N_LAYERS):
        kv_cache[layer_idx] = (
            past.layers[layer_idx].keys.clone(),
            past.layers[layer_idx].values.clone(),
        )

    logits = out.logits[0, -1]
    next_tok = logits.argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]

    telem = {"q_heads": [], "kv_heads": [], "layers": [], "rotations": []}

    # Track KV rank for depth measurement
    prev_kv_rank = None

    for step in range(max_new_tokens - 1):
        seq_len += 1

        h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
        pos_ids = torch.tensor([[seq_len - 1]], device=device)
        cos, sin = model.model.rotary_emb(h, pos_ids)

        exit_layer = N_LAYERS
        h_prev = h.clone()
        rotations = []

        with torch.no_grad():
            # ── PHASE 1: coarse layers at full width ──
            for i in range(min(coarse_layers, N_LAYERS)):
                all_q = list(range(N_HEADS))
                all_kv = list(range(N_KV))
                h, kv_cache = sparse_layer(h, model.model.layers[i], all_q, all_kv,
                                           cos, sin, kv_cache, i)

                rot = measure_rotation(h_prev, h)
                rotations.append(rot)
                h_prev = h.clone()

            # ── MEASURE: head geometry after coarse layers ──
            h_norm = model.model.layers[coarse_layers].input_layernorm(h) if coarse_layers < N_LAYERS else h
            sorted_heads, distinctness = measure_head_geometry(
                model.model.layers[min(coarse_layers, N_LAYERS-1)].input_layernorm(h),
                model.model.layers[min(coarse_layers, N_LAYERS-1)]
            )

            # Width: keep heads above median distinctness
            median_d = distinctness.median().item()
            active_q = [hd for hd in sorted_heads if distinctness[hd].item() > median_d * 0.5]
            active_q = active_q[:max(min_heads, len(active_q))]
            if len(active_q) < min_heads:
                active_q = sorted_heads[:min_heads]

            # KV depth: measure rank growth
            ref_layer = min(coarse_layers - 1, N_LAYERS - 1)
            if ref_layer in kv_cache:
                k_ref = kv_cache[ref_layer][0][0].float()  # [N_KV, T, HD]
                k_flat = k_ref.permute(1, 0, 2).reshape(k_ref.shape[1], -1)
                if k_flat.shape[0] > 1:
                    svd = torch.linalg.svdvals(k_flat)
                    s_norm = svd / svd.sum()
                    s_ent = -(s_norm * (s_norm + 1e-10).log()).sum()
                    kv_rank = torch.exp(s_ent).item()
                else:
                    kv_rank = 1.0

                # If rank barely grew from last step, reduce KV heads
                if prev_kv_rank is not None:
                    rank_growth = kv_rank - prev_kv_rank
                    if rank_growth < 0.3:
                        n_kv = max(min_kv, N_KV // 4)  # manifold saturating
                    elif rank_growth < 0.5:
                        n_kv = max(min_kv, N_KV // 2)
                    else:
                        n_kv = N_KV  # still growing, need all angles
                else:
                    n_kv = N_KV
                prev_kv_rank = kv_rank
            else:
                n_kv = N_KV

            # Ensure KV heads cover the Q heads' GQA groups
            needed_kv = sorted(set(qh // GQA_RATIO for qh in active_q))
            active_kv = list(needed_kv)
            for kv_h in range(N_KV):
                if len(active_kv) >= n_kv:
                    break
                if kv_h not in active_kv:
                    active_kv.append(kv_h)
            active_kv = sorted(active_kv)

            # ── PHASE 2: remaining layers with adapted width ──
            for i in range(coarse_layers, N_LAYERS):
                h, kv_cache = sparse_layer(h, model.model.layers[i], active_q, active_kv,
                                           cos, sin, kv_cache, i)

                rot = measure_rotation(h_prev, h)
                rotations.append(rot)
                h_prev = h.clone()

                # Length: exit when rotation stabilizes
                if i >= 10 and rot > rotation_threshold:
                    exit_layer = i + 1
                    break

        # Final projection
        h_out = final_norm(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        next_tok = logits.argmax(-1).item()
        gen_tokens.append(next_tok)

        telem["q_heads"].append(len(active_q))
        telem["kv_heads"].append(len(active_kv))
        telem["layers"].append(exit_layer)
        telem["rotations"].append(rotations[-1] if rotations else 0)

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

# Baseline
print(f"\n{'='*60}")
print("BASELINE")
print(f"{'='*60}")

ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

base_results = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_results.append({"tps": tps, "text": text})
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

avg_base = sum(r["tps"] for r in base_results) / len(base_results)
print(f"  Average: {avg_base:.1f} tok/s")

# Measurement-based
print(f"\n{'='*60}")
print("MEASUREMENT-BASED THIN SLICES")
print(f"{'='*60}")

configs = [
    {"name": "tight",     "coarse": 5, "rot_thresh": 0.998, "min_h": 4, "min_kv": 1},
    {"name": "moderate",  "coarse": 5, "rot_thresh": 0.995, "min_h": 6, "min_kv": 2},
    {"name": "relaxed",   "coarse": 5, "rot_thresh": 0.990, "min_h": 8, "min_kv": 2},
    {"name": "coarse_10", "coarse": 10, "rot_thresh": 0.995, "min_h": 4, "min_kv": 1},
]

for cfg in configs:
    print(f"\n  Config: {cfg['name']} (coarse={cfg['coarse']}, rot={cfg['rot_thresh']}, "
          f"min_h={cfg['min_h']}, min_kv={cfg['min_kv']})")

    cfg_tps = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            tokens, telem = measurement_generate(
                model, ids, max_new_tokens=N_GEN,
                coarse_layers=cfg["coarse"],
                rotation_threshold=cfg["rot_thresh"],
                min_heads=cfg["min_h"],
                min_kv=cfg["min_kv"],
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        n_gen = len(tokens)
        tps = n_gen / elapsed
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        avg_q = sum(telem["q_heads"]) / max(len(telem["q_heads"]), 1)
        avg_kv = sum(telem["kv_heads"]) / max(len(telem["kv_heads"]), 1)
        avg_l = sum(telem["layers"]) / max(len(telem["layers"]), 1)
        speedup = tps / avg_base

        cfg_tps.append(tps)
        print(f"    {tps:>5.1f} tok/s ({speedup:.2f}x) Q={avg_q:.0f}/{N_HEADS} "
              f"KV={avg_kv:.0f}/{N_KV} L={avg_l:.0f}/{N_LAYERS}")
        print(f"    [{text[:70]}]")

    avg_tps = sum(cfg_tps) / len(cfg_tps)
    print(f"  Average: {avg_tps:.1f} tok/s ({avg_tps/avg_base:.2f}x)")

# Quality comparison
print(f"\n{'='*60}")
print("TEXT QUALITY")
print(f"{'='*60}")

for prompt in prompts[:2]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt}'")

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    base_text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  Full:  '{base_text[:80]}'")

    with torch.no_grad():
        tokens, telem = measurement_generate(model, ids, max_new_tokens=N_GEN,
                                             coarse_layers=5, rotation_threshold=0.995,
                                             min_heads=6, min_kv=2)
    meas_text = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"  Meas:  '{meas_text[:80]}'")

    base_ids = out[0][ids.shape[1]:ids.shape[1]+len(tokens)].tolist()
    match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
    print(f"  Match: {match}/{min(len(base_ids), len(tokens))} = {match/max(min(len(base_ids),len(tokens)),1)*100:.0f}%")
    print(f"  Q={sum(telem['q_heads'])/max(len(telem['q_heads']),1):.0f} "
          f"KV={sum(telem['kv_heads'])/max(len(telem['kv_heads']),1):.0f} "
          f"L={sum(telem['layers'])/max(len(telem['layers']),1):.0f}")

print(f"\nDone.", flush=True)

with open("machines/strix_halo/results/measurement_engine.json", "w") as f:
    json.dump({"baseline": avg_base}, f, indent=2)
print("Saved.", flush=True)
