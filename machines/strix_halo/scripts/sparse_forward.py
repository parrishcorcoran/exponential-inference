"""Sparse forward: actual compute skip via weight slicing.

Router says (heads, layers) → forward computes ONLY those heads/layers.
No zeroing, no hooks, no wasted compute.
"""
import torch
import torch.nn.functional as F
import time
device = 'cuda'

print("="*70, flush=True)
print("SPARSE FORWARD: actual head skip via weight slicing", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

N_LAYERS = model.config.num_hidden_layers      # 40
N_HEADS = model.config.num_attention_heads      # 40
N_KV = model.config.num_key_value_heads         # 8
HEAD_DIM = model.config.hidden_size // N_HEADS  # 128
H = model.config.hidden_size                    # 5120
GQA_RATIO = N_HEADS // N_KV                     # 5

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV} gqa={GQA_RATIO}:1", flush=True)

ids = tokenizer("The future of AI will", return_tensors='pt').input_ids.to(device)
N_GEN = 64

# Baseline
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): base = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
torch.cuda.synchronize()
base_tps = N_GEN / (time.time() - t0)
base_text = tokenizer.decode(base[0][ids.shape[1]:], skip_special_tokens=True)
print(f"Baseline: {base_tps:.1f} tok/s [{base_text[:60]}]", flush=True)

# Precompute weight slices for each head count
# Q: [N_HEADS * HEAD_DIM, H] → per-head blocks of [HEAD_DIM, H]
# K: [N_KV * HEAD_DIM, H] → per-kv-head blocks
# V: same as K
# O: [H, N_HEADS * HEAD_DIM] → per-head blocks of [H, HEAD_DIM]

def sparse_forward(model, input_ids, active_q_heads, exit_layer):
    """Run only active_q_heads through exit_layer layers.

    active_q_heads: list of query head indices to activate
    exit_layer: stop after this many layers
    """
    B, T = input_ids.shape
    n_active = len(active_q_heads)

    # Figure out which KV heads we need (GQA: each KV head serves GQA_RATIO Q heads)
    active_kv_heads = sorted(set(h // GQA_RATIO for h in active_q_heads))
    n_active_kv = len(active_kv_heads)

    h = model.model.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = model.model.rotary_emb(h, pos_ids)
    cos = cos.unsqueeze(1)  # [B, 1, T, HD]
    sin = sin.unsqueeze(1)

    for i in range(exit_layer):
        layer = model.model.layers[i]
        attn = layer.self_attn

        residual = h
        h_norm = layer.input_layernorm(h)

        # SPARSE Q: only compute active query heads
        q_weight_full = attn.q_proj.weight.view(N_HEADS, HEAD_DIM, H)
        q_weight = q_weight_full[active_q_heads].reshape(n_active * HEAD_DIM, H)
        q = (h_norm @ q_weight.T).view(B, T, n_active, HEAD_DIM)

        # SPARSE K, V: only compute active KV heads
        k_weight_full = attn.k_proj.weight.view(N_KV, HEAD_DIM, H)
        k_weight = k_weight_full[active_kv_heads].reshape(n_active_kv * HEAD_DIM, H)
        k = (h_norm @ k_weight.T).view(B, T, n_active_kv, HEAD_DIM)

        v_weight_full = attn.v_proj.weight.view(N_KV, HEAD_DIM, H)
        v_weight = v_weight_full[active_kv_heads].reshape(n_active_kv * HEAD_DIM, H)
        v = (h_norm @ v_weight.T).view(B, T, n_active_kv, HEAD_DIM)

        # QK norms
        if attn.q_norm is not None:
            q = attn.q_norm(q)
        if attn.k_norm is not None:
            k = attn.k_norm(k)

        q = q.transpose(1, 2)  # [B, n_active, T, HD]
        k = k.transpose(1, 2)  # [B, n_active_kv, T, HD]
        v = v.transpose(1, 2)

        # Rotary
        rd = HEAD_DIM // 2
        q1, q2 = q[..., :rd], q[..., rd:]
        q = torch.cat([q1*cos[...,:rd] - q2*sin[...,:rd], q2*cos[...,:rd] + q1*sin[...,:rd]], -1)
        k1, k2 = k[..., :rd], k[..., rd:]
        k = torch.cat([k1*cos[...,:rd] - k2*sin[...,:rd], k2*cos[...,:rd] + k1*sin[...,:rd]], -1)

        # GQA expand for active heads only
        # Map active Q heads to their KV head index within active_kv_heads
        kv_index_map = {kv: idx for idx, kv in enumerate(active_kv_heads)}
        q_to_kv = [kv_index_map[qh // GQA_RATIO] for qh in active_q_heads]
        k_expanded = k[:, q_to_kv]  # [B, n_active, T, HD]
        v_expanded = v[:, q_to_kv]

        # Attention (only n_active heads, not N_HEADS)
        attn_out = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=True)
        # [B, n_active, T, HD]

        # SPARSE O projection: only project active heads back
        o_weight_full = attn.o_proj.weight.view(H, N_HEADS, HEAD_DIM)
        o_weight = o_weight_full[:, active_q_heads, :].reshape(H, n_active * HEAD_DIM)

        attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, n_active * HEAD_DIM)
        attn_proj = (attn_flat @ o_weight.T)  # [B, T, H]

        # Scale to compensate for missing heads
        attn_proj = attn_proj * (N_HEADS / n_active)

        if attn.o_proj.bias is not None:
            attn_proj = attn_proj + attn.o_proj.bias

        h = residual + attn_proj

        # MLP (full — depth is always full)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return model.lm_head(model.model.norm(h))


# Test: speed and quality at different (heads, layers)
print(f"\nSparse forward (actual compute skip):", flush=True)
print(f"{'Heads':>8} {'Layers':>7} {'Compute%':>9} {'tok/s':>7} {'speedup':>8} text", flush=True)
print("-" * 80, flush=True)

configs = [
    (list(range(40)), 40),   # full
    (list(range(20)), 40),   # half heads, full length
    (list(range(20)), 30),   # half heads, 75% length
    (list(range(10)), 40),   # quarter heads, full length
    (list(range(10)), 30),   # quarter heads, 75% length
    (list(range(5)), 40),    # 1/8 heads, full length
]

for heads, layers in configs:
    n_h = len(heads)
    compute = (n_h / N_HEADS) * (layers / N_LAYERS) * 100

    gen_ids = ids.clone()
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for _ in range(N_GEN):
            logits = sparse_forward(model, gen_ids, heads, layers)
            tok = logits[0, -1:].argmax(-1)
            gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(gen_ids[0][ids.shape[1]:], skip_special_tokens=True)

    print(f"  {n_h:2d}/40h {layers:3d}/40L {compute:7.1f}% {tps:6.1f} {tps/base_tps:7.2f}x "
          f"{text[:40]}", flush=True)

print(f"\nBaseline: {base_tps:.1f} tok/s", flush=True)

with open("machines/strix_halo/results/sparse_forward.json", "w") as f:
    import json
    json.dump({"baseline_tps": base_tps}, f, indent=2)
print("Saved sparse_forward.json", flush=True)
