"""Manifold Inference: one pass, per-token width × length from the router.

Each token gets:
  - width: number of active heads (1 to N_HEADS), from the manifold
  - length: number of active layers (1 to N_LAYERS), from the manifold

Inactive heads are zeroed in attention output. Exited layers pass through
on residual. One batch, one forward, continuous resolution per token.
"""
import torch
import torch.nn.functional as F
import time
device = 'cuda'

from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*70, flush=True)
print("MANIFOLD INFERENCE: per-token width × length", flush=True)
print("="*70, flush=True)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

N_LAYERS = model.config.num_hidden_layers      # 40
N_HEADS = model.config.num_attention_heads      # 40
N_KV = model.config.num_key_value_heads         # 8
HEAD_DIM = model.config.hidden_size // N_HEADS  # 128
H = model.config.hidden_size                    # 5120

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}", flush=True)

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

# ═══════════════════════════════════════════════════════
# Custom forward with per-token width mask
# ═══════════════════════════════════════════════════════
def manifold_forward(model, input_ids, n_active_heads, n_active_layers):
    """One forward pass. Per-token head masking + layer exit.

    n_active_heads: int, how many heads each token uses (uniform for now)
    n_active_layers: int, after this layer, pass through on residual
    """
    h = model.model.embed_tokens(input_ids)
    B, T, D = h.shape
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    pos_emb = model.model.rotary_emb(h, pos_ids)

    # Build head mask: keep first n_active_heads, zero the rest
    # Applied after attention output, before residual add
    head_scale = torch.zeros(N_HEADS, device=device, dtype=h.dtype)
    head_scale[:n_active_heads] = N_HEADS / n_active_heads  # scale up to compensate
    # Reshape for broadcast: [1, 1, N_HEADS, 1]
    head_scale = head_scale.view(1, 1, N_HEADS, 1)

    for i in range(N_LAYERS):
        layer = model.model.layers[i]

        if i >= n_active_layers:
            # Past exit: residual pass-through
            continue

        # Run attention with head masking
        residual = h
        h_norm = layer.input_layernorm(h)

        attn = layer.self_attn
        # Compute Q, K, V and reshape
        q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
        k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
        v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

        # Apply Q/K norm (Qwen3 qk_norm operates on per-head dim)
        if hasattr(attn, 'q_norm') and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, 'k_norm') and attn.k_norm is not None:
            k = attn.k_norm(k)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, KV, T, D]
        v = v.transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = pos_emb
        q, k = apply_rotary(q, k, cos, sin)

        # GQA: expand K, V to match Q heads
        n_rep = N_HEADS // N_KV
        k = k.repeat_interleave(n_rep, dim=1)  # [B, H, T, D]
        v = v.repeat_interleave(n_rep, dim=1)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # [B, H, T, D]

        # HEAD MASKING: scale active heads, zero inactive
        attn_out = attn_out * head_scale  # [B, H, T, D] × [1, 1, H, 1]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        # MLP (full — bulk, no compression)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return model.lm_head(model.model.norm(h))


def apply_rotary(q, k, cos, sin):
    """Apply rotary position embeddings."""
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)

    def rotate(x, cos, sin):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rotate(q, cos, sin), rotate(k, cos, sin)


# Test: width × length grid
print(f"\n{'Config':>25} {'tok/s':>7} {'speedup':>8} text", flush=True)
print("-" * 80, flush=True)

for n_heads in [40, 20, 10, 5, 2, 1]:
    for n_layers in [40, 30, 20]:
        gen_ids = ids.clone()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            for _ in range(N_GEN):
                logits = manifold_forward(model, gen_ids, n_heads, n_layers)
                tok = logits[0, -1:].argmax(-1)
                gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
        torch.cuda.synchronize()
        tps = N_GEN / (time.time() - t0)
        text = tokenizer.decode(gen_ids[0][ids.shape[1]:], skip_special_tokens=True)
        label = f"W={n_heads:2d}/{N_HEADS} L={n_layers:2d}/{N_LAYERS}"
        pct = (n_heads/N_HEADS) * (n_layers/N_LAYERS) * 100
        print(f"  {label}  ({pct:4.1f}%)  {tps:6.1f}  {tps/base_tps:6.2f}x  {text[:40]}", flush=True)

print(f"\nBaseline: {base_tps:.1f} tok/s", flush=True)
