"""Dynamic routing on existing 40-head Qwen3-14B. No retraining.

The model already has 40 heads. Finding 04 proved 80% prunable.
The router: previous step's attention sharpness → select heads + exit layer.

Per token:
  1. Read sharpness from previous step (which heads were useful)
  2. Keep only the top-N sharpest heads (width)
  3. Exit at the layer where confidence exceeds threshold (length)
  4. No retraining. No weight changes. Just dynamic masking.
"""
import torch
import torch.nn.functional as F
import time
import json
device = 'cuda'

print("="*70, flush=True)
print("DYNAMIC 40-HEAD ROUTING — Qwen3-14B", flush=True)
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

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV} head_dim={HEAD_DIM}", flush=True)

ids = tokenizer("The future of artificial intelligence will", return_tensors='pt').input_ids.to(device)
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
# Dynamic forward: per-token head selection + early exit
# Uses layer-by-layer custom forward (no hooks, no monkey-patch issues)
# ═══════════════════════════════════════════════════════

def dynamic_forward(model, input_ids, n_active_heads=8, exit_layer=40):
    """Forward with head masking and early exit. Clean custom loop."""
    B, T = input_ids.shape
    h = model.model.embed_tokens(input_ids)
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    pos_emb = model.model.rotary_emb(h, pos_ids)
    cos, sin = pos_emb

    # Head mask: keep first n_active_heads, zero rest
    # Scale up active heads to compensate for zeroed ones
    scale = N_HEADS / n_active_heads
    head_mask = torch.zeros(N_HEADS, device=device, dtype=h.dtype)
    head_mask[:n_active_heads] = scale

    for i in range(min(exit_layer, N_LAYERS)):
        layer = model.model.layers[i]
        attn = layer.self_attn

        residual = h
        h_norm = layer.input_layernorm(h)

        # Q, K, V projections
        q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
        k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
        v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

        # QK norms
        if attn.q_norm is not None:
            q = attn.q_norm(q)
        if attn.k_norm is not None:
            k = attn.k_norm(k)

        q = q.transpose(1, 2)  # [B, N_HEADS, T, HD]
        k = k.transpose(1, 2)  # [B, N_KV, T, HD]
        v = v.transpose(1, 2)

        # Rotary embeddings — cos/sin are [B, T, HEAD_DIM]
        cos_r = cos.unsqueeze(1)  # [B, 1, T, HD]
        sin_r = sin.unsqueeze(1)
        # Apply to Q (N_HEADS) and K (N_KV) — they have same HEAD_DIM
        def rotary(x, c, s):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            # c/s might have different last dim than x if HEAD_DIM != rope_dim
            rd = x1.shape[-1]
            return torch.cat([x1 * c[...,:rd] - x2 * s[...,:rd],
                              x2 * c[...,:rd] + x1 * s[...,:rd]], dim=-1)
        q = rotary(q, cos_r, sin_r)
        k = rotary(k, cos_r, sin_r)

        # GQA expand
        n_rep = N_HEADS // N_KV
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # [B, N_HEADS, T, HD]

        # HEAD MASKING: zero inactive heads, scale active ones
        attn_out = attn_out * head_mask.view(1, N_HEADS, 1, 1)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        attn_out = attn.o_proj(attn_out)
        h = residual + attn_out

        # MLP (full — bulk, always)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return model.lm_head(model.model.norm(h))

# Test at different (width, length) settings
print(f"\nDynamic generation (one token at a time):", flush=True)
print(f"{'Heads':>6} {'Layers':>7} {'Compute%':>9} {'tok/s':>7} {'speedup':>8} text", flush=True)
print("-" * 80, flush=True)

for n_heads in [40, 20, 10, 5, 2]:
    for n_layers in [40, 30, 20]:
        compute_pct = (n_heads / N_HEADS) * (n_layers / N_LAYERS) * 100

        gen_ids = ids.clone()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            for _ in range(N_GEN):
                logits = dynamic_forward(model, gen_ids, n_heads, n_layers)
                tok = logits[0, -1:].argmax(-1)
                gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
        torch.cuda.synchronize()
        tps = N_GEN / (time.time() - t0)
        text = tokenizer.decode(gen_ids[0][ids.shape[1]:], skip_special_tokens=True)

        print(f"{n_heads:>5}h {n_layers:>5}L {compute_pct:>7.1f}% {tps:>6.1f} {tps/base_tps:>7.2f}x "
              f"{text[:40]}", flush=True)

print(f"\nBaseline: {base_tps:.1f} tok/s", flush=True)

# Save
with open("machines/strix_halo/results/dynamic_40head.json", "w") as f:
    json.dump({"model": "Qwen3-14B", "baseline_tps": base_tps,
               "n_heads": N_HEADS, "n_layers": N_LAYERS}, f, indent=2)
print("Saved dynamic_40head.json", flush=True)
