"""No-scale masking: zero inactive heads, DON'T scale active heads.

Previous tests ALL used scale = N_HEADS / n_active. This assumes
equal contribution per head. What if we just... don't scale?

The active heads contribute what they contribute. The missing heads
contribute zero. The residual stream gets less from attention.
The MLP sees a slightly different input.

Also test: scale=1 for active, scale=0 for inactive (pure masking).
And: what happens at different fractions WITHOUT any scaling?
"""
import torch
import torch.nn.functional as F
import time

device = "cuda"

print("=" * 70)
print("NO-SCALE MASKING: does removing the scale fix quality?")
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

lm_head_weight = model.lm_head.weight
final_norm_layer = model.model.norm

print(f"L={N_LAYERS} H={H}")


def masked_generate(model, input_ids, max_new_tokens, head_mask, split_layer=0):
    """Generate with head masking applied to layers split_layer..N_LAYERS.

    head_mask: [N_HEADS] tensor applied as attn_out * mask before O-proj reshape.
    split_layer: layers before this use full width.
    """
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    kv = {}
    for i in range(N_LAYERS):
        kv[i] = (past.layers[i].keys.clone(), past.layers[i].values.clone())

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]

    for step in range(max_new_tokens - 1):
        seq_len += 1
        with torch.no_grad():
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
            pos_ids = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos_ids)

            for i in range(N_LAYERS):
                layer = model.model.layers[i]
                attn = layer.self_attn
                B, T, D = h.shape

                residual = h
                h_norm = layer.input_layernorm(h)

                q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
                k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
                v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

                if attn.q_norm is not None: q = attn.q_norm(q)
                if attn.k_norm is not None: k = attn.k_norm(k)

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                rd = HEAD_DIM // 2
                cos_r = cos.unsqueeze(1)
                sin_r = sin.unsqueeze(1)
                q1, q2 = q[..., :rd], q[..., rd:]
                q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                               q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
                k1, k2 = k[..., :rd], k[..., rd:]
                k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                               k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

                k_full = torch.cat([kv[i][0], k], dim=2)
                v_full = torch.cat([kv[i][1], v], dim=2)
                kv[i] = (k_full, v_full)

                k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
                v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

                attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

                # Apply mask to layers >= split_layer
                if i >= split_layer:
                    attn_out = attn_out * head_mask.view(1, N_HEADS, 1, 1)

                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
                attn_out = attn.o_proj(attn_out)

                h = residual + attn_out
                residual = h
                h = residual + layer.mlp(layer.post_attention_layernorm(h))

        h_out = final_norm_layer(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        next_tok = logits.argmax(-1).item()
        gen_tokens.append(next_tok)

        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens


# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "In computer science, the most fundamental data structure is",
    "The history of mathematics spans thousands of years and includes",
]
N_GEN = 64

print(f"\nBASELINE:")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)

base_texts = {}
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_texts[prompt] = (out[0][ids.shape[1]:].tolist(), text)
    print(f"  [{text[:60]}]")

# Test different masks and scales
print(f"\nMASKING EXPERIMENTS (all layers):")
print(f"{'Config':>35} {'Match%':>7} {'Text sample':>50}")
print("-" * 95)

for n_active in [40, 30, 20, 10]:
    for scale_type in ["full_scale", "no_scale", "sqrt_scale", "half_scale"]:
        mask = torch.zeros(N_HEADS, device=device, dtype=torch.bfloat16)

        if scale_type == "full_scale":
            s = N_HEADS / n_active
        elif scale_type == "no_scale":
            s = 1.0
        elif scale_type == "sqrt_scale":
            s = (N_HEADS / n_active) ** 0.5
        elif scale_type == "half_scale":
            s = 1.0 + (N_HEADS / n_active - 1.0) * 0.5  # halfway between 1 and full

        mask[:n_active] = s

        all_match = []
        sample_text = ""
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            with torch.no_grad():
                tokens = masked_generate(model, ids, N_GEN, mask, split_layer=0)

            base_ids = base_texts[prompt][0][:len(tokens)]
            match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
            all_match.append(match / max(len(base_ids), 1) * 100)

            if not sample_text:
                sample_text = tokenizer.decode(tokens, skip_special_tokens=True)[:45]

        avg_match = sum(all_match) / len(all_match)
        name = f"{n_active}h_{scale_type}"
        print(f"{name:>35} {avg_match:>6.0f}% [{sample_text}]")

# Also test: mask only LAST N layers (keep first M full)
print(f"\nMASKING ONLY LATE LAYERS (20h, no_scale):")
for split in [0, 10, 20, 30, 35]:
    mask = torch.zeros(N_HEADS, device=device, dtype=torch.bfloat16)
    mask[:20] = 1.0  # no scale

    all_match = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            tokens = masked_generate(model, ids, N_GEN, mask, split_layer=split)
        base_ids = base_texts[prompt][0][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        all_match.append(match / max(len(base_ids), 1) * 100)

    avg_match = sum(all_match) / len(all_match)
    print(f"  split@L{split:>2}, 20h noscale: match={avg_match:.0f}%")

print(f"\nDone.", flush=True)
