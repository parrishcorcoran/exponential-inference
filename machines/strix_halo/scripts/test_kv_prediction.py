"""Test: does the KV cache after prefill encode future tokens?

The claim: the KV cache is the holographic reconstruction of the manifold.
Future tokens' KV should be derivable from existing KV + rotation curve.

Test 1: After prefill, predict token N+1 normally. Then predict token N+2
         using only KV cache + the known rotation structure. Compare to
         what the full forward pass produces.

Test 2: For each generated token, measure how much of its per-layer hidden
         state is predictable from the PREVIOUS layer's hidden state + the
         rotation curve. If fractal, early layers predict late layers.

Test 3: Given the full KV cache from prefill + token N+1's embedding,
         run ONLY attention (skip MLP) through all layers. Does the output
         still predict the correct N+2? If yes: the KV cache carries the
         manifold, the MLP is just materializing what's already determined.
"""
import torch
import torch.nn.functional as F
import json
import time

device = "cuda"

print("=" * 70)
print("KV PREDICTION TEST: does the KV cache encode future tokens?")
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
HIDDEN = model.config.hidden_size

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Test prompts
# ═══════════════════════════════════════════════════════
prompts = [
    "The theory of general relativity describes gravity as",
    "In the beginning, there was nothing but darkness and",
    "To solve a quadratic equation, you can use the",
    "The capital of France is Paris, which is known for",
    "Neural networks learn by adjusting weights through",
    "Water freezes at zero degrees Celsius and boils at",
    "The Fibonacci sequence starts with zero and one, then each",
    "Photosynthesis converts light energy into chemical energy in",
]

N_FUTURE = 10  # How many future tokens to test

# ═══════════════════════════════════════════════════════
# Test 1: Layer-to-layer KV predictability (rotation curve)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 1: Layer-to-layer KV predictability")
print("Does KV at layer L predict KV at layer L+1?")
print(f"{'='*60}")

layer_cosines = []  # [prompt][layer] = cosine(KV_L, KV_L+1)

for prompt in prompts[:4]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
        hidden_states = out.hidden_states  # (L+1) × [1, T, H]

        # Measure cosine similarity between consecutive layers' hidden states
        # at the last token position (the one that predicts next)
        cos_per_layer = []
        for l in range(1, len(hidden_states) - 1):
            h_l = hidden_states[l][0, -1]      # [H]
            h_l1 = hidden_states[l + 1][0, -1]  # [H]
            cos = F.cosine_similarity(h_l.unsqueeze(0).float(),
                                       h_l1.unsqueeze(0).float()).item()
            cos_per_layer.append(cos)
        layer_cosines.append(cos_per_layer)

# Average across prompts
avg_cos = [sum(lc[l] for lc in layer_cosines) / len(layer_cosines)
           for l in range(len(layer_cosines[0]))]

print(f"\nLayer-to-layer cosine similarity (last token position):")
print(f"{'Layers':>10} {'Cosine':>8}")
for l in range(0, len(avg_cos), 4):
    print(f"  {l+1}→{l+2:>3}    {avg_cos[l]:>7.4f}")
print(f"  Mean:      {sum(avg_cos)/len(avg_cos):>7.4f}")
print(f"  Min:       {min(avg_cos):>7.4f} (layers {avg_cos.index(min(avg_cos))+1}→{avg_cos.index(min(avg_cos))+2})")
print(f"  Max:       {max(avg_cos):>7.4f}")

# ═══════════════════════════════════════════════════════
# Test 2: KV cache contains future token information
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 2: Can we predict future tokens from KV cache?")
print("Generate N tokens. For each, measure how much the KV cache")
print("already 'knew' about it before it was computed.")
print(f"{'='*60}")

for prompt in prompts[:4]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt}'")

    with torch.no_grad():
        # Prefill
        out = model(ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        prefill_hidden = out.hidden_states  # save for later

        # Generate N tokens, recording hidden states at each step
        gen_tokens = []
        gen_hidden_states = []  # per-step, per-layer hidden states

        next_tok = out.logits[0, -1:].argmax(-1)
        gen_tokens.append(next_tok.item())

        for step in range(N_FUTURE - 1):
            out = model(next_tok.unsqueeze(0), past_key_values=past,
                       use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            gen_hidden_states.append([h[0, -1].clone() for h in out.hidden_states])
            next_tok = out.logits[0, -1:].argmax(-1)
            gen_tokens.append(next_tok.item())

        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(f"  Generated: '{gen_text[:60]}'")

        # Now test: for each generated token, how similar is its hidden state
        # to what could be predicted from the PREVIOUS step's hidden state
        # via the rotation?
        if len(gen_hidden_states) >= 2:
            print(f"  Step-to-step hidden state predictability:")
            for s in range(1, min(len(gen_hidden_states), 6)):
                prev_hs = gen_hidden_states[s-1]  # list of per-layer [H]
                curr_hs = gen_hidden_states[s]

                # Per-layer cosine between consecutive steps at each layer
                cos_per_layer = []
                for l in range(len(prev_hs)):
                    cos = F.cosine_similarity(
                        prev_hs[l].unsqueeze(0).float(),
                        curr_hs[l].unsqueeze(0).float()
                    ).item()
                    cos_per_layer.append(cos)

                early_cos = sum(cos_per_layer[:10]) / 10
                mid_cos = sum(cos_per_layer[10:30]) / 20
                late_cos = sum(cos_per_layer[30:]) / max(len(cos_per_layer[30:]), 1)

                tok_prev = tokenizer.decode([gen_tokens[s-1]])
                tok_curr = tokenizer.decode([gen_tokens[s]])
                print(f"    step {s}: '{tok_prev}'→'{tok_curr}' "
                      f"early={early_cos:.3f} mid={mid_cos:.3f} late={late_cos:.3f}")

# ═══════════════════════════════════════════════════════
# Test 3: Attention-only forward (skip MLP)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 3: Attention-only forward (skip MLP)")
print("If KV cache IS the manifold, attention alone should")
print("propagate the information. MLP materializes but doesn't add.")
print(f"{'='*60}")

for prompt in prompts[:4]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        # Full forward: prefill + generate 1 token normally
        out = model(ids, use_cache=True)
        past_full = out.past_key_values
        tok1 = out.logits[0, -1].argmax().item()

        # Generate token 2 with FULL forward
        out2 = model(torch.tensor([[tok1]], device=device),
                     past_key_values=past_full, use_cache=True,
                     output_hidden_states=True)
        tok2_full = out2.logits[0, -1].argmax().item()
        hidden_full = out2.hidden_states[-1][0, -1]  # final hidden state

        # Now: generate token 2 with ATTENTION ONLY (skip MLP)
        # Start from tok1's embedding
        h = model.model.embed_tokens(torch.tensor([[tok1]], device=device))

        # Build position ids
        seq_len = ids.shape[1] + 1  # prompt + tok1
        pos_ids = torch.tensor([[seq_len - 1]], device=device)
        cos, sin = model.model.rotary_emb(h, pos_ids)

        for layer_idx in range(N_LAYERS):
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn

            residual = h
            h_norm = layer.input_layernorm(h)

            # Full attention computation with KV cache
            B, T, D = h_norm.shape
            q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
            k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
            v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

            if attn.q_norm is not None:
                q = attn.q_norm(q)
            if attn.k_norm is not None:
                k = attn.k_norm(k)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Rotary
            cos_r = cos.unsqueeze(1)
            sin_r = sin.unsqueeze(1)
            rd = HEAD_DIM // 2
            q1, q2 = q[..., :rd], q[..., rd:]
            q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                           q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
            k1, k2 = k[..., :rd], k[..., rd:]
            k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                           k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

            # Concatenate with cached KV
            k_cache = past_full.layers[layer_idx].keys  # [B, N_KV, cache_len, HD]
            v_cache = past_full.layers[layer_idx].values
            k_full = torch.cat([k_cache, k], dim=2)
            v_full = torch.cat([v_cache, v], dim=2)

            # GQA expand
            n_rep = N_HEADS // N_KV
            k_exp = k_full.repeat_interleave(n_rep, dim=1)
            v_exp = v_full.repeat_interleave(n_rep, dim=1)

            # Attention
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, HIDDEN)
            attn_proj = attn.o_proj(attn_out)

            h = residual + attn_proj

            # SKIP MLP — this is the test
            # h = residual_mlp + layer.mlp(layer.post_attention_layernorm(h))
            # Instead: just pass through without MLP

        h_attn_only = model.model.norm(h)
        logits_attn = model.lm_head(h_attn_only)
        tok2_attn = logits_attn[0, -1].argmax().item()

        # Compare
        cos_hidden = F.cosine_similarity(
            hidden_full.unsqueeze(0).float(),
            h_attn_only[0, -1].unsqueeze(0).float()
        ).item()

        tok2_full_str = tokenizer.decode([tok2_full])
        tok2_attn_str = tokenizer.decode([tok2_attn])
        match = "MATCH" if tok2_full == tok2_attn else "DIFF"

        print(f"\n  Prompt: '{prompt[:40]}...'")
        print(f"  tok1: '{tokenizer.decode([tok1])}'")
        print(f"  tok2 (full):     '{tok2_full_str}'")
        print(f"  tok2 (attn-only): '{tok2_attn_str}'  [{match}]")
        print(f"  Hidden cosine (full vs attn-only): {cos_hidden:.4f}")

# ═══════════════════════════════════════════════════════
# Test 4: Can KV from layer L predict the token directly?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEST 4: Per-layer token prediction (logit lens on KV)")
print("At which layer does the KV already encode the next token?")
print(f"{'='*60}")

for prompt in prompts[:4]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
        hidden_states = out.hidden_states
        final_tok = out.logits[0, -1].argmax().item()

        # At each layer, project hidden state through lm_head
        print(f"\n  Prompt: '{prompt[:40]}...'")
        print(f"  Final prediction: '{tokenizer.decode([final_tok])}'")
        print(f"  {'Layer':>7} {'Token':>15} {'Match':>6} {'Prob':>8}")

        for l in [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 38, 39, 40]:
            if l >= len(hidden_states):
                continue
            h = hidden_states[l][0, -1]  # [H]
            h_normed = model.model.norm(h.unsqueeze(0).unsqueeze(0))
            logits = model.lm_head(h_normed)[0, 0]
            probs = F.softmax(logits.float(), dim=-1)
            pred_tok = logits.argmax().item()
            pred_prob = probs[final_tok].item()
            match = "YES" if pred_tok == final_tok else ""
            pred_str = tokenizer.decode([pred_tok])
            print(f"  {l:>5}L {pred_str:>15} {match:>6} {pred_prob:>8.4f}")

# Save results
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Layer cosine mean: {sum(avg_cos)/len(avg_cos):.4f}")
print(f"Layer cosine min:  {min(avg_cos):.4f}")
print(f"Layer cosine max:  {max(avg_cos):.4f}")

with open("machines/strix_halo/results/kv_prediction.json", "w") as f:
    json.dump({
        "layer_cosines_avg": avg_cos,
        "n_prompts": len(prompts),
        "n_future": N_FUTURE,
    }, f, indent=2)
print(f"\nSaved results.", flush=True)
