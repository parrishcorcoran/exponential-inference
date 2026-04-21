"""Strict early exit: require N consecutive matching layers.

Previous test: 2 consecutive matches exits at L18 → 2% token match (too early).
The argmax bounces. Real stabilization = stays consistent through MANY layers.

Test stricter criteria:
- 3, 5, 8, 10 consecutive matching layers
- Track: at what layer do tokens ACTUALLY stabilize for this model?
"""
import torch
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("STRICT EARLY EXIT — require N consecutive matches")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
HEAD_DIM = model.config.hidden_size // N_HEADS
H = model.config.hidden_size
N_KV = model.config.num_key_value_heads
GQA_RATIO = N_HEADS // N_KV

lm_head_weight = model.lm_head.weight
final_norm_layer = model.model.norm

print(f"L={N_LAYERS} H={H} heads={N_HEADS}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def full_layer_manual(h, layer, cos, sin, k_cached, v_cached):
    B, T, D = h.shape
    attn = layer.self_attn
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

    k_full = torch.cat([k_cached, k], dim=2)
    v_full = torch.cat([v_cached, v], dim=2)

    k_exp = k_full.repeat_interleave(GQA_RATIO, dim=1)
    v_exp = v_full.repeat_interleave(GQA_RATIO, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
    attn_out = attn.o_proj(attn_out)

    h = residual + attn_out
    residual = h
    h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return h, k_full, v_full


def strict_exit_generate(model, input_ids, max_new_tokens=64,
                         n_consecutive=5, min_layer=10):
    """Exit when argmax is stable for n_consecutive layers in a row."""
    B = input_ids.shape[0]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    kv = {}
    for i in range(N_LAYERS):
        kv[i] = (past.layers[i].keys.clone(), past.layers[i].values.clone())

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]
    telem = {"exit_layers": [], "stable_pred": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1

        with torch.no_grad():
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
            pos_ids = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos_ids)

            consecutive = 0
            prev_pred = None
            exit_layer = N_LAYERS
            stable_pred = None

            for i in range(N_LAYERS):
                h, k_new, v_new = full_layer_manual(h, model.model.layers[i],
                                                     cos, sin, kv[i][0], kv[i][1])
                kv[i] = (k_new, v_new)

                # Check logit lens at every layer from min_layer
                if i >= min_layer:
                    h_check = final_norm_layer(h)
                    curr_pred = F.linear(h_check, lm_head_weight)[0, 0].argmax(-1).item()

                    if curr_pred == prev_pred:
                        consecutive += 1
                        if consecutive >= n_consecutive:
                            exit_layer = i + 1
                            stable_pred = curr_pred
                            # BUT: still need KV for remaining layers in cache
                            # Run remaining layers to fill cache (needed for next steps)
                            for j in range(i + 1, N_LAYERS):
                                h, k_new, v_new = full_layer_manual(
                                    h, model.model.layers[j],
                                    cos, sin, kv[j][0], kv[j][1])
                                kv[j] = (k_new, v_new)
                            break
                    else:
                        consecutive = 1  # current layer starts a new streak
                    prev_pred = curr_pred

        # Use the stable prediction if found, otherwise final
        h_out = final_norm_layer(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        final_pred = logits.argmax(-1).item()

        # Did early exit agree with final?
        if stable_pred is not None:
            next_tok = stable_pred  # use the stable prediction
        else:
            next_tok = final_pred

        gen_tokens.append(next_tok)
        telem["exit_layers"].append(exit_layer)
        telem["stable_pred"].append(stable_pred == final_pred if stable_pred is not None else None)

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

print(f"\nBASELINE:")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)

base_texts = {}
base_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps.append(tps)
    base_texts[prompt] = out[0][ids.shape[1]:].tolist()
    print(f"  {tps:.1f} tok/s [{text[:60]}]")
avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

# Strict exit configs
# Note: we still run all layers for KV cache, but use the EARLY prediction
# This tests: does early exit give the RIGHT token, even if speed is the same?
# (Speed savings require not computing remaining layers — needs sparse KV fill)

print(f"\nSTRICT EARLY EXIT (use stable prediction, still fill KV):")
for n_consec in [2, 3, 5, 8, 10]:
    all_match = []
    all_exit = []
    all_agree = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            tokens, telem = strict_exit_generate(
                model, ids, max_new_tokens=N_GEN,
                n_consecutive=n_consec, min_layer=10
            )

        base_ids = base_texts[prompt][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        match_pct = match / max(len(base_ids), 1) * 100

        avg_exit = sum(telem["exit_layers"]) / max(len(telem["exit_layers"]), 1)
        # How often did stable pred agree with final layer pred?
        agrees = [x for x in telem["stable_pred"] if x is not None]
        agree_pct = sum(agrees) / max(len(agrees), 1) * 100 if agrees else 0
        n_early = sum(1 for x in telem["exit_layers"] if x < N_LAYERS)

        all_match.append(match_pct)
        all_exit.append(avg_exit)
        all_agree.append(agree_pct)

    avg_match = sum(all_match) / len(all_match)
    avg_exit_l = sum(all_exit) / len(all_exit)
    avg_agree = sum(all_agree) / len(all_agree)
    print(f"  n={n_consec:>2}: match={avg_match:>4.0f}% exit@L{avg_exit_l:>4.1f} "
          f"agree={avg_agree:.0f}% (stable pred = final pred)")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/strict_exit.json", "w") as f:
    json.dump({"baseline": avg_base}, f, indent=2)
print("Saved.", flush=True)
