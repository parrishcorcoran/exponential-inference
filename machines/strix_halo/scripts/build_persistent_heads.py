"""Persistent head selection: use the SAME heads as previous step.

The problem with measurement-based selection:
- Measuring at layer L doesn't predict layers L+1..N
- Head importance is layer-dependent

New approach: run FULL width on step 0. Identify sharp/contributing heads.
Use those SAME heads for subsequent steps. Re-measure every M steps.

If the manifold trajectory is smooth (step-to-step cosine 0.7-0.9),
the important heads shouldn't change much between consecutive tokens.

Also test: run full width EVERY step but exit early. No head pruning.
Just pure early exit based on stabilization (logit lens at every layer).
This is the simplest possible thin slice.
"""
import torch
import torch.nn.functional as F
import time
import json

device = "cuda"

print("=" * 70)
print("PERSISTENT HEADS + PURE EARLY EXIT")
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

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


def full_layer_manual(h, layer, cos, sin, past_layer):
    """Full-width layer with manual KV cache. Returns h, updated cache, per-head attn norms."""
    B, T, D = h.shape
    attn = layer.self_attn

    residual = h
    h_norm = layer.input_layernorm(h)

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

    rd = HEAD_DIM // 2
    cos_r = cos.unsqueeze(1)
    sin_r = sin.unsqueeze(1)
    q1, q2 = q[..., :rd], q[..., rd:]
    q = torch.cat([q1*cos_r[...,:rd] - q2*sin_r[...,:rd],
                   q2*cos_r[...,:rd] + q1*sin_r[...,:rd]], -1)
    k1, k2 = k[..., :rd], k[..., rd:]
    k = torch.cat([k1*cos_r[...,:rd] - k2*sin_r[...,:rd],
                   k2*cos_r[...,:rd] + k1*sin_r[...,:rd]], -1)

    k_cached, v_cached = past_layer
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

    return h, (k_full, v_full)


def early_exit_generate(model, input_ids, max_new_tokens=64, check_interval=1, min_layer=15):
    """Pure early exit: full width, exit when logit lens stabilizes.

    Check logit lens every check_interval layers starting from min_layer.
    Exit when the argmax matches the final layer's argmax from 2 consecutive checks.
    """
    B = input_ids.shape[0]

    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values

    # Convert cache
    kv_cache = {}
    for i in range(N_LAYERS):
        kv_cache[i] = (past.layers[i].keys.clone(), past.layers[i].values.clone())

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    seq_len = input_ids.shape[1]

    telem = {"exit_layers": []}

    for step in range(max_new_tokens - 1):
        seq_len += 1

        with torch.no_grad():
            h = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))
            pos_ids = torch.tensor([[seq_len - 1]], device=device)
            cos, sin = model.model.rotary_emb(h, pos_ids)

            prev_pred = None
            exit_layer = N_LAYERS

            for i in range(N_LAYERS):
                h, kv_cache[i] = full_layer_manual(h, model.model.layers[i],
                                                    cos, sin, kv_cache[i])

                # Logit lens check
                if i >= min_layer and (i - min_layer) % check_interval == 0:
                    h_check = final_norm_layer(h)
                    logits_check = F.linear(h_check, lm_head_weight)[0, 0]
                    curr_pred = logits_check.argmax(-1).item()

                    if prev_pred is not None and curr_pred == prev_pred:
                        # Stabilized — exit
                        exit_layer = i + 1
                        break
                    prev_pred = curr_pred

        h_out = final_norm_layer(h)
        logits = F.linear(h_out, lm_head_weight)[0, 0]
        next_tok = logits.argmax(-1).item()
        gen_tokens.append(next_tok)
        telem["exit_layers"].append(exit_layer)

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
with torch.no_grad():
    model.generate(ids, max_new_tokens=5, do_sample=False)

base_texts = {}
base_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    base_tps.append(tps)
    base_texts[prompt] = out[0][ids.shape[1]:].tolist()
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

# Early exit configs
print(f"\nEARLY EXIT (full width, stabilization-based):")
configs = [
    {"name": "check_every1_min15", "interval": 1, "min": 15},
    {"name": "check_every1_min20", "interval": 1, "min": 20},
    {"name": "check_every2_min15", "interval": 2, "min": 15},
    {"name": "check_every1_min25", "interval": 1, "min": 25},
    {"name": "check_every1_min30", "interval": 1, "min": 30},
    {"name": "check_every3_min10", "interval": 3, "min": 10},
]

for cfg in configs:
    all_tps = []
    all_match = []
    all_exit = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            tokens, telem = early_exit_generate(
                model, ids, max_new_tokens=N_GEN,
                check_interval=cfg["interval"], min_layer=cfg["min"]
            )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        tps = len(tokens) / elapsed

        base_ids = base_texts[prompt][:len(tokens)]
        match = sum(1 for a, b in zip(base_ids, tokens) if a == b)
        match_pct = match / max(len(base_ids), 1) * 100

        avg_exit = sum(telem["exit_layers"]) / max(len(telem["exit_layers"]), 1)
        all_tps.append(tps)
        all_match.append(match_pct)
        all_exit.append(avg_exit)

    avg_tps = sum(all_tps) / len(all_tps)
    avg_match = sum(all_match) / len(all_match)
    avg_exit_l = sum(all_exit) / len(all_exit)
    print(f"  {cfg['name']:>25}: {avg_tps:.1f} ({avg_tps/avg_base:.2f}x) "
          f"match={avg_match:.0f}% exit@L{avg_exit_l:.0f}")

print(f"\nDone.", flush=True)
with open("machines/strix_halo/results/persistent_heads.json", "w") as f:
    json.dump({"baseline": avg_base}, f, indent=2)
print("Saved.", flush=True)
