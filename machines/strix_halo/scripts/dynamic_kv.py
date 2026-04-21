"""Engine B: Dynamic KV cache — keep only what the manifold says matters.

Per token, per layer: score each cached K/V by relevance to current query.
Keep the top-N most relevant + local window. N determined dynamically
by the attention distribution's own sharpness — NOT a fixed percentage.

Sharp attention = few keys matter = aggressive eviction.
Diffuse attention = many keys matter = keep more.
The manifold decides.
"""
import torch
import torch.nn.functional as F
import time
import json
device = 'cuda'

print("="*70, flush=True)
print("ENGINE B: Dynamic KV Cache — manifold-driven eviction", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.config.hidden_size // N_HEADS
H = model.config.hidden_size
LOCAL_WINDOW = 32  # always keep recent tokens (grammar needs locality)

print(f"L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}", flush=True)

# Build a long context to test KV compression
print("\nBuilding long context (needle in haystack)...", flush=True)
import random
random.seed(42)
filler = [
    "The weather forecast predicted rain for the upcoming weekend.",
    "Advanced quantum computing research continues to push boundaries.",
    "The local library hosted a book fair attracting hundreds.",
    "Marine biologists discovered a new deep-sea fish species.",
    "Software engineers developed a more efficient sorting algorithm.",
    "Renewable energy investments surpassed fossil fuel investments.",
    "The stock exchange experienced unusual volatility this week.",
    "Climate scientists warn that Arctic ice is at record lows.",
    "Professional chess popularity surged driven by online streaming.",
    "New regulations aimed at reducing plastic waste were announced.",
] * 20
random.shuffle(filler)
needle = "The secret password is 'Supernova'."
filler.insert(len(filler)//2, needle)
prompt = " ".join(filler) + "\n\nBased on the text above, the secret password is: '"
ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
ctx_len = ids.shape[1]
print(f"Context: {ctx_len} tokens, needle at middle", flush=True)

# Baseline: find the needle with full KV cache
print("\nBaseline (full KV cache)...", flush=True)
with torch.no_grad():
    base = model.generate(ids, max_new_tokens=5, do_sample=False)
base_text = tokenizer.decode(base[0][ctx_len:], skip_special_tokens=True)
base_found = "supernova" in base_text.lower()
print(f"  Output: '{base_text}'  Needle: {base_found}", flush=True)

# ═══════════════════════════════════════════════════════
# Dynamic KV: custom forward with per-step cache eviction
# ═══════════════════════════════════════════════════════

def dynamic_kv_generate(model, input_ids, max_new_tokens, local_window=32):
    """Generate with dynamic KV eviction based on attention sharpness.

    After each step, for each layer:
    - Compute attention scores between new query and all cached keys
    - Measure sharpness of the attention distribution
    - Sharp → keep few keys (manifold position is precise)
    - Diffuse → keep more keys (need broader context)
    - Always keep the local window (recent tokens for grammar)
    """
    # Prefill: run full context, build initial KV cache
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[0, -1:].argmax(-1)

    generated = [next_tok.item()]
    gen_ids = torch.cat([input_ids, next_tok.unsqueeze(0)], dim=-1)

    telemetry = {"cache_sizes": [], "eviction_rates": []}

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            # Forward with cached KV
            out = model(
                next_tok.unsqueeze(0),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_tok = out.logits[0, -1:].argmax(-1)
            generated.append(next_tok.item())

            # Dynamic eviction: check cache size and trim
            # Get current cache length
            cache_len = past[0][0].shape[2]  # [B, KV_heads, seq_len, head_dim]

            if cache_len > local_window * 2:
                # For each layer: score cached keys by relevance
                new_past = []
                total_kept = 0
                total_possible = 0

                for layer_idx in range(len(past)):
                    k_cache = past[layer_idx][0]  # [B, KV, seq, HD]
                    v_cache = past[layer_idx][1]

                    # Current token's key as query proxy
                    q_current = k_cache[:, :, -1:, :]  # [B, KV, 1, HD]
                    k_past = k_cache[:, :, :-1, :]      # [B, KV, seq-1, HD]

                    # Attention score: how relevant is each cached position?
                    scores = (q_current * k_past).sum(-1)  # [B, KV, seq-1]
                    scores = scores / (HEAD_DIM ** 0.5)

                    # Average across KV heads for a single relevance score per position
                    relevance = scores.mean(dim=1)[0]  # [seq-1]

                    # DYNAMIC threshold: based on attention sharpness
                    # Sharp distribution → high max, keep fewer
                    # Diffuse → low max, keep more
                    attn_probs = F.softmax(relevance, dim=-1)
                    sharpness = attn_probs.max().item()

                    # Dynamic keep count: inversely proportional to sharpness
                    # sharpness near 1.0 → keep very few (maybe 5%)
                    # sharpness near 0.001 → keep most (maybe 80%)
                    keep_frac = max(0.05, min(0.8, 1.0 - sharpness * 2))
                    n_keep = max(local_window, int(len(relevance) * keep_frac))

                    # Keep: top-n_keep by relevance + last local_window
                    if n_keep < len(relevance):
                        topk_idx = relevance.topk(n_keep).indices
                        local_idx = torch.arange(
                            max(0, len(relevance) - local_window),
                            len(relevance), device=device
                        )
                        keep_idx = torch.unique(torch.cat([topk_idx, local_idx]))
                        keep_idx = keep_idx.sort().values

                        # Add the current token (last position)
                        keep_idx_full = torch.cat([
                            keep_idx,
                            torch.tensor([k_cache.shape[2] - 1], device=device)
                        ])

                        k_new = k_cache[:, :, keep_idx_full, :]
                        v_new = v_cache[:, :, keep_idx_full, :]
                        new_past.append((k_new, v_new))

                        total_kept += len(keep_idx_full)
                        total_possible += cache_len
                    else:
                        new_past.append((k_cache, v_cache))
                        total_kept += cache_len
                        total_possible += cache_len

                past = tuple(new_past)

                eviction_rate = 1.0 - (total_kept / total_possible)
                telemetry["cache_sizes"].append(total_kept // len(past))
                telemetry["eviction_rates"].append(eviction_rate)

        if next_tok.item() == tokenizer.eos_token_id:
            break

    return generated, telemetry

# Run with dynamic KV
print("\nDynamic KV generation...", flush=True)
t0 = time.time()
tokens, telem = dynamic_kv_generate(model, ids, 10, LOCAL_WINDOW)
elapsed = time.time() - t0

text = tokenizer.decode(tokens, skip_special_tokens=True)
found = "supernova" in text.lower()

avg_cache = sum(telem["cache_sizes"]) / max(len(telem["cache_sizes"]), 1)
avg_evict = sum(telem["eviction_rates"]) / max(len(telem["eviction_rates"]), 1)

print(f"\n{'='*60}", flush=True)
print(f"DYNAMIC KV RESULTS", flush=True)
print(f"  Context: {ctx_len} tokens", flush=True)
print(f"  Output: '{text}'", flush=True)
print(f"  Needle found: {found}", flush=True)
print(f"  Baseline found: {base_found}", flush=True)
print(f"  Avg cache size per layer: {avg_cache:.0f} / {ctx_len} ({avg_cache/ctx_len*100:.0f}%)", flush=True)
print(f"  Avg eviction rate: {avg_evict*100:.0f}%", flush=True)
print(f"  Time: {elapsed:.1f}s", flush=True)
print(f"{'='*60}", flush=True)

with open("machines/strix_halo/results/dynamic_kv.json", "w") as f:
    json.dump({"ctx_len": ctx_len, "found": found, "base_found": base_found,
               "avg_cache_pct": avg_cache/ctx_len*100, "avg_evict": avg_evict,
               "text": text}, f, indent=2)
print("Saved dynamic_kv.json", flush=True)
