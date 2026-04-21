"""Test: KV is depth. Layer-selective KV eviction.

If KV = depth (holographic projections), then:
- Late-layer KV (where tokens resolve) carries the depth signal
- Early-layer KV (where projection curves fast) is less critical
- Keeping only late-layer KV should preserve quality

Test: generate with full model, then regenerate with partial KV caches
where we zero out early-layer KV for all but the most recent tokens.
Compare: does keeping only late-layer KV still produce correct tokens?
"""
import torch
import torch.nn.functional as F
import json

device = "cuda"

print("=" * 70)
print("KV IS DEPTH: layer-selective KV eviction test")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
print(f"Model loaded. {N_LAYERS} layers. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

prompts = [
    "The theory of general relativity describes gravity as the curvature of spacetime caused by",
    "The capital of France is Paris, which is known for its iconic landmarks such as",
    "In computer science, a hash table is a data structure that maps keys to",
    "Water freezes at zero degrees Celsius and boils at one hundred degrees, which means",
    "The Fibonacci sequence starts with zero and one, then each subsequent number is the sum of",
    "Neural networks learn by adjusting weights through a process called backpropagation which minimizes the",
]

N_GEN = 15

# ═══════════════════════════════════════════════════════
# For each prompt: generate baseline, then generate with
# layer-selective KV eviction and compare
# ═══════════════════════════════════════════════════════

# Layer keep strategies: which layers' KV to preserve for OLD positions
# (most recent token always gets full KV)
strategies = {
    "full":       list(range(40)),           # all layers (baseline)
    "late_only":  list(range(30, 40)),       # only layers 30-39 (25%)
    "mid_late":   list(range(20, 40)),       # layers 20-39 (50%)
    "early_late": list(range(0, 5)) + list(range(35, 40)),  # first 5 + last 5 (25%)
    "early_only": list(range(0, 10)),        # only layers 0-9 (25%)
    "every_4th":  list(range(0, 40, 4)),     # every 4th layer (25%)
    "last_10":    list(range(30, 40)),       # last 10 layers
    "last_5":     list(range(35, 40)),       # last 5 layers
    "last_1":     [39],                      # just the final layer
}

print(f"\n{'Strategy':>15} {'Layers kept':>12} {'KV %':>6}  Results per prompt...", flush=True)
print("=" * 80)

all_results = {}

for strategy_name, keep_layers in strategies.items():
    keep_set = set(keep_layers)
    kv_pct = len(keep_layers) / N_LAYERS * 100
    matches_total = 0
    tokens_total = 0
    all_texts = []

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        prompt_len = ids.shape[1]

        with torch.no_grad():
            # Step 1: Prefill with full model
            out = model(ids, use_cache=True)
            past = out.past_key_values

            # Step 2: Generate tokens one at a time
            baseline_tokens = []
            strategy_tokens = []

            # For baseline: just generate normally
            past_baseline = out.past_key_values
            next_tok = out.logits[0, -1].argmax(-1)
            baseline_tokens.append(next_tok.item())

            # Deep copy past for strategy version
            # We'll modify it per strategy

            for step in range(N_GEN - 1):
                # Baseline generation
                out_b = model(next_tok.view(1, 1), past_key_values=past_baseline,
                             use_cache=True)
                past_baseline = out_b.past_key_values
                next_tok = out_b.logits[0, -1].argmax(-1)
                baseline_tokens.append(next_tok.item())

        # Now regenerate with layer-selective KV eviction
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past_strat = out.past_key_values
            next_tok_s = out.logits[0, -1].argmax(-1)
            strategy_tokens.append(next_tok_s.item())

            for step in range(N_GEN - 1):
                # Before each step: zero out KV at non-kept layers
                # for all positions EXCEPT the most recent few
                # (recent tokens need full KV for local grammar)
                LOCAL_WINDOW = 4
                cache_len = past_strat.layers[0].keys.shape[2]

                if cache_len > LOCAL_WINDOW:
                    for layer_idx in range(N_LAYERS):
                        if layer_idx not in keep_set:
                            # Zero out old positions' KV at this layer
                            k = past_strat.layers[layer_idx].keys
                            v = past_strat.layers[layer_idx].values
                            # Zero everything except last LOCAL_WINDOW positions
                            k[:, :, :-LOCAL_WINDOW, :] = 0
                            v[:, :, :-LOCAL_WINDOW, :] = 0

                out_s = model(next_tok_s.view(1, 1), past_key_values=past_strat,
                             use_cache=True)
                past_strat = out_s.past_key_values
                next_tok_s = out_s.logits[0, -1].argmax(-1)
                strategy_tokens.append(next_tok_s.item())

        # Compare
        matches = sum(1 for a, b in zip(baseline_tokens, strategy_tokens) if a == b)
        matches_total += matches
        tokens_total += len(baseline_tokens)

        strat_text = tokenizer.decode(strategy_tokens, skip_special_tokens=True)
        all_texts.append(strat_text[:50])

    match_pct = matches_total / max(tokens_total, 1) * 100
    all_results[strategy_name] = {
        "keep_layers": keep_layers,
        "kv_pct": kv_pct,
        "match_pct": match_pct,
        "matches": matches_total,
        "total": tokens_total,
    }
    print(f"{strategy_name:>15} {len(keep_layers):>4} layers {kv_pct:>5.0f}%  "
          f"{matches_total}/{tokens_total} = {match_pct:.0f}% match", flush=True)

# ═══════════════════════════════════════════════════════
# Show text comparison for key strategies
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TEXT COMPARISON")
print(f"{'='*60}")

for prompt in prompts[:3]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt[:50]}...'")

    with torch.no_grad():
        # Baseline
        out = model(ids, use_cache=True)
        past_b = out.past_key_values
        gen_b = []
        next_tok = out.logits[0, -1].argmax(-1)
        gen_b.append(next_tok.item())
        for _ in range(N_GEN - 1):
            out_b = model(next_tok.view(1, 1), past_key_values=past_b, use_cache=True)
            past_b = out_b.past_key_values
            next_tok = out_b.logits[0, -1].argmax(-1)
            gen_b.append(next_tok.item())

    print(f"  Full:      '{tokenizer.decode(gen_b, skip_special_tokens=True)[:60]}'")

    for sname in ["late_only", "last_5", "early_only"]:
        keep_set = set(strategies[sname])

        with torch.no_grad():
            out = model(ids, use_cache=True)
            past_s = out.past_key_values
            gen_s = []
            next_tok = out.logits[0, -1].argmax(-1)
            gen_s.append(next_tok.item())

            for _ in range(N_GEN - 1):
                cache_len = past_s.layers[0].keys.shape[2]
                if cache_len > 4:
                    for layer_idx in range(N_LAYERS):
                        if layer_idx not in keep_set:
                            past_s.layers[layer_idx].keys[:, :, :-4, :] = 0
                            past_s.layers[layer_idx].values[:, :, :-4, :] = 0

                out_s = model(next_tok.unsqueeze(0), past_key_values=past_s, use_cache=True)
                past_s = out_s.past_key_values
                next_tok = out_s.logits[0, -1].argmax(-1)
                gen_s.append(next_tok.item())

        text = tokenizer.decode(gen_s, skip_special_tokens=True)[:60]
        print(f"  {sname:>12}: '{text}'")

# Save
with open("machines/strix_halo/results/kv_is_depth.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results.", flush=True)
