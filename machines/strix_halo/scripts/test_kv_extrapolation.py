"""Test: can we extrapolate the KV trajectory to predict future tokens?

The KV cache defines a curve on the manifold. Each token extends it.
If the curve is smooth at defined positions, extrapolating the KV
should predict the next token WITHOUT a forward pass.

Method:
1. Prefill a prompt → KV cache
2. Generate tokens one at a time (full forward) → record actual KV at each step
3. At each step, EXTRAPOLATE the next KV from the trajectory so far
4. Project extrapolated KV through attention + lm_head → predicted token
5. Compare predicted token to actual token

If they match: the KV trajectory was determined. The forward pass was redundant.
"""
import torch
import torch.nn.functional as F
import json

device = "cuda"

print("=" * 70)
print("KV EXTRAPOLATION TEST")
print("Can the KV trajectory predict future tokens?")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_KV = model.config.num_key_value_heads
HEAD_DIM = model.config.hidden_size // model.config.num_attention_heads
HIDDEN = model.config.hidden_size
N_HEADS = model.config.num_attention_heads

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

prompts = [
    "The theory of general relativity describes gravity as",
    "The capital of France is Paris, which is known for",
    "To solve a quadratic equation, you can use the",
    "Water freezes at zero degrees Celsius and boils at",
    "The Fibonacci sequence starts with zero and one, then each",
    "Neural networks learn by adjusting weights through a process called",
    "In the beginning, there was nothing but darkness and",
    "The speed of light in a vacuum is approximately",
]

N_GEN = 15  # tokens to generate and test

print(f"\n{'='*60}")
print("EXTRAPOLATION METHODS")
print(f"{'='*60}")

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    prompt_len = ids.shape[1]

    print(f"\nPrompt: '{prompt}'")

    with torch.no_grad():
        # Generate N tokens, storing hidden states at each step
        out = model(ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values

        # Collect per-step final hidden states (last token position)
        step_hidden = [out.hidden_states[-1][0, -1].clone()]  # [H]
        step_tokens = [out.logits[0, -1].argmax().item()]

        # Also collect per-step KV deltas at a reference layer
        ref_layer = 20
        prev_k = past.layers[ref_layer].keys.clone()  # [1, N_KV, T, HD]
        prev_v = past.layers[ref_layer].values.clone()

        kv_keys_history = [prev_k[0, :, -1, :].clone()]  # last position's K
        kv_vals_history = [prev_v[0, :, -1, :].clone()]  # last position's V

        for step in range(N_GEN - 1):
            tok_input = torch.tensor([[step_tokens[-1]]], device=device)
            out = model(tok_input, past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = out.past_key_values

            step_hidden.append(out.hidden_states[-1][0, -1].clone())
            step_tokens.append(out.logits[0, -1].argmax().item())

            # Record KV at reference layer (just the new position)
            kv_keys_history.append(past.layers[ref_layer].keys[0, :, -1, :].clone())
            kv_vals_history.append(past.layers[ref_layer].values[0, :, -1, :].clone())

    gen_text = tokenizer.decode(step_tokens, skip_special_tokens=True)
    print(f"  Generated: '{gen_text[:70]}'")

    # Now test extrapolation methods
    step_hidden = torch.stack(step_hidden)    # [N, H]
    kv_keys = torch.stack(kv_keys_history)    # [N, N_KV, HD]
    kv_vals = torch.stack(kv_vals_history)    # [N, N_KV, HD]

    # ── Method 1: Linear extrapolation of hidden state ──
    # h[t+1] ≈ h[t] + (h[t] - h[t-1])
    print(f"\n  Method 1: Linear extrapolation of hidden state")
    print(f"  {'Step':>6} {'Actual':>15} {'Predicted':>15} {'Match':>6} {'Cosine':>8}")

    m1_matches = 0
    m1_total = 0
    for t in range(2, len(step_hidden)):
        # Extrapolate
        delta = step_hidden[t-1] - step_hidden[t-2]
        h_pred = step_hidden[t-1] + delta

        # Project through lm_head
        h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0))
        logits = model.lm_head(h_normed)[0, 0]
        pred_tok = logits.argmax().item()
        actual_tok = step_tokens[t]

        cos = F.cosine_similarity(
            h_pred.unsqueeze(0).float(),
            step_hidden[t].unsqueeze(0).float()
        ).item()

        match = "YES" if pred_tok == actual_tok else ""
        if pred_tok == actual_tok:
            m1_matches += 1
        m1_total += 1

        if t < 10 or pred_tok == actual_tok:
            print(f"  {t:>6} {tokenizer.decode([actual_tok]):>15} "
                  f"{tokenizer.decode([pred_tok]):>15} {match:>6} {cos:>8.4f}")

    print(f"  Linear extrapolation: {m1_matches}/{m1_total} = {m1_matches/max(m1_total,1)*100:.0f}%")

    # ── Method 2: Quadratic extrapolation ──
    # h[t+1] ≈ h[t] + delta + 0.5*(delta - prev_delta)  [acceleration]
    print(f"\n  Method 2: Quadratic extrapolation of hidden state")
    m2_matches = 0
    m2_total = 0
    for t in range(3, len(step_hidden)):
        delta1 = step_hidden[t-1] - step_hidden[t-2]
        delta2 = step_hidden[t-2] - step_hidden[t-3]
        accel = delta1 - delta2
        h_pred = step_hidden[t-1] + delta1 + 0.5 * accel

        h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0))
        logits = model.lm_head(h_normed)[0, 0]
        pred_tok = logits.argmax().item()
        actual_tok = step_tokens[t]

        cos = F.cosine_similarity(
            h_pred.unsqueeze(0).float(),
            step_hidden[t].unsqueeze(0).float()
        ).item()

        match = "YES" if pred_tok == actual_tok else ""
        if pred_tok == actual_tok:
            m2_matches += 1
        m2_total += 1

        if t < 10 or pred_tok == actual_tok:
            print(f"  {t:>6} {tokenizer.decode([actual_tok]):>15} "
                  f"{tokenizer.decode([pred_tok]):>15} {match:>6} {cos:>8.4f}")

    print(f"  Quadratic extrapolation: {m2_matches}/{m2_total} = {m2_matches/max(m2_total,1)*100:.0f}%")

    # ── Method 3: KV trajectory extrapolation ──
    # Extrapolate in KV space instead of hidden state space
    print(f"\n  Method 3: KV trajectory extrapolation (layer {ref_layer})")
    m3_matches = 0
    m3_total = 0

    for t in range(2, len(kv_keys)):
        # Extrapolate K and V
        k_delta = kv_keys[t-1] - kv_keys[t-2]
        v_delta = kv_vals[t-1] - kv_vals[t-2]
        k_pred = kv_keys[t-1] + k_delta  # [N_KV, HD]
        v_pred = kv_vals[t-1] + v_delta

        # Cosine between predicted and actual KV
        k_cos = F.cosine_similarity(
            k_pred.reshape(1, -1).float(),
            kv_keys[t].reshape(1, -1).float()
        ).item()
        v_cos = F.cosine_similarity(
            v_pred.reshape(1, -1).float(),
            kv_vals[t].reshape(1, -1).float()
        ).item()

        actual_tok = step_tokens[t]
        m3_total += 1

        if t < 10:
            print(f"  {t:>6} {tokenizer.decode([actual_tok]):>15} "
                  f"k_cos={k_cos:>.4f} v_cos={v_cos:>.4f}")

    # ── Method 4: Weighted average of recent hidden states ──
    print(f"\n  Method 4: Momentum-weighted extrapolation")
    m4_matches = 0
    m4_total = 0

    for t in range(4, len(step_hidden)):
        # Exponential moving average of deltas
        d1 = step_hidden[t-1] - step_hidden[t-2]
        d2 = step_hidden[t-2] - step_hidden[t-3]
        d3 = step_hidden[t-3] - step_hidden[t-4]
        avg_delta = 0.6 * d1 + 0.3 * d2 + 0.1 * d3
        h_pred = step_hidden[t-1] + avg_delta

        h_normed = model.model.norm(h_pred.unsqueeze(0).unsqueeze(0))
        logits = model.lm_head(h_normed)[0, 0]
        pred_tok = logits.argmax().item()
        actual_tok = step_tokens[t]

        match = "YES" if pred_tok == actual_tok else ""
        if pred_tok == actual_tok:
            m4_matches += 1
        m4_total += 1

        if t < 10 or pred_tok == actual_tok:
            print(f"  {t:>6} {tokenizer.decode([actual_tok]):>15} "
                  f"{tokenizer.decode([pred_tok]):>15} {match:>6}")

    print(f"  Momentum extrapolation: {m4_matches}/{m4_total} = {m4_matches/max(m4_total,1)*100:.0f}%")

    # ── Method 5: Per-position cosine between extrapolated and actual ──
    # Just measure trajectory smoothness: is the manifold curve predictable?
    print(f"\n  Trajectory smoothness (hidden state):")
    for t in range(1, min(len(step_hidden), 12)):
        if t >= 2:
            delta = step_hidden[t-1] - step_hidden[t-2]
            h_pred = step_hidden[t-1] + delta
            cos = F.cosine_similarity(
                h_pred.unsqueeze(0).float(),
                step_hidden[t].unsqueeze(0).float()
            ).item()
        else:
            cos = float('nan')

        tok = tokenizer.decode([step_tokens[t]])
        top1_logit = model.lm_head(model.model.norm(
            step_hidden[t-1].unsqueeze(0).unsqueeze(0)))[0, 0]
        top1_prob = F.softmax(top1_logit.float(), dim=-1)[step_tokens[t]].item()

        defined = "DEFINED" if top1_prob > 0.8 else "branch" if top1_prob < 0.3 else ""
        print(f"    t={t:>2} '{tok:>12}' cos={cos:>.4f} p={top1_prob:.3f} {defined}")

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")

with open("machines/strix_halo/results/kv_extrapolation.json", "w") as f:
    json.dump({"prompts": len(prompts), "n_gen": N_GEN}, f, indent=2)
print("Saved results.", flush=True)
