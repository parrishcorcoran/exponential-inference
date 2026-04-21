"""Test: does each token's KV point to the next token?

After generating token N, its KV is added to the cache.
Does the KV cache (with the new angle) already contain enough
information to determine token N+1 at an EARLIER layer than
where token N resolved?

Method:
1. Generate tokens one at a time with full forward
2. At each step, after the token is found:
   - Record which layer it resolved at (logit lens stabilization)
   - Record the KV cache state
   - At each layer, apply lm_head to see if the NEXT token is already
     visible in the hidden state — meaning the current token's KV
     added the angle that makes the next token determinable
3. If the next token appears at layer L, that means the KV depth
   from the current token made it visible at layer L.
   If L < current token's resolution layer, the chain is accelerating.
"""
import torch
import torch.nn.functional as F
import json

device = "cuda"

print("=" * 70)
print("KV CHAIN TEST: does each token's KV point to the next?")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
HIDDEN = model.config.hidden_size
lm_head_weight = model.lm_head.weight
final_norm = model.model.norm

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

prompts = [
    "The theory of general relativity describes gravity as the curvature of",
    "The capital of France is Paris, which is known for its",
    "Water freezes at zero degrees Celsius and boils at one hundred",
    "Neural networks learn by adjusting weights through a process called",
    "The Fibonacci sequence starts with zero and one, then each number is",
    "In the beginning there was nothing but silence and then suddenly",
]

N_GEN = 15

print(f"\n{'='*60}")
print("Per-token: at which layer does EACH token resolve,")
print("and at which layer is the NEXT token already visible?")
print(f"{'='*60}")

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"\nPrompt: '{prompt}'")

    with torch.no_grad():
        gen_tokens = []
        resolve_layers = []     # layer where this token resolved
        next_visible_layers = []  # layer where NEXT token first appears

        # Prefill
        out = model(ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        hidden_states = out.hidden_states

        # Where does the first generated token resolve?
        final_tok = out.logits[0, -1].argmax().item()
        gen_tokens.append(final_tok)

        # Logit lens: at which layer did this token first appear?
        first_resolve = N_LAYERS
        for l in range(1, len(hidden_states)):
            h = hidden_states[l][0, -1]
            h_normed = final_norm(h.unsqueeze(0).unsqueeze(0))
            pred = F.linear(h_normed, lm_head_weight)[0, 0].argmax().item()
            if pred == final_tok:
                first_resolve = l
                break
        resolve_layers.append(first_resolve)

        # Generate subsequent tokens
        for step in range(N_GEN - 1):
            # Run next token through full forward with hidden states
            tok_input = torch.tensor([[gen_tokens[-1]]], device=device)
            out = model(tok_input, past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = out.past_key_values
            hidden_states = out.hidden_states

            next_tok = out.logits[0, -1].argmax().item()
            gen_tokens.append(next_tok)

            # At which layer did THIS token resolve?
            this_resolve = N_LAYERS
            for l in range(1, len(hidden_states)):
                h = hidden_states[l][0, -1]
                h_normed = final_norm(h.unsqueeze(0).unsqueeze(0))
                pred = F.linear(h_normed, lm_head_weight)[0, 0].argmax().item()
                if pred == next_tok:
                    this_resolve = l
                    break
            resolve_layers.append(this_resolve)

            # At which layer was THIS token already visible in the
            # PREVIOUS step's hidden states? (i.e., the previous token's
            # forward pass at layer L already predicted this token)
            # We saved previous hidden_states... but we overwrote them.
            # So let's track it differently: at each layer of THIS step,
            # is the NEXT token (step+2) already visible?
            # We don't know step+2 yet. So instead, check:
            # at each layer of this step, what token does logit lens predict?
            # If it matches what step+1 WILL generate, the KV made it visible.

        # Now: for each generated token, check at which layer the PREVIOUS
        # step's computation already predicted it.
        # We need to rerun with tracking. Let's do a clean pass.

        # Clean rerun: for each step, save per-layer predictions
        out = model(ids, use_cache=True, output_hidden_states=True)
        past2 = out.past_key_values
        hs = out.hidden_states

        per_step_per_layer_preds = []

        # Prefill: per-layer prediction at last position
        layer_preds = []
        for l in range(len(hs)):
            h = hs[l][0, -1]
            h_normed = final_norm(h.unsqueeze(0).unsqueeze(0))
            pred = F.linear(h_normed, lm_head_weight)[0, 0].argmax().item()
            layer_preds.append(pred)
        per_step_per_layer_preds.append(layer_preds)

        for step in range(N_GEN - 1):
            tok_input = torch.tensor([[gen_tokens[step]]], device=device)
            out = model(tok_input, past_key_values=past2, use_cache=True,
                       output_hidden_states=True)
            past2 = out.past_key_values
            hs = out.hidden_states

            layer_preds = []
            for l in range(len(hs)):
                h = hs[l][0, -1]
                h_normed = final_norm(h.unsqueeze(0).unsqueeze(0))
                pred = F.linear(h_normed, lm_head_weight)[0, 0].argmax().item()
                layer_preds.append(pred)
            per_step_per_layer_preds.append(layer_preds)

    # Analysis: for each token N, at which layer of step N-1
    # was token N already the argmax prediction?
    print(f"\n  {'Step':>5} {'Token':>12} {'Resolves':>9} {'Visible@':>9} {'Delta':>6} {'Chain':>6}")
    print("  " + "-" * 55)

    chain_data = []
    for i in range(len(gen_tokens)):
        tok_str = tokenizer.decode([gen_tokens[i]])
        res_layer = resolve_layers[i]

        # At which layer of step i-1 was token i predicted?
        if i > 0:
            prev_preds = per_step_per_layer_preds[i - 1]
            visible_at = N_LAYERS + 1  # not visible
            for l in range(len(prev_preds)):
                if prev_preds[l] == gen_tokens[i]:
                    visible_at = l
                    break

            # Also check: at which layer of step i-1 was token i the
            # argmax AND stayed the argmax through remaining layers?
            stable_at = N_LAYERS + 1
            for l in range(len(prev_preds)):
                if all(prev_preds[ll] == gen_tokens[i] for ll in range(l, len(prev_preds))):
                    stable_at = l
                    break

            delta = visible_at - res_layer if visible_at <= N_LAYERS else None
            delta_str = f"{delta:>+5}" if delta is not None else "  N/A"

            chain = ""
            if visible_at <= N_LAYERS:
                if visible_at < resolve_layers[i - 1]:
                    chain = "ACCEL"  # next token visible earlier than current resolved
                elif visible_at == resolve_layers[i - 1]:
                    chain = "SAME"
                else:
                    chain = "slower"

            vis_str = f"L{visible_at}" if visible_at <= N_LAYERS else "never"
            print(f"  {i:>5} {tok_str:>12} L{res_layer:>2}      "
                  f"{vis_str:>7}  {delta_str} {chain:>6}")

            chain_data.append({
                "token": tok_str, "resolve": res_layer,
                "visible_at_prev": visible_at if visible_at <= N_LAYERS else None,
                "stable_at_prev": stable_at if stable_at <= N_LAYERS else None,
            })
        else:
            print(f"  {i:>5} {tok_str:>12} L{res_layer:>2}          —      —      —")

    # Summary for this prompt
    visible_count = sum(1 for d in chain_data if d["visible_at_prev"] is not None)
    accel_count = sum(1 for j, d in enumerate(chain_data)
                      if d["visible_at_prev"] is not None
                      and d["visible_at_prev"] < resolve_layers[j])
    print(f"\n  {visible_count}/{len(chain_data)} tokens visible in previous step's layers")
    print(f"  {accel_count}/{len(chain_data)} accelerating (visible earlier than previous resolved)")

# ═══════════════════════════════════════════════════════
# Summary across all prompts
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("The question: does each token's KV add depth that")
print("makes the next token visible at an earlier layer?")
print(f"{'='*60}")

with open("machines/strix_halo/results/kv_chain.json", "w") as f:
    json.dump({"n_gen": N_GEN, "n_prompts": len(prompts)}, f, indent=2)
print("Saved results.", flush=True)
