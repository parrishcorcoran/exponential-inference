"""Chained decoding: KV-Medusa + iterative token prediction.

Instead of predicting all tokens from h_t (dies at t+2),
chain through predicted KV:

1. h_t → predict token t+1 (top-k candidates)
2. For each candidate: embed it, inject predicted KV, run
   model forward to get h_{t+1}
3. h_{t+1} → predict token t+2 (top-k candidates)
4. Build a TREE of candidate sequences
5. Verify the entire tree in one batched forward pass
6. Accept the longest matching branch

The key: KV-Medusa provides cache at each position (cos 0.72+),
so each chain step only needs ONE forward pass of the model
with the draft token, not from scratch.

Tree width k=3 (top-3 candidates per position)
Tree depth d=10 (chain 10 positions)
Total candidates: up to 3^10 = 59K (pruned to ~100 best)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda"
CHECKPOINT = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
N_TOKENS = 50

print("=" * 60)
print("CHAINED DECODING: KV-Medusa + tree search")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = d // model.config.num_attention_heads
L = model.config.num_hidden_layers
vocab_size = model.config.vocab_size

PROMPT = "The theory of general relativity describes gravity as"
ids = tokenizer(PROMPT, return_tensors='pt').input_ids.to(device)

# ═══════════════════════════════════════════════════════
# Standard generation (ground truth)
# ═══════════════════════════════════════════════════════
print("\n--- Standard generation (ground truth) ---", flush=True)
with torch.no_grad():
    out_std = model.generate(ids, max_new_tokens=N_TOKENS, do_sample=False)
standard_tokens = out_std[0][ids.shape[1]:].tolist()
standard_text = tokenizer.decode(out_std[0][ids.shape[1]:], skip_special_tokens=True)
print(f"  {standard_text[:80]}", flush=True)

# Time standard generation
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    model.generate(ids, max_new_tokens=N_TOKENS, do_sample=False)
torch.cuda.synchronize()
standard_time = time.time() - t0
standard_tps = N_TOKENS / standard_time
print(f"  {standard_tps:.1f} tok/s ({standard_time:.2f}s)", flush=True)

# ═══════════════════════════════════════════════════════
# Chained decoding
# ═══════════════════════════════════════════════════════
print("\n--- Chained decoding ---", flush=True)

# Method: greedy chain with verification
# At each step:
# 1. Run full forward pass → get next token + hidden state
# 2. From hidden state, greedily predict next K tokens
#    by running model forward with each predicted token
#    using use_cache=True (appending to KV cache)
# 3. Verify: compare draft tokens with what a fresh forward
#    pass would produce
# 4. Accept matching prefix, reject the rest

# Simpler version first: draft N tokens using cached forward passes
# (each draft step reuses the KV cache from the previous step)
# Then verify all at once

DRAFT_LENGTH = 10  # draft 10 tokens per step

generated = ids[0].tolist()
total_forward_passes = 0
total_tokens = 0
total_steps = 0
accepted_counts = []

torch.cuda.synchronize()
t0_chain = time.time()

with torch.no_grad():
    while total_tokens < N_TOKENS:
        total_steps += 1

        # Step 1: Full forward pass on current sequence
        inp = torch.tensor([generated], dtype=torch.long, device=device)
        out = model(inp, use_cache=True)
        past_kv = out.past_key_values
        total_forward_passes += 1

        # Get the first draft token (greedy)
        next_logits = out.logits[0, -1]
        draft_tokens = [next_logits.argmax().item()]

        # Step 2: Chain forward — draft more tokens using KV cache
        current_past = past_kv
        for d in range(DRAFT_LENGTH - 1):
            draft_inp = torch.tensor([[draft_tokens[-1]]], dtype=torch.long, device=device)
            out_draft = model(draft_inp, past_key_values=current_past, use_cache=True)
            current_past = out_draft.past_key_values
            next_tok = out_draft.logits[0, -1].argmax().item()
            draft_tokens.append(next_tok)
            total_forward_passes += 1

        # Step 3: Verify — run a fresh forward pass with all draft tokens
        # to see how many match
        verify_inp = torch.tensor([generated + draft_tokens], dtype=torch.long, device=device)
        out_verify = model(verify_inp, use_cache=False)
        total_forward_passes += 1

        # Compare: for each position, does the verified token match the draft?
        accepted = 0
        for j in range(len(draft_tokens)):
            pos = len(generated) + j
            if pos >= out_verify.logits.shape[1]:
                break
            verified_tok = out_verify.logits[0, pos - 1].argmax().item()
            if verified_tok == draft_tokens[j]:
                accepted += 1
            else:
                # First mismatch — accept the verified token instead
                draft_tokens[j] = verified_tok
                accepted += 1
                break

        # Accept tokens
        for j in range(accepted):
            generated.append(draft_tokens[j])
            total_tokens += 1
            if total_tokens >= N_TOKENS:
                break

        accepted_counts.append(accepted)

torch.cuda.synchronize()
chain_time = time.time() - t0_chain
chain_tps = total_tokens / chain_time

chain_text = tokenizer.decode(torch.tensor(generated[ids.shape[1]:]), skip_special_tokens=True)

print(f"  Text: {chain_text[:80]}")
print(f"  Match: {'YES' if chain_text[:60] == standard_text[:60] else 'PARTIAL'}")
print(f"  Steps: {total_steps}")
print(f"  Forward passes: {total_forward_passes}")
print(f"  Avg accepted/step: {sum(accepted_counts)/len(accepted_counts):.1f}")
print(f"  Speed: {chain_tps:.1f} tok/s ({chain_time:.2f}s)", flush=True)

# ═══════════════════════════════════════════════════════
# Chained with larger draft (20 tokens)
# ═══════════════════════════════════════════════════════
print("\n--- Chained decoding (draft=20) ---", flush=True)

DRAFT_LENGTH = 20
generated2 = ids[0].tolist()
total_forward_passes2 = 0
total_tokens2 = 0
total_steps2 = 0
accepted_counts2 = []

torch.cuda.synchronize()
t0_chain2 = time.time()

with torch.no_grad():
    while total_tokens2 < N_TOKENS:
        total_steps2 += 1

        inp = torch.tensor([generated2], dtype=torch.long, device=device)
        out = model(inp, use_cache=True)
        past_kv = out.past_key_values
        total_forward_passes2 += 1

        draft_tokens = [out.logits[0, -1].argmax().item()]
        current_past = past_kv
        for d in range(DRAFT_LENGTH - 1):
            draft_inp = torch.tensor([[draft_tokens[-1]]], dtype=torch.long, device=device)
            out_draft = model(draft_inp, past_key_values=current_past, use_cache=True)
            current_past = out_draft.past_key_values
            draft_tokens.append(out_draft.logits[0, -1].argmax().item())
            total_forward_passes2 += 1

        verify_inp = torch.tensor([generated2 + draft_tokens], dtype=torch.long, device=device)
        out_verify = model(verify_inp, use_cache=False)
        total_forward_passes2 += 1

        accepted = 0
        for j in range(len(draft_tokens)):
            pos = len(generated2) + j
            if pos >= out_verify.logits.shape[1]: break
            verified_tok = out_verify.logits[0, pos - 1].argmax().item()
            if verified_tok == draft_tokens[j]:
                accepted += 1
            else:
                draft_tokens[j] = verified_tok
                accepted += 1
                break

        for j in range(accepted):
            generated2.append(draft_tokens[j])
            total_tokens2 += 1
            if total_tokens2 >= N_TOKENS: break

        accepted_counts2.append(accepted)

torch.cuda.synchronize()
chain_time2 = time.time() - t0_chain2
chain_tps2 = total_tokens2 / chain_time2

chain_text2 = tokenizer.decode(torch.tensor(generated2[ids.shape[1]:]), skip_special_tokens=True)
print(f"  Text: {chain_text2[:80]}")
print(f"  Match: {'YES' if chain_text2[:60] == standard_text[:60] else 'PARTIAL'}")
print(f"  Steps: {total_steps2}")
print(f"  Forward passes: {total_forward_passes2}")
print(f"  Avg accepted/step: {sum(accepted_counts2)/len(accepted_counts2):.1f}")
print(f"  Speed: {chain_tps2:.1f} tok/s ({chain_time2:.2f}s)", flush=True)

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"CHAINED DECODE SUMMARY")
print(f"{'='*60}")
print(f"  Standard:       {standard_tps:.1f} tok/s | {N_TOKENS} passes for {N_TOKENS} tokens")
print(f"  Chain (d=10):   {chain_tps:.1f} tok/s | {total_forward_passes} passes for {total_tokens} tokens | {sum(accepted_counts)/len(accepted_counts):.1f} accepted/step")
print(f"  Chain (d=20):   {chain_tps2:.1f} tok/s | {total_forward_passes2} passes for {total_tokens2} tokens | {sum(accepted_counts2)/len(accepted_counts2):.1f} accepted/step")
print(f"\n  Speedup (d=10): {chain_tps/standard_tps:.2f}x")
print(f"  Speedup (d=20): {chain_tps2/standard_tps:.2f}x")
print(f"{'='*60}", flush=True)

with open("results/chained_decode.json", "w") as f:
    json.dump({
        "standard_tps": round(standard_tps, 2),
        "chain_10_tps": round(chain_tps, 2),
        "chain_20_tps": round(chain_tps2, 2),
        "chain_10_accepted_avg": round(sum(accepted_counts)/len(accepted_counts), 2),
        "chain_20_accepted_avg": round(sum(accepted_counts2)/len(accepted_counts2), 2),
        "chain_10_speedup": round(chain_tps/standard_tps, 2),
        "chain_20_speedup": round(chain_tps2/standard_tps, 2),
        "standard_text": standard_text[:100],
        "chain_10_text": chain_text[:100],
        "chain_20_text": chain_text2[:100],
    }, f, indent=2)
print(f"Saved results/chained_decode.json", flush=True)

del model; torch.cuda.empty_cache()
