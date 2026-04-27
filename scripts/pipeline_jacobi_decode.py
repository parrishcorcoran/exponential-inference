"""Jacobi parallel decoding: iterate until fixed point.

For each generation step, we want to produce K tokens at once:
  1. Append K placeholder tokens to the prompt (init: replicate last token).
  2. Run forward pass on [prefix; placeholders].
  3. Update each placeholder to argmax(logits at prior position).
  4. If updated tokens == previous, converged. Else iterate.
  5. Append the converged K tokens. Move to the next chunk.

The fixed point of this iteration IS greedy autoregressive decoding —
because at convergence, each position's input matches the prior position's
argmax, which is the definition of greedy decode. The question is how
many iterations it takes to converge.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
N_GENERATE = 60
K_CHUNK = 5
MAX_ITERS = 6
PROMPTS = [
    "The future of artificial intelligence depends on",
    "In the early morning, the city was quiet, and",
    "Once upon a time, there was a small village where",
]
RESULTS_PATH = Path("results/pipeline_jacobi_decode.json")


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False


@torch.no_grad()
def baseline_greedy(prompt, n_tokens):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    forwards = 0
    for _ in range(n_tokens):
        out = model(ids, use_cache=False)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
        forwards += 1
    return ids, forwards


@torch.no_grad()
def jacobi_decode(prompt, n_tokens, k_chunk=K_CHUNK, max_iters=MAX_ITERS):
    """Parallel decode k_chunk tokens at a time via Jacobi iteration."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    forwards = 0
    iters_per_chunk = []
    while ids.shape[1] - len(tokenizer.encode(prompt)) < n_tokens:
        # Append k_chunk placeholders (replicate last token)
        prefix_len = ids.shape[1]
        last = ids[:, -1:].repeat(1, k_chunk)
        ids = torch.cat([ids, last], dim=1)

        prev_chunk = ids[:, prefix_len:].clone()
        converged_iter = max_iters
        for it in range(max_iters):
            out = model(ids, use_cache=False)
            forwards += 1
            # Update each future position with argmax of logit at prior position
            new_chunk = out.logits[:, prefix_len - 1:-1, :].argmax(-1)
            ids[:, prefix_len:] = new_chunk
            if torch.equal(new_chunk, prev_chunk):
                converged_iter = it + 1
                break
            prev_chunk = new_chunk.clone()
        iters_per_chunk.append(converged_iter)

    return ids, forwards, iters_per_chunk


@torch.no_grad()
def self_perplexity(ids):
    out = model(ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item(), math.exp(nll.mean().item())


results = []
for prompt in PROMPTS:
    print(f"\n{'='*70}\nPROMPT: {prompt!r}\n{'='*70}")

    base_ids, base_forwards = baseline_greedy(prompt, N_GENERATE)
    jac_ids, jac_forwards, iters = jacobi_decode(prompt, N_GENERATE)

    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    jac_text = tokenizer.decode(jac_ids[0], skip_special_tokens=True)

    base_nll, base_ppl = self_perplexity(base_ids)
    jac_nll, jac_ppl = self_perplexity(jac_ids)

    n_tokens_jac = jac_ids.shape[1] - len(tokenizer.encode(prompt))
    speedup = base_forwards / jac_forwards if jac_forwards > 0 else 0

    print(f"\n--- BASELINE (greedy) ---")
    print(base_text)
    print(f"\nforwards: {base_forwards}, self-PPL: {base_ppl:.2f}")

    print(f"\n--- JACOBI (k={K_CHUNK}) ---")
    print(jac_text)
    print(f"\nforwards: {jac_forwards}, iters per chunk: {iters}, "
          f"speedup: {speedup:.2f}x, self-PPL: {jac_ppl:.2f}")

    # Sanity: jacobi text should match baseline text (both are greedy fixed point)
    match = (base_ids[0, :len(jac_ids[0])] == jac_ids[0, :len(base_ids[0])]).all().item() \
        if base_ids.shape[1] == jac_ids.shape[1] else False
    print(f"text-identical to baseline: {match}")

    results.append({"prompt": prompt,
                    "baseline_forwards": base_forwards, "jacobi_forwards": jac_forwards,
                    "speedup": speedup, "iters_per_chunk": iters,
                    "baseline_text": base_text, "jacobi_text": jac_text,
                    "baseline_ppl": base_ppl, "jacobi_ppl": jac_ppl,
                    "text_match": match})

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_generate": N_GENERATE, "k_chunk": K_CHUNK,
               "max_iters": MAX_ITERS, "results": results}, f, indent=2)
print(f"\n\nSaved {RESULTS_PATH}")
