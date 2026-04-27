"""Jacobi + Medusa-5 init: lookahead decoding architecture.

Each step:
  1. Run main model on current prompt -> h_final[-1].
  2. Run 5 Medusa heads on h_final[-1] -> 5 candidate tokens.
  3. Append candidates to prompt as the Jacobi initial guess.
  4. Iterate: run forward, update each draft to argmax(prior position logit).
  5. Stop when iteration is stable (fixed point reached).
  6. Append converged chunk. Move forward.

Output is text-identical to greedy autoregressive (Jacobi fixed point = greedy).
Speedup comes from how many iters Jacobi needs to converge — should be much
fewer with Medusa init than with placeholder init.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
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
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_jacobi_medusa_init.json")


class MedusaHead(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])
    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
lm_head_weight = model.lm_head.weight.detach()

print("Loading 5 Medusa heads...")
medusa_heads = []
for k in range(1, K_CHUNK + 1):
    h = MedusaHead(d_model, n_layers=1).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h.eval()
    medusa_heads.append(h)


@torch.no_grad()
def medusa_init(ids):
    """Run model + heads to get K candidate tokens."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    h_last = out.hidden_states[-1][:, -1:].float()
    candidates = []
    for head in medusa_heads:
        logits = head(h_last, lm_head_weight).float()
        candidates.append(logits.argmax(-1))
    return torch.cat(candidates, dim=1)  # [1, K]


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
def jacobi_with_medusa_init(prompt, n_tokens, k_chunk=K_CHUNK, max_iters=MAX_ITERS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = ids.shape[1]
    forwards = 0
    iters_per_chunk = []

    while ids.shape[1] - prompt_len < n_tokens:
        prefix_len = ids.shape[1]

        # Init: run Medusa heads to get k_chunk candidates
        init_chunk = medusa_init(ids)  # [1, K]
        forwards += 1  # 1 forward for medusa init
        ids = torch.cat([ids, init_chunk], dim=1)

        prev_chunk = ids[:, prefix_len:].clone()
        converged_iter = max_iters
        for it in range(max_iters):
            out = model(ids, use_cache=False)
            forwards += 1
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
    jac_ids, jac_forwards, iters = jacobi_with_medusa_init(prompt, N_GENERATE)

    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    jac_text = tokenizer.decode(jac_ids[0], skip_special_tokens=True)
    base_nll, base_ppl = self_perplexity(base_ids)
    jac_nll, jac_ppl = self_perplexity(jac_ids)
    speedup = base_forwards / jac_forwards if jac_forwards > 0 else 0
    match = (base_ids[0, :len(jac_ids[0])] == jac_ids[0, :len(base_ids[0])]).all().item() \
        if base_ids.shape[1] == jac_ids.shape[1] else False

    print(f"\n--- BASELINE ---")
    print(base_text)
    print(f"forwards: {base_forwards}, self-PPL: {base_ppl:.2f}")
    print(f"\n--- JACOBI + MEDUSA-5 INIT ---")
    print(jac_text)
    print(f"forwards: {jac_forwards}, iters per chunk: {iters}")
    print(f"speedup: {speedup:.2f}x, self-PPL: {jac_ppl:.2f}, text-match: {match}")

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
