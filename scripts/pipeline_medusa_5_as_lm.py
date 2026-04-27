"""Medusa-as-LM (no verify): use 5 standard Medusa heads to produce 5 tokens per step.

Each step:
  1. Run main model on current prompt -> h_final[-1].
  2. All 5 heads on h_final[-1] -> 5 token predictions (one per offset).
  3. Append all 5 to the prompt. Advance 5 positions.
  4. Repeat.

This is the user's reframe: don't verify, just take the 5 picks as "the
next 5 tokens of generation." Speedup is 5× per main forward pass.
Text quality is the question.
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
N_GENERATE = 80  # divisible by 5
N_OFFSETS = 5
PROMPTS = [
    "The future of artificial intelligence depends on",
    "In the early morning, the city was quiet, and",
    "Once upon a time, there was a small village where",
]
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_medusa_5_as_lm.json")


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

print("Loading 5 standard Medusa heads...")
medusa_heads = []
for k in range(1, N_OFFSETS + 1):
    h = MedusaHead(d_model, n_layers=1).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"medusa_head_{k}.pt", map_location=device))
    h.eval()
    medusa_heads.append(h)


@torch.no_grad()
def generate_baseline(prompt, n_tokens):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(n_tokens):
        out = model(ids, use_cache=False)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids


@torch.no_grad()
def generate_medusa_5(prompt, n_tokens):
    """Each step: 1 forward pass → 5 Medusa heads → 5 tokens appended."""
    assert n_tokens % N_OFFSETS == 0, "n_tokens must be divisible by N_OFFSETS"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(n_tokens // N_OFFSETS):
        out = model(ids, output_hidden_states=True, use_cache=False)
        h_final_last = out.hidden_states[-1][:, -1:].float()  # [1, 1, d_model]
        next_5 = []
        for head in medusa_heads:
            logits = head(h_final_last, lm_head_weight).float()
            next_5.append(logits.argmax(-1))  # [1, 1]
        # Concatenate all 5 along sequence dim
        new_tokens = torch.cat(next_5, dim=1)  # [1, 5]
        ids = torch.cat([ids, new_tokens], dim=1)
    return ids


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

    base_ids = generate_baseline(prompt, N_GENERATE)
    med_ids = generate_medusa_5(prompt, N_GENERATE)

    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
    med_text = tokenizer.decode(med_ids[0], skip_special_tokens=True)

    base_nll, base_ppl = self_perplexity(base_ids)
    med_nll, med_ppl = self_perplexity(med_ids)

    print(f"\n--- BASELINE (greedy LM head) ---")
    print(base_text)
    print(f"\nself-PPL: {base_ppl:.2f}, mean NLL: {base_nll:.3f}")

    print(f"\n--- MEDUSA-5 AS LM (5 heads top-1, no verify) ---")
    print(med_text)
    print(f"\nself-PPL: {med_ppl:.2f}, mean NLL: {med_nll:.3f}")

    results.append({"prompt": prompt,
                    "baseline_text": base_text, "baseline_ppl": base_ppl, "baseline_nll": base_nll,
                    "medusa_text": med_text, "medusa_ppl": med_ppl, "medusa_nll": med_nll})

with open(RESULTS_PATH, "w") as f:
    json.dump({"checkpoint": CHECKPOINT, "n_generate": N_GENERATE, "results": results}, f, indent=2)
print(f"\n\nSaved {RESULTS_PATH}")
