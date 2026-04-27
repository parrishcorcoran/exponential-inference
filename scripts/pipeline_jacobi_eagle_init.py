"""Jacobi + Eagle drafter init — use autoregressive Eagle drafts as Jacobi seed.

Each step:
  1. Run main model on current prompt -> h_final[-1].
  2. Run Eagle drafter autoregressively for K steps -> K candidate tokens.
     Each Eagle step uses the previous draft as input (real autoregression).
  3. Append Eagle's K drafts as the Jacobi seed.
  4. Iterate: run main forward, update drafts to argmax(prior position logit).
  5. Stop when stable.
  6. Append converged chunk; advance.

Eagle's autoregressive nature means its 5 drafts are *coordinated* (each
draft conditions on the prior). Should give Jacobi a much better init than
Medusa's independent heads, especially at deep offsets.
"""
import math
import json
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
RESULTS_PATH = Path("results/pipeline_jacobi_eagle_init.json")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(self.weight.dtype)


class EagleDrafter(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, ffn_mult=4):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.head_dim = head_dim
        self.fc_in = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        ffn_dim = ffn_mult * d_model
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, h_prev, tok_embeds):
        x = self.fc_in(torch.cat([h_prev, tok_embeds], dim=-1))
        B, S, _ = x.shape
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device, dtype=scores.dtype), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ V).transpose(1, 2).reshape(B, S, -1)
        x = x + self.o_proj(attn_out)
        x_norm2 = self.norm2(x)
        x = x + self.down(F.silu(self.gate(x_norm2)) * self.up(x_norm2))
        return x


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

d_model = model.config.hidden_size
n_attn_heads = model.config.num_attention_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // n_attn_heads)
embed_layer = model.model.embed_tokens
lm_head_weight = model.lm_head.weight.detach()

print("Loading Eagle drafter...")
drafter = EagleDrafter(d_model, n_attn_heads, head_dim).to(device)
drafter.load_state_dict(torch.load(CKPT_DIR / "eagle_drafter.pt", map_location=device))
drafter.eval()


@torch.no_grad()
def eagle_init(ids):
    """Autoregressively draft K tokens via Eagle."""
    out = model(ids, output_hidden_states=True, use_cache=False)
    h_t = out.hidden_states[-1][:, -1:].float()
    tok_t = ids[:, -1:]
    drafts = []
    for _ in range(K_CHUNK):
        te = embed_layer(tok_t).float()
        pred_h = drafter(h_t, te)
        logits = F.linear(pred_h.to(lm_head_weight.dtype), lm_head_weight).float()
        next_tok = logits.argmax(-1)
        drafts.append(next_tok)
        h_t = pred_h
        tok_t = next_tok
    return torch.cat(drafts, dim=1)  # [1, K]


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
def jacobi_with_eagle_init(prompt, n_tokens, k_chunk=K_CHUNK, max_iters=MAX_ITERS):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = ids.shape[1]
    main_forwards = 0
    iters_per_chunk = []

    while ids.shape[1] - prompt_len < n_tokens:
        prefix_len = ids.shape[1]

        init_chunk = eagle_init(ids)
        main_forwards += 1  # Eagle init runs 1 main model forward (for h_t)
        ids = torch.cat([ids, init_chunk], dim=1)

        prev_chunk = ids[:, prefix_len:].clone()
        converged_iter = max_iters
        for it in range(max_iters):
            out = model(ids, use_cache=False)
            main_forwards += 1
            new_chunk = out.logits[:, prefix_len - 1:-1, :].argmax(-1)
            ids[:, prefix_len:] = new_chunk
            if torch.equal(new_chunk, prev_chunk):
                converged_iter = it + 1
                break
            prev_chunk = new_chunk.clone()
        iters_per_chunk.append(converged_iter)

    return ids, main_forwards, iters_per_chunk


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
    jac_ids, jac_forwards, iters = jacobi_with_eagle_init(prompt, N_GENERATE)

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
    print(f"\n--- JACOBI + EAGLE INIT ---")
    print(jac_text)
    print(f"main forwards: {jac_forwards}, iters per chunk: {iters}")
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
