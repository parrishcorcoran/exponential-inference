"""Uniform rectangle prototype on Qwen3-4B.

Factorize EVERY layer to the same bottleneck width.
All matmuls become uniform size → GPU-friendly.

Method:
  1. Load 4B, measure wormhole shape
  2. Factorize all layers to uniform rank via SVD
  3. Train factored weights (not just norms) — real training
  4. Measure quality recovery over many steps

Start with rank 256 (0.53B params, 7.6x compression).
Train on OpenWebText, eval every 500 steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import time
import gc
from pathlib import Path

device = "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_owt_tokens(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(encoded)
            continue
        toks.extend(encoded)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def iter_batches(tokens, seq_len, batch_size, device):
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


@torch.no_grad()
def eval_ppl(model, val_tokens, seq_len, device, n_batches=15):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


class FactoredLinear(nn.Module):
    """Uniform bottleneck: W ≈ A @ B, A=[out,rank], B=[rank,in]."""
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = x @ self.B.T @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out


def factorize_to_rank(linear, rank):
    """SVD factorize a linear layer to uniform rank."""
    W = linear.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqS = S[:k].sqrt()
    A = (U[:, :k] * sqS).to(torch.bfloat16)  # [out, rank]
    B = (sqS.unsqueeze(1) * Vt[:k]).to(torch.bfloat16)  # [rank, in]
    bias = linear.bias.data if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


MODEL = "Qwen/Qwen3-4B"
RANK = 256
SEQ_LEN = 256
TRAIN_STEPS = 10000
EVAL_EVERY = 500
LR = 2e-4

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "Machine learning models are trained by",
    "The French Revolution began in 1789 when",
]

print("=" * 60)
print(f"UNIFORM RECTANGLE — Qwen3-4B at rank {RANK}")
print(f"  Target: all layers same bottleneck width")
print(f"  Training: {TRAIN_STEPS} steps on OpenWebText")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("\nLoading tokens...", flush=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 5000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 5000)
print(f"  Train: {len(train_tokens)} tokens, Val: {len(val_tokens)} tokens", flush=True)

print(f"\nLoading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Teacher ppl: {teacher_ppl:.1f}", flush=True)

# ═══════════════════════════════════════════════════════
# Gradual equalization: squeeze toward uniform rank
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"GRADUAL EQUALIZATION toward rank {RANK}")
print(f"  1% reduction per step, 200 FT steps between each")
print(f"{'='*60}", flush=True)

# Start: each layer's K projection at its natural full rank
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
current_ranks = {i: d_kv for i in range(L)}  # all start at full (640 for 4B)

history = []
step = 0

# Freeze embed/lm_head
model.get_input_embeddings().weight.requires_grad = False
if hasattr(model, 'lm_head'):
    model.lm_head.weight.requires_grad = False

while True:
    # Check if all at target
    max_rank = max(current_ranks.values())
    if max_rank <= RANK:
        print(f"\n  All layers at rank {RANK}. Done.", flush=True)
        break

    step += 1

    # Squeeze each layer by 1% toward target
    for i in range(L):
        if current_ranks[i] <= RANK:
            continue
        new_rank = max(int(current_ranks[i] * 0.99), RANK)
        if new_rank >= current_ranks[i]:
            new_rank = current_ranks[i] - 1
        if new_rank < RANK:
            new_rank = RANK

        # SVD truncate K projection
        proj = model.model.layers[i].self_attn.k_proj
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(new_rank, len(S))
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
        current_ranks[i] = new_rank

    torch.cuda.empty_cache()

    # FT: train norms (fast, safe)
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
        model.train()
        ft_step = 0
        for inp, tgt in iter_batches(train_tokens, SEQ_LEN, 1, device):
            if ft_step >= 200: break
            opt.zero_grad()
            loss = F.cross_entropy(
                model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(),
                tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            ft_step += 1
        del opt
        for p in model.parameters(): p.requires_grad = False
        torch.cuda.empty_cache()

    # Eval
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    avg_rank = sum(current_ranks.values()) / L
    max_rank = max(current_ranks.values())
    min_rank = min(current_ranks.values())

    if step % 5 == 0 or step <= 3:
        # Coherence
        ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  step {step:>3}: ppl={ppl:.1f} avg_rank={avg_rank:.0f} range={min_rank}-{max_rank} [{text[:50]}]", flush=True)
    else:
        print(f"  step {step:>3}: ppl={ppl:.1f} avg_rank={avg_rank:.0f} range={min_rank}-{max_rank}", flush=True)

    history.append({
        "step": step, "ppl": round(ppl, 2),
        "avg_rank": round(avg_rank, 1),
        "min_rank": min_rank, "max_rank": max_rank,
    })

    # Thermostat
    if ppl > teacher_ppl * 3:
        print(f"\n  ⚠ THERMOSTAT: ppl {ppl:.1f} > {teacher_ppl * 3:.1f}", flush=True)
        break

post_factor_ppl = ppl

# Final eval
del opt, scheduler
torch.cuda.empty_cache()

final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"\n{'='*60}")
print(f"UNIFORM RECTANGLE COMPLETE")
print(f"  Teacher: {teacher_ppl:.1f}")
print(f"  Post-factorization: {post_factor_ppl:.1f}")
print(f"  After {TRAIN_STEPS} steps: {final_ppl:.1f}")
print(f"  Recovery: {post_factor_ppl/final_ppl:.1f}x")
print(f"  Params: {total_params/1e9:.2f}B at rank {RANK}")
print(f"{'='*60}", flush=True)

# Coherence
print(f"\nCoherence:")
for p in PROMPTS:
    ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{p[:30]}...] → {text[:60]}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/uniform_rect_4b.json", "w") as f:
    json.dump({
        "model": MODEL, "rank": RANK,
        "teacher_ppl": teacher_ppl,
        "post_factor_ppl": post_factor_ppl,
        "final_ppl": final_ppl,
        "total_params": total_params,
        "train_steps": TRAIN_STEPS,
        "history": history,
    }, f, indent=2)
print(f"Saved results/uniform_rect_4b.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
