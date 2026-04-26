"""Uniform rectangle 4B with Cholesky whitening (SVD-LLM style).

Same as the plain version but uses Cholesky whitening before SVD:
  1. Cache activations (one forward pass per step)
  2. Compute Cholesky of activation covariance: L = chol(X^T X)
  3. Whiten weights: W_white = W @ L
  4. SVD truncate in whitened space
  5. Un-whiten: W_compressed = W_trunc @ L^{-1}
  6. Fine-tune norms 200 steps

In whitened space, each singular value directly equals its
compression loss — provably optimal truncation.

Compare with plain SVD version: does whitening give smoother
convergence and deeper compression?
"""
import torch
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


def cache_layer_inputs(model, tokens, seq_len, device):
    """One forward pass — cache input activations per layer."""
    layer_inputs = {}
    handles = []
    def make_hook(idx):
        def hook(module, args, kwargs):
            layer_inputs[idx] = args[0].detach().float()
        return hook
    for i in range(len(model.model.layers)):
        h = model.model.layers[i].register_forward_pre_hook(make_hook(i), with_kwargs=True)
        handles.append(h)
    inp = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    with torch.no_grad():
        model(inp, use_cache=False)
    for h in handles:
        h.remove()
    return layer_inputs


def whitened_svd_compress(proj, X, rank):
    """SVD-LLM style: Cholesky whiten → SVD → truncate → un-whiten."""
    W = proj.weight.data.float()  # [d_out, d_in]

    # Covariance of input activations: X is [seq, d_in]
    XtX = X.T @ X  # [d_in, d_in]

    # Add small regularization for numerical stability
    XtX += torch.eye(XtX.shape[0], device=XtX.device) * 1e-6

    # Cholesky decomposition: XtX = L @ L^T
    try:
        L_chol = torch.linalg.cholesky(XtX)
    except:
        # Fallback to plain SVD if Cholesky fails
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(min(rank, len(S)), 1)
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
        return

    # Whiten: W_white = W @ L (in whitened space, loss = singular value)
    W_white = W @ L_chol

    # SVD in whitened space
    U, S, Vt = torch.linalg.svd(W_white, full_matrices=False)
    k = max(min(rank, len(S)), 1)

    # Truncate
    W_trunc = (U[:, :k] * S[:k]) @ Vt[:k]

    # Un-whiten: W_compressed = W_trunc @ L^{-1}
    L_inv = torch.linalg.inv(L_chol)
    W_restored = W_trunc @ L_inv

    proj.weight.data = W_restored.to(proj.weight.dtype)

    # Cleanup
    del W, XtX, L_chol, W_white, U, S, Vt, W_trunc, L_inv, W_restored
    torch.cuda.empty_cache()


def finetune_norms(model, train_tokens, seq_len, device, steps=200, lr=5e-5):
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable: return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train(); step = 0
    for inp, tgt in iter_batches(train_tokens, seq_len, 1, device):
        if step >= steps: break
        opt.zero_grad()
        loss = F.cross_entropy(
            model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(),
            tgt.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); step += 1
    del opt
    for p in model.parameters(): p.requires_grad = False
    torch.cuda.empty_cache()


MODEL = "Qwen/Qwen3-4B"
RANK_TARGET = 256
SEQ_LEN = 256
SQUEEZE = 0.99  # 1% per step
MAX_STEPS = 100

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "Machine learning models are trained by",
    "The French Revolution began in 1789 when",
]

print("=" * 60)
print(f"UNIFORM RECTANGLE 4B — WHITENED (SVD-LLM style)")
print(f"  Cholesky whitening before SVD truncation")
print(f"  1% squeeze per step, 200 FT steps between each")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("\nLoading tokens...", flush=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 5000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 5000)
print(f"  Train: {len(train_tokens)}, Val: {len(val_tokens)}", flush=True)

print(f"\nLoading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
thermostat = teacher_ppl * 3
print(f"  Teacher ppl: {teacher_ppl:.1f}")
print(f"  Thermostat: {thermostat:.1f}")
print(f"  K full rank: {d_kv}", flush=True)

current_ranks = {i: d_kv for i in range(L)}
history = []

for step in range(1, MAX_STEPS + 1):
    # Compute target rank for this step
    for i in range(L):
        new_rank = max(int(current_ranks[i] * SQUEEZE), RANK_TARGET)
        if new_rank >= current_ranks[i]:
            new_rank = current_ranks[i] - 1
        if new_rank < RANK_TARGET:
            new_rank = RANK_TARGET
        current_ranks[i] = new_rank

    # Check if at target
    if max(current_ranks.values()) <= RANK_TARGET:
        print(f"\n  All at target rank {RANK_TARGET}. Done.", flush=True)
        break

    # Cache activations for whitening
    layer_inputs = cache_layer_inputs(model, val_tokens, SEQ_LEN, device)

    # Whitened SVD compress each layer's K projection
    for i in range(L):
        if current_ranks[i] < d_kv:
            X = layer_inputs[i][0]  # [seq, d_model]
            whitened_svd_compress(model.model.layers[i].self_attn.k_proj, X, current_ranks[i])

    del layer_inputs
    torch.cuda.empty_cache()

    # Fine-tune norms
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=200)

    # Eval
    ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    avg_rank = sum(current_ranks.values()) / L

    if step % 5 == 0 or step <= 3:
        ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  step {step:>3}: ppl={ppl:.1f} avg_rank={avg_rank:.0f} [{text[:50]}]", flush=True)
    else:
        print(f"  step {step:>3}: ppl={ppl:.1f} avg_rank={avg_rank:.0f}", flush=True)

    history.append({
        "step": step, "ppl": round(ppl, 2), "avg_rank": round(avg_rank, 1),
    })

    if ppl > thermostat:
        print(f"\n  ⚠ THERMOSTAT: ppl {ppl:.1f} > {thermostat:.1f}", flush=True)
        break

# Final
print(f"\n{'='*60}")
print(f"WHITENED UNIFORM RECTANGLE COMPLETE")
print(f"  Teacher: {teacher_ppl:.1f}")
if history:
    print(f"  Final: ppl={history[-1]['ppl']:.1f} rank={history[-1]['avg_rank']:.0f}")
    best = min(history, key=lambda h: h['ppl'])
    print(f"  Best: ppl={best['ppl']:.1f} at rank={best['avg_rank']:.0f}")
print(f"  Steps: {len(history)}")

# Coherence
print(f"\nCoherence:")
for p in PROMPTS:
    ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{p[:30]}...] → {text[:60]}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/uniform_rect_4b_whitened.json", "w") as f:
    json.dump({
        "model": MODEL, "method": "cholesky_whitened_svd",
        "teacher_ppl": teacher_ppl,
        "history": history,
    }, f, indent=2)
print(f"\nSaved results/uniform_rect_4b_whitened.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
