"""KV Gradient Separation: forward → backward at each KV depth.

Step 1: 1 token → forward → backward → what matters with no KV
Step 2: 2 tokens → forward → backward → what matters with 1 KV entry
Step 3: 3 tokens → forward → backward → what matters with 2 KV entries
...

The DIFFERENCE between consecutive steps = what each KV entry contributes.
Separates KV dimensionality from the normal forward axis.
"""
import torch
import torch.nn.functional as F
import numpy as np

device = "cuda"

print("=" * 70)
print("KV GRADIENT SEPARATION")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device)

# Need gradients — set model to eval but enable grad on input
model.eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads

text = "The theory of general relativity describes gravity as the curvature of spacetime caused by mass"
ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
T = ids.shape[1]

print(f"Sequence: {T} tokens")
print(f"Running forward+backward at each KV depth...\n")

# For each prefix length: forward → get logit for next token → backward → measure gradients
per_step = []

for n_tokens in range(1, min(T, 16)):
    prefix = ids[:, :n_tokens]

    # Enable gradients on embeddings
    model.zero_grad()
    embeds = model.model.embed_tokens(prefix)
    embeds = embeds.detach().requires_grad_(True)

    # Manual forward through layers
    h = embeds
    pos = torch.arange(n_tokens, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(h, pos)
    pos_emb = (cos, sin)

    # Track per-layer output norms
    layer_outputs = []

    for i in range(N_LAYERS):
        layer_out = model.model.layers[i](h, position_embeddings=pos_emb)
        h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        layer_outputs.append(h[:, -1, :].detach().norm().item())

    h = model.model.norm(h)
    logits = model.lm_head(h)

    # Get the predicted next token's logit
    next_logit = logits[0, -1].max()

    # Backward
    next_logit.backward()

    # Measure: gradient on embeddings tells us what input mattered
    if embeds.grad is not None:
        grad = embeds.grad[0]  # [n_tokens, H]
        # Per-position gradient norm — which input tokens mattered?
        per_pos_grad = grad.float().norm(dim=-1).cpu().numpy()  # [n_tokens]
        # Total gradient magnitude
        total_grad = grad.float().norm().item()
        # Gradient on last token vs earlier tokens
        last_grad = per_pos_grad[-1]
        context_grad = per_pos_grad[:-1].sum() if n_tokens > 1 else 0
    else:
        per_pos_grad = np.zeros(n_tokens)
        total_grad = 0
        last_grad = 0
        context_grad = 0

    # Token and prediction
    pred_tok = logits[0, -1].argmax().item()
    pred_str = tokenizer.decode([pred_tok])
    tok_str = tokenizer.decode(prefix[0, -1:])

    per_step.append({
        "n_tokens": n_tokens,
        "total_grad": total_grad,
        "last_grad": float(last_grad),
        "context_grad": float(context_grad),
        "per_pos_grad": per_pos_grad.tolist(),
        "layer_norms": layer_outputs,
        "pred": pred_str,
    })

    kv_ratio = context_grad / (last_grad + 1e-10)
    print(f"  n={n_tokens:>2} '{tok_str:>12}' → '{pred_str:>12}' | "
          f"last_grad={last_grad:.2f} ctx_grad={context_grad:.2f} ratio={kv_ratio:.2f}")

# Analysis
print(f"\n{'='*60}")
print("KV CONTRIBUTION ANALYSIS")
print(f"{'='*60}")

# How does context gradient grow with KV depth?
ctx_grads = [s["context_grad"] for s in per_step]
last_grads = [s["last_grad"] for s in per_step]

print(f"\n  KV depth vs context gradient (how much context matters):")
for s in per_step:
    n = s["n_tokens"]
    bar_ctx = "█" * int(s["context_grad"] * 5)
    bar_last = "▓" * int(s["last_grad"] * 5)
    print(f"    n={n:>2}: ctx={s['context_grad']:>6.2f} {bar_ctx}")
    print(f"          last={s['last_grad']:>6.2f} {bar_last}")

# Delta: how much does each NEW KV entry add to context gradient?
print(f"\n  Per-token KV contribution (delta in context gradient):")
for i in range(1, len(per_step)):
    delta = per_step[i]["context_grad"] - per_step[i-1]["context_grad"]
    print(f"    Token {i+1}: delta_ctx = {delta:+.3f}")

# Per-position gradient pattern: does the model attend to specific positions?
print(f"\n  Gradient attention pattern at n=10:")
if len(per_step) >= 10:
    grads = per_step[9]["per_pos_grad"]
    total = sum(grads)
    for p, g in enumerate(grads):
        tok = tokenizer.decode(ids[0, p:p+1])
        bar = "█" * int(g / total * 50)
        print(f"    pos={p:>2} '{tok:>12}': {g/total*100:>5.1f}% {bar}")

print(f"\nDone.", flush=True)
