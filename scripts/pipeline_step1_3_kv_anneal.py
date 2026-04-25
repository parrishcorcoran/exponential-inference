"""Pipeline Step 1.3 — KV rank anneal with ASVD init + FT.

The biggest lever first. ASVD-initialized, per-layer K and V separate,
wormhole-shaped targets, super slow 5% steps, thermostat controlled.

K schedule (per-layer, wormhole-shaped):
  Mouth (L0-6, L26-39): conservative, anneal slowly
  Throat (L7-21): aggressive, target rank 5-7
  Passage (L22-25): moderate

V schedule (uniform): modest reduction across all layers

Each step:
  1. Cache activations (one forward pass)
  2. ASVD compress K projections per-layer to target rank
  3. ASVD compress V projections uniformly to target rank
  4. Fine-tune norms + KV, 500 steps on OpenWebText
  5. Eval PPL on OpenWebText holdout
  6. Thermostat: accept if < 1.5x baseline, back off if not
  7. Save checkpoint periodically

Uses saved tokens from Step 1.1 for consistent eval.
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=30):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_sample(model, tokenizer, prompt, n=60):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def asvd_compress(proj, X, rank):
    """ASVD: activation-weighted SVD compression of a linear projection."""
    W = proj.weight.data.float()  # [d_out, d_in]

    # Covariance of input activations
    XtX = X.T @ X  # [d_in, d_in]

    # Eigendecompose for whitening
    eigvals, eigvecs = torch.linalg.eigh(XtX)
    eigvals = eigvals.clamp(min=1e-6)
    sqrt_eig = eigvals.sqrt()

    # Whiten: W_white = W @ eigvecs @ diag(sqrt_eig)
    W_white = W @ eigvecs @ torch.diag(sqrt_eig)

    # SVD of whitened weights
    U, S, Vt = torch.linalg.svd(W_white, full_matrices=False)
    k = min(rank, len(S))

    # Truncate and un-whiten
    W_trunc = (U[:, :k] * S[:k]) @ Vt[:k]
    inv_sqrt = 1.0 / sqrt_eig
    W_restored = W_trunc @ torch.diag(inv_sqrt) @ eigvecs.T

    proj.weight.data = W_restored.to(proj.weight.dtype)
    del W, XtX, eigvals, eigvecs, W_white, U, S, Vt, W_trunc
    torch.cuda.empty_cache()


def finetune(model, train_tokens, seq_len, device, steps=500, lr=5e-5):
    """Fine-tune norms + KV projections."""
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
            p.requires_grad = True
            trainable.append(p)
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train()
    step = 0
    for inp, tgt in iter_batches(train_tokens, seq_len, 1, device):
        if step >= steps:
            break
        opt.zero_grad()
        loss = F.cross_entropy(
            model(inp, use_cache=False).logits.reshape(-1, model.config.vocab_size).float(),
            tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
    del opt
    for p in model.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()


def cache_layer_inputs(model, tokens, seq_len, device):
    """Run one forward pass and cache per-layer input activations."""
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


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 256
PROMPT = "The theory of general relativity describes gravity as"
THERMOSTAT = 1.5  # accept if ppl < baseline * thermostat

# Wormhole regions from Step 1.1
THROAT = list(range(7, 22))   # L7-L21 (r99=1)
PASSAGE = list(range(22, 26)) # L22-L25
MOUTH = list(range(0, 7)) + list(range(26, 40))  # L0-6, L26-39

print("=" * 60)
print("PIPELINE STEP 1.3 — KV RANK ANNEAL (ASVD + FT)")
print(f"  Throat: L{THROAT[0]}-L{THROAT[-1]}")
print(f"  Thermostat: {THERMOSTAT}x baseline")
print(f"  FT: 500 steps per rank step, OpenWebText")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Load saved tokens from Step 1.1
token_path = Path("checkpoints/pipeline/step0_baseline/tokens.pt")
saved = torch.load(str(token_path), weights_only=True)
train_tokens = saved["train"]
val_tokens = saved["val"]
print(f"  Loaded {len(train_tokens)} train, {len(val_tokens)} val tokens", flush=True)

print(f"\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Baseline: ppl={baseline_ppl:.2f}", flush=True)

# Current ranks per layer (start at full)
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
k_ranks = {i: d_kv for i in range(L)}  # 1024 for all
v_rank = d_kv  # uniform V rank

history = []
step_num = 0

# ═══════════════════════════════════════════════════════
# K anneal: per-layer wormhole-shaped, 5% steps
# ═══════════════════════════════════════════════════════
# Target ranks by region
K_TARGETS = {}
for i in THROAT:
    K_TARGETS[i] = 7    # rank 7 target
for i in PASSAGE:
    K_TARGETS[i] = 64   # moderate
for i in MOUTH:
    K_TARGETS[i] = 256  # conservative

# Anneal K: reduce each layer by 5% toward its target each step
print(f"\n{'='*60}")
print("K ANNEAL — per-layer wormhole-shaped")
print(f"{'='*60}")

while True:
    step_num += 1
    any_moved = False

    # Check if any layer still above target
    for i in range(L):
        target = K_TARGETS[i]
        if k_ranks[i] > target:
            any_moved = True
            break

    if not any_moved:
        print(f"\n  All K ranks at target. Done.", flush=True)
        break

    print(f"\n{'─'*50}")
    print(f"  K STEP {step_num}", flush=True)

    # Cache activations
    print(f"  Caching activations...", end="", flush=True)
    layer_inputs = cache_layer_inputs(model, val_tokens, SEQ_LEN, device)
    print(f" done", flush=True)

    # Reduce each layer by 5% toward target
    for i in range(L):
        target = K_TARGETS[i]
        if k_ranks[i] <= target:
            continue
        new_rank = max(int(k_ranks[i] * 0.95), target)
        if new_rank >= k_ranks[i]:
            new_rank = k_ranks[i] - 1
        if new_rank < 1:
            new_rank = 1

        X = layer_inputs[i][0]  # [seq, d_model]
        asvd_compress(model.model.layers[i].self_attn.k_proj, X, new_rank)
        k_ranks[i] = new_rank

    # Free cached activations
    del layer_inputs
    torch.cuda.empty_cache()

    # Show rank snapshot
    throat_avg = sum(k_ranks[i] for i in THROAT) / len(THROAT)
    mouth_avg = sum(k_ranks[i] for i in MOUTH) / len(MOUTH)
    print(f"  K ranks: throat_avg={throat_avg:.0f} mouth_avg={mouth_avg:.0f}", flush=True)

    # Eval pre-FT
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-FT: ppl={pre_ppl:.1f}", flush=True)

    # Fine-tune
    print(f"  Fine-tuning 500 steps...", end="", flush=True)
    finetune(model, train_tokens, SEQ_LEN, device, steps=500)
    print(f" done", flush=True)

    # Eval post-FT
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    delta = post_ppl - baseline_ppl

    print(f"  post-FT: ppl={post_ppl:.1f} (Δ={delta:+.1f})")
    print(f"  [{text[:70]}]", flush=True)

    history.append({
        "step": step_num, "type": "K_anneal",
        "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
        "delta": round(delta, 2),
        "throat_avg_rank": round(throat_avg, 1),
        "mouth_avg_rank": round(mouth_avg, 1),
        "text": text[:100],
    })

    # Thermostat check
    if post_ppl > baseline_ppl * THERMOSTAT:
        print(f"\n  ⚠ THERMOSTAT: ppl {post_ppl:.1f} > {baseline_ppl * THERMOSTAT:.1f} ({THERMOSTAT}x baseline)")
        print(f"  Backing off — freezing K ranks at current levels", flush=True)
        break

    # Save checkpoint every 5 steps
    if step_num % 5 == 0:
        save_path = Path(f"checkpoints/pipeline/step1_3_kv_rank_s{step_num}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved checkpoint: {save_path}", flush=True)

# ═══════════════════════════════════════════════════════
# V anneal: uniform, 5% steps
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("V ANNEAL — uniform reduction")
print(f"{'='*60}")

V_TARGET = 192  # from stage 142: V 192 = sweet spot

while v_rank > V_TARGET:
    step_num += 1
    new_v = max(int(v_rank * 0.95), V_TARGET)
    if new_v >= v_rank:
        new_v = v_rank - 1

    print(f"\n{'─'*50}")
    print(f"  V STEP {step_num}: V {v_rank} → {new_v}", flush=True)

    # Cache activations
    layer_inputs = cache_layer_inputs(model, val_tokens, SEQ_LEN, device)

    for i in range(L):
        X = layer_inputs[i][0]
        asvd_compress(model.model.layers[i].self_attn.v_proj, X, new_v)

    del layer_inputs
    torch.cuda.empty_cache()
    v_rank = new_v

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-FT: ppl={pre_ppl:.1f}", flush=True)

    finetune(model, train_tokens, SEQ_LEN, device, steps=500)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    delta = post_ppl - baseline_ppl

    print(f"  post-FT: ppl={post_ppl:.1f} (Δ={delta:+.1f}) V_rank={v_rank}")
    print(f"  [{text[:70]}]", flush=True)

    history.append({
        "step": step_num, "type": "V_anneal",
        "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
        "delta": round(delta, 2), "v_rank": v_rank,
        "text": text[:100],
    })

    if post_ppl > baseline_ppl * THERMOSTAT:
        print(f"\n  ⚠ THERMOSTAT: V anneal stopped at rank {v_rank}", flush=True)
        break


# ═══════════════════════════════════════════════════════
# Save final checkpoint
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1.3 COMPLETE — KV RANK ANNEAL")
print(f"{'='*60}")
print(f"  Baseline: {baseline_ppl:.2f}")
if history:
    print(f"  Final: {history[-1]['post_ppl']:.2f}")
print(f"  K ranks: throat={[k_ranks[i] for i in THROAT[:3]]}... mouth={[k_ranks[i] for i in MOUTH[:3]]}...")
print(f"  V rank: {v_rank}")
print(f"  Steps: {step_num}")

save_path = Path("checkpoints/pipeline/step1_3_kv_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  Saved final: {save_path}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_step1_3_kv_anneal.json", "w") as f:
    json.dump({
        "baseline_ppl": baseline_ppl,
        "final_k_ranks": {str(i): k_ranks[i] for i in range(L)},
        "final_v_rank": v_rank,
        "thermostat": THERMOSTAT,
        "history": history,
    }, f, indent=2)
print(f"  Saved results/pipeline_step1_3_kv_anneal.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
