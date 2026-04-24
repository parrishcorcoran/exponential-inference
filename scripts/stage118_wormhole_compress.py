"""Stage 118 — Wormhole-shaped compression on 14B.

Use the measured wormhole shape (stage 117) as the compression schedule.
Throat layers (L7-14, r99=1): maximum compression.
Passage layers (L15-27, r99=3-72): moderate compression.
Mouth layers (L0-6, L28-40, r99=100-211): minimal compression.

Per-layer compression intensity based on r99:
  - KV rank: proportional to r99 (throat → rank 16, mouth → rank 512+)
  - Weight quant: throat → Q4, passage → Q5, mouth → Q6
  - MLP width: throat → 70%, passage → 85%, mouth → 100%

Progressive anneal: apply in 5 steps with norm fine-tune between each.
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


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=20):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_sample(model, tokenizer, prompt, n=40):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def compress_kv_layer(layer, rank):
    for name in ("k_proj", "v_proj"):
        proj = getattr(layer.self_attn, name)
        W = proj.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = max(min(rank, len(S)), 1)
        proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)


def quantize_layer(layer, bits):
    half = 2 ** (bits - 1)
    for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                          (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
        for name in names:
            w = getattr(parent, name).weight
            W = w.data.float()
            scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
            w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)


def prune_mlp(layer, keep_pct):
    full = layer.mlp.gate_proj.weight.shape[0]
    keep = int(full * keep_pct / 100)
    layer.mlp.gate_proj.weight.data[keep:] = 0
    layer.mlp.up_proj.weight.data[keep:] = 0
    layer.mlp.down_proj.weight.data[:, keep:] = 0


def finetune_kv_norms(model, train_tokens, seq_len, device, steps=200, lr=5e-5):
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


MODEL = "Qwen/Qwen3-14B"
SEQ_LEN = 128
PROMPT = "The theory of general relativity describes gravity as"

# Wormhole shape from stage 117
# r99 per layer (L0-L40, but we use L0-L39 for the 40 transformer layers)
WORMHOLE_R99 = {
    0: 116, 1: 152, 2: 162, 3: 168, 4: 174, 5: 174, 6: 179,  # mouth
    7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 2,     # throat
    15: 3, 16: 4, 17: 6, 18: 9, 19: 12, 20: 9, 21: 13,        # narrow passage
    22: 20, 23: 27, 24: 35, 25: 42, 26: 53, 27: 72,            # re-opening
    28: 95, 29: 115, 30: 134, 31: 149, 32: 161, 33: 171,       # exit mouth
    34: 180, 35: 187, 36: 194, 37: 201, 38: 206, 39: 211,
}

# Compression schedule per layer based on wormhole region
def get_layer_schedule(layer_idx, step, total_steps):
    """Returns (kv_rank, weight_bits, mlp_pct) for this layer at this step.
    Progressive: step 1 = gentle, step N = full compression."""
    r99 = WORMHOLE_R99.get(layer_idx, 200)
    progress = step / total_steps  # 0 → 1

    if r99 <= 2:  # THROAT
        kv_target = 32
        bits_target = 4
        mlp_target = 70
    elif r99 <= 20:  # NARROW PASSAGE
        kv_target = 128
        bits_target = 5
        mlp_target = 80
    elif r99 <= 80:  # RE-OPENING
        kv_target = 256
        bits_target = 5
        mlp_target = 90
    else:  # MOUTH
        kv_target = 512
        bits_target = 6
        mlp_target = 100

    # Progressive: interpolate from full to target
    full_kv = 1280  # Qwen3-14B KV dim = 8 heads × 128
    kv_rank = int(full_kv - (full_kv - kv_target) * progress)
    bits = max(int(16 - (16 - bits_target) * progress), bits_target)
    mlp_pct = 100 - (100 - mlp_target) * progress

    return kv_rank, bits, mlp_pct


print("=" * 60)
print("STAGE 118 — WORMHOLE-SHAPED COMPRESSION")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Teacher: ppl={teacher_ppl:.1f}  [{teacher_text[:60]}]", flush=True)

TOTAL_STEPS = 5
history = []

for step in range(1, TOTAL_STEPS + 1):
    t0 = time.time()
    print(f"\n{'─'*50}")
    print(f"  STEP {step}/{TOTAL_STEPS} — progress {step/TOTAL_STEPS:.0%}", flush=True)

    # Apply per-layer compression
    for i in range(L):
        kv_rank, bits, mlp_pct = get_layer_schedule(i, step, TOTAL_STEPS)
        layer = model.model.layers[i]

        # KV compression
        compress_kv_layer(layer, kv_rank)

        # Weight quantization (only apply at final precision, not re-round)
        if bits < 16:
            quantize_layer(layer, bits)

        # MLP pruning
        if mlp_pct < 100:
            prune_mlp(layer, mlp_pct)

    # Embed Q6 at final step
    if step == TOTAL_STEPS:
        w = model.get_input_embeddings().weight
        half = 32
        s = w.float().abs().max().item() / (half - 1)
        w.data = (w.float()/s).round().clamp(-half+1, half-1).mul(s).to(w.dtype)

    torch.cuda.empty_cache()

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-tune: ppl={pre_ppl:.1f}", flush=True)

    finetune_kv_norms(model, train_tokens, SEQ_LEN, device, steps=200)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    elapsed = time.time() - t0

    # Show per-region compression state
    throat_kv = [get_layer_schedule(i, step, TOTAL_STEPS)[0] for i in range(7, 15)]
    mouth_kv = [get_layer_schedule(i, step, TOTAL_STEPS)[0] for i in range(0, 7)]

    print(f"  post-tune: ppl={post_ppl:.1f} (Δ={post_ppl-teacher_ppl:+.1f})")
    print(f"  [{text[:60]}]")
    print(f"  throat KV={throat_kv[0]}, mouth KV={mouth_kv[0]}")
    print(f"  elapsed={elapsed:.0f}s", flush=True)

    history.append({
        "step": step, "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "text": text[:80], "elapsed": elapsed,
        "sample_schedule": {
            "L0_mouth": get_layer_schedule(0, step, TOTAL_STEPS),
            "L10_throat": get_layer_schedule(10, step, TOTAL_STEPS),
            "L20_passage": get_layer_schedule(20, step, TOTAL_STEPS),
            "L35_mouth": get_layer_schedule(35, step, TOTAL_STEPS),
        }
    })

    if post_ppl > teacher_ppl * 4:
        print(f"  ⚠ STOPPED: {post_ppl:.1f} > 4x teacher")
        break


# Final summary
print(f"\n{'='*60}")
print("WORMHOLE COMPRESSION COMPLETE")
print(f"{'='*60}")
print(f"  Teacher: {teacher_ppl:.1f}")
if history:
    final = history[-1]
    print(f"  Final:   {final['post_ppl']:.1f} (Δ={final['post_ppl']-teacher_ppl:+.1f})")
    print(f"  Ratio:   {final['post_ppl']/teacher_ppl:.2f}x")

print(f"\n  Final per-region compression:")
for name, layers in [("Mouth entry", range(0,7)), ("Throat", range(7,15)),
                     ("Passage", range(15,28)), ("Mouth exit", range(28,40))]:
    kv, bits, mlp = get_layer_schedule(list(layers)[len(layers)//2], TOTAL_STEPS, TOTAL_STEPS)
    print(f"    {name:>12}: KV={kv:>4}  Q{bits}  MLP={mlp:.0f}%")

print(f"\n  Text: {generate_sample(model, tokenizer, PROMPT, n=60)}")

Path("results").mkdir(exist_ok=True)
with open("results/stage118_wormhole_compress.json", "w") as f:
    json.dump({
        "teacher_ppl": teacher_ppl,
        "wormhole_r99": WORMHOLE_R99,
        "history": history,
    }, f, indent=2, default=str)
print(f"\nSaved results/stage118_wormhole_compress.json", flush=True)

# Save model
if history and history[-1]["post_ppl"] < teacher_ppl * 2.5:
    save_path = Path("checkpoints/qwen_halo/wormhole_compressed")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved model to {save_path}", flush=True)
