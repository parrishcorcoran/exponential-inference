"""Stage 117 — Total annealing: squeeze everything, everywhere, all at once.

Every compressible axis in every layer gets reduced by 5% per round.
Fine-tune norms after each round. The model fights to keep what matters.

Per-layer axes:
  - KV rank (SVD on k_proj + v_proj — small matrices, fast)
  - MLP width (zero out bottom rows of gate/up, cols of down)
  - Weight precision (per-channel quantization, decreasing bits via noise)

Global axes:
  - Embed precision (quantization)

Each round:
  1. SVD truncate KV to 95% of current rank
  2. Zero out 5% more MLP intermediate
  3. Add quantization noise (increasing each round)
  4. Fine-tune norms (150 steps)
  5. Eval + record per-layer state

The final profile shows what the model chose to keep.
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=15):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_sample(model, tokenizer, prompt, n=30):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def svd_truncate_inplace(weight, rank):
    """SVD truncation on GPU. For KV projections (small: 1280×5120)."""
    W = weight.data.float()
    k = max(min(rank, min(W.shape)), 1)
    if k >= min(W.shape):
        return min(W.shape)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(weight.dtype)
    return k


def finetune_norms(model, train_tokens, seq_len, device, steps=150, lr=5e-5):
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
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
SQUEEZE = 0.95  # 5% per round
MIN_KV_RANK = 16
MIN_MLP_PCT = 50  # don't go below 50% MLP
MAX_ROUNDS = 30

print("=" * 60)
print("STAGE 117 — TOTAL ANNEALING (fast version)")
print(f"  Squeeze: {SQUEEZE} per round | KV via SVD | MLP via zeroing | Weight via quant")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 500, split="train")
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

print("\nLoading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size

teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
teacher_text = generate_sample(model, tokenizer, PROMPT)
print(f"  Teacher: ppl={teacher_ppl:.1f}  L={L}  d={d}  [{teacher_text[:60]}]", flush=True)

# State trackers
kv_ranks = {}
mlp_keep_pct = {}
for i in range(L):
    layer = model.model.layers[i]
    kv_ranks[i] = min(layer.self_attn.k_proj.weight.shape)  # 1280 for 14B (8 heads × 128 dim)
    mlp_keep_pct[i] = 100.0

# Weight quant: start at effective 16-bit, reduce effective bits each round
# We do this by applying per-channel quantization at decreasing precision
# Round 1: ~12 bit, Round 2: ~11 bit, ... Round N: ~4 bit
# Effective bits = 16 - round * 0.4
weight_effective_bits = 16.0

# Embed effective bits
embed_effective_bits = 16.0

history = []

for rnd in range(1, MAX_ROUNDS + 1):
    t0 = time.time()
    print(f"\n{'─'*50}")
    print(f"  ROUND {rnd}/{MAX_ROUNDS} — squeeze to {SQUEEZE**rnd*100:.1f}% of original", flush=True)

    # ── KV rank: SVD truncate each layer's k_proj + v_proj ──
    for i in range(L):
        new_rank = max(int(kv_ranks[i] * SQUEEZE), MIN_KV_RANK)
        if new_rank < kv_ranks[i]:
            layer = model.model.layers[i]
            for name in ("k_proj", "v_proj"):
                proj = getattr(layer.self_attn, name)
                svd_truncate_inplace(proj.weight, new_rank)
            kv_ranks[i] = new_rank

    # ── MLP width: zero out more rows ──
    for i in range(L):
        new_pct = max(mlp_keep_pct[i] * SQUEEZE, MIN_MLP_PCT)
        if new_pct < mlp_keep_pct[i]:
            layer = model.model.layers[i]
            full_gate = layer.mlp.gate_proj.weight.shape[0]
            old_keep = int(full_gate * mlp_keep_pct[i] / 100)
            new_keep = int(full_gate * new_pct / 100)
            if new_keep < old_keep:
                # Zero out the newly pruned rows
                layer.mlp.gate_proj.weight.data[new_keep:old_keep] = 0
                layer.mlp.up_proj.weight.data[new_keep:old_keep] = 0
                layer.mlp.down_proj.weight.data[:, new_keep:old_keep] = 0
            mlp_keep_pct[i] = new_pct

    # ── Weight quantization: reduce effective precision ──
    weight_effective_bits = max(weight_effective_bits - 0.4, 4.0)
    actual_bits = max(int(round(weight_effective_bits)), 4)
    half = 2 ** (actual_bits - 1)
    for i in range(L):
        layer = model.model.layers[i]
        for parent, names in [(layer.self_attn, ["q_proj", "o_proj"]),
                              (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
            for name in names:
                w = getattr(parent, name).weight
                W = w.data.float()
                scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
                w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)

    # ── Embed quantization ──
    embed_effective_bits = max(embed_effective_bits - 0.4, 4.0)
    actual_embed_bits = max(int(round(embed_effective_bits)), 4)
    half_e = 2 ** (actual_embed_bits - 1)
    w = model.get_input_embeddings().weight
    s = w.float().abs().max().item() / max(half_e - 1, 1)
    if s > 1e-10:
        w.data = (w.float() / s).round().clamp(-half_e+1, half_e-1).mul(s).to(w.dtype)

    torch.cuda.empty_cache()

    # ── Eval pre-tune ──
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)

    # ── Fine-tune norms ──
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=150)

    # ── Eval post-tune ──
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    text = generate_sample(model, tokenizer, PROMPT)
    elapsed = time.time() - t0

    # Stats
    avg_kv = sum(kv_ranks.values()) / L
    avg_mlp = sum(mlp_keep_pct.values()) / L
    kv_at_floor = sum(1 for v in kv_ranks.values() if v <= MIN_KV_RANK)

    print(f"  pre={pre_ppl:.1f} → post={post_ppl:.1f}  (Δ from teacher: {post_ppl-teacher_ppl:+.1f})")
    print(f"  [{text[:60]}]")
    print(f"  kv_avg={avg_kv:.0f} mlp_avg={avg_mlp:.0f}% wt_bits~{actual_bits} emb_bits~{actual_embed_bits}")
    print(f"  kv_floor={kv_at_floor}/{L}  elapsed={elapsed:.0f}s", flush=True)

    # Sample per-layer
    sample = [0, L//4, L//2, 3*L//4, L-1]
    rank_snap = {}
    for s_i in sample:
        rank_snap[f"L{s_i}"] = {"kv": kv_ranks[s_i], "mlp_pct": round(mlp_keep_pct[s_i], 1)}

    history.append({
        "round": rnd,
        "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "text": text[:80],
        "avg_kv_rank": avg_kv, "avg_mlp_pct": avg_mlp,
        "weight_bits": actual_bits, "embed_bits": actual_embed_bits,
        "kv_at_floor": kv_at_floor,
        "rank_snapshot": rank_snap,
        "elapsed": elapsed,
    })

    if post_ppl > teacher_ppl * 3:
        print(f"\n  ⚠ STOPPED: {post_ppl:.1f} > 3× teacher ({teacher_ppl*3:.1f})")
        break

# ═══════════════════════════════════════════════════════
# Final shape analysis
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TOTAL ANNEAL — FINAL SHAPE")
print(f"{'='*60}")
print(f"  Teacher: {teacher_ppl:.1f}")
if history:
    print(f"  Final:   {history[-1]['post_ppl']:.1f} after {len(history)} rounds")

print(f"\n  Per-layer final profile:")
print(f"  {'Layer':>5} {'KV rank':>8} {'MLP %':>7}")
for i in range(L):
    print(f"  L{i:>3}  {kv_ranks[i]:>8} {mlp_keep_pct[i]:>7.1f}")

# Bathtub check
third = L // 3
print(f"\n  Region averages:")
for name, s, e in [("Early", 0, third), ("Middle", third, 2*third), ("Late", 2*third, L)]:
    avg_kv = sum(kv_ranks[i] for i in range(s, e)) / (e - s)
    avg_mlp = sum(mlp_keep_pct[i] for i in range(s, e)) / (e - s)
    print(f"    {name:>6} (L{s}-{e-1}): kv={avg_kv:.0f} mlp={avg_mlp:.0f}%")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/stage117_total_anneal.json", "w") as f:
    json.dump({
        "teacher_ppl": teacher_ppl,
        "rounds": len(history),
        "history": history,
        "final_kv_ranks": {str(i): kv_ranks[i] for i in range(L)},
        "final_mlp_pct": {str(i): round(mlp_keep_pct[i], 1) for i in range(L)},
        "final_weight_bits": actual_bits,
        "final_embed_bits": actual_embed_bits,
    }, f, indent=2)
print(f"\nSaved results/stage117_total_anneal.json", flush=True)

# Save model if reasonable quality
if history and history[-1]["post_ppl"] < teacher_ppl * 2.5:
    save_path = Path("checkpoints/qwen_halo/total_annealed")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved model to {save_path}", flush=True)
