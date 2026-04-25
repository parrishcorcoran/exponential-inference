"""Pipeline Step 1.5b — Per-layer cavity anneal.

Each layer gets its own K rank, its own thermostat, its own pace.
Cheap layers (cavities) go deep toward rank 5-7.
Expensive layers (walls) stop early.
The wormhole shape emerges from the data.

Method: cycle through all 40 layers. For each layer:
  1. Try reducing K rank by 5%
  2. Eval PPL (fast, 5 batches)
  3. If ppl increased > layer_thermostat: back off, freeze this layer
  4. If OK: keep reduction, move to next layer
  5. Fine-tune norms globally every full cycle (all 40 layers)

This is ONE-LAYER-AT-A-TIME compression with global FT between cycles.
Cheap layers naturally go deeper. Expensive layers freeze early.
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=10):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(val_tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))


def generate_samples(model, tokenizer, prompts, n=60):
    results = []
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": p, "output": text})
    return results


def svd_compress_k(layer, rank):
    proj = layer.self_attn.k_proj
    W = proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(min(rank, len(S)), 1)
    proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def finetune_norms(model, train_tokens, seq_len, device, steps=300, lr=5e-5):
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


CHECKPOINT = "checkpoints/pipeline/step1_5_segment_final"
SEQ_LEN = 256
MIN_RANK = 5
PER_LAYER_BUDGET = 3.0  # max ppl increase per layer compression
MAX_CYCLES = 20

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "In quantum mechanics, the uncertainty principle states that",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "The mitochondria is often called the powerhouse of the cell because",
]

print("=" * 60)
print("PIPELINE STEP 1.5b — PER-LAYER CAVITY ANNEAL")
print(f"  Per-layer budget: {PER_LAYER_BUDGET} ppl increase max")
print(f"  Min rank: {MIN_RANK}")
print(f"  Max cycles: {MAX_CYCLES}")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

token_path = Path("checkpoints/pipeline/step0_baseline/tokens.pt")
saved = torch.load(str(token_path), weights_only=True)
train_tokens = saved["train"]
val_tokens = saved["val"]

print(f"\nLoading checkpoint...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
base_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Starting ppl: {base_ppl:.1f}", flush=True)

# Load current K ranks from step 1.5
with open("results/pipeline_step1_5_segment_anneal.json") as f:
    prev = json.load(f)
k_ranks = {int(k): v for k, v in prev["final_k_ranks"].items()}

# Track which layers are frozen
frozen = {i: False for i in range(L)}

history = []

for cycle in range(1, MAX_CYCLES + 1):
    print(f"\n{'='*50}")
    print(f"  CYCLE {cycle}/{MAX_CYCLES}")
    print(f"{'='*50}", flush=True)

    any_compressed = False
    cycle_results = []

    for i in range(L):
        if frozen[i]:
            continue
        if k_ranks[i] <= MIN_RANK:
            frozen[i] = True
            continue

        # Try 5% reduction
        old_rank = k_ranks[i]
        new_rank = max(int(old_rank * 0.95), MIN_RANK)
        if new_rank >= old_rank:
            new_rank = old_rank - 1
        if new_rank < MIN_RANK:
            new_rank = MIN_RANK

        # Save original weight
        k_orig = model.model.layers[i].self_attn.k_proj.weight.data.clone()

        # Compress
        svd_compress_k(model.model.layers[i], new_rank)

        # Quick eval
        new_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=5)
        damage = new_ppl - base_ppl

        if damage > PER_LAYER_BUDGET:
            # Too expensive — restore and freeze
            model.model.layers[i].self_attn.k_proj.weight.data = k_orig
            frozen[i] = True
            print(f"  L{i:>2}: rank {old_rank}→{new_rank} REJECTED (damage={damage:+.1f}), FROZEN at {old_rank}", flush=True)
            cycle_results.append({"layer": i, "action": "frozen", "rank": old_rank, "damage": round(damage, 2)})
        else:
            # Accept
            k_ranks[i] = new_rank
            any_compressed = True
            if i % 5 == 0 or damage > 1.0:
                print(f"  L{i:>2}: rank {old_rank}→{new_rank} accepted (damage={damage:+.1f})", flush=True)
            cycle_results.append({"layer": i, "action": "accepted", "rank": new_rank, "damage": round(damage, 2)})

        del k_orig
        torch.cuda.empty_cache()

    if not any_compressed:
        print(f"\n  All layers frozen. Done.", flush=True)
        break

    # Global FT after each cycle
    print(f"\n  Fine-tuning norms (300 steps)...", end="", flush=True)
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=300)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    base_ppl = post_ppl  # update baseline for next cycle
    print(f" ppl={post_ppl:.1f}", flush=True)

    # Count frozen vs active
    n_frozen = sum(1 for v in frozen.values() if v)
    n_active = L - n_frozen

    # Rank stats
    throat_ranks = [k_ranks[i] for i in range(7, 22)]
    mouth_ranks = [k_ranks[i] for i in list(range(0, 7)) + list(range(26, 40))]
    print(f"  Active: {n_active}/{L}  Frozen: {n_frozen}/{L}")
    print(f"  Throat min/avg: {min(throat_ranks)}/{sum(throat_ranks)/len(throat_ranks):.0f}")
    print(f"  Mouth min/avg: {min(mouth_ranks)}/{sum(mouth_ranks)/len(mouth_ranks):.0f}", flush=True)

    history.append({
        "cycle": cycle, "post_ppl": round(post_ppl, 2),
        "n_frozen": n_frozen, "n_active": n_active,
        "throat_min": min(throat_ranks), "throat_avg": round(sum(throat_ranks)/len(throat_ranks), 1),
        "mouth_min": min(mouth_ranks), "mouth_avg": round(sum(mouth_ranks)/len(mouth_ranks), 1),
        "per_layer": cycle_results,
    })

    # Checkpoint every 5 cycles
    if cycle % 5 == 0:
        save_path = Path(f"checkpoints/pipeline/step1_5b_cavity_c{cycle}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# ═══════════════════════════════════════════════════════
# Coherence test
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COHERENCE TEST")
print(f"{'='*60}")

samples = generate_samples(model, tokenizer, PROMPTS)
for s in samples:
    print(f"  [{s['prompt'][:40]}...] → {s['output'][:70]}", flush=True)

# ═══════════════════════════════════════════════════════
# Final rank profile — THE WORMHOLE SHAPE
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL PER-LAYER K RANK PROFILE — THE WORMHOLE SHAPE")
print(f"{'='*60}")

for i in range(L):
    bar = "█" * max(int(k_ranks[i] / 10), 1)
    status = "FROZEN" if frozen[i] else "active"
    print(f"  L{i:>2}: rank {k_ranks[i]:>4}  {bar}  [{status}]")

# Save
final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"\n  Final ppl: {final_ppl:.1f}")

save_path = Path("checkpoints/pipeline/step1_5b_cavity_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_step1_5b_cavity.json", "w") as f:
    json.dump({
        "final_ppl": final_ppl,
        "final_k_ranks": {str(i): k_ranks[i] for i in range(L)},
        "frozen": {str(i): frozen[i] for i in range(L)},
        "history": history,
        "coherence_samples": samples,
    }, f, indent=2)
print(f"  Saved results + model", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
