"""Pipeline Step 1.5 — Segment anneal: throat + passage deeper, mouths frozen.

Loads the step1_3 checkpoint (K~295, V~642) and continues annealing:
  - Mouth (L0-6, L26-39): FROZEN at current rank
  - Throat (L7-21): anneal K together as segment, 5% steps → target rank 7
  - Passage (L22-25): anneal K at gentler rate, 10% steps → target rank 64

Each segment anneals all its layers simultaneously (LASER effect).
Plain SVD + 500 FT steps between each. (ASVD too slow for 1% steps.)
Thermostat per segment: 2.0x baseline (looser — we already paid 1.5x).

After K segments, do coherence test with 10 prompts.
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


def generate_samples(model, tokenizer, prompts, n=60):
    results = []
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=n, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        results.append({"prompt": p, "output": text})
    return results



def svd_compress(proj, rank):
    """Plain SVD truncation — fast, FT does the heavy lifting."""
    W = proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(min(rank, len(S)), 1)
    proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)
    torch.cuda.empty_cache()


def finetune(model, train_tokens, seq_len, device, steps=500, lr=5e-5):
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "k_proj" in name or "v_proj" in name or "norm" in name.lower():
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


CHECKPOINT = "checkpoints/pipeline/step1_3_kv_final"
SEQ_LEN = 256
THERMOSTAT = 2.0  # looser — we already paid for 1.5x

THROAT = list(range(7, 22))    # L7-L21
PASSAGE = list(range(22, 26))  # L22-L25
MOUTH = list(range(0, 7)) + list(range(26, 40))

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "In quantum mechanics, the uncertainty principle states that",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "The mitochondria is often called the powerhouse of the cell because",
    "To solve a system of linear equations, one common method is",
    "Climate change is primarily driven by",
    "The human genome contains approximately",
    "In economics, supply and demand interact to",
    "Shakespeare wrote Hamlet as a meditation on",
]

print("=" * 60)
print("PIPELINE STEP 1.5 — SEGMENT ANNEAL")
print(f"  Throat (L7-21): anneal K together, 5% steps → rank 7")
print(f"  Passage (L22-25): anneal K, 10% steps → rank 64")
print(f"  Mouth: FROZEN")
print(f"  Thermostat: {THERMOSTAT}x original baseline")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

# Load tokens
token_path = Path("checkpoints/pipeline/step0_baseline/tokens.pt")
saved = torch.load(str(token_path), weights_only=True)
train_tokens = saved["train"]
val_tokens = saved["val"]

# Load original baseline for thermostat reference
with open("results/pipeline_step1_baseline.json") as f:
    orig_baseline = json.load(f)["baseline_ppl"]
thermostat_limit = orig_baseline * THERMOSTAT
print(f"  Original baseline: {orig_baseline:.1f}")
print(f"  Thermostat limit: {thermostat_limit:.1f}", flush=True)

print(f"\nLoading checkpoint...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
checkpoint_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Checkpoint ppl: {checkpoint_ppl:.1f}", flush=True)

# Load previous K ranks from step 1.3
with open("results/pipeline_step1_3_kv_anneal.json") as f:
    prev = json.load(f)
k_ranks = {int(k): v for k, v in prev["final_k_ranks"].items()}
print(f"  Current K ranks: throat~{k_ranks[10]} mouth~{k_ranks[0]}", flush=True)

history = []
step_num = 0

# ═══════════════════════════════════════════════════════
# THROAT ANNEAL — all throat layers together, 5% steps
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("THROAT SEGMENT ANNEAL (L7-L21)")
print(f"{'='*60}")

throat_target = 7

while True:
    # Check if throat is at target
    throat_min = min(k_ranks[i] for i in THROAT)
    if throat_min <= throat_target:
        print(f"\n  Throat at target rank {throat_target}. Done.", flush=True)
        break

    step_num += 1
    print(f"\n{'─'*50}")
    print(f"  THROAT STEP {step_num}", flush=True)

    # Compress all throat layers by 1% (plain SVD — fast, FT recovers)
    for i in THROAT:
        new_rank = max(int(k_ranks[i] * 0.99), throat_target)
        if new_rank >= k_ranks[i]:
            new_rank = k_ranks[i] - 1
        if new_rank < throat_target:
            new_rank = throat_target
        if new_rank < k_ranks[i]:
            svd_compress(model.model.layers[i].self_attn.k_proj, new_rank)
            k_ranks[i] = new_rank

    torch.cuda.empty_cache()

    throat_avg = sum(k_ranks[i] for i in THROAT) / len(THROAT)
    print(f"  Throat avg rank: {throat_avg:.0f}", flush=True)

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  pre-FT: ppl={pre_ppl:.1f}", flush=True)

    finetune(model, train_tokens, SEQ_LEN, device, steps=500)

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  post-FT: ppl={post_ppl:.1f} (Δ from orig={post_ppl-orig_baseline:+.1f})", flush=True)

    history.append({
        "step": step_num, "segment": "throat",
        "throat_avg_rank": round(throat_avg, 1),
        "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
    })

    if post_ppl > thermostat_limit:
        print(f"\n  ⚠ THERMOSTAT: {post_ppl:.1f} > {thermostat_limit:.1f}")
        print(f"  Throat frozen at avg rank {throat_avg:.0f}", flush=True)
        break

    # Checkpoint every 10 steps
    if step_num % 10 == 0:
        save_path = Path(f"checkpoints/pipeline/step1_5_segment_s{step_num}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# ═══════════════════════════════════════════════════════
# PASSAGE ANNEAL — passage layers together, 10% steps
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PASSAGE SEGMENT ANNEAL (L22-L25)")
print(f"{'='*60}")

passage_target = 64

while True:
    passage_min = min(k_ranks[i] for i in PASSAGE)
    if passage_min <= passage_target:
        print(f"\n  Passage at target rank {passage_target}. Done.", flush=True)
        break

    step_num += 1
    print(f"\n{'─'*50}")
    print(f"  PASSAGE STEP {step_num}", flush=True)

    for i in PASSAGE:
        new_rank = max(int(k_ranks[i] * 0.95), passage_target)
        if new_rank >= k_ranks[i]:
            new_rank = k_ranks[i] - 1
        if new_rank < passage_target:
            new_rank = passage_target
        if new_rank < k_ranks[i]:
            svd_compress(model.model.layers[i].self_attn.k_proj, new_rank)
            k_ranks[i] = new_rank

    torch.cuda.empty_cache()

    passage_avg = sum(k_ranks[i] for i in PASSAGE) / len(PASSAGE)
    print(f"  Passage avg rank: {passage_avg:.0f}", flush=True)

    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    finetune(model, train_tokens, SEQ_LEN, device, steps=500)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    print(f"  post-FT: ppl={post_ppl:.1f} (Δ={post_ppl-orig_baseline:+.1f})", flush=True)

    history.append({
        "step": step_num, "segment": "passage",
        "passage_avg_rank": round(passage_avg, 1),
        "post_ppl": round(post_ppl, 2),
    })

    if post_ppl > thermostat_limit:
        print(f"\n  ⚠ THERMOSTAT: Passage frozen at avg rank {passage_avg:.0f}", flush=True)
        break

# ═══════════════════════════════════════════════════════
# Coherence test
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COHERENCE TEST — 10 prompts")
print(f"{'='*60}")

samples = generate_samples(model, tokenizer, PROMPTS)
for s in samples:
    print(f"  [{s['prompt'][:40]}...] → {s['output'][:60]}", flush=True)

# ═══════════════════════════════════════════════════════
# Save final
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STEP 1.5 COMPLETE")
print(f"{'='*60}")
print(f"  Original baseline: {orig_baseline:.1f}")
final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"  Final ppl: {final_ppl:.1f}")
print(f"  Throat ranks: {[k_ranks[i] for i in THROAT]}")
print(f"  Passage ranks: {[k_ranks[i] for i in PASSAGE]}")
print(f"  Mouth ranks: {[k_ranks[i] for i in MOUTH[:3]]}...")

save_path = Path("checkpoints/pipeline/step1_5_segment_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"  Saved: {save_path}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_step1_5_segment_anneal.json", "w") as f:
    json.dump({
        "orig_baseline": orig_baseline,
        "checkpoint_ppl": checkpoint_ppl,
        "final_ppl": final_ppl,
        "final_k_ranks": {str(i): k_ranks[i] for i in range(L)},
        "history": history,
        "coherence_samples": samples,
    }, f, indent=2)
print(f"  Saved results/pipeline_step1_5_segment_anneal.json", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
