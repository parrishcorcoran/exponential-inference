"""Pipeline Step 1.1 — Load Q4 base + baseline eval.

Load Qwen3-14B-AWQ, eval on OpenWebText holdout, generate samples,
measure wormhole shape, save baseline checkpoint.
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
    """Load tokens from OpenWebText via streaming."""
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
        print(f"  [{p[:40]}...] → {text[:60]}", flush=True)
    return results


MODEL = "Qwen/Qwen3-14B-AWQ"
SEQ_LEN = 256

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
print("PIPELINE STEP 1.1 — Q4 BASELINE")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("\nLoading OpenWebText tokens...", flush=True)
# Training tokens (first 500K tokens)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 2000)
# Val tokens (skip first 500K, take next 100K)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 2000)
print(f"  Train: {len(train_tokens)} tokens, Val: {len(val_tokens)} tokens", flush=True)

print(f"\nLoading {MODEL}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True, attn_implementation="eager"
).to(device).eval()

L = model.config.num_hidden_layers
d = model.config.hidden_size
print(f"  L={L} d={d}", flush=True)
print(f"  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
print(f"  Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ── Baseline PPL ──
print(f"\n--- Baseline PPL (OpenWebText) ---", flush=True)
baseline_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=30)
print(f"  val_ppl = {baseline_ppl:.2f}", flush=True)

# ── Generation samples ──
print(f"\n--- Generation samples ---", flush=True)
samples = generate_samples(model, tokenizer, PROMPTS)

# ── Wormhole shape ──
print(f"\n--- Wormhole shape (r99 per layer) ---", flush=True)
inp = torch.tensor([val_tokens[:SEQ_LEN]], dtype=torch.long, device=device)
with torch.no_grad():
    out = model(inp, use_cache=False, output_hidden_states=True)

wormhole = []
for i, h in enumerate(out.hidden_states):
    H = h[0].float()
    H_c = H - H.mean(0, keepdim=True)
    _, S, _ = torch.linalg.svd(H_c, full_matrices=False)
    s2 = S ** 2
    cumvar = torch.cumsum(s2, 0) / s2.sum()
    r99 = (cumvar < 0.99).sum().item() + 1
    pr = (s2.sum() ** 2 / (s2 ** 2).sum()).item()
    wormhole.append({"layer": i, "r99": r99, "pr": round(pr, 1)})
    if i % 5 == 0 or i == L:
        region = "THROAT" if r99 <= 3 else "passage" if r99 <= 50 else "mouth"
        print(f"  L{i:>2}: r99={r99:>4} pr={pr:>6.1f}  {region}", flush=True)

del out; torch.cuda.empty_cache()

# ── Wall clock speed ──
print(f"\n--- Wall clock benchmark ---", flush=True)
ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): model.generate(ids, max_new_tokens=100, do_sample=False)
torch.cuda.synchronize()
tps = 100 / (time.time() - t0)
print(f"  Speed: {tps:.1f} tok/s", flush=True)

# ── Save results ──
results = {
    "model": MODEL,
    "baseline_ppl": baseline_ppl,
    "L": L, "d": d,
    "params_B": sum(p.numel() for p in model.parameters()) / 1e9,
    "memory_GB": torch.cuda.memory_allocated() / 1e9,
    "tok_per_sec": tps,
    "wormhole_shape": wormhole,
    "generation_samples": samples,
    "train_tokens": len(train_tokens),
    "val_tokens": len(val_tokens),
    "eval_data": "OpenWebText (Skylion007/openwebtext)",
}

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_step1_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/pipeline_step1_baseline.json", flush=True)

# ── Save baseline checkpoint ──
save_path = Path("checkpoints/pipeline/step0_baseline")
save_path.mkdir(parents=True, exist_ok=True)
# Don't re-save the AWQ model — just symlink or note the path
with open(save_path / "source.txt", "w") as f:
    f.write(f"Source model: {MODEL}\nBaseline PPL: {baseline_ppl}\n")
print(f"  Baseline noted at {save_path}", flush=True)

# ── Save training/val tokens for consistency ──
torch.save({"train": train_tokens, "val": val_tokens}, str(save_path / "tokens.pt"))
print(f"  Saved tokens for consistent eval across pipeline steps", flush=True)

print(f"\n{'='*60}")
print(f"STEP 1.1 COMPLETE")
print(f"  Model: {MODEL}")
print(f"  PPL: {baseline_ppl:.2f}")
print(f"  Speed: {tps:.1f} tok/s")
print(f"  Wormhole throat: L{[w['layer'] for w in wormhole if w['r99'] <= 3]}")
print(f"{'='*60}", flush=True)
