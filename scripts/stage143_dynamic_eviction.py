"""Stage 143 — Dynamic eviction on KV-256 base.

Use the model's own output entropy as the eviction signal.
Stage 140 showed: entropy 3.4 at pos 0-10, drops to 0.5-0.8 after.
High-entropy positions need full cache. Low-entropy can be evicted.

Method:
  1. Forward pass with full KV cache, record per-position entropy
  2. Evict cache entries where entropy was below threshold
  3. Continue generating — measure quality degradation

Test eviction rates: keep top 100%, 80%, 60%, 40%, 20%, 10% by entropy.
Also test: H2O-style (keep by attention score) for comparison.

Measure on the KV-256 saved model to see how eviction stacks with
rank compression.
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


MODEL_PATH = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
PROMPT = "The theory of general relativity describes gravity as"

print("=" * 60)
print("STAGE 143 — DYNAMIC EVICTION")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
val_tokens = load_tokens(tokenizer, max_tokens=SEQ_LEN * 30, split="validation")

print("Loading KV-256 base...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    trust_remote_code=True, attn_implementation="eager"
).to(device).eval()

L = model.config.num_hidden_layers

# ═══════════════════════════════════════════════════════
# Baseline: full cache PPL
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def eval_ppl_full(model, tokens, seq_len, n_batches=15):
    model.eval()
    total = 0; n = 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        n += 1
        if n >= n_batches: break
    return math.exp(total / max(n, 1))

baseline_ppl = eval_ppl_full(model, val_tokens, SEQ_LEN)
print(f"  Baseline (full cache): ppl={baseline_ppl:.1f}", flush=True)

# ═══════════════════════════════════════════════════════
# Measure per-position entropy on multiple sequences
# ═══════════════════════════════════════════════════════
print("\n--- Measuring per-position entropy ---", flush=True)

@torch.no_grad()
def eval_with_eviction(model, tokens, seq_len, keep_pct, strategy="entropy", n_batches=10):
    """
    Evaluate with cache eviction.
    Strategy: 'entropy' = keep highest-entropy positions,
              'h2o' = keep highest attention-score positions,
              'recent' = keep most recent positions (sliding window).
    """
    model.eval()
    total_loss = 0; n = 0

    for inp, tgt in iter_batches(tokens, seq_len, 1, device):
        seq_len_actual = inp.shape[1]
        keep_count = max(int(seq_len_actual * keep_pct / 100), 2)

        # Full forward to get logits, attentions, and per-position entropy
        out = model(inp, use_cache=False, output_attentions=True)
        logits = out.logits  # [1, seq, vocab]
        attentions = out.attentions  # tuple of [1, heads, seq, seq]

        # Per-position entropy from logits
        probs = F.softmax(logits[0].float(), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(-1)  # [seq]

        # Choose which positions to keep
        if strategy == "entropy":
            # Keep highest-entropy positions (most uncertain = most info)
            _, keep_idx = entropy.topk(keep_count)
            keep_idx = keep_idx.sort()[0]
        elif strategy == "h2o":
            # Keep positions with highest total attention received
            # Sum attention across all layers, all heads, all query positions
            attn_scores = torch.zeros(seq_len_actual, device=device)
            for layer_attn in attentions:
                # [1, heads, seq, seq] — sum over heads and query positions
                attn_scores += layer_attn[0].float().sum(dim=(0, 1))  # sum over heads and queries
            _, keep_idx = attn_scores.topk(keep_count)
            keep_idx = keep_idx.sort()[0]
        elif strategy == "recent":
            # Keep most recent positions
            keep_idx = torch.arange(seq_len_actual - keep_count, seq_len_actual, device=device)
        elif strategy == "random":
            perm = torch.randperm(seq_len_actual, device=device)
            keep_idx = perm[:keep_count].sort()[0]

        # Now evaluate: mask out evicted positions
        # We can't truly evict from a non-cached forward, so we simulate:
        # Zero out the input embeddings at evicted positions and re-forward
        # This is an approximation — true eviction would modify KV cache

        # Better approach: forward only the kept positions
        # But that changes position IDs... use attention mask instead
        attn_mask = torch.zeros(1, seq_len_actual, device=device, dtype=torch.bool)
        attn_mask[0, keep_idx] = True

        # Create a causal mask that also masks evicted positions
        # For each query position, it can only attend to kept positions that come before it
        causal_mask = torch.zeros(1, 1, seq_len_actual, seq_len_actual, device=device, dtype=torch.bfloat16)
        causal_mask.fill_(float('-inf'))
        for q in range(seq_len_actual):
            for k_pos in keep_idx:
                if k_pos <= q:
                    causal_mask[0, 0, q, k_pos] = 0.0

        # Re-forward with eviction mask
        out2 = model(inp, attention_mask=causal_mask, use_cache=False)
        loss = F.cross_entropy(out2.logits.reshape(-1, out2.logits.shape[-1]).float(), tgt.reshape(-1))
        total_loss += loss.item()
        n += 1
        if n >= n_batches: break

    return math.exp(total_loss / max(n, 1))


# ═══════════════════════════════════════════════════════
# Test eviction strategies at various keep rates
# ═══════════════════════════════════════════════════════
print("\n--- Testing eviction strategies ---", flush=True)

results = {"baseline_ppl": baseline_ppl, "seq_len": SEQ_LEN}
eviction_results = []

for strategy in ["entropy", "h2o", "recent", "random"]:
    print(f"\n  Strategy: {strategy}", flush=True)
    for keep_pct in [90, 80, 70, 60, 50, 40, 30, 20]:
        try:
            ppl = eval_with_eviction(model, val_tokens, SEQ_LEN, keep_pct, strategy, n_batches=8)
            delta = ppl - baseline_ppl
            print(f"    keep {keep_pct}%: ppl={ppl:.1f} (Δ={delta:+.1f})", flush=True)
            eviction_results.append({
                "strategy": strategy, "keep_pct": keep_pct,
                "ppl": round(ppl, 2), "delta": round(delta, 2),
                "cache_compression": round(100 / keep_pct, 1),
            })
        except Exception as e:
            print(f"    keep {keep_pct}%: ERROR — {e}", flush=True)
            eviction_results.append({
                "strategy": strategy, "keep_pct": keep_pct,
                "error": str(e),
            })

results["eviction_results"] = eviction_results

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("DYNAMIC EVICTION SUMMARY")
print(f"{'='*60}")
print(f"  Baseline: ppl={baseline_ppl:.1f}")

for strategy in ["entropy", "h2o", "recent", "random"]:
    print(f"\n  {strategy}:")
    strat_results = [r for r in eviction_results if r["strategy"] == strategy and "ppl" in r]
    for r in strat_results:
        status = "✓" if r["delta"] < 2 else "⚠" if r["delta"] < 10 else "✗"
        print(f"    {status} keep {r['keep_pct']}%: ppl={r['ppl']:.1f} ({r['cache_compression']:.1f}x cache)")

# Best per strategy at <2 ppl cost
print(f"\n  Best eviction (< 2 ppl cost):")
for strategy in ["entropy", "h2o", "recent", "random"]:
    good = [r for r in eviction_results if r["strategy"] == strategy and "ppl" in r and r["delta"] < 2]
    if good:
        best = min(good, key=lambda x: x["keep_pct"])
        print(f"    {strategy}: keep {best['keep_pct']}% ({best['cache_compression']:.1f}x)")

Path("results").mkdir(exist_ok=True)
with open("results/stage143_dynamic_eviction.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results/stage143_dynamic_eviction.json", flush=True)
