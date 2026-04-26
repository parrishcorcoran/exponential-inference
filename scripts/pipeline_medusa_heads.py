"""Pipeline — Medusa heads on KV-256 base.

Add speculative decode heads one at a time to the KV-compressed model.
Each head predicts a future token offset. Train heavily, measure
acceptance rate. Then add KV-Medusa heads that predict future K/V.

Head k predicts token at position t+k given hidden state at t.
Architecture: small MLP → shared LM head.

On the KV-256 base (ppl 9.7, better than teacher), Medusa heads
should train well because the model is already high quality.

With KV compression, each head's cache cost is 4x cheaper than
normal, so we can afford many more heads → higher decode throughput.
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
        if len(window) < seq_len + 1: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t
            batch = []


class MedusaHead(nn.Module):
    """Predict token at offset +k from hidden state."""
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])

    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


CHECKPOINT = "checkpoints/qwen_halo/kv256_base"
SEQ_LEN = 256
MAX_HEADS = 5
STEPS_PER_HEAD = 500
EVAL_EVERY = 100
LR = 1e-4

print("=" * 60)
print("MEDUSA HEADS ON KV-256 BASE")
print(f"  {MAX_HEADS} heads, {STEPS_PER_HEAD} steps each")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

print("\nLoading tokens...", flush=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)
print(f"  Train: {len(train_tokens)}, Val: {len(val_tokens)}", flush=True)

print(f"\nLoading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

# Freeze base model
for p in model.parameters():
    p.requires_grad = False

print(f"  d_model={d_model}, vocab={vocab_size}")
print(f"  Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

all_heads = []
all_results = []

for head_idx in range(1, MAX_HEADS + 1):
    offset = head_idx  # predict t+offset

    print(f"\n{'='*60}")
    print(f"  MEDUSA HEAD {head_idx} — predict t+{offset}")
    print(f"{'='*60}", flush=True)

    head = MedusaHead(d_model, n_layers=1).to(device).to(torch.float32)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)

    head.train()
    step = 0
    running_loss = []
    running_acc = []
    history = []

    for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
        if step >= STEPS_PER_HEAD:
            break

        # Get hidden states from base model
        with torch.no_grad():
            out = model(batch[:, :-offset], use_cache=False, output_hidden_states=True)
            # Last hidden state before final norm
            h = out.hidden_states[-1].detach()  # [1, seq-offset, d_model]

        # Target: tokens at position +offset
        targets = batch[:, offset:]  # [1, seq-offset]

        # Trim to match
        min_len = min(h.shape[1], targets.shape[1])
        h = h[:, :min_len]
        targets = targets[:, :min_len]

        # Forward through Medusa head
        logits = head(h.float(), lm_head_weight.float())  # [1, seq, vocab]

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        # Accuracy
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == targets).float().mean().item()

        running_loss.append(loss.item())
        running_acc.append(acc)
        step += 1

        if step % EVAL_EVERY == 0:
            avg_loss = sum(running_loss[-EVAL_EVERY:]) / EVAL_EVERY
            avg_acc = sum(running_acc[-EVAL_EVERY:]) / EVAL_EVERY

            # Val accuracy
            head.eval()
            val_accs = []
            val_count = 0
            for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
                if val_count >= 10: break
                with torch.no_grad():
                    out = model(vbatch[:, :-offset], use_cache=False, output_hidden_states=True)
                    h_val = out.hidden_states[-1]
                    targets_val = vbatch[:, offset:]
                    min_l = min(h_val.shape[1], targets_val.shape[1])
                    logits_val = head(h_val[:, :min_l].float(), lm_head_weight.float())
                    preds_val = logits_val.argmax(-1)
                    val_accs.append((preds_val == targets_val[:, :min_l]).float().mean().item())
                val_count += 1
            val_acc = sum(val_accs) / len(val_accs) if val_accs else 0
            head.train()

            print(f"  step {step:>5}: loss={avg_loss:.3f} train_acc={avg_acc:.3f} val_acc={val_acc:.3f}", flush=True)
            history.append({
                "step": step, "loss": round(avg_loss, 4),
                "train_acc": round(avg_acc, 4), "val_acc": round(val_acc, 4),
            })

    # Final val accuracy
    head.eval()
    final_accs = []
    for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
        if len(final_accs) >= 20: break
        with torch.no_grad():
            out = model(vbatch[:, :-offset], use_cache=False, output_hidden_states=True)
            h_val = out.hidden_states[-1]
            targets_val = vbatch[:, offset:]
            min_l = min(h_val.shape[1], targets_val.shape[1])
            logits_val = head(h_val[:, :min_l].float(), lm_head_weight.float())
            preds_val = logits_val.argmax(-1)
            final_accs.append((preds_val == targets_val[:, :min_l]).float().mean().item())
    final_acc = sum(final_accs) / len(final_accs)

    print(f"\n  HEAD {head_idx} FINAL: val_acc={final_acc:.3f} ({final_acc*100:.1f}%)", flush=True)

    all_heads.append(head)
    all_results.append({
        "head": head_idx, "offset": offset,
        "final_val_acc": round(final_acc, 4),
        "history": history,
    })

    # Save head
    torch.save(head.state_dict(), f"checkpoints/qwen_halo/medusa_head_{head_idx}.pt")
    print(f"  Saved medusa_head_{head_idx}.pt", flush=True)

    del opt; torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("MEDUSA HEADS SUMMARY")
print(f"{'='*60}")
print(f"  Model: KV-256 base (ppl 9.7)")
for r in all_results:
    print(f"  Head {r['head']} (t+{r['offset']}): val_acc={r['final_val_acc']:.3f} ({r['final_val_acc']*100:.1f}%)")

# Estimated decode speedup
# With acceptance rates a1, a2, a3...
# Expected tokens per step ≈ 1 + a1 + a1*a2 + a1*a2*a3 + ...
accs = [r["final_val_acc"] for r in all_results]
expected_tokens = 1.0
product = 1.0
for a in accs:
    product *= a
    expected_tokens += product
print(f"\n  Expected tokens per decode step: {expected_tokens:.2f}")
print(f"  Estimated decode speedup: {expected_tokens:.1f}x")

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_medusa_heads.json", "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "results": all_results,
        "expected_tokens_per_step": round(expected_tokens, 3),
    }, f, indent=2)
print(f"\nSaved results/pipeline_medusa_heads.json", flush=True)

# Save all heads together
torch.save(
    {f"head_{i+1}": h.state_dict() for i, h in enumerate(all_heads)},
    "checkpoints/qwen_halo/medusa_all_heads.pt"
)
print(f"Saved medusa_all_heads.pt", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
