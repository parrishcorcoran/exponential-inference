"""Train 5 standard token-Medusa heads on Qwen3-0.6B.

Standard Medusa: each head predicts the token at offset +k directly from
final hidden state h_t. Architecture: residual SiLU MLP -> shared LM head.
Train with cross-entropy on actual next-token-at-offset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import gc
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32


def empty_cache():
    if device == "cuda": torch.cuda.empty_cache()
    elif device == "mps": torch.mps.empty_cache()


def load_owt_tokens(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []
    skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
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
            yield torch.tensor(batch, dtype=torch.long, device=device)
            batch = []


class MedusaHead(nn.Module):
    """Predict token at offset +k from final hidden state."""
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])

    def forward(self, h, lm_head_weight):
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
MAX_HEADS = 5
STEPS_PER_HEAD = 300
EVAL_EVERY = 50
LR = 1e-4
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_medusa_heads_06b.json")

print(f"device={device} dtype={dtype}")
print(f"  {MAX_HEADS} standard Medusa heads, {STEPS_PER_HEAD} steps each")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
print("Loading tokens...", flush=True)
train_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 3000)
val_tokens = load_owt_tokens(tokenizer, max_tokens=SEQ_LEN * 200, skip_tokens=SEQ_LEN * 3000)

print(f"Loading {CHECKPOINT}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
vocab_size = model.config.vocab_size
lm_head_weight = model.lm_head.weight.detach()

for p in model.parameters():
    p.requires_grad = False

print(f"  d_model={d_model}, vocab={vocab_size}")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

all_results = []

for head_idx in range(1, MAX_HEADS + 1):
    offset = head_idx
    print(f"\n{'='*60}\n  MEDUSA HEAD {head_idx} — predict t+{offset}\n{'='*60}", flush=True)

    head = MedusaHead(d_model, n_layers=1).to(device).to(torch.float32)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"  Head params: {head_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    head.train()
    step = 0
    history = []

    for batch in iter_batches(train_tokens, SEQ_LEN, 1, device):
        if step >= STEPS_PER_HEAD:
            break

        with torch.no_grad():
            out = model(batch[:, :-offset], use_cache=False, output_hidden_states=True)
            h = out.hidden_states[-1].float()  # [1, seq, d]

        targets = batch[:, offset:]  # [1, seq-offset]
        min_l = min(h.shape[1], targets.shape[1])
        h = h[:, :min_l]
        targets = targets[:, :min_l]

        logits = head(h, lm_head_weight.float())
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()

        step += 1

        if step % EVAL_EVERY == 0:
            with torch.no_grad():
                preds = logits.argmax(-1)
                acc = (preds == targets).float().mean().item()
            print(f"  step {step:>4}: loss={loss.item():.3f} acc={acc:.3f}", flush=True)
            history.append({"step": step, "loss": round(loss.item(), 4), "acc": round(acc, 4)})

    # Final val
    head.eval()
    val_accs = []
    val_count = 0
    for vbatch in iter_batches(val_tokens, SEQ_LEN, 1, device):
        if val_count >= 20: break
        with torch.no_grad():
            out = model(vbatch[:, :-offset], use_cache=False, output_hidden_states=True)
            h_val = out.hidden_states[-1].float()
            tgt_val = vbatch[:, offset:]
            ml = min(h_val.shape[1], tgt_val.shape[1])
            logits_val = head(h_val[:, :ml], lm_head_weight.float())
            preds_val = logits_val.argmax(-1)
            val_accs.append((preds_val == tgt_val[:, :ml]).float().mean().item())
        val_count += 1

    final_acc = sum(val_accs) / len(val_accs)
    print(f"\n  HEAD {head_idx} FINAL val_acc={final_acc:.3f}", flush=True)

    all_results.append({
        "offset": offset,
        "final_val_acc": round(final_acc, 4),
        "head_params_M": round(head_params/1e6, 2),
        "history": history,
    })

    torch.save(head.state_dict(), CKPT_DIR / f"medusa_head_{head_idx}.pt")
    del head, opt; empty_cache()

print(f"\n{'='*60}\nMEDUSA HEADS 0.6B SUMMARY\n{'='*60}")
for r in all_results:
    print(f"  t+{r['offset']}: val_acc={r['final_val_acc']:.3f} ({r['final_val_acc']*100:.1f}%)")

# Chained tokens-per-step assuming independence
chained = 1.0; prod = 1.0
for r in all_results:
    prod *= r["final_val_acc"]; chained += prod
print(f"\n  Chained tokens/step (1 + a1 + a1*a2 + ...): {chained:.3f}")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "device": device,
        "steps_per_head": STEPS_PER_HEAD,
        "results": all_results,
        "chained_tokens_per_step": round(chained, 4),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")

del model; gc.collect(); empty_cache()
