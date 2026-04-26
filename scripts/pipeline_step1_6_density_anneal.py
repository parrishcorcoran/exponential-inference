"""Pipeline Step 1.6 — Per-layer density anneal (weight quantization).

Each layer gets its own bit-width floor. No region assumptions.
Start from step 1.5b checkpoint (shape done), now find density.

Per-layer quantization anneal:
  - Start at effective bf16
  - Reduce precision one step at a time: 16 → 8 → 7 → 6 → 5 → 4
  - Per-channel quantization (proven best from stage 112)
  - Accept if damage < budget, freeze if not
  - Global FT (300 steps norms) after each cycle
  - Each layer finds its own floor independently

The density map overlays on the shape map from step 1.5b.
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


def quantize_layer_perchannel(layer, bits):
    """Per-channel quantization of all weights in a layer."""
    half = 2 ** (bits - 1)
    for parent, names in [(layer.self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                          (layer.mlp, ["gate_proj", "up_proj", "down_proj"])]:
        for name in names:
            w = getattr(parent, name).weight
            W = w.data.float()
            scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / (half - 1)
            w.data = (torch.round(W / scale).clamp(-(half-1), half-1) * scale).to(w.dtype)


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


CHECKPOINT = "checkpoints/pipeline/step1_5b_cavity_final"
SEQ_LEN = 256
PER_LAYER_BUDGET = 3.0
BIT_SCHEDULE = [8, 7, 6, 5, 4]  # steps to try per layer

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "In quantum mechanics, the uncertainty principle states that",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "The mitochondria is often called the powerhouse of the cell because",
]

print("=" * 60)
print("PIPELINE STEP 1.6 — PER-LAYER DENSITY ANNEAL")
print(f"  Bit schedule: {BIT_SCHEDULE}")
print(f"  Per-layer budget: {PER_LAYER_BUDGET} ppl")
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

# Per-layer current bits (start at 16)
layer_bits = {i: 16 for i in range(L)}
frozen = {i: False for i in range(L)}

history = []

# For each bit level, try all layers
for target_bits in BIT_SCHEDULE:
    print(f"\n{'='*50}")
    print(f"  DENSITY PASS: trying Q{target_bits} per layer")
    print(f"{'='*50}", flush=True)

    any_accepted = False

    for i in range(L):
        if frozen[i]:
            continue
        if layer_bits[i] <= target_bits:
            continue

        # Save original weights
        orig_weights = {}
        for parent, names in [(model.model.layers[i].self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                              (model.model.layers[i].mlp, ["gate_proj", "up_proj", "down_proj"])]:
            for name in names:
                orig_weights[name] = getattr(parent, name).weight.data.clone()

        # Quantize this layer
        quantize_layer_perchannel(model.model.layers[i], target_bits)

        # Quick eval
        new_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=5)
        damage = new_ppl - base_ppl

        if damage > PER_LAYER_BUDGET:
            # Restore and freeze at current bits
            for parent, names in [(model.model.layers[i].self_attn, ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                  (model.model.layers[i].mlp, ["gate_proj", "up_proj", "down_proj"])]:
                for name in names:
                    getattr(parent, name).weight.data = orig_weights[name]
            frozen[i] = True
            print(f"  L{i:>2}: Q{target_bits} REJECTED (damage={damage:+.1f}), FROZEN at Q{layer_bits[i]}", flush=True)
        else:
            layer_bits[i] = target_bits
            any_accepted = True
            if i % 5 == 0 or damage > 1.0:
                print(f"  L{i:>2}: Q{target_bits} accepted (damage={damage:+.1f})", flush=True)

        del orig_weights
        torch.cuda.empty_cache()

    if not any_accepted:
        print(f"\n  No layers accepted Q{target_bits}. Moving to next level.", flush=True)
        # Unfreeze for next bit level attempt
        for i in range(L):
            frozen[i] = False
        continue

    # Global FT after each bit level
    print(f"\n  Fine-tuning norms (300 steps)...", end="", flush=True)
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=300)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    base_ppl = post_ppl
    print(f" ppl={post_ppl:.1f}", flush=True)

    # Stats
    n_at_bits = {}
    for b in [16, 8, 7, 6, 5, 4]:
        count = sum(1 for v in layer_bits.values() if v == b)
        if count > 0:
            n_at_bits[f"Q{b}"] = count

    print(f"  Distribution: {n_at_bits}", flush=True)

    history.append({
        "target_bits": target_bits, "post_ppl": round(post_ppl, 2),
        "distribution": n_at_bits,
        "per_layer": {str(i): layer_bits[i] for i in range(L)},
    })

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
# Final density profile
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL PER-LAYER DENSITY PROFILE")
print(f"{'='*60}")

# Load shape from step 1.5b for combined view
with open("results/pipeline_step1_5b_cavity.json") as f:
    shape_data = json.load(f)
k_ranks = {int(k): v for k, v in shape_data["final_k_ranks"].items()}

for i in range(L):
    bits = layer_bits[i]
    rank = k_ranks.get(i, "?")
    bar_rank = "█" * max(int(rank / 20), 1) if isinstance(rank, int) else ""
    bar_bits = "▓" * bits
    print(f"  L{i:>2}: rank={rank:>4}  Q{bits}  {bar_bits}")

final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
print(f"\n  Final ppl: {final_ppl:.1f}")

# Save
save_path = Path("checkpoints/pipeline/step1_6_density_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

Path("results").mkdir(exist_ok=True)
with open("results/pipeline_step1_6_density.json", "w") as f:
    json.dump({
        "final_ppl": final_ppl,
        "layer_bits": {str(i): layer_bits[i] for i in range(L)},
        "k_ranks": {str(i): k_ranks.get(i, None) for i in range(L)},
        "history": history,
        "coherence_samples": samples,
    }, f, indent=2)
print(f"  Saved results + model", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
