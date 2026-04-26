"""Pipeline Step 1.6 — Simultaneous 1% squeeze on ALL axes.

All orthogonal axes compressed together by 1% each step.
Per layer: V rank, Q rank, O rank, MLP width, weight bits.
All at once, all together, tiny steps, FT between each.

The LASER effect from compressing everything simultaneously
should provide better recovery than sequential per-axis.

Start from step 1.3 checkpoint (ppl ~24, more headroom).
K ranks already set from steps 1.3/1.5b — don't touch those.
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


def svd_compress_proj(proj, rank):
    W = proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(min(rank, len(S)), 1)
    proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(proj.weight.dtype)


def quantize_layer_perchannel(layer, bits):
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


def finetune_norms(model, train_tokens, seq_len, device, steps=500, lr=5e-5):
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


# Resume from step 10 checkpoint (V=921, MLP=90%, Q15)
CHECKPOINT = "checkpoints/pipeline/step1_6_sim_s10"
SEQ_LEN = 256
SQUEEZE = 0.99  # 1% per step
MAX_STEPS = 100
MIN_RANK = 32
MIN_MLP_PCT = 60
MIN_BITS = 4
THERMOSTAT = 2.5  # relative to starting ppl

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "In quantum mechanics, the uncertainty principle states that",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "The mitochondria is often called the powerhouse of the cell because",
]

print("=" * 60)
print("PIPELINE STEP 1.6 — SIMULTANEOUS 1% SQUEEZE")
print(f"  All axes, all layers, 1% per step, all together")
print(f"  Squeeze: {SQUEEZE} | Min rank: {MIN_RANK} | Min bits: Q{MIN_BITS}")
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
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
d_qo = model.config.hidden_size

base_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
thermostat_limit = base_ppl * THERMOSTAT
print(f"  Starting ppl: {base_ppl:.1f}")
print(f"  Thermostat: {thermostat_limit:.1f} ({THERMOSTAT}x)", flush=True)

# Initialize per-layer state — resume from step 10
state = {}
for i in range(L):
    state[i] = {
        "v_rank": 921,      # V rank at step 10
        "mlp_pct": 90.0,    # MLP at step 10
    }

history = []

for step in range(1, MAX_STEPS + 1):
    t0 = time.time()

    # === Squeeze ALL axes by 1% on ALL layers ===
    for i in range(L):
        layer = model.model.layers[i]
        s = state[i]

        # V rank: 1% reduction (small matrix 1280×5120 — fast SVD)
        new_v = max(int(s["v_rank"] * SQUEEZE), MIN_RANK)
        if new_v < s["v_rank"]:
            svd_compress_proj(layer.self_attn.v_proj, new_v)
            s["v_rank"] = new_v

        # Q/O rank: SKIP — froze at full in per-axis test, don't compress
        # Q/O compression comes from weight bits instead

        # MLP: 1% width reduction
        new_mlp = max(s["mlp_pct"] * SQUEEZE, MIN_MLP_PCT)
        if new_mlp < s["mlp_pct"]:
            prune_mlp(layer, new_mlp)
            s["mlp_pct"] = new_mlp

    # Weight bits: SKIPPED — apply as final export step (GGUF Q4 / AWQ)
    # V rank + MLP are the axes we uniquely optimize through anneal

    torch.cuda.empty_cache()

    # Eval pre-FT
    pre_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)

    # FT
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=500)

    # Eval post-FT
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
    elapsed = time.time() - t0

    # Sample stats
    avg_v = sum(state[i]["v_rank"] for i in range(L)) / L
    avg_mlp = sum(state[i]["mlp_pct"] for i in range(L)) / L

    print(f"  Step {step:>3}: pre={pre_ppl:.1f} → post={post_ppl:.1f} | "
          f"V={avg_v:.0f} MLP={avg_mlp:.0f}% | "
          f"{elapsed:.0f}s", flush=True)

    history.append({
        "step": step, "pre_ppl": round(pre_ppl, 2), "post_ppl": round(post_ppl, 2),
        "avg_v": round(avg_v, 1),
        "avg_mlp": round(avg_mlp, 1),
    })

    # Thermostat
    if post_ppl > thermostat_limit:
        print(f"\n  ⚠ THERMOSTAT: {post_ppl:.1f} > {thermostat_limit:.1f}")
        break

    # Checkpoint every 10 steps
    if step % 10 == 0:
        save_path = Path(f"checkpoints/pipeline/step1_6_sim_s{step}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

        # Coherence check every 10 steps
        print(f"  Coherence:", flush=True)
        samples = generate_samples(model, tokenizer, PROMPTS[:3])
        for s in samples:
            print(f"    [{s['prompt'][:30]}...] → {s['output'][:60]}", flush=True)

# ═══════════════════════════════════════════════════════
# Final
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FINAL COHERENCE TEST")
print(f"{'='*60}")
samples = generate_samples(model, tokenizer, PROMPTS)
for s in samples:
    print(f"  [{s['prompt'][:40]}...] → {s['output'][:70]}", flush=True)

print(f"\n{'='*60}")
print("FINAL COMPRESSION PROFILE")
print(f"{'='*60}")
print(f"  {'L':>3} {'V':>6} {'Q':>6} {'O':>6} {'MLP%':>5} {'Wbits':>5}")
for i in range(L):
    s = state[i]
    print(f"  L{i:>2} {s['v_rank']:>6} {s['q_rank']:>6} {s['o_rank']:>6} {s['mlp_pct']:>5.0f} Q{s['w_bits']:>3}")

final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=15)
print(f"\n  Final ppl: {final_ppl:.1f}")

save_path = Path("checkpoints/pipeline/step1_6_simultaneous_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

with open("results/pipeline_step1_6_simultaneous.json", "w") as f:
    json.dump({
        "final_ppl": final_ppl,
        "state": {str(i): state[i] for i in range(L)},
        "history": history,
        "coherence_samples": samples,
    }, f, indent=2)
print(f"  Saved everything.", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
