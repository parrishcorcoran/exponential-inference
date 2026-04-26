"""Pipeline Step 1.6 — ALL density axes per layer.

Every compressible axis, every layer, independently annealed.
No region assumptions. Each finds its own floor.

Axes per layer:
  1. K projection rank (already done in 1.5b — loaded from checkpoint)
  2. V projection rank (SVD per layer)
  3. Q projection rank (SVD per layer)
  4. O projection rank (SVD per layer)
  5. MLP width (zero out rows per layer)
  6. Weight bits (per-channel quant per layer)

Method per cycle:
  - For each layer, for each axis: try 5% reduction
  - Accept if damage < budget, reject and freeze that axis for that layer
  - After full sweep: global FT norms 300 steps
  - Repeat until all axes on all layers frozen

This produces the COMPLETE compression profile: shape + density
for every component in every layer.
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
def eval_ppl(model, val_tokens, seq_len, device, n_batches=5):
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
    torch.cuda.empty_cache()


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
BUDGET = 2.0  # per-axis per-layer ppl budget
MAX_CYCLES = 15
MIN_RANK = 16
MIN_MLP_PCT = 50

PROMPTS = [
    "The theory of general relativity describes gravity as",
    "In quantum mechanics, the uncertainty principle states that",
    "The French Revolution began in 1789 when",
    "Machine learning models are trained by",
    "The mitochondria is often called the powerhouse of the cell because",
]

print("=" * 60)
print("PIPELINE STEP 1.6 — ALL AXES PER LAYER")
print(f"  Axes: V_rank, Q_rank, O_rank, MLP_width, weight_bits")
print(f"  Budget: {BUDGET} ppl per axis per layer")
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
base_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=10)
print(f"  Starting ppl: {base_ppl:.1f}", flush=True)

# Load K ranks from step 1.5b
with open("results/pipeline_step1_5b_cavity.json") as f:
    shape_data = json.load(f)
k_ranks = {int(k): v for k, v in shape_data["final_k_ranks"].items()}

# Initialize per-layer state for new axes
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
d_qo = model.config.hidden_size

state = {}
for i in range(L):
    state[i] = {
        "v_rank": d_kv,       # current V rank (start full)
        "q_rank": d_qo,       # current Q rank
        "o_rank": d_qo,       # current O rank
        "mlp_pct": 100.0,     # MLP width percent
        "w_bits": 16,         # weight bits
        "v_frozen": False,
        "q_frozen": False,
        "o_frozen": False,
        "mlp_frozen": False,
        "w_frozen": False,
    }

history = []

for cycle in range(1, MAX_CYCLES + 1):
    print(f"\n{'='*50}")
    print(f"  CYCLE {cycle}/{MAX_CYCLES}")
    print(f"{'='*50}", flush=True)

    any_moved = False

    for i in range(L):
        layer = model.model.layers[i]
        s = state[i]

        # === V rank ===
        if not s["v_frozen"] and s["v_rank"] > MIN_RANK:
            new_rank = max(int(s["v_rank"] * 0.95), MIN_RANK)
            if new_rank < s["v_rank"]:
                v_orig = layer.self_attn.v_proj.weight.data.clone()
                svd_compress_proj(layer.self_attn.v_proj, new_rank)
                ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
                if ppl - base_ppl > BUDGET:
                    layer.self_attn.v_proj.weight.data = v_orig
                    s["v_frozen"] = True
                    if i % 10 == 0: print(f"  L{i:>2} V_rank FROZEN at {s['v_rank']}", flush=True)
                else:
                    s["v_rank"] = new_rank
                    any_moved = True
                del v_orig

        # === Q rank ===
        if not s["q_frozen"] and s["q_rank"] > MIN_RANK:
            new_rank = max(int(s["q_rank"] * 0.95), MIN_RANK)
            if new_rank < s["q_rank"]:
                q_orig = layer.self_attn.q_proj.weight.data.clone()
                svd_compress_proj(layer.self_attn.q_proj, new_rank)
                ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
                if ppl - base_ppl > BUDGET:
                    layer.self_attn.q_proj.weight.data = q_orig
                    s["q_frozen"] = True
                    if i % 10 == 0: print(f"  L{i:>2} Q_rank FROZEN at {s['q_rank']}", flush=True)
                else:
                    s["q_rank"] = new_rank
                    any_moved = True
                del q_orig

        # === O rank ===
        if not s["o_frozen"] and s["o_rank"] > MIN_RANK:
            new_rank = max(int(s["o_rank"] * 0.95), MIN_RANK)
            if new_rank < s["o_rank"]:
                o_orig = layer.self_attn.o_proj.weight.data.clone()
                svd_compress_proj(layer.self_attn.o_proj, new_rank)
                ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
                if ppl - base_ppl > BUDGET:
                    layer.self_attn.o_proj.weight.data = o_orig
                    s["o_frozen"] = True
                    if i % 10 == 0: print(f"  L{i:>2} O_rank FROZEN at {s['o_rank']}", flush=True)
                else:
                    s["o_rank"] = new_rank
                    any_moved = True
                del o_orig

        # === MLP width ===
        if not s["mlp_frozen"] and s["mlp_pct"] > MIN_MLP_PCT:
            new_pct = max(s["mlp_pct"] * 0.95, MIN_MLP_PCT)
            if new_pct < s["mlp_pct"]:
                # Save MLP weights
                g_orig = layer.mlp.gate_proj.weight.data.clone()
                u_orig = layer.mlp.up_proj.weight.data.clone()
                d_orig = layer.mlp.down_proj.weight.data.clone()
                prune_mlp(layer, new_pct)
                ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
                if ppl - base_ppl > BUDGET:
                    layer.mlp.gate_proj.weight.data = g_orig
                    layer.mlp.up_proj.weight.data = u_orig
                    layer.mlp.down_proj.weight.data = d_orig
                    s["mlp_frozen"] = True
                    if i % 10 == 0: print(f"  L{i:>2} MLP FROZEN at {s['mlp_pct']:.0f}%", flush=True)
                else:
                    s["mlp_pct"] = new_pct
                    any_moved = True
                del g_orig, u_orig, d_orig

        # === Weight bits ===
        if not s["w_frozen"] and s["w_bits"] > 4:
            next_bits = s["w_bits"] - 1
            # Save all weights
            all_orig = {}
            for parent, names in [(layer.self_attn, ["q_proj","k_proj","v_proj","o_proj"]),
                                  (layer.mlp, ["gate_proj","up_proj","down_proj"])]:
                for name in names:
                    all_orig[name] = getattr(parent, name).weight.data.clone()
            quantize_layer_perchannel(layer, next_bits)
            ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
            if ppl - base_ppl > BUDGET:
                for parent, names in [(layer.self_attn, ["q_proj","k_proj","v_proj","o_proj"]),
                                      (layer.mlp, ["gate_proj","up_proj","down_proj"])]:
                    for name in names:
                        getattr(parent, name).weight.data = all_orig[name]
                s["w_frozen"] = True
                if i % 10 == 0: print(f"  L{i:>2} W_bits FROZEN at Q{s['w_bits']}", flush=True)
            else:
                s["w_bits"] = next_bits
                any_moved = True
            del all_orig

        torch.cuda.empty_cache()

    if not any_moved:
        print(f"\n  All axes on all layers frozen. Done.", flush=True)
        break

    # Global FT
    print(f"\n  Fine-tuning norms (300 steps)...", end="", flush=True)
    finetune_norms(model, train_tokens, SEQ_LEN, device, steps=300)
    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=10)
    base_ppl = post_ppl
    print(f" ppl={post_ppl:.1f}", flush=True)

    # Count active axes
    total_axes = L * 5
    frozen_axes = sum(1 for i in range(L) for k in ["v_frozen","q_frozen","o_frozen","mlp_frozen","w_frozen"] if state[i][k])
    print(f"  Active axes: {total_axes - frozen_axes}/{total_axes}  Frozen: {frozen_axes}/{total_axes}", flush=True)

    # Sample layer stats
    for si in [0, 10, 20, 30, 39]:
        s = state[si]
        print(f"  L{si:>2}: V={s['v_rank']:>4} Q={s['q_rank']:>4} O={s['o_rank']:>4} MLP={s['mlp_pct']:.0f}% W=Q{s['w_bits']}", flush=True)

    history.append({
        "cycle": cycle, "post_ppl": round(post_ppl, 2),
        "frozen_axes": frozen_axes, "total_axes": total_axes,
    })

    if cycle % 5 == 0:
        save_path = Path(f"checkpoints/pipeline/step1_6_c{cycle}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# ═══════════════════════════════════════════════════════
# Final profile + coherence
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("COHERENCE TEST")
print(f"{'='*60}")
samples = generate_samples(model, tokenizer, PROMPTS)
for s in samples:
    print(f"  [{s['prompt'][:40]}...] → {s['output'][:70]}", flush=True)

print(f"\n{'='*60}")
print("COMPLETE COMPRESSION PROFILE")
print(f"{'='*60}")
print(f"  {'L':>3} {'K_rank':>6} {'V_rank':>6} {'Q_rank':>6} {'O_rank':>6} {'MLP%':>5} {'Wbits':>5}")
for i in range(L):
    s = state[i]
    print(f"  L{i:>2} {k_ranks[i]:>6} {s['v_rank']:>6} {s['q_rank']:>6} {s['o_rank']:>6} {s['mlp_pct']:>5.0f} Q{s['w_bits']:>3}")

final_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n_batches=15)
print(f"\n  Final ppl: {final_ppl:.1f}")

save_path = Path("checkpoints/pipeline/step1_6_all_axes_final")
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(save_path))
tokenizer.save_pretrained(str(save_path))

with open("results/pipeline_step1_6_all_axes.json", "w") as f:
    json.dump({
        "final_ppl": final_ppl,
        "state": {str(i): state[i] for i in range(L)},
        "k_ranks": {str(i): k_ranks[i] for i in range(L)},
        "history": history,
        "coherence_samples": samples,
    }, f, indent=2)
print(f"  Saved everything.", flush=True)

del model; gc.collect(); torch.cuda.empty_cache()
