"""PID-controlled multi-axis continuous compression.

Track ppl at exactly target_ratio × teacher (e.g. 1.05x = 5% above).
Squeeze ALL orthogonal axes simultaneously at 0.1% per step.
If ppl drifts above target → more FT to pull back.
If ppl below target → squeeze faster.

Axes squeezed simultaneously:
  1. K rank (SVD truncation, all layers)
  2. MLP width (row zeroing, all layers)
  3. Spectral norm (top SV via power iteration)
  4. Magnitude (uniform weight scaling)

PID controller:
  error = current_ppl - target_ppl
  P: squeeze_rate proportional to -error (squeeze more when below target)
  I: accumulated error (prevents drift)
  D: rate of change (dampen oscillations)

Pre-tokenized OWT. Coherence check every 20 steps.
"""
import torch
import torch.nn.functional as F
import math, json, time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

device = "cuda"
MODEL = "Qwen/Qwen3-4B"
SEQ_LEN = 256
TARGET_RATIO = 1.05  # stay at 5% above teacher
MAX_STEPS = 500
BASE_FT = 300
MIN_SQUEEZE = 0.998  # minimum squeeze per step (0.2%)
MAX_SQUEEZE = 0.9999  # maximum squeeze per step (0.01%)

PROMPTS = ["The theory of general relativity describes gravity as",
           "Machine learning models are trained by",
           "The French Revolution began in 1789 when"]

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Pre-tokenize
print("Pre-tokenizing OWT...", flush=True)
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
all_toks = []
for item in ds:
    t = item.get("text", "")
    if not t.strip(): continue
    all_toks.extend(tokenizer.encode(t, add_special_tokens=False))
    if len(all_toks) >= SEQ_LEN * 5000: break
train_tokens = all_toks[:SEQ_LEN * 4000]
val_tokens = all_toks[SEQ_LEN * 4000:]
print(f"  Train: {len(train_tokens)}, Val: {len(val_tokens)}", flush=True)

def iter_batches(tokens, seq_len, device, n=999):
    import random
    idxs = list(range((len(tokens)-1)//seq_len))
    random.shuffle(idxs)
    for i in idxs[:n]:
        s = i * seq_len
        w = tokens[s:s+seq_len+1]
        if len(w) < seq_len+1: continue
        yield torch.tensor([w], dtype=torch.long, device=device)

def eval_ppl(model, val_tokens, seq_len, device, n=15):
    model.eval(); total = 0; c = 0
    for batch in iter_batches(val_tokens, seq_len, device, n):
        with torch.no_grad():
            logits = model(batch[:, :-1], use_cache=False).logits
            total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1)).item()
            c += 1
    return math.exp(total / max(c, 1))

def power_iteration_squeeze(weight, squeeze):
    W = weight.data.float()
    v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
    v = v / v.norm()
    for _ in range(15):
        u = W @ v; u = u / u.norm()
        v = W.T @ u; v = v / v.norm()
    sigma = (u @ W @ v).item()
    weight.data = (W - (1 - squeeze) * sigma * u.unsqueeze(1) * v.unsqueeze(0)).to(torch.bfloat16)

def svd_truncate_k(layer, rank):
    proj = layer.self_attn.k_proj
    W = proj.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = max(min(rank, len(S)), 1)
    proj.weight.data = ((U[:, :k] * S[:k]) @ Vt[:k]).to(torch.bfloat16)

print("=" * 60)
print("PID MULTI-AXIS CONTINUOUS COMPRESSION")
print(f"  Target: {TARGET_RATIO}x teacher ({(TARGET_RATIO-1)*100:.0f}% above)")
print(f"  Axes: K rank, MLP width, spectral norm, magnitude")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

L = model.config.num_hidden_layers
d_kv = model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)
d_ffn = model.config.intermediate_size

teacher_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device)
target_ppl = teacher_ppl * TARGET_RATIO
print(f"  Teacher: {teacher_ppl:.1f}")
print(f"  Target:  {target_ppl:.1f}", flush=True)

# State tracking
k_rank = d_kv  # 640 for 4B
mlp_pct = 100.0
spectral_squeezed = 0  # cumulative % removed
magnitude = 1.0

# PID state
integral = 0
prev_error = 0

history = []

for step in range(1, MAX_STEPS + 1):
    t0 = time.time()

    current_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n=10)

    # PID error: negative = below target (room to squeeze), positive = above target (ease off)
    error = current_ppl - target_ppl
    integral += error
    derivative = error - prev_error
    prev_error = error

    # PID output → squeeze rate
    # When error < 0 (below target): squeeze more (smaller squeeze factor)
    # When error > 0 (above target): squeeze less (larger squeeze factor, more FT)
    Kp = 0.0001
    Ki = 0.00001
    Kd = 0.00005

    pid_output = Kp * error + Ki * integral + Kd * derivative

    # Map PID output to squeeze factor: 0.998 to 0.9999
    # Negative PID (below target) → more squeeze (0.998)
    # Positive PID (above target) → less squeeze (0.9999) + more FT
    squeeze = max(MIN_SQUEEZE, min(MAX_SQUEEZE, 0.999 + pid_output))

    # FT steps: inverse of squeeze aggressiveness
    ft_steps = int(BASE_FT * (1 + max(error / teacher_ppl, 0) * 5))
    ft_steps = min(ft_steps, 3000)

    # === SQUEEZE ALL AXES SIMULTANEOUSLY ===

    # 1. K rank
    new_k = max(int(k_rank * squeeze), 16)
    if new_k < k_rank:
        for i in range(L):
            svd_truncate_k(model.model.layers[i], new_k)
        k_rank = new_k

    # 2. MLP width
    new_mlp = mlp_pct * squeeze
    if new_mlp < mlp_pct and new_mlp >= 50:
        for i in range(L):
            full = model.model.layers[i].mlp.gate_proj.weight.shape[0]
            old_keep = int(full * mlp_pct / 100)
            new_keep = int(full * new_mlp / 100)
            if new_keep < old_keep:
                model.model.layers[i].mlp.gate_proj.weight.data[new_keep:old_keep] = 0
                model.model.layers[i].mlp.up_proj.weight.data[new_keep:old_keep] = 0
                model.model.layers[i].mlp.down_proj.weight.data[:, new_keep:old_keep] = 0
        mlp_pct = new_mlp

    # 3. Spectral norm (top SV)
    for i in range(L):
        layer = model.model.layers[i]
        for parent, names in [(layer.self_attn, ["q_proj","k_proj","v_proj","o_proj"]),
                              (layer.mlp, ["gate_proj","up_proj","down_proj"])]:
            for name in names:
                power_iteration_squeeze(getattr(parent, name).weight, squeeze)
    spectral_squeezed = 1 - squeeze ** step

    # 4. Magnitude
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name.lower() and "embed" not in name.lower() and "lm_head" not in name.lower():
                p.data *= squeeze
    magnitude *= squeeze

    torch.cuda.empty_cache()

    # === FINE-TUNE ===
    for p in model.parameters(): p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            p.requires_grad = True; trainable.append(p)
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.01)
        model.train(); ft = 0
        for batch in iter_batches(train_tokens, SEQ_LEN, device, ft_steps):
            if ft >= ft_steps: break
            opt.zero_grad()
            logits = model(batch[:, :-1], use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), batch[:, 1:].reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); ft += 1
        del opt
        for p in model.parameters(): p.requires_grad = False
        torch.cuda.empty_cache()

    post_ppl = eval_ppl(model, val_tokens, SEQ_LEN, device, n=10)
    elapsed = time.time() - t0

    if step % 10 == 0 or step <= 5:
        ids = tokenizer(PROMPTS[0], return_tensors='pt').input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=25, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  step {step:>3}: ppl={post_ppl:.1f} (target={target_ppl:.1f}) | K={k_rank} MLP={mlp_pct:.1f}% mag={magnitude:.3f} | sq={squeeze:.4f} ft={ft_steps} | [{text[:40]}]", flush=True)
    else:
        print(f"  step {step:>3}: ppl={post_ppl:.1f} (target={target_ppl:.1f}) | K={k_rank} MLP={mlp_pct:.1f}% mag={magnitude:.3f} | sq={squeeze:.4f} ft={ft_steps}", flush=True)

    history.append({
        "step": step, "ppl": round(post_ppl, 2), "target": round(target_ppl, 2),
        "k_rank": k_rank, "mlp_pct": round(mlp_pct, 2),
        "magnitude": round(magnitude, 4), "squeeze": round(squeeze, 5),
        "ft_steps": ft_steps, "error": round(error, 2),
    })

    # Hard stop if text degrades
    if post_ppl > target_ppl * 3:
        print(f"\n  HARD STOP: {post_ppl:.1f} > {target_ppl*3:.1f}", flush=True)
        break

    if step % 50 == 0:
        save_path = Path(f"checkpoints/pipeline/pid_s{step}")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Saved: {save_path}", flush=True)

# Final
print(f"\n{'='*60}")
print(f"PID MULTI-AXIS COMPRESSION COMPLETE")
print(f"  Teacher:    {teacher_ppl:.1f}")
print(f"  Target:     {target_ppl:.1f}")
if history:
    print(f"  Final:      {history[-1]['ppl']:.1f}")
    print(f"  K rank:     {k_rank} (from {d_kv})")
    print(f"  MLP:        {mlp_pct:.1f}%")
    print(f"  Magnitude:  {magnitude:.3f}")
    print(f"  Steps:      {len(history)}")

print(f"\nCoherence:")
for p in PROMPTS:
    ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{p[:30]}...] → {text[:60]}", flush=True)

Path("results").mkdir(exist_ok=True)
with open("results/pid_multiaxis_4b.json", "w") as f:
    json.dump({"teacher": teacher_ppl, "target": target_ppl, "history": history}, f, indent=2)
print(f"Saved results/pid_multiaxis_4b.json", flush=True)

del model; torch.cuda.empty_cache()
