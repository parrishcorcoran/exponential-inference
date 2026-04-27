"""KV-Medusa 0.6B — wallclock benchmark.

Measures:
  - Baseline autoregressive greedy decode tok/s
  - Heads inference time (10 heads on a single h_t)
  - Verification forward pass time (one full forward with prefix length)
  - Projected KV-Medusa tok/s using chained tokens/step from the acceptance test

Reports speedup vs baseline.
"""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

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


CHECKPOINT = "Qwen/Qwen3-0.6B"
PREFIX_LEN = 128
N_GEN = 64
N_REPEATS = 5
N_OFFSETS = 10
TARGET_LAYER = 14
CKPT_DIR = Path("checkpoints/qwen_06b")
RESULTS_PATH = Path("results/pipeline_kv_medusa_06b_speed.json")
TOKENS_PER_STEP = 8.76  # from acceptance test


class KVMedusaHead(nn.Module):
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, n_kv_heads * head_dim, bias=False),
        )
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def forward(self, h):
        k = self.k_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(h.shape[0], h.shape[1], self.n_kv_heads, self.head_dim)
        return k, v


def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


print(f"device={device} dtype={dtype}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

d_model = model.config.hidden_size
n_kv_heads = model.config.num_key_value_heads
head_dim = getattr(model.config, "head_dim", None) or (d_model // model.config.num_attention_heads)

print(f"Loading {N_OFFSETS} heads...")
heads = []
for k in range(1, N_OFFSETS + 1):
    h = KVMedusaHead(d_model, n_kv_heads, head_dim).to(device)
    h.load_state_dict(torch.load(CKPT_DIR / f"kv_medusa_head_{k}.pt", map_location=device))
    h.eval()
    heads.append(h)

# ── 1. Baseline autoregressive greedy decode ─────────────────────────────
prompt = "The future of artificial intelligence depends on understanding the structure of"
prefix_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prefix_ids = prefix_ids[:, :PREFIX_LEN] if prefix_ids.shape[1] > PREFIX_LEN else prefix_ids

print(f"\nBenchmark setup: prefix_len={prefix_ids.shape[1]}, generating {N_GEN} tokens, {N_REPEATS} reps")

# Warmup
with torch.no_grad():
    _ = model.generate(prefix_ids, max_new_tokens=4, do_sample=False, use_cache=True)
sync()

baseline_times = []
for r in range(N_REPEATS):
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(prefix_ids, max_new_tokens=N_GEN, do_sample=False, use_cache=True)
    sync()
    t1 = time.perf_counter()
    baseline_times.append(t1 - t0)

baseline_dur = sum(baseline_times) / len(baseline_times)
baseline_tps = N_GEN / baseline_dur
print(f"\nBaseline greedy: {baseline_dur*1000:.1f} ms for {N_GEN} tokens => {baseline_tps:.2f} tok/s")
print(f"  Per-step: {baseline_dur/N_GEN*1000:.2f} ms")

# ── 2. Heads inference time (running 10 heads on a single h_t) ──────────
# Run a forward to get a realistic h_t
with torch.no_grad():
    out = model(prefix_ids, output_hidden_states=True, use_cache=True)
    h_t = out.hidden_states[-1][:, -1:].float()  # [1, 1, d]

sync()
heads_times = []
for r in range(N_REPEATS + 2):  # extra warmup
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for hd in heads:
            _ = hd(h_t)
    sync()
    t1 = time.perf_counter()
    if r >= 2:
        heads_times.append(t1 - t0)

heads_dur = sum(heads_times) / len(heads_times)
print(f"\n10 heads inference: {heads_dur*1000:.2f} ms")

# ── 3. Verification forward pass time ────────────────────────────────────
# This is a forward pass on prefix_len + N_OFFSETS positions (drafting 10 tokens ahead)
draft_input = torch.cat([prefix_ids, prefix_ids[:, -N_OFFSETS:]], dim=1)  # synthetic — just shape

sync()
verify_times = []
for r in range(N_REPEATS + 2):
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(draft_input, use_cache=False)
    sync()
    t1 = time.perf_counter()
    if r >= 2:
        verify_times.append(t1 - t0)

verify_dur = sum(verify_times) / len(verify_times)
print(f"Verification forward pass (len {draft_input.shape[1]}): {verify_dur*1000:.2f} ms")

# Single-token forward pass for comparison (this is what baseline does per step)
sync()
single_times = []
single_input = prefix_ids[:, -1:]
with torch.no_grad():
    cache_out = model(prefix_ids[:, :-1], use_cache=True)
    pkv = cache_out.past_key_values

for r in range(N_REPEATS + 2):
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        # Single token forward with cache
        _ = model(single_input, past_key_values=pkv, use_cache=True)
    sync()
    t1 = time.perf_counter()
    if r >= 2:
        single_times.append(t1 - t0)

single_dur = sum(single_times) / len(single_times)
print(f"Single-token forward (with cache): {single_dur*1000:.2f} ms")

# ── 4. Projected KV-Medusa tok/s ─────────────────────────────────────────
# One Medusa step: heads_time + verify_time → produces TOKENS_PER_STEP accepted tokens
medusa_step_dur = heads_dur + verify_dur
medusa_tps = TOKENS_PER_STEP / medusa_step_dur
speedup = medusa_tps / baseline_tps

print(f"\n{'=' * 60}")
print("KV-MEDUSA 0.6B WALLCLOCK SUMMARY")
print(f"{'=' * 60}")
print(f"  Baseline greedy:           {baseline_tps:>8.2f} tok/s   ({baseline_dur/N_GEN*1000:.2f} ms/tok)")
print(f"  Heads inference:           {heads_dur*1000:>8.2f} ms     (10 heads, single h_t)")
print(f"  Verify forward pass:       {verify_dur*1000:>8.2f} ms     (len {draft_input.shape[1]})")
print(f"  Single-tok forward:        {single_dur*1000:>8.2f} ms")
print(f"  Medusa step duration:      {medusa_step_dur*1000:>8.2f} ms     (heads + verify)")
print(f"  Tokens per step:           {TOKENS_PER_STEP:>8.2f}        (from acceptance test)")
print(f"  Projected Medusa tok/s:    {medusa_tps:>8.2f} tok/s")
print(f"  Projected speedup:         {speedup:>8.2f}x")

with open(RESULTS_PATH, "w") as f:
    json.dump({
        "checkpoint": CHECKPOINT,
        "device": device,
        "dtype": str(dtype),
        "prefix_len": int(prefix_ids.shape[1]),
        "n_gen": N_GEN,
        "n_repeats": N_REPEATS,
        "tokens_per_step": TOKENS_PER_STEP,
        "baseline_tps": round(baseline_tps, 3),
        "baseline_ms_per_tok": round(baseline_dur / N_GEN * 1000, 3),
        "heads_ms": round(heads_dur * 1000, 3),
        "verify_ms": round(verify_dur * 1000, 3),
        "single_tok_ms": round(single_dur * 1000, 3),
        "medusa_step_ms": round(medusa_step_dur * 1000, 3),
        "medusa_projected_tps": round(medusa_tps, 3),
        "speedup": round(speedup, 3),
    }, f, indent=2)
print(f"\nSaved {RESULTS_PATH}")
