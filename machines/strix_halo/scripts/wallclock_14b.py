"""Wall-clock speedup: factored rank-32 vs full model on Qwen3-14B."""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
device = 'cuda'

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

# Load full model
print("Loading Qwen3-14B...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

prompt = "The future of artificial intelligence will transform"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
N_GEN = 64

# Baseline: full model
print("\n=== BASELINE (full model) ===", flush=True)
with torch.no_grad():
    model.generate(input_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    base_out = model.generate(input_ids, max_new_tokens=N_GEN, do_sample=False)
torch.cuda.synchronize()
base_time = time.time() - t0
base_tps = N_GEN / base_time
base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"  {base_tps:.1f} tok/s ({N_GEN} in {base_time:.1f}s)", flush=True)
print(f"  {base_text[:80]}", flush=True)

# Now factorize at rank-32 and measure
print("\n=== Factorizing to rank-32 ===", flush=True)
from machines.strix_halo.scripts.train_matryoshka import (
    MatryoshkaFactoredLinear, RankController
)
TARGET = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
H = model.config.hidden_size
K = 32
controller = RankController(K)
controller.global_k = K

# Quick SVD factorization (no training — just measure raw speed of factored arch)
n_fact = 0
for name, mod in list(model.named_modules()):
    for cn, child in list(mod.named_children()):
        if not isinstance(child, nn.Linear) or cn not in TARGET:
            continue
        W = child.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        # Use top-K as basis
        P = Vt[:K].T.contiguous()  # [in, K]
        try:
            li = int(f"{name}.{cn}".split("model.layers.")[1].split(".")[0])
        except:
            li = -1
        fact = MatryoshkaFactoredLinear(child, P, controller, li, trainable=False)
        setattr(mod, cn, fact)
        n_fact += 1
        del child

torch.cuda.empty_cache()
print(f"  {n_fact} layers factorized to rank-{K}", flush=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# Factored model speed (note: SVD factorization gives wrong tokens but RIGHT speed)
print(f"\n=== RANK-32 FACTORED (wall-clock only — tokens won't match SVD) ===", flush=True)
with torch.no_grad():
    model.generate(input_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    fact_out = model.generate(input_ids, max_new_tokens=N_GEN, do_sample=False)
torch.cuda.synchronize()
fact_time = time.time() - t0
fact_tps = N_GEN / fact_time
fact_text = tokenizer.decode(fact_out[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"  {fact_tps:.1f} tok/s ({N_GEN} in {fact_time:.1f}s)", flush=True)
print(f"  {fact_text[:80]}", flush=True)

speedup = fact_tps / base_tps
print(f"\n{'='*60}", flush=True)
print(f"WALL-CLOCK RESULT", flush=True)
print(f"  Baseline:     {base_tps:.1f} tok/s", flush=True)
print(f"  Rank-32:      {fact_tps:.1f} tok/s", flush=True)
print(f"  Speedup:      {speedup:.2f}×", flush=True)
print(f"  Compression:  {H/K:.0f}× (5120 → 32 dims)", flush=True)
print(f"{'='*60}", flush=True)

# Save
out = {"baseline_tps": base_tps, "factored_tps": fact_tps,
       "speedup": speedup, "rank": K, "n_gen": N_GEN}
Path("machines/strix_halo/results").mkdir(exist_ok=True)
with open("machines/strix_halo/results/wallclock_14b.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved wallclock_14b.json", flush=True)
