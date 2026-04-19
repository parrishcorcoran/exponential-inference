"""Strix Halo: OOM-safe 14B training with proper teacher-vs-student eval.

Combines:
- Per-layer covariance collection (avoids 28GB CPU OOM)
- train_matryoshka.py factorization + distillation (proper Matryoshka)
- Eval: factored student vs ORIGINAL unfactored teacher
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, math, random, gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
device = 'cuda'
random.seed(42); torch.manual_seed(42)

K_MIN = 64; K_MAX = 128; STEPS = 8000; LR = 1e-4

print("="*70, flush=True)
print("STRIX: Qwen3-14B Holographic Matryoshka (OOM-safe, proper eval)", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model + corpus
print("\n[1/5] Loading model + corpus...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; N_LAYERS = model.config.num_hidden_layers
print(f"  H={H} L={N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

corpus = torch.load("machines/strix_halo/scratch/corpora/corpus.pt", weights_only=False)
batches = [s.unsqueeze(0) if s.dim()==1 else s for s in corpus["sequences"] if len(s)>=16]
print(f"  Corpus: {len(batches)} batches, {corpus['total_tokens']} tokens", flush=True)

# Baseline: save original predictions on held-out texts
EVAL_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres.",
    "Topological insulators conduct electricity along their surface but not interior.",
    "The Higgs field gives mass to elementary particles through symmetry breaking.",
    "Vector clocks extend Lamport timestamps for distributed system event ordering.",
    "The Curie temperature is where a ferromagnetic material loses permanent magnetism.",
    "Recombinant DNA technology emerged in the 1970s with restriction enzymes.",
    "The Antikythera mechanism is an ancient analog computer from the second century BCE.",
    "Edge-triggered flip-flops store one bit and change state only on the clock edge.",
]
GEN_PROMPTS = [
    "The future of artificial intelligence will",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The French Revolution began in 1789 when",
]

print("\n  Saving original baseline...", flush=True)
orig_preds = {}; orig_gens = {}
with torch.inference_mode():
    for text in EVAL_TEXTS:
        ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
        orig_preds[text] = model(ids, use_cache=False).logits[0,:-1].argmax(-1).cpu()
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        orig_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"    '{p[:40]}' → {orig_gens[p][:60]}", flush=True)

# Baseline wall-clock
test_ids = tokenizer("The future of artificial intelligence", return_tensors='pt').input_ids.to(device)
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
baseline_tps = 64 / (time.time() - t0)
print(f"  Baseline: {baseline_tps:.1f} tok/s", flush=True)

# [2/5] Per-layer covariance (OOM-safe)
print("\n[2/5] Per-layer covariances...", flush=True)
TARGET = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
bases = {}; t0 = time.time(); N_CAL = min(20, len(batches))

for layer_idx in range(N_LAYERS):
    covs_layer = {}; handles = []
    prefix = f"model.layers.{layer_idx}."
    for name, mod in model.named_modules():
        if not name.startswith(prefix): continue
        last = name.rsplit(".",1)[-1]
        if isinstance(mod, nn.Linear) and last in TARGET:
            def mk(n, d):
                def hook(m, inp, out):
                    x = inp[0].detach().reshape(-1, d).to(torch.float32)
                    if n not in covs_layer:
                        covs_layer[n] = torch.zeros(d,d,dtype=torch.float32,device=device)
                    covs_layer[n] += x.T @ x
                return hook
            handles.append(mod.register_forward_hook(mk(name, mod.in_features)))
    if not handles: continue
    with torch.inference_mode():
        for b in batches[:N_CAL]:
            model(input_ids=b.to(device), use_cache=False)
    for h in handles: h.remove()
    for n, c in covs_layer.items():
        ev, evec = torch.linalg.eigh(c.to(torch.float64))
        bases[n] = evec[:,-K_MAX:].flip(1).contiguous().to(torch.float32)
    del covs_layer; torch.cuda.empty_cache()
    if (layer_idx+1) % 10 == 0:
        print(f"  layer {layer_idx+1}/{N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
print(f"  {len(bases)} bases in {time.time()-t0:.0f}s", flush=True)

# [3/5] Factorize
print("\n[3/5] Factorizing...", flush=True)
from machines.strix_halo.scripts.train_matryoshka import (
    MatryoshkaFactoredLinear, RankController, freeze_non_factored,
    matryoshka_distill, distribution_eval_at_k
)
controller = RankController(K_MAX)
n_rep=0; fp=0; factp=0
for name, mod in list(model.named_modules()):
    for cn, child in list(mod.named_children()):
        if not isinstance(child, nn.Linear) or cn not in TARGET: continue
        fn = f"{name}.{cn}" if name else cn
        if fn not in bases: continue
        try: li = int(fn.split("model.layers.")[1].split(".")[0])
        except: li = -1
        fact = MatryoshkaFactoredLinear(child, bases[fn], controller, li, True)
        setattr(mod, cn, fact)
        fp += child.in_features*child.out_features
        factp += K_MAX*(child.in_features+child.out_features)
        n_rep += 1; del child
gc.collect(); torch.cuda.empty_cache()
trainable = freeze_non_factored(model)
print(f"  {n_rep} layers, {factp/1e6:.0f}M factored ({factp/fp:.1%})", flush=True)
print(f"  Trainable: {trainable/1e6:.1f}M  GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# [4/5] Distill — model is BOTH teacher (K_MAX) and student (sampled k)
print(f"\n[4/5] Distilling {STEPS} steps...", flush=True)
history = matryoshka_distill(
    model, model, batches, STEPS, LR, device,
    controller, K_MIN, K_MAX, log_every=200, warmup=200)

# [5/5] Evaluate factored vs ORIGINAL
print(f"\n[5/5] Evaluating factored vs ORIGINAL...", flush=True)
model.eval()

for k in [64, 96, 128]:
    controller.global_k = k
    total_m = 0; total_t = 0
    with torch.inference_mode():
        for text in EVAL_TEXTS:
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
            fact_preds = model(ids, use_cache=False).logits[0,:-1].argmax(-1).cpu()
            orig = orig_preds[text]
            total_m += (fact_preds == orig).sum().item()
            total_t += len(orig)
    match = total_m / total_t * 100
    print(f"  k={k:4d}  match vs ORIGINAL: {match:.1f}%", flush=True)

# Generation comparison
controller.global_k = K_MIN
print(f"\n  Generation comparison (k={K_MIN}):", flush=True)
fact_gens = {}
with torch.inference_mode():
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        fact_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"    Original: {orig_gens[p][:70]}", flush=True)
        print(f"    Factored: {fact_gens[p][:70]}", flush=True)
        print(flush=True)

# Wall-clock factored
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
factored_tps = 64 / (time.time() - t0)
controller.global_k = None

print(f"\n{'='*70}", flush=True)
print(f"RESULTS: Qwen3-14B Holographic Matryoshka k=[{K_MIN},{K_MAX}]", flush=True)
print(f"  Baseline:  {baseline_tps:.1f} tok/s", flush=True)
print(f"  Factored:  {factored_tps:.1f} tok/s  ({factored_tps/baseline_tps:.2f}×)", flush=True)
print(f"  Params:    {factp/1e6:.0f}M factored ({factp/fp:.1%})", flush=True)
print(f"{'='*70}", flush=True)

# Save
results = {
    "model": "Qwen3-14B", "k_min": K_MIN, "k_max": K_MAX, "steps": STEPS,
    "baseline_tps": baseline_tps, "factored_tps": factored_tps,
    "speedup": factored_tps/baseline_tps,
    "factored_params_M": factp/1e6, "ratio": factp/fp,
    "generation_samples": {p: {"original": orig_gens[p], "factored": fact_gens[p]} for p in GEN_PROMPTS},
    "history": history,
}
with open("machines/strix_halo/results/holographic_14b_r64_128.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved holographic_14b_r64_128.json", flush=True)
