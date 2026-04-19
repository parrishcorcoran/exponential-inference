"""Holographic Matryoshka: cache teacher logits THEN factorize.

The critical fix: teacher logits are cached from the ORIGINAL unfactored
model BEFORE any factorization. The student then distills against these
cached logits, not against itself.

Per Finding 10: only factorize boundary weights (Q/K/V/O projections),
keep MLP intermediate (bulk) at full dimension.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, math, random, gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
device = 'cuda'
random.seed(42); torch.manual_seed(42)

K_MIN = 64; K_MAX = 128; STEPS = 4000; LR = 1e-4

print("="*70, flush=True)
print("HOLOGRAPHIC MATRYOSHKA: cached teacher → factorize → distill", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; N_LAYERS = model.config.num_hidden_layers
print(f"  H={H} L={N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# Load corpus
corpus = torch.load("machines/strix_halo/scratch/corpora/corpus.pt", weights_only=False)
batches = [s.unsqueeze(0) if s.dim()==1 else s for s in corpus["sequences"] if len(s)>=16]
print(f"  Corpus: {len(batches)} batches, {corpus['total_tokens']} tokens", flush=True)

# === STEP 1: CACHE ORIGINAL TEACHER LOGITS ===
print("\n[1/5] Caching ORIGINAL teacher logits (before factorization)...", flush=True)
t0 = time.time()
teacher_cache = []  # list of (input_ids, teacher_logits) on CPU
with torch.inference_mode():
    for i, batch in enumerate(batches):
        b = batch.to(device)
        logits = model(input_ids=b, use_cache=False).logits.detach().cpu()
        teacher_cache.append((batch, logits))
        if (i+1) % 100 == 0:
            print(f"  cached {i+1}/{len(batches)}  [{time.time()-t0:.0f}s]", flush=True)
print(f"  {len(teacher_cache)} batches cached in {time.time()-t0:.0f}s", flush=True)

# Save baseline generation
GEN_PROMPTS = [
    "The future of artificial intelligence will",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The French Revolution began in 1789 when",
]
orig_gens = {}
with torch.inference_mode():
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        orig_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  '{p[:35]}' → {orig_gens[p][:60]}", flush=True)

# Baseline speed
test_ids = tokenizer("The future of artificial intelligence", return_tensors='pt').input_ids.to(device)
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
baseline_tps = 64/(time.time()-t0)
print(f"  Baseline: {baseline_tps:.1f} tok/s", flush=True)

# Save original eval predictions
EVAL_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres.",
    "Topological insulators conduct electricity along their surface.",
    "The Higgs field gives mass to elementary particles.",
    "Vector clocks extend Lamport timestamps for distributed systems.",
    "The Curie temperature is where ferromagnets lose permanent magnetism.",
]
orig_preds = {}
with torch.inference_mode():
    for text in EVAL_TEXTS:
        ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
        orig_preds[text] = model(ids, use_cache=False).logits[0,:-1].argmax(-1).cpu()

# === STEP 2: PER-LAYER COVARIANCE + FACTORIZE ===
# Per Finding 10: ONLY boundary weights (Q/K/V/O), NOT MLP gate/up/down
print("\n[2/5] Per-layer covariances (boundary only)...", flush=True)
BOUNDARY_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
# Note: Finding 10 says MLP projections are boundary too (gate/up/down project
# from residual to bulk and back). But MLP INTERMEDIATE dim must stay full.
# The factorization targets the boundary-side of MLP: the d_model → d_int and
# d_int → d_model projections. Both are boundary weights.
ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

bases = {}; t0 = time.time()
for layer_idx in range(N_LAYERS):
    covs_layer = {}; handles = []
    prefix = f"model.layers.{layer_idx}."
    for name, mod in model.named_modules():
        if not name.startswith(prefix): continue
        last = name.rsplit(".",1)[-1]
        if isinstance(mod, nn.Linear) and last in ALL_TARGETS:
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
        for b in batches[:10]:
            model(input_ids=b.to(device), use_cache=False)
    for h in handles: h.remove()
    for n, c in covs_layer.items():
        ev, evec = torch.linalg.eigh(c.to(torch.float64))
        bases[n] = evec[:,-K_MAX:].flip(1).contiguous().to(torch.float32)
    del covs_layer; torch.cuda.empty_cache()
    if (layer_idx+1) % 10 == 0:
        print(f"  layer {layer_idx+1}/{N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
print(f"  {len(bases)} bases in {time.time()-t0:.0f}s", flush=True)

# Factorize
print("\n[3/5] Factorizing boundary weights...", flush=True)
from machines.strix_halo.scripts.train_matryoshka import (
    MatryoshkaFactoredLinear, RankController, freeze_non_factored
)
controller = RankController(K_MAX)
n_rep=0; fp=0; factp=0
for name, mod in list(model.named_modules()):
    for cn, child in list(mod.named_children()):
        if not isinstance(child, nn.Linear) or cn not in ALL_TARGETS: continue
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
print(f"  {n_rep} layers factored, {factp/1e6:.0f}M params  GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# === STEP 3: DISTILL AGAINST CACHED ORIGINAL LOGITS ===
print(f"\n[4/5] Distilling against CACHED ORIGINAL teacher ({STEPS} steps)...", flush=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
model.train()
t0 = time.time(); step = 0
lkmin, lkmax = math.log(K_MIN), math.log(K_MAX)
history = []

while step < STEPS:
    random.shuffle(teacher_cache)
    for batch_cpu, t_logits_cpu in teacher_cache:
        if step >= STEPS: break
        lr = LR*(step+1)/200 if step<200 else LR*0.5*(1+math.cos(math.pi*(step-200)/max(STEPS-200,1)))
        for g in opt.param_groups: g["lr"] = lr

        k = int(round(math.exp(random.uniform(lkmin, lkmax))))
        k = max(K_MIN, min(k, K_MAX))
        controller.global_k = k

        b = batch_cpu.to(device)
        t_logits = t_logits_cpu.to(device)  # ORIGINAL teacher logits

        s_logits = model(input_ids=b, use_cache=False).logits

        # KL divergence: student should match ORIGINAL teacher
        kl = F.kl_div(
            F.log_softmax(s_logits.float()/2, dim=-1),
            F.softmax(t_logits.float()/2, dim=-1),
            reduction="batchmean"
        ) * 4

        opt.zero_grad(); kl.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()

        if step % 200 == 0 or step == STEPS-1:
            print(f"  step {step:5d} k={k:4d} kl={kl.item():.4f} lr={lr:.2e} "
                  f"GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB [{time.time()-t0:.0f}s]", flush=True)
            history.append({"step":step,"k":k,"kl":float(kl.item())})
        step += 1

controller.global_k = None
model.eval()

# === STEP 4: EVALUATE vs ORIGINAL ===
print(f"\n[5/5] Evaluating factored vs ORIGINAL...", flush=True)
for k in [64, 96, 128]:
    controller.global_k = k
    total_m = 0; total_t = 0
    with torch.inference_mode():
        for text in EVAL_TEXTS:
            ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
            fact_preds = model(ids, use_cache=False).logits[0,:-1].argmax(-1).cpu()
            total_m += (fact_preds == orig_preds[text]).sum().item()
            total_t += len(orig_preds[text])
    print(f"  k={k:4d}  match vs ORIGINAL: {total_m/total_t*100:.1f}%", flush=True)

# Generation
controller.global_k = K_MIN
print(f"\n  Generation (k={K_MIN}):", flush=True)
fact_gens = {}
with torch.inference_mode():
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        fact_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  Original: {orig_gens[p][:70]}", flush=True)
        print(f"  Factored: {fact_gens[p][:70]}\n", flush=True)

# Wall-clock
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode(): model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
factored_tps = 64/(time.time()-t0)
controller.global_k = None

print(f"\n{'='*70}", flush=True)
print(f"HOLOGRAPHIC MATRYOSHKA RESULTS — Qwen3-14B", flush=True)
print(f"  Baseline:  {baseline_tps:.1f} tok/s", flush=True)
print(f"  Factored:  {factored_tps:.1f} tok/s ({factored_tps/baseline_tps:.2f}×)", flush=True)
print(f"{'='*70}", flush=True)

results = {
    "model": "Qwen3-14B", "k_min": K_MIN, "k_max": K_MAX, "steps": STEPS,
    "baseline_tps": baseline_tps, "factored_tps": factored_tps,
    "speedup": factored_tps/baseline_tps,
    "generation": {p: {"original": orig_gens[p], "factored": fact_gens[p]} for p in GEN_PROMPTS},
    "history": history,
}
with open("machines/strix_halo/results/holographic_cached_teacher_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved holographic_cached_teacher_14b.json", flush=True)
