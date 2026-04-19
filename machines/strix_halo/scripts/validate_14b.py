"""Full validation: distill + compare factored vs ORIGINAL unfactored.

1. Load original model, generate baseline text + compute baseline predictions
2. Factorize + distill (reuses the working pipeline)
3. Compare: factored-rank-32 vs ORIGINAL (not vs factored-rank-128)
4. MMLU-style eval (5-shot multiple choice)
5. Generation samples side by side
6. Wall-clock speedup
7. Save everything for plots
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, math, random, gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
device = 'cuda'
random.seed(42); torch.manual_seed(42)
RANK = 32; K_MAX = 128; STEPS = 2000; LR = 1e-4

print("="*70, flush=True)
print("FULL VALIDATION: factored vs ORIGINAL Qwen3-14B", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)

# === 1. ORIGINAL MODEL BASELINE ===
print("\n[1/7] Loading original model + baseline...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
H = model.config.hidden_size; N_LAYERS = model.config.num_hidden_layers
print(f"  H={H} L={N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# Baseline generation samples
GEN_PROMPTS = [
    "The future of artificial intelligence will",
    "In quantum mechanics, the uncertainty principle states that",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The French Revolution began in 1789 when",
    "To solve a quadratic equation of the form ax² + bx + c = 0,",
]

EVAL_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres across North America.",
    "Topological insulators behave as insulators in their interior but conduct electricity along their surface.",
    "Recombinant DNA technology emerged in the 1970s with the discovery of restriction enzymes.",
    "The Antikythera mechanism is an ancient analog computer from the second century BCE.",
    "Edge-triggered flip-flops store one bit and change state only on the clock edge.",
    "The Curie temperature is where a ferromagnetic material loses its permanent magnetism.",
    "Vector clocks extend Lamport timestamps for distributed system event ordering.",
    "Germanium is a metalloid in the carbon group widely used in fiber-optic systems.",
    "The Higgs field gives mass to elementary particles through spontaneous symmetry breaking.",
    "Operational amplifiers implement signal processing through high gain and negative feedback.",
    "Climate change is primarily driven by greenhouse gas emissions from fossil fuel combustion.",
    "The human genome contains approximately 3 billion base pairs encoding 20000 protein-coding genes.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy from the sun.",
    "The theory of general relativity describes gravity as the curvature of spacetime by mass and energy.",
    "Machine learning algorithms optimize parameters by minimizing a loss function via gradient descent.",
]

# MMLU-style questions (5 diverse topics)
MMLU_QS = [
    {"q": "What is the capital of Australia?", "choices": ["Sydney", "Melbourne", "Canberra", "Perth"], "answer": 2},
    {"q": "Which element has atomic number 79?", "choices": ["Silver", "Gold", "Platinum", "Copper"], "answer": 1},
    {"q": "What does HTTP stand for?", "choices": ["HyperText Transfer Protocol", "High Tech Transfer Protocol", "HyperText Transmission Process", "High Transfer Text Protocol"], "answer": 0},
    {"q": "Who wrote 'The Origin of Species'?", "choices": ["Newton", "Einstein", "Darwin", "Galileo"], "answer": 2},
    {"q": "What is the derivative of x²?", "choices": ["x", "2x", "x²", "2x²"], "answer": 1},
    {"q": "Which planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": 2},
    {"q": "What is the speed of light in m/s?", "choices": ["3×10⁶", "3×10⁸", "3×10¹⁰", "3×10⁴"], "answer": 1},
    {"q": "What year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": 2},
    {"q": "Which data structure uses LIFO?", "choices": ["Queue", "Stack", "Heap", "Array"], "answer": 1},
    {"q": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi"], "answer": 2},
]

print("\n  Baseline generation samples:", flush=True)
baseline_gens = {}
with torch.no_grad():
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        baseline_gens[p] = text
        print(f"    '{p[:40]}...' → {text[:60]}", flush=True)

# Baseline predictions on eval texts
print("\n  Baseline eval predictions...", flush=True)
baseline_preds = {}
with torch.no_grad():
    for text in EVAL_TEXTS:
        ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
        logits = model(ids, use_cache=False).logits[0, :-1]
        baseline_preds[text] = logits.argmax(-1).cpu()

# Baseline MMLU
print("  Baseline MMLU...", flush=True)
baseline_mmlu = 0
with torch.no_grad():
    for q in MMLU_QS:
        prompt = f"Question: {q['q']}\nChoices:\n"
        for i, c in enumerate(q['choices']):
            prompt += f"  {chr(65+i)}. {c}\n"
        prompt += "Answer: "
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        logits = model(ids, use_cache=False).logits[0, -1]
        # Check which choice letter has highest logit
        choice_ids = [tokenizer.encode(chr(65+i))[-1] for i in range(4)]
        choice_logits = logits[choice_ids]
        pred = choice_logits.argmax().item()
        if pred == q['answer']:
            baseline_mmlu += 1
print(f"  Baseline MMLU: {baseline_mmlu}/{len(MMLU_QS)} ({baseline_mmlu/len(MMLU_QS)*100:.0f}%)", flush=True)

# Baseline wall-clock
print("\n  Baseline wall-clock...", flush=True)
test_ids = tokenizer("The future of artificial intelligence", return_tensors='pt').input_ids.to(device)
with torch.no_grad():
    model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
baseline_tps = 64 / (time.time() - t0)
print(f"  Baseline: {baseline_tps:.1f} tok/s", flush=True)

# === 2-4. COVARIANCE + FACTORIZE + DISTILL (reuse pipeline) ===
print("\n[2/7] Collecting covariances (per-layer)...", flush=True)
for p in model.parameters(): p.requires_grad_(False)

from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
batches = []
total_tokens = 0
for i, item in enumerate(ds):
    if len(item['text']) < 200: continue
    ids = tokenizer(item['text'][:2000], return_tensors='pt', truncation=True, max_length=256).input_ids
    if ids.shape[1] >= 16:
        batches.append(ids)
        total_tokens += ids.shape[1]
    if total_tokens >= 50000: break
print(f"  {len(batches)} batches, {total_tokens} tokens", flush=True)

TARGET = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
bases = {}
t0 = time.time()
n_cal = 10
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
        for b in batches[:n_cal]:
            model(input_ids=b.to(device), use_cache=False)
    for h in handles: h.remove()
    for n, c in covs_layer.items():
        ev, evec = torch.linalg.eigh(c.to(torch.float64))
        bases[n] = evec[:,-K_MAX:].flip(1).contiguous().to(torch.float32)
    del covs_layer; torch.cuda.empty_cache()
    if (layer_idx+1) % 10 == 0:
        print(f"  layer {layer_idx+1}/{N_LAYERS}  GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
print(f"  {len(bases)} bases in {time.time()-t0:.0f}s", flush=True)

print("\n[3/7] Factorizing...", flush=True)
from machines.strix_halo.scripts.train_matryoshka import (
    MatryoshkaFactoredLinear, RankController, freeze_non_factored
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
print(f"  {n_rep} layers, {factp/1e6:.0f}M factored  GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

print("\n[4/7] Distilling (2000 steps)...", flush=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
t0 = time.time(); step = 0
lkmin, lkmax = math.log(RANK), math.log(K_MAX)
model.train()
history = []
while step < STEPS:
    random.shuffle(batches)
    for batch in batches:
        if step >= STEPS: break
        lr = LR*(step+1)/100 if step<100 else LR*0.5*(1+math.cos(math.pi*(step-100)/max(STEPS-100,1)))
        for g in opt.param_groups: g["lr"] = lr
        k = int(round(math.exp(random.uniform(lkmin, lkmax))))
        k = max(RANK, min(k, K_MAX))
        b = batch.to(device)
        controller.global_k = K_MAX
        with torch.inference_mode():
            t_logits = model(b, use_cache=False).logits.detach()
        controller.global_k = k
        s_logits = model(b, use_cache=False).logits
        kl = F.kl_div(F.log_softmax(s_logits.float()/2,-1),
                       F.softmax(t_logits.float()/2,-1), reduction="batchmean")*4
        opt.zero_grad(); kl.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        if step%200==0 or step==STEPS-1:
            print(f"  step {step:5d} k={k:4d} kl={kl.item():.4f} [{time.time()-t0:.0f}s]", flush=True)
            history.append({"step":step,"k":k,"kl":float(kl.item())})
        step += 1
controller.global_k = None
model.eval()

# === 5. COMPARE FACTORED vs ORIGINAL ===
print("\n[5/7] Comparing factored-rank-32 vs ORIGINAL...", flush=True)
controller.global_k = RANK

# Token match on eval texts
total_match = 0; total_tok = 0
with torch.inference_mode():
    for text in EVAL_TEXTS:
        ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
        fact_preds = model(ids, use_cache=False).logits[0, :-1].argmax(-1).cpu()
        orig_preds = baseline_preds[text]
        total_match += (fact_preds == orig_preds).sum().item()
        total_tok += len(orig_preds)

match_vs_original = total_match / total_tok * 100
print(f"  Token match vs ORIGINAL: {match_vs_original:.1f}% ({total_match}/{total_tok})", flush=True)

# === 6. MMLU on factored model ===
print("\n[6/7] MMLU on factored model...", flush=True)
factored_mmlu = 0
with torch.inference_mode():
    for q in MMLU_QS:
        prompt = f"Question: {q['q']}\nChoices:\n"
        for i, c in enumerate(q['choices']):
            prompt += f"  {chr(65+i)}. {c}\n"
        prompt += "Answer: "
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        logits = model(ids, use_cache=False).logits[0, -1]
        choice_ids = [tokenizer.encode(chr(65+i))[-1] for i in range(4)]
        pred = logits[choice_ids].argmax().item()
        if pred == q['answer']:
            factored_mmlu += 1
print(f"  Factored MMLU: {factored_mmlu}/{len(MMLU_QS)} ({factored_mmlu/len(MMLU_QS)*100:.0f}%)", flush=True)
print(f"  Baseline MMLU: {baseline_mmlu}/{len(MMLU_QS)} ({baseline_mmlu/len(MMLU_QS)*100:.0f}%)", flush=True)

# === 7. Generation samples + wall-clock ===
print("\n[7/7] Generation samples + wall-clock...", flush=True)
factored_gens = {}
with torch.inference_mode():
    for p in GEN_PROMPTS:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=50, do_sample=False)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        factored_gens[p] = text

# Wall-clock
with torch.inference_mode():
    model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.inference_mode():
    model.generate(test_ids, max_new_tokens=64, do_sample=False)
torch.cuda.synchronize()
factored_tps = 64 / (time.time() - t0)
speedup = factored_tps / baseline_tps

controller.global_k = None

# === FINAL REPORT ===
print(f"\n{'='*70}", flush=True)
print(f"FULL VALIDATION REPORT — Qwen3-14B rank-32", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Token match vs ORIGINAL:  {match_vs_original:.1f}%", flush=True)
print(f"  MMLU baseline:            {baseline_mmlu}/{len(MMLU_QS)} ({baseline_mmlu/len(MMLU_QS)*100:.0f}%)", flush=True)
print(f"  MMLU factored:            {factored_mmlu}/{len(MMLU_QS)} ({factored_mmlu/len(MMLU_QS)*100:.0f}%)", flush=True)
print(f"  Wall-clock baseline:      {baseline_tps:.1f} tok/s", flush=True)
print(f"  Wall-clock factored:      {factored_tps:.1f} tok/s", flush=True)
print(f"  Speedup:                  {speedup:.2f}×", flush=True)
print(f"  Compression:              {H/RANK:.0f}×", flush=True)
print(f"\n  Generation comparison:", flush=True)
for p in GEN_PROMPTS:
    print(f"  Prompt: '{p[:50]}...'", flush=True)
    print(f"    Original: {baseline_gens[p][:70]}", flush=True)
    print(f"    Factored: {factored_gens[p][:70]}", flush=True)
    print(flush=True)

# Save all results
results = {
    "model": "Qwen3-14B", "rank": RANK, "k_max": K_MAX,
    "match_vs_original_pct": match_vs_original,
    "mmlu_baseline": baseline_mmlu, "mmlu_factored": factored_mmlu,
    "mmlu_total": len(MMLU_QS),
    "baseline_tps": baseline_tps, "factored_tps": factored_tps,
    "speedup": speedup, "compression": H/RANK,
    "factored_params_M": factp/1e6,
    "generation_samples": {p: {"original": baseline_gens[p], "factored": factored_gens[p]} for p in GEN_PROMPTS},
    "history": history,
}
with open("machines/strix_halo/results/validation_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved validation_14b.json", flush=True)
