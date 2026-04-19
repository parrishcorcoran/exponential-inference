"""Strix Halo: Qwen3-14B Matryoshka distillation at rank-32→128.

Generates corpus from C4, collects covariances, factorizes in-place,
distills via KL, evaluates at multiple ranks. Single model copy.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import time, json, math, random, gc, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

device = 'cuda'
random.seed(42); torch.manual_seed(42)
RANK = 32; K_MAX = 128; STEPS = 2000; LR = 1e-4

print("="*70, flush=True)
print("STRIX HALO: Qwen3-14B rank-32→128 distillation", flush=True)
print("="*70, flush=True)

# Step 1: Load
print("\n[1/6] Loading Qwen3-14B...", flush=True)
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
print(f"  H={H} L={N_LAYERS} loaded in {time.time()-t0:.0f}s", flush=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# Step 2: Corpus
print("\n[2/6] Generating corpus from C4...", flush=True)
from datasets import load_dataset
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
batches = []
total_tokens = 0
for i, item in enumerate(ds):
    if len(item['text']) < 200: continue
    ids = tokenizer(item['text'][:2000], return_tensors='pt',
                    truncation=True, max_length=256).input_ids
    if ids.shape[1] >= 16:
        batches.append(ids)
        total_tokens += ids.shape[1]
    if total_tokens >= 50000: break
    if (i+1) % 100 == 0:
        print(f"  {total_tokens} tokens...", flush=True)
print(f"  {len(batches)} batches, {total_tokens} tokens", flush=True)

# Step 3: Covariances
print("\n[3/6] Collecting covariances...", flush=True)
t0 = time.time()
TARGET = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
# Collect covariances ONE LAYER AT A TIME to avoid OOM
# Each cov is 5120×5120 float32 = 100MB. 280 at once = 28GB.
# Instead: process one layer group at a time, compute PCA, free cov.
bases = {}
n_cal_batches = 10  # fewer batches for memory safety

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
                    # Keep on GPU to avoid CPU OOM, use float32
                    if n not in covs_layer:
                        covs_layer[n] = torch.zeros(d,d,dtype=torch.float32,device=device)
                    covs_layer[n] += x.T @ x
                return hook
            handles.append(mod.register_forward_hook(mk(name, mod.in_features)))

    if not handles: continue
    with torch.inference_mode():
        for b in batches[:n_cal_batches]:
            model(input_ids=b.to(device), use_cache=False)
    for h in handles: h.remove()

    # Compute PCA for this layer's projections immediately, then free covs
    for n, c in covs_layer.items():
        ev, evec = torch.linalg.eigh(c.to(torch.float64))
        bases[n] = evec[:,-K_MAX:].flip(1).contiguous().to(torch.float32)
    del covs_layer; torch.cuda.empty_cache()

    if (layer_idx+1) % 10 == 0:
        print(f"  layer {layer_idx+1}/{N_LAYERS}  bases={len(bases)}  "
              f"GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

print(f"  {len(bases)} bases in {time.time()-t0:.0f}s", flush=True)

# Step 4: Factorize
print("\n[4/6] Factorizing...", flush=True)
from machines.strix_halo.scripts.train_matryoshka import (
    MatryoshkaFactoredLinear, RankController, freeze_non_factored
)
# bases already computed per-layer above

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

# Step 5: Distill
print("\n[5/6] Distilling...", flush=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
history = []; t0 = time.time(); step = 0
lkmin, lkmax = math.log(RANK), math.log(K_MAX)
model.train()

while step < STEPS:
    random.shuffle(batches)
    for batch in batches:
        if step >= STEPS: break
        lr = LR * (step+1)/100 if step<100 else LR*0.5*(1+math.cos(math.pi*(step-100)/max(STEPS-100,1)))
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
        if step%100==0 or step==STEPS-1:
            print(f"  step {step:5d} k={k:4d} kl={kl.item():.4f} lr={lr:.2e} "
                  f"gpu={torch.cuda.memory_allocated()/1e9:.1f}GB [{time.time()-t0:.0f}s]", flush=True)
            history.append({"step":step,"k":k,"kl":float(kl.item())})
        step += 1
controller.global_k = None

# Step 6: Eval
print("\n[6/6] Evaluating...", flush=True)
model.eval()
texts = ["The migratory patterns of monarch butterflies span thousands of kilometres.",
         "Topological insulators conduct electricity along their surface.",
         "The Higgs field gives mass to elementary particles.",
         "Vector clocks extend Lamport timestamps for distributed systems.",
         "The Curie temperature is where ferromagnets lose permanent magnetism."]
for k in [32, 48, 64, 96, 128]:
    tm=0; tt=0
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
            controller.global_k = K_MAX
            tp = model(ids, use_cache=False).logits[0,:-1].argmax(-1)
            controller.global_k = k
            sp = model(ids, use_cache=False).logits[0,:-1].argmax(-1)
            tm += (tp==sp).sum().item(); tt += len(tp)
    print(f"  k={k:4d} match={tm/tt*100:.1f}% ({H/k:.0f}× compression)", flush=True)

out_path = Path("/home/cpinchington/Exponential-Inference/machines/strix_halo/results/qwen3_14b_r32_128.json")
with open(out_path, "w") as f:
    json.dump({"model":"Qwen3-14B","H":H,"L":N_LAYERS,"k_min":RANK,"k_max":K_MAX,
               "steps":STEPS,"tokens":total_tokens,"factored_M":factp/1e6,
               "ratio":factp/fp,"history":history}, f, indent=2)
print(f"\nSaved {out_path}", flush=True)
