"""Early exit ONLY on Qwen3-14B. No head pruning yet (GQA complicates hooks).

Tests Finding 09's stabilization_depth signal: exit when the prediction
has locked in. Measures wall-clock + quality.
"""
import torch, torch.nn.functional as F, time, json
from pathlib import Path

device = 'cuda'
print("="*70, flush=True)
print("EARLY EXIT on Qwen3-14B (40 layers)", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",
).to(device).eval()
N_LAYERS = model.config.num_hidden_layers  # 40
print(f"  L={N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

prompts = [
    "The future of artificial intelligence will",
    "In quantum mechanics the uncertainty principle states",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The French Revolution began in 1789 when",
    "To solve a quadratic equation ax² + bx + c = 0",
]
N_GEN = 64

# Baseline
print("\n--- Baseline ---", flush=True)
baseline_gens = {}
with torch.no_grad():
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
        baseline_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

test_ids = tokenizer("The future of artificial intelligence", return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): model.generate(test_ids, max_new_tokens=N_GEN, do_sample=False)
torch.cuda.synchronize()
baseline_tps = N_GEN / (time.time() - t0)
print(f"  {baseline_tps:.1f} tok/s", flush=True)
for p in prompts[:2]:
    print(f"  '{p[:40]}' → {baseline_gens[p][:60]}", flush=True)

# Early exit: check at multiple layers, skip remaining when confident
print("\n--- Early Exit Engine ---", flush=True)

# Sweep tau to find the sweet spot
for TAU in [0.99, 0.95, 0.90, 0.80]:
    CHECK_EVERY = 5  # check every 5 layers starting at layer 15
    CHECK_LAYERS = list(range(15, N_LAYERS, CHECK_EVERY))

    exit_state = {"layer": None, "hidden": None}

    def make_check(layer_idx):
        def hook(module, input, output):
            if exit_state["layer"] is not None:
                return  # already exited
            h = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                logits = model.lm_head(model.model.norm(h[:, -1:, :]))
                conf = F.softmax(logits.float(), dim=-1).max(-1).values.item()
            if conf > TAU:
                exit_state["layer"] = layer_idx
                exit_state["hidden"] = h
        return hook

    def make_skip(layer_idx):
        def hook(module, input, output):
            if exit_state["hidden"] is not None and exit_state["layer"] is not None:
                if layer_idx > exit_state["layer"]:
                    h = exit_state["hidden"]
                    return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    handles = []
    for cl in CHECK_LAYERS:
        handles.append(model.model.layers[cl].register_forward_hook(make_check(cl)))
    for i in range(CHECK_LAYERS[0] + 1, N_LAYERS):
        handles.append(model.model.layers[i].register_forward_hook(make_skip(i)))

    # Generate with early exit
    exits = 0; total = 0; layers_used = []
    ee_gens = {}
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
            gen_ids = ids.clone()
            for _ in range(N_GEN):
                exit_state["layer"] = None; exit_state["hidden"] = None
                out = model(gen_ids)
                tok = out.logits[0, -1:].argmax(-1)
                if exit_state["layer"] is not None:
                    exits += 1
                    layers_used.append(exit_state["layer"])
                else:
                    layers_used.append(N_LAYERS)
                total += 1
                gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
                if tok.item() == tokenizer.eos_token_id: break
            ee_gens[p] = tokenizer.decode(gen_ids[0][ids.shape[1]:], skip_special_tokens=True)

    # Speed measurement
    exit_state["layer"] = None; exit_state["hidden"] = None
    with torch.no_grad(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize(); t0 = time.time()
    gen_ids = test_ids.clone()
    with torch.no_grad():
        for _ in range(N_GEN):
            exit_state["layer"] = None; exit_state["hidden"] = None
            out = model(gen_ids)
            tok = out.logits[0, -1:].argmax(-1)
            gen_ids = torch.cat([gen_ids, tok.unsqueeze(0)], dim=-1)
    torch.cuda.synchronize()
    ee_tps = N_GEN / (time.time() - t0)

    for h in handles: h.remove()

    avg_l = sum(layers_used)/max(len(layers_used),1)
    skip_pct = (N_LAYERS - avg_l)/N_LAYERS * 100

    # Token match
    tm = 0; tt = 0
    for p in prompts:
        bt = tokenizer.encode(baseline_gens[p])
        et = tokenizer.encode(ee_gens[p])
        ml = min(len(bt), len(et))
        tm += sum(1 for a,b in zip(bt[:ml], et[:ml]) if a==b)
        tt += ml
    match = tm/max(tt,1)*100

    print(f"\n  τ={TAU:.2f}: {ee_tps:.1f} tok/s ({ee_tps/baseline_tps:.2f}×)  "
          f"match={match:.0f}%  exits={exits}/{total} ({exits/total*100:.0f}%)  "
          f"avg_layers={avg_l:.0f}/{N_LAYERS} (skip {skip_pct:.0f}%)", flush=True)
    for p in prompts[:2]:
        same = "✓" if baseline_gens[p][:40] == ee_gens[p][:40] else "≠"
        print(f"    {same} {ee_gens[p][:65]}", flush=True)

print(f"\n  Baseline: {baseline_tps:.1f} tok/s", flush=True)

# Save best result
results = {"model": "Qwen3-14B", "n_layers": N_LAYERS,
           "baseline_tps": baseline_tps, "tau_sweep": TAU}
with open("machines/strix_halo/results/early_exit_14b.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved early_exit_14b.json", flush=True)
