"""Measure stabilization depth on Qwen3-14B.

At each layer, project through lm_head and check: does this layer's
prediction match the final layer's prediction?

This validates Finding 09 on 14B and tells us exactly how many layers
we can skip for each token category.
No hooks, no skipping, no format issues. Pure measurement.
"""
import torch, torch.nn.functional as F, time, json
device = 'cuda'

print("="*70, flush=True)
print("STABILIZATION DEPTH — Qwen3-14B", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to(device).eval()
N_LAYERS = model.config.num_hidden_layers
print(f"  L={N_LAYERS} GPU:{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# Capture hidden state at EVERY layer
layer_hiddens = {}
handles = []
for i in range(N_LAYERS):
    def make_hook(idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_hiddens[idx] = h.detach()
        return hook
    handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))

# Run diverse prompts
prompts = [
    "The future of artificial intelligence will transform the way we live and work in many",
    "In quantum mechanics particles can exist in superposition meaning they occupy multiple states",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(",
    "The French Revolution began in 1789 and fundamentally changed European politics when the people",
    "Water is composed of two hydrogen atoms and one oxygen atom bonded together through covalent bonds",
    "Machine learning algorithms learn patterns from data by minimizing a loss function using gradient",
    "Shakespeare wrote Hamlet which tells the story of a Danish prince seeking revenge against his",
    "The speed of light in a vacuum is approximately 299792458 meters per second which is a fundamental",
]

print(f"\nMeasuring per-layer stabilization on {len(prompts)} prompts...", flush=True)

all_results = []
for pi, prompt in enumerate(prompts):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    T = ids.shape[1]

    with torch.no_grad():
        layer_hiddens.clear()
        out = model(ids, use_cache=False, output_hidden_states=False)
        final_logits = out.logits[0]  # [T, V]
        final_preds = final_logits.argmax(-1)  # [T]

    # At each layer: project hidden through norm+lm_head, compare to final
    per_layer_match = []
    per_layer_conf = []
    for layer_idx in range(N_LAYERS):
        h = layer_hiddens[layer_idx][0]  # [T, H]
        with torch.no_grad():
            layer_logits = model.lm_head(model.model.norm(h))  # [T, V]
            layer_preds = layer_logits.argmax(-1)
            layer_conf = F.softmax(layer_logits.float(), dim=-1).max(-1).values.mean().item()

        match = (layer_preds == final_preds).float().mean().item() * 100
        per_layer_match.append(match)
        per_layer_conf.append(layer_conf)

    all_results.append({
        "prompt": prompt[:50],
        "n_tokens": T,
        "per_layer_match": per_layer_match,
        "per_layer_conf": per_layer_conf,
    })
    print(f"\n  Prompt {pi+1}: '{prompt[:45]}...' ({T} tokens)", flush=True)
    for l in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        if l < N_LAYERS:
            print(f"    layer {l:2d}: {per_layer_match[l]:5.1f}% match  conf={per_layer_conf[l]:.3f}", flush=True)

for h in handles: h.remove()

# Aggregate: at which layer do 90%+ of tokens match the final prediction?
print(f"\n{'='*70}", flush=True)
print(f"AGGREGATE STABILIZATION CURVE (avg across {len(prompts)} prompts)", flush=True)
print(f"{'='*70}", flush=True)
print(f"{'Layer':>6} {'Match%':>8} {'Conf':>8} {'Layers skippable':>18}", flush=True)
for l in range(N_LAYERS):
    avg_match = sum(r["per_layer_match"][l] for r in all_results) / len(all_results)
    avg_conf = sum(r["per_layer_conf"][l] for r in all_results) / len(all_results)
    skippable = N_LAYERS - l - 1
    marker = " ← 90%+" if avg_match >= 90 else " ← 80%+" if avg_match >= 80 else ""
    print(f"  {l:4d}   {avg_match:6.1f}%  {avg_conf:7.3f}  skip {skippable:2d} layers{marker}", flush=True)

# Save
with open("machines/strix_halo/results/stabilization_14b.json", "w") as f:
    json.dump({"model": "Qwen3-14B", "n_layers": N_LAYERS, "results": all_results}, f, indent=2)
print(f"\nSaved stabilization_14b.json", flush=True)
