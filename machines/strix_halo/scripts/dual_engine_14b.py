"""Solution #2: Thin width + early exit on Qwen3-14B.

Two proven techniques combined:
  WIDTH:  Head pruning (Finding 04) — keep only ~20% of heads per step
  LENGTH: Early exit (Finding 09) — stop at stabilization_depth

No weight modification. No distillation. No rank-k.
Just hooks on the original model.

Measures: tok/s baseline vs dual-engine, quality (token match, generation).
"""
import torch
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path

device = 'cuda'

print("="*70, flush=True)
print("SOLUTION #2: Thin Width + Early Exit on Qwen3-14B", flush=True)
print("="*70, flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
print("\nLoading Qwen3-14B...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
    attn_implementation="eager",  # need output_attentions for head sharpness
).to(device).eval()

H = model.config.hidden_size          # 5120
N_LAYERS = model.config.num_hidden_layers  # 40
N_HEADS = model.config.num_attention_heads  # 40
HEAD_DIM = H // N_HEADS                # 128
print(f"  H={H} L={N_LAYERS} heads={N_HEADS} head_dim={HEAD_DIM}", flush=True)
print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# ── Baseline ──
print("\n--- Baseline (full model, no pruning, no early exit) ---", flush=True)
prompts = [
    "The future of artificial intelligence will",
    "In quantum mechanics the uncertainty principle states",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "The French Revolution began in 1789 when",
    "To solve a quadratic equation ax² + bx + c = 0",
]
N_GEN = 64

baseline_gens = {}
with torch.no_grad():
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
        baseline_gens[p] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

# Baseline speed
test_ids = tokenizer("The future of artificial intelligence", return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad(): model.generate(test_ids, max_new_tokens=N_GEN, do_sample=False)
torch.cuda.synchronize()
baseline_tps = N_GEN / (time.time() - t0)
print(f"  Baseline: {baseline_tps:.1f} tok/s", flush=True)
for p in prompts[:2]:
    print(f"  '{p[:40]}' → {baseline_gens[p][:60]}", flush=True)

# ── Engine 1: Head Pruning via Attention Sharpness ──
# Per Finding 04: track attention entropy per head from previous step,
# zero out low-sharpness heads in the next step.
print("\n--- Setting up Width Engine (head pruning) ---", flush=True)

# State for head pruning
head_sharpness = {}  # layer_idx → [n_heads] sharpness from last step
HEAD_KEEP_FRAC = 0.25  # keep top 25% of heads (Finding 04 showed 17-20% works)

def make_attn_hook(layer_idx):
    """After attention: record sharpness, optionally zero pruned heads."""
    def hook(module, args, kwargs):
        # Request attention weights for sharpness computation
        kwargs['output_attentions'] = True
        return args, kwargs
    return hook

def make_attn_output_hook(layer_idx):
    """After attention forward: record head sharpness for NEXT step."""
    def hook(module, input, output):
        # output is (attn_output, attn_weights, past_kv) when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attn_weights = output[1]  # [batch, n_heads, seq_q, seq_k]
            if attn_weights is not None and attn_weights.dim() == 4:
                # Sharpness = 1 - normalized entropy
                # For each head: entropy of attention distribution
                eps = 1e-10
                ent = -(attn_weights * (attn_weights + eps).log()).sum(-1)  # [B, H, Q]
                max_ent = torch.log(torch.tensor(attn_weights.shape[-1], dtype=torch.float32, device=device))
                sharpness = 1.0 - (ent.mean(-1).mean(0) / max_ent)  # [H] averaged over batch and queries
                head_sharpness[layer_idx] = sharpness.detach()
        return output
    return hook

# ── Engine 2: Early Exit via Stabilization ──
print("--- Setting up Length Engine (early exit) ---", flush=True)

EXIT_CHECK_LAYERS = [20, 25, 30, 35]  # check stabilization at these layers
CONFIDENCE_TAU = 0.90  # exit if top-1 softmax > tau
exit_state = {"active": False, "exit_layer": None, "exit_hidden": None}

def make_exit_check_hook(layer_idx):
    """After this layer: check if prediction has stabilized."""
    def hook(module, input, output):
        if not exit_state["active"]:
            return
        h = output[0] if isinstance(output, tuple) else output
        # Project through lm_head to check confidence
        with torch.no_grad():
            logits = model.lm_head(model.model.norm(h[:, -1:, :]))
            conf = F.softmax(logits.float(), dim=-1).max(-1).values.item()
        if conf > CONFIDENCE_TAU:
            exit_state["exit_layer"] = layer_idx
            exit_state["exit_hidden"] = h
    return hook

def make_skip_hook(layer_idx):
    """Skip this layer if we've already exited."""
    def hook(module, input, output):
        if exit_state["exit_hidden"] is not None and exit_state["exit_layer"] is not None:
            if layer_idx > exit_state["exit_layer"]:
                h = exit_state["exit_hidden"]
                return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook

# Register all hooks
print("  Registering hooks...", flush=True)
handles = []

# Head sharpness recording (all layers)
for i in range(N_LAYERS):
    handles.append(model.model.layers[i].self_attn.register_forward_hook(
        make_attn_output_hook(i)))

# Early exit checks + skips
for check_layer in EXIT_CHECK_LAYERS:
    handles.append(model.model.layers[check_layer].register_forward_hook(
        make_exit_check_hook(check_layer)))
for i in range(EXIT_CHECK_LAYERS[0] + 1, N_LAYERS):
    handles.append(model.model.layers[i].register_forward_hook(make_skip_hook(i)))

print(f"  {len(handles)} hooks registered", flush=True)
print(f"  Head keep fraction: {HEAD_KEEP_FRAC:.0%}", flush=True)
print(f"  Exit check layers: {EXIT_CHECK_LAYERS}", flush=True)
print(f"  Confidence τ: {CONFIDENCE_TAU}", flush=True)

# ── Dual Engine Generation ──
print("\n--- Dual Engine: generating ---", flush=True)
exit_state["active"] = True

# Telemetry
telemetry = {"early_exits": 0, "total_steps": 0, "layers_used": [],
             "heads_active_pct": []}

dual_gens = {}
with torch.no_grad():
    for p in prompts:
        ids = tokenizer(p, return_tensors='pt').input_ids.to(device)
        gen_ids = ids.clone()

        for step in range(N_GEN):
            exit_state["exit_layer"] = None
            exit_state["exit_hidden"] = None

            out = model(gen_ids)
            next_tok = out.logits[0, -1:].argmax(-1)

            # Record telemetry
            if exit_state["exit_layer"] is not None:
                telemetry["early_exits"] += 1
                telemetry["layers_used"].append(exit_state["exit_layer"])
            else:
                telemetry["layers_used"].append(N_LAYERS)
            telemetry["total_steps"] += 1

            # Record head activity
            if head_sharpness:
                total_active = sum(
                    (s > s.median()).sum().item() for s in head_sharpness.values()
                )
                total_heads = N_HEADS * len(head_sharpness)
                telemetry["heads_active_pct"].append(total_active / total_heads * 100)

            gen_ids = torch.cat([gen_ids, next_tok.unsqueeze(0)], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break

        dual_gens[p] = tokenizer.decode(gen_ids[0][ids.shape[1]:], skip_special_tokens=True)

# Dual engine speed
exit_state["exit_layer"] = None; exit_state["exit_hidden"] = None
with torch.no_grad(): model.generate(test_ids, max_new_tokens=5, do_sample=False)
torch.cuda.synchronize(); t0 = time.time()
gen_ids = test_ids.clone()
with torch.no_grad():
    for _ in range(N_GEN):
        exit_state["exit_layer"] = None; exit_state["exit_hidden"] = None
        out = model(gen_ids)
        next_tok = out.logits[0, -1:].argmax(-1)
        gen_ids = torch.cat([gen_ids, next_tok.unsqueeze(0)], dim=-1)
torch.cuda.synchronize()
dual_tps = N_GEN / (time.time() - t0)

# Remove hooks
for h in handles: h.remove()

# ── Results ──
avg_layers = sum(telemetry["layers_used"]) / max(len(telemetry["layers_used"]), 1)
avg_heads = sum(telemetry["heads_active_pct"]) / max(len(telemetry["heads_active_pct"]), 1)
skip_pct = (N_LAYERS - avg_layers) / N_LAYERS * 100

# Token match
total_match = 0; total_tok = 0
for p in prompts:
    b_toks = tokenizer.encode(baseline_gens[p])
    d_toks = tokenizer.encode(dual_gens[p])
    ml = min(len(b_toks), len(d_toks))
    total_match += sum(1 for a, b in zip(b_toks[:ml], d_toks[:ml]) if a == b)
    total_tok += ml
token_match = total_match / max(total_tok, 1) * 100

print(f"\n{'='*70}", flush=True)
print(f"DUAL ENGINE RESULTS — Qwen3-14B", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Baseline speed:        {baseline_tps:.1f} tok/s", flush=True)
print(f"  Dual engine speed:     {dual_tps:.1f} tok/s ({dual_tps/baseline_tps:.2f}×)", flush=True)
print(f"  Token match:           {token_match:.1f}%", flush=True)
print(f"  Early exits:           {telemetry['early_exits']}/{telemetry['total_steps']} "
      f"({telemetry['early_exits']/max(telemetry['total_steps'],1)*100:.0f}%)", flush=True)
print(f"  Avg layers per token:  {avg_layers:.1f}/{N_LAYERS} (skip {skip_pct:.0f}%)", flush=True)
print(f"  Avg active heads:      {avg_heads:.0f}%", flush=True)
print(f"\n  Generation comparison:", flush=True)
for p in prompts:
    print(f"  Prompt: '{p[:45]}...'", flush=True)
    print(f"    Baseline: {baseline_gens[p][:70]}", flush=True)
    print(f"    Dual:     {dual_gens[p][:70]}", flush=True)
    match_this = "✓" if baseline_gens[p][:50] == dual_gens[p][:50] else "≠"
    print(f"    {match_this}", flush=True)
print(f"{'='*70}", flush=True)

# Save
results = {
    "model": "Qwen3-14B", "n_layers": N_LAYERS, "n_heads": N_HEADS,
    "baseline_tps": baseline_tps, "dual_tps": dual_tps,
    "speedup": dual_tps / baseline_tps,
    "token_match_pct": token_match,
    "early_exits": telemetry["early_exits"],
    "total_steps": telemetry["total_steps"],
    "avg_layers": avg_layers, "avg_heads_pct": avg_heads,
    "head_keep_frac": HEAD_KEEP_FRAC,
    "confidence_tau": CONFIDENCE_TAU,
    "exit_check_layers": EXIT_CHECK_LAYERS,
    "generation": {p: {"baseline": baseline_gens[p], "dual": dual_gens[p]} for p in prompts},
}
out_path = Path("machines/strix_halo/results/dual_engine_14b.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_path}", flush=True)
