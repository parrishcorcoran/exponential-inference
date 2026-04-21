"""Exponential Model: cycled layers with rotation-curve scaling.

Base model layers cycled as extension. Residual scaled by the
rotation curve — later cycles contribute less, matching the
natural slowdown of the manifold projection.

No training. Stock weights. Just cycle + scale.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda"

print("=" * 70)
print("EXPONENTIAL: cycled layers + rotation curve scaling")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers  # 36
H = model.config.hidden_size              # 2560

print(f"Base: {N_LAYERS}L, H={H}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Rotation curve: derived from Finding 02
# Sharp drop early, then plateau. Normalize to [0, 1] depth.
# Scale factor = how much each layer should contribute at that depth.
# ═══════════════════════════════════════════════════════

def rotation_scale(depth, total_depth):
    """Rotation curve scaling factor at a given depth.

    Finding 02: sharp early, plateau late.
    We model this as exponential decay: scale = exp(-k * depth/total)
    Normalized so first layer = 1.0, deep layers approach a floor.
    """
    t = depth / total_depth
    # Exponential decay with floor
    # Early layers (t~0): scale ~1.0
    # Late layers (t~1): scale ~0.1
    return max(0.05, np.exp(-3.0 * t))


# ═══════════════════════════════════════════════════════
# Cycled forward: base layers repeated with scaling
# ═══════════════════════════════════════════════════════

def cycled_forward(model, input_ids, total_depth):
    """Run model layers cycled to reach total_depth, with rotation scaling.

    total_depth=36: normal forward (one cycle, full scale)
    total_depth=72: two cycles, second cycle scaled down
    total_depth=108: three cycles, each progressively scaled
    """
    h = model.model.embed_tokens(input_ids)
    B, T, D = h.shape
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(h, pos)
    pos_emb = (cos, sin)

    for depth in range(total_depth):
        layer_idx = depth % N_LAYERS  # cycle
        layer = model.model.layers[layer_idx]

        scale = rotation_scale(depth, total_depth)

        # Run layer
        residual = h
        layer_out = layer(h, position_embeddings=pos_emb)
        if isinstance(layer_out, tuple):
            h_new = layer_out[0]
        else:
            h_new = layer_out

        # Scaled residual: later depths contribute less
        # h_new already includes the residual (transformer layer does h + attn + mlp)
        # So the delta is h_new - residual
        delta = h_new - residual
        h = residual + scale * delta

    h = model.model.norm(h)
    return model.lm_head(h)


def cycled_generate(model, input_ids, max_new_tokens, total_depth):
    """Generate with cycled forward at specified depth."""
    # Prefill
    with torch.no_grad():
        logits = cycled_forward(model, input_ids, total_depth)
    next_tok = logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    gen_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=-1)

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            logits = cycled_forward(model, gen_ids, total_depth)
        next_tok = logits[0, -1].argmax(-1).item()
        gen_tokens.append(next_tok)
        gen_ids = torch.cat([gen_ids, torch.tensor([[next_tok]], device=device)], dim=-1)

        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens


# ═══════════════════════════════════════════════════════
# Test: baseline vs cycled at various depths
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
]

N_GEN = 40

print(f"\nBASELINE (stock model):")
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{text[:70]}]")

# Test cycled depths
for total_depth in [36, 54, 72, 108]:
    n_cycles = total_depth / N_LAYERS
    print(f"\nCYCLED depth={total_depth} ({n_cycles:.1f} cycles):")

    # Show rotation scale at key points
    scales = [rotation_scale(d, total_depth) for d in range(total_depth)]
    print(f"  Scale: start={scales[0]:.2f} mid={scales[total_depth//2]:.2f} end={scales[-1]:.2f}")

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        t0 = time.time()
        with torch.no_grad():
            tokens = cycled_generate(model, ids, N_GEN, total_depth)
        elapsed = time.time() - t0
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        tps = len(tokens) / elapsed
        print(f"  {tps:.1f} tok/s [{text[:70]}]")

# ═══════════════════════════════════════════════════════
# Velocity measurement: does cycling + scaling converge?
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("VELOCITY: does cycled forward converge?")
print(f"{'='*60}")

ids = tokenizer("The theory of general relativity describes gravity as the curvature of",
                return_tensors='pt').input_ids.to(device)

h = model.model.embed_tokens(ids)
pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
cos, sin = model.model.rotary_emb(h, pos)
pos_emb = (cos, sin)

total_depth = 108
h_prev = h.clone()

print(f"\n{'Depth':>6} {'Layer':>6} {'Scale':>7} {'Velocity':>10} {'Norm':>10}")
print("-" * 45)

with torch.no_grad():
    for depth in range(total_depth):
        layer_idx = depth % N_LAYERS
        layer = model.model.layers[layer_idx]
        scale = rotation_scale(depth, total_depth)

        residual = h
        layer_out = layer(h, position_embeddings=pos_emb)
        h_new = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        delta = h_new - residual
        h = residual + scale * delta

        if depth % 6 == 0 or depth == total_depth - 1:
            vel = (h - h_prev).norm(dim=-1).mean().item()
            norm = h.norm(dim=-1).mean().item()
            print(f"{depth:>6} L{layer_idx:>3} {scale:>7.3f} {vel:>10.1f} {norm:>10.1f}")
        h_prev = h.clone()

print(f"\nDone.", flush=True)
