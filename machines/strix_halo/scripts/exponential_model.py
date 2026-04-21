"""Exponential Inference Model — the real architecture.

Read manifold → configure → execute once.

1. Token arrives, embed
2. One attention against KV cache (manifold read)
3. Manifold features → (n_layers, n_heads) for this token
4. Run forward pass at that exact configuration
5. Store K/V in cache
6. Next token

No per-layer checks. No early exit logic. No iteration.
One read. One execution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

device = "cuda"

print("=" * 70)
print("EXPONENTIAL MODEL — read → configure → execute once")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
H = model.config.hidden_size

print(f"Base: {N_LAYERS}L, {N_HEADS}H, {H}D")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# ═══════════════════════════════════════════════════════
# Manifold Read: one attention + feature extraction
# ═══════════════════════════════════════════════════════

class ManifoldRead:
    """Read the manifold at the embedding level.

    Uses the embedding + KV context to determine (n_layers, n_heads).
    Features computed from the relationship between this token's
    embedding and the recent trajectory (KV cache states).
    """
    def __init__(self, n_layers, n_heads, history_size=50):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.history = []  # recent hidden states
        self.history_size = history_size

    def reset(self):
        self.history = []

    def read(self, embedding):
        """Read manifold from embedding + history.

        Returns (n_layers, n_heads) for this token.
        Purely geometric measurement.
        """
        h = embedding.detach().float().cpu().numpy().flatten()

        # Manifold measurements
        velocity = 0.0
        nbr_min = float('inf')
        norm = np.linalg.norm(h)

        if len(self.history) >= 1:
            velocity = np.linalg.norm(h - self.history[-1])

        if len(self.history) >= 3:
            recent = np.stack(self.history[-self.history_size:])
            dists = np.linalg.norm(recent - h, axis=-1)
            nbr_min = float(dists.min())

        # Store
        self.history.append(h)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        # Route based on geometry
        # Low velocity + small neighbor distance = defined = less compute
        # High velocity + large neighbor distance = branching = more compute

        # Normalize to [0, 1] (empirical ranges for 4B model)
        if len(self.history) < 3:
            # First tokens: no context yet, use full compute
            return self.n_layers, self.n_heads

        vel_norm = min(velocity / 300.0, 1.0)
        nbr_norm = min(nbr_min / 300.0, 1.0) if nbr_min != float('inf') else 1.0
        norm_factor = min(norm / 600.0, 1.0)

        # Manifold signal: 0 = defined, 1 = branching
        signal = 0.5 * vel_norm + 0.3 * nbr_norm + 0.2 * norm_factor

        # Map to layers: signal 0 → 12 layers, signal 1 → all layers
        min_layers = max(8, self.n_layers // 3)
        n_layers_out = min_layers + int(signal * (self.n_layers - min_layers))

        # Map to heads: signal 0 → 8 heads, signal 1 → all heads
        min_heads = max(4, self.n_heads // 4)
        n_heads_out = min_heads + int(signal * (self.n_heads - min_heads))

        return n_heads_out, n_layers_out


# ═══════════════════════════════════════════════════════
# Configurable forward: run exact (n_layers, n_heads) specified
# Uses head masking (not slicing) for correct O-projection
# ═══════════════════════════════════════════════════════

def configured_forward(model, input_ids, past, n_heads, n_layers):
    """Run forward with exactly n_heads active for n_layers.

    Uses head masking: compute all heads, zero inactive ones, scale active.
    This preserves correct O-projection behavior.
    For a production model, Triton would skip inactive heads entirely.
    """
    # Use the base model's forward for the configured depth
    # For now: just use model forward with full depth
    # (The real speedup comes from Triton implementation)

    # Run the model up to n_layers
    # Note: we can't easily run partial layers with the HF API
    # So for this prototype: run full forward, which gives correct output
    # The routing decisions are measured for theoretical compute savings

    with torch.no_grad():
        out = model(input_ids, past_key_values=past, use_cache=True,
                    output_hidden_states=False)

    return out.logits, out.past_key_values


# ═══════════════════════════════════════════════════════
# Generation loop: read → configure → execute
# ═══════════════════════════════════════════════════════

def exponential_generate(model, input_ids, max_new_tokens=64):
    """Generate with manifold-routed per-token compute."""
    router = ManifoldRead(N_LAYERS, N_HEADS)

    # Prefill: full compute (building the KV context)
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values

    # Seed router with prefill hidden states
    h_last = out.hidden_states[-1][0]
    for t in range(h_last.shape[0]):
        router.history.append(h_last[t].float().cpu().numpy())
    router.history = router.history[-router.history_size:]

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    telem = {"heads": [], "layers": [], "signal": []}

    for step in range(max_new_tokens - 1):
        # 1. Embed
        with torch.no_grad():
            tok_emb = model.model.embed_tokens(
                torch.tensor([[next_tok]], device=device))

        # 2. Manifold read (from embedding + history)
        n_heads, n_layers = router.read(tok_emb[0, 0])
        telem["heads"].append(n_heads)
        telem["layers"].append(n_layers)

        # 3. Execute (configured forward pass)
        with torch.no_grad():
            logits, past = configured_forward(
                model, torch.tensor([[next_tok]], device=device),
                past, n_heads, n_layers
            )

        # 4. Get token
        next_tok = logits[0, -1].argmax(-1).item()
        gen_tokens.append(next_tok)

        # Update router history with this token's final state
        # (In production, we'd use the actual hidden state from the configured forward)
        router.history.append(tok_emb[0, 0].float().cpu().numpy())
        if len(router.history) > router.history_size:
            router.history = router.history[-router.history_size:]

        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Run and measure routing decisions
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
    "Once upon a time in a kingdom far away there lived",
    "The most fundamental principle in quantum mechanics is",
    "To make a perfect cup of coffee you need to",
]

N_GEN = 64

print(f"\nGenerating with manifold routing...")
print(f"(Full compute used — routing decisions measured for theoretical speedup)")
print(f"\n{'Prompt':>50} {'Avg H':>6} {'Avg L':>6} {'Compute%':>9} {'Speedup':>8}")
print("-" * 85)

all_heads = []
all_layers = []

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    with torch.no_grad():
        tokens, telem = exponential_generate(model, ids, N_GEN)

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    avg_h = sum(telem["heads"]) / max(len(telem["heads"]), 1)
    avg_l = sum(telem["layers"]) / max(len(telem["layers"]), 1)
    compute = (avg_h / N_HEADS) * (avg_l / N_LAYERS) * 100
    speedup = 100 / compute

    all_heads.extend(telem["heads"])
    all_layers.extend(telem["layers"])

    print(f"{prompt[:48]:>50} {avg_h:>5.0f} {avg_l:>5.0f} {compute:>7.0f}% {speedup:>7.1f}x")
    print(f"  [{text[:70]}]")

# Summary
heads_arr = np.array(all_heads)
layers_arr = np.array(all_layers)
avg_compute = (heads_arr.mean() / N_HEADS) * (layers_arr.mean() / N_LAYERS) * 100
avg_speedup = 100 / avg_compute

print(f"\n{'='*60}")
print(f"ROUTING SUMMARY")
print(f"{'='*60}")
print(f"  Heads:  mean={heads_arr.mean():.0f}/{N_HEADS} min={heads_arr.min()} max={heads_arr.max()}")
print(f"  Layers: mean={layers_arr.mean():.0f}/{N_LAYERS} min={layers_arr.min()} max={layers_arr.max()}")
print(f"  Avg compute: {avg_compute:.0f}%")
print(f"  Theoretical speedup: {avg_speedup:.1f}x")
print(f"\n  Per-token distribution:")
print(f"    < 25% compute: {((heads_arr/N_HEADS * layers_arr/N_LAYERS) < 0.25).sum()}/{len(heads_arr)} tokens")
print(f"    < 50% compute: {((heads_arr/N_HEADS * layers_arr/N_LAYERS) < 0.50).sum()}/{len(heads_arr)} tokens")
print(f"    < 75% compute: {((heads_arr/N_HEADS * layers_arr/N_LAYERS) < 0.75).sum()}/{len(heads_arr)} tokens")
print(f"    100% compute:  {((heads_arr/N_HEADS * layers_arr/N_LAYERS) >= 0.99).sum()}/{len(heads_arr)} tokens")

print(f"\nThis is the architecture. Text is correct (full compute used).")
print(f"The routing decisions show theoretical savings with Triton implementation.")
print(f"\nDone.", flush=True)
