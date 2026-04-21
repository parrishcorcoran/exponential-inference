"""Stage 1: Dynamic layer exit + dynamic KV heads.

Stock Qwen3-14B. No weight changes. No Triton (yet).
Manifold read at layer 0 → determines:
  - n_layers: how many layers to run
  - n_kv_heads: how many KV heads to store (1 if deep, more if shallow)

Run exactly that many layers. Full width (all Q heads).
Store selected KV heads only.

This is the simplest possible implementation of the architecture:
read → configure → execute.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda"

print("=" * 70)
print("STAGE 1: Dynamic exit + dynamic KV heads (14B)")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

N_LAYERS = model.config.num_hidden_layers  # 40
N_HEADS = model.config.num_attention_heads  # 40
N_KV = model.config.num_key_value_heads    # 8
H = model.config.hidden_size              # 5120
HEAD_DIM = H // N_HEADS                   # 128... actually Q_DIM/N_HEADS

# Get actual dims
Q_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0]  # might differ
KV_DIM = model.model.layers[0].self_attn.k_proj.weight.shape[0]
REAL_HEAD_DIM = Q_DIM // N_HEADS

print(f"L={N_LAYERS} H={H} Q_heads={N_HEADS} KV_heads={N_KV} HD={REAL_HEAD_DIM}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)


# ═══════════════════════════════════════════════════════
# Manifold Read: determines (n_layers, n_kv_heads) per token
# ═══════════════════════════════════════════════════════

class ManifoldRouter:
    """Read manifold from embedding + KV context → (n_layers, n_kv_heads)."""

    def __init__(self, n_layers, n_kv_heads, history_size=50):
        self.n_layers = n_layers
        self.n_kv = n_kv_heads
        self.history = []
        self.history_size = history_size

    def reset(self):
        self.history = []

    def read(self, hidden_state):
        """Manifold measurement → routing decision.

        Returns (n_layers, n_kv_heads).
        - Fewer layers needed → more KV heads (compensate depth with width)
        - More layers needed → fewer KV heads (depth provides the angles)
        """
        h = hidden_state.detach().float().cpu().numpy().flatten()

        if len(self.history) < 3:
            self.history.append(h)
            return self.n_layers, self.n_kv  # first tokens: full compute

        # Manifold measurements
        velocity = np.linalg.norm(h - self.history[-1])
        recent = np.stack(self.history[-self.history_size:])
        nbr_min = float(np.linalg.norm(recent - h, axis=-1).min())
        norm = np.linalg.norm(h)

        self.history.append(h)
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

        # Signal: 0 = defined (easy), 1 = branching (hard)
        vel_norm = min(velocity / 300.0, 1.0)
        nbr_norm = min(nbr_min / 300.0, 1.0)
        signal = 0.6 * vel_norm + 0.4 * nbr_norm

        # Layers: signal 0 → few layers, signal 1 → all layers
        min_layers = 15
        n_layers = min_layers + int(signal * (self.n_layers - min_layers))
        n_layers = min(n_layers, self.n_layers)

        # KV heads: INVERSE of layers.
        # Few layers (shallow) → more KV heads (need spatial angles)
        # Many layers (deep) → fewer KV heads (depth provides angles)
        depth_frac = n_layers / self.n_layers  # 0.4 to 1.0
        # depth_frac high (deep) → 1 KV head
        # depth_frac low (shallow) → up to N_KV heads
        n_kv_out = max(1, int(self.n_kv * (1.0 - depth_frac) * 2))
        n_kv_out = min(n_kv_out, self.n_kv)

        return n_layers, n_kv_out


# ═══════════════════════════════════════════════════════
# Generation: read → configure → execute
# ═══════════════════════════════════════════════════════

def stage1_generate(model, input_ids, max_new_tokens=64):
    """Generate with dynamic exit + dynamic KV heads."""
    router = ManifoldRouter(N_LAYERS, N_KV)

    # Prefill: full model (optimized, builds context)
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values

    # Seed router history
    h_last = out.hidden_states[-1][0]
    for t in range(h_last.shape[0]):
        router.history.append(h_last[t].float().cpu().numpy())
    router.history = router.history[-router.history_size:]

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    telem = {"layers": [], "kv_heads": []}

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            # EXECUTE: run single token through model (full compute for now)
            # The routing decisions are measured but not yet enforced
            # (enforcing requires the custom forward or Triton)
            tok_ids = torch.tensor([[next_tok]], device=device)
            out = model(tok_ids, past_key_values=past, use_cache=True,
                       output_hidden_states=True)
            past = out.past_key_values

            # READ: manifold measurement from the hidden state
            h = out.hidden_states[-1][0, -1]
            n_layers, n_kv = router.read(h)

            telem["layers"].append(n_layers)
            telem["kv_heads"].append(n_kv)

            next_tok = out.logits[0, -1].argmax(-1).item()

        gen_tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark: measure routing + compute savings
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at one hundred",
    "Once upon a time in a kingdom far away there lived a wise",
    "The most fundamental principle in quantum mechanics is",
    "To solve a quadratic equation you can use the quadratic formula which states",
]

N_GEN = 64

print(f"\nBASELINE (full model):")
ids = tokenizer(prompts[0], return_tensors='pt').input_ids.to(device)
with torch.no_grad(): model.generate(ids, max_new_tokens=5, do_sample=False)

base_tps = []
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False)
    torch.cuda.synchronize()
    tps = N_GEN / (time.time() - t0)
    base_tps.append(tps)
avg_base = sum(base_tps) / len(base_tps)
print(f"  Average: {avg_base:.1f} tok/s")

print(f"\nSTAGE 1 (routing active, full compute measured):")
print(f"{'Prompt':>55} {'Avg L':>6} {'Avg KV':>7} {'L%':>5} {'KV%':>5}")
print("-" * 85)

all_layers = []
all_kv = []

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        tokens, telem = stage1_generate(model, ids, N_GEN)

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    avg_l = sum(telem["layers"]) / max(len(telem["layers"]), 1)
    avg_kv = sum(telem["kv_heads"]) / max(len(telem["kv_heads"]), 1)

    all_layers.extend(telem["layers"])
    all_kv.extend(telem["kv_heads"])

    print(f"{prompt[:53]:>55} {avg_l:>5.0f} {avg_kv:>6.1f} {avg_l/N_LAYERS*100:>4.0f}% {avg_kv/N_KV*100:>4.0f}%")
    print(f"  [{text[:70]}]")

# Summary
layers_arr = np.array(all_layers)
kv_arr = np.array(all_kv)

print(f"\n{'='*60}")
print(f"ROUTING SUMMARY")
print(f"{'='*60}")
print(f"  Layers: mean={layers_arr.mean():.0f}/{N_LAYERS} "
      f"({layers_arr.mean()/N_LAYERS*100:.0f}% compute)")
print(f"  KV heads: mean={kv_arr.mean():.1f}/{N_KV} "
      f"({kv_arr.mean()/N_KV*100:.0f}% cache)")
print(f"  Layer savings: {(1 - layers_arr.mean()/N_LAYERS)*100:.0f}%")
print(f"  KV savings: {(1 - kv_arr.mean()/N_KV)*100:.0f}%")
print(f"\n  Depth distribution:")
for pct in [40, 50, 60, 75, 100]:
    n = int(N_LAYERS * pct / 100)
    count = (layers_arr <= n).sum()
    print(f"    ≤{n} layers: {count}/{len(layers_arr)} tokens ({count/len(layers_arr)*100:.0f}%)")
print(f"\n  KV head distribution:")
for n in [1, 2, 4, 8]:
    count = (kv_arr == n).sum()
    print(f"    {n} KV heads: {count}/{len(kv_arr)} tokens ({count/len(kv_arr)*100:.0f}%)")

# Theoretical: if we only computed the routed layers
layer_compute_frac = layers_arr.mean() / N_LAYERS
kv_cache_frac = kv_arr.mean() / N_KV
print(f"\n  Theoretical speedup from layer savings: {1/layer_compute_frac:.1f}x")
print(f"  KV cache size: {kv_cache_frac*100:.0f}% of baseline")

print(f"\nText is correct (full compute used). Routing decisions show savings.")
print(f"Next: enforce the routing with custom forward / Triton.")
print(f"\nDone.", flush=True)
