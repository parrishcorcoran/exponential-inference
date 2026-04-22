"""Holographic Router: SAE encoder → routing decisions → execute.

The SAE reads the manifold. The router translates SAE output into
(n_layers, active_heads, n_kv_heads) per token. The forward pass
executes exactly that configuration.

Based on mega matrix findings:
- LENGTH: SAE features + early logit lens
- HEADS: Q head 4 dominates, then 1, 13, 12, 8
- KV: KV4 and KV6 carry signal, others nearly redundant

No proxies. No entropy. No sharpness. SAE reads the manifold directly.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda"

print("=" * 70)
print("HOLOGRAPHIC ROUTER — SAE manifold read → route → execute")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads

# Load SAE (layer 0 = reads embedding directly, no layers needed)
SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"
sae_0 = torch.load(f"{SAE_PATH}/ae_0.pt", map_location=device, weights_only=False)
sae_enc_w = sae_0["encoder.weight"].float().to(device)  # [16384, 1024]
sae_enc_b = sae_0["encoder.bias"].float().to(device)

# SAE features that distinguish defined from branching (from deep dive)
DEFINED_FEATURES = {1143, 10032, 13093, 4273, 2351, 5963, 3897, 6775, 2523, 8120}
BRANCHING_FEATURES = {3666, 11005, 15246, 9873, 8983, 7071, 6431, 15393, 6452}

# Head importance ranking (from mega matrix)
# Q heads ranked by |r(top1)|: 4, 1, 13, 12, 8, 0, 2, 3, 10, 5, 15, 9, 11, 6, 14, 7
HEAD_RANK = [4, 1, 13, 12, 8, 0, 2, 3, 10, 5, 15, 9, 11, 6, 14, 7]

# KV heads: 4 and 6 are informative, others are redundant
KV_ESSENTIAL = [4, 6]
KV_FULL = list(range(N_KV))

print(f"Model: H={H} L={N_LAYERS} heads={N_HEADS} kv={N_KV}")
print(f"SAE: {sae_enc_w.shape[0]} features")
print(f"Defined features: {len(DEFINED_FEATURES)}")
print(f"Branching features: {len(BRANCHING_FEATURES)}")
print(f"Head rank: {HEAD_RANK[:5]} (top 5)")
print(f"Essential KV: {KV_ESSENTIAL}")


class HolographicRouter:
    """SAE encoder → manifold read → routing decisions."""

    def __init__(self):
        self.sae_w = sae_enc_w
        self.sae_b = sae_enc_b

    def read(self, hidden_state):
        """Read manifold via SAE. Returns routing decision.

        Args:
            hidden_state: [H] tensor — embedding or early-layer hidden

        Returns:
            n_layers: int — how many layers to run
            active_heads: list[int] — which Q heads to activate
            n_kv: int — how many KV heads
        """
        # SAE encode
        h = hidden_state.float()
        pre = h @ self.sae_w.T + self.sae_b
        acts = F.relu(pre)  # [16384]

        # Which features are active
        active_indices = set((acts > 0).nonzero(as_tuple=True)[0].cpu().tolist())

        # Count defined vs branching features
        n_defined = len(active_indices & DEFINED_FEATURES)
        n_branching = len(active_indices & BRANCHING_FEATURES)

        # Total activation magnitude
        total_act = acts.sum().item()
        mean_act = acts[acts > 0].mean().item() if (acts > 0).any() else 0

        # Routing decisions based on manifold position:

        # LENGTH: defined tokens resolve early, branching need full depth
        if n_defined > n_branching + 2:
            # Strongly defined — early exit
            n_layers = max(8, N_LAYERS // 2)
        elif n_defined > n_branching:
            # Mildly defined
            n_layers = max(14, int(N_LAYERS * 0.7))
        elif n_branching > n_defined + 2:
            # Strongly branching — need full depth
            n_layers = N_LAYERS
        else:
            # Ambiguous — use most of the depth
            n_layers = max(20, int(N_LAYERS * 0.85))

        # HEADS: use top-ranked heads, more for branching
        if n_defined > n_branching + 2:
            n_active = max(4, N_HEADS // 4)  # 4 heads
        elif n_defined > n_branching:
            n_active = max(8, N_HEADS // 2)  # 8 heads
        else:
            n_active = N_HEADS  # all heads for branching

        active_heads = HEAD_RANK[:n_active]

        # KV: essential 2 for defined, all 8 for branching
        if n_defined > n_branching:
            n_kv = len(KV_ESSENTIAL)
        else:
            n_kv = N_KV

        return n_layers, sorted(active_heads), n_kv, {
            "n_defined": n_defined,
            "n_branching": n_branching,
            "total_act": total_act,
            "mean_act": mean_act,
        }


# ═══════════════════════════════════════════════════════
# Generate with holographic routing
# ═══════════════════════════════════════════════════════

def holographic_generate(model, input_ids, max_new_tokens=64):
    """Generate with SAE-routed per-token compute."""
    router = HolographicRouter()

    # Prefill: full model (builds KV cache)
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=True)
        past = out.past_key_values

    next_tok = out.logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    telem = {"layers": [], "heads": [], "kv": [], "defined": [], "branching": []}

    for step in range(max_new_tokens - 1):
        # Embed
        with torch.no_grad():
            h_emb = model.model.embed_tokens(torch.tensor([[next_tok]], device=device))

        # SAE READ: manifold → routing
        n_layers, active_heads, n_kv, info = router.read(h_emb[0, 0])

        telem["layers"].append(n_layers)
        telem["heads"].append(len(active_heads))
        telem["kv"].append(n_kv)
        telem["defined"].append(info["n_defined"])
        telem["branching"].append(info["n_branching"])

        # EXECUTE: full model forward (routing decisions measured, not enforced yet)
        with torch.no_grad():
            tok_ids = torch.tensor([[next_tok]], device=device)
            out = model(tok_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[0, -1].argmax(-1).item()

        gen_tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
    "Once upon a time in a kingdom far away there lived",
    "The most fundamental concept in quantum mechanics is",
    "To solve a quadratic equation you can use the",
]

N_GEN = 64

print(f"\nGenerating with holographic router...")
print(f"\n{'Prompt':>50} {'Avg L':>6} {'Avg H':>6} {'Avg KV':>7} {'Compute%':>9}")
print("-" * 85)

all_layers = []
all_heads = []
all_kv = []

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        tokens, telem = holographic_generate(model, ids, N_GEN)

    text = tokenizer.decode(tokens, skip_special_tokens=True)
    avg_l = sum(telem["layers"]) / len(telem["layers"])
    avg_h = sum(telem["heads"]) / len(telem["heads"])
    avg_kv = sum(telem["kv"]) / len(telem["kv"])
    compute = (avg_h / N_HEADS) * (avg_l / N_LAYERS) * 100

    all_layers.extend(telem["layers"])
    all_heads.extend(telem["heads"])
    all_kv.extend(telem["kv"])

    print(f"{prompt[:48]:>50} {avg_l:>5.0f} {avg_h:>5.0f} {avg_kv:>6.1f} {compute:>7.0f}%")
    print(f"  [{text[:70]}]")

# Summary
la = np.array(all_layers)
ha = np.array(all_heads)
kva = np.array(all_kv)

avg_compute = (ha.mean() / N_HEADS) * (la.mean() / N_LAYERS) * 100
kv_savings = (1 - kva.mean() / N_KV) * 100

print(f"\n{'='*60}")
print(f"HOLOGRAPHIC ROUTER SUMMARY")
print(f"{'='*60}")
print(f"  Layers: {la.mean():.0f}/{N_LAYERS} ({la.mean()/N_LAYERS*100:.0f}%)")
print(f"  Heads:  {ha.mean():.0f}/{N_HEADS} ({ha.mean()/N_HEADS*100:.0f}%)")
print(f"  KV:     {kva.mean():.1f}/{N_KV} ({kva.mean()/N_KV*100:.0f}%)")
print(f"  Compute: {avg_compute:.0f}% of baseline")
print(f"  KV cache savings: {kv_savings:.0f}%")
print(f"  Theoretical speedup: {100/avg_compute:.1f}x")

# Distribution
print(f"\n  Per-token routing decisions:")
for n_l in sorted(set(all_layers)):
    count = sum(1 for x in all_layers if x == n_l)
    print(f"    {n_l}L: {count}/{len(all_layers)} tokens ({count/len(all_layers)*100:.0f}%)")

print(f"\n  Head count distribution:")
for n_h in sorted(set(all_heads)):
    count = sum(1 for x in all_heads if x == n_h)
    print(f"    {n_h}H: {count}/{len(all_heads)} tokens ({count/len(all_heads)*100:.0f}%)")

print(f"\n  KV head distribution:")
for n_kv in sorted(set(all_kv)):
    count = sum(1 for x in all_kv if x == n_kv)
    print(f"    {n_kv}KV: {count}/{len(all_kv)} tokens ({count/len(all_kv)*100:.0f}%)")

print(f"\nText is correct (full compute used). Routing shows theoretical savings.")
print(f"Next: enforce routing with custom forward / Triton kernels.")
print(f"\nDone.", flush=True)
