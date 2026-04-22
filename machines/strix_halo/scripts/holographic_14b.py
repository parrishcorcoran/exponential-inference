"""Holographic Transformer on Qwen3-14B.

14B has 40 layers vs 0.6B's 28 — more room to skip.
Split: 8 layers full → SAE → route remaining 32 layers.
The SAE features should separate defined/branching better
at this scale because the model has more capacity.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

device = "cuda"

print("=" * 70)
print("HOLOGRAPHIC TRANSFORMER — 14B")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True
).to(device).eval()

H = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
N_HEADS = model.config.num_attention_heads
N_KV = model.config.num_key_value_heads

print(f"Model: L={N_LAYERS} H={H} heads={N_HEADS} kv={N_KV}")
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# Load 14B SAE
SAE_BASE = os.path.expanduser("~/.cache/huggingface/hub/models--adamkarvonen--qwen3-14b-saes")
sae_snap = None
for root, dirs, files in os.walk(SAE_BASE):
    for f in files:
        if f.endswith('.pt') or f.endswith('.safetensors'):
            sae_snap = root
            break
    if sae_snap:
        break

if sae_snap is None:
    print("14B SAE not found! Checking available files...")
    for root, dirs, files in os.walk(SAE_BASE):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    exit(1)

# Find SAE files
sae_files = sorted([f for f in os.listdir(sae_snap) if f.endswith('.pt')])
print(f"SAE files: {sae_files[:5]}...")

# Load SAE at ~20% depth (layer 8 of 40)
sae_file = None
for f in sae_files:
    # Try to find one near 20%
    if '8' in f or '20' in f or 'layer_8' in f:
        sae_file = f
        break
if sae_file is None:
    sae_file = sae_files[min(2, len(sae_files)-1)]  # take 3rd file

print(f"Loading SAE: {sae_file}")
sae_ckpt = torch.load(os.path.join(sae_snap, sae_file), map_location=device, weights_only=False)

if isinstance(sae_ckpt, dict):
    print(f"SAE keys: {list(sae_ckpt.keys())[:10]}")
    if "encoder.weight" in sae_ckpt:
        sae_enc_w = sae_ckpt["encoder.weight"].float().to(device)
        sae_enc_b = sae_ckpt["encoder.bias"].float().to(device)
        SAE_DIM = sae_enc_w.shape[0]
        print(f"SAE: {SAE_DIM} features, encoder [{sae_enc_w.shape}]")
    else:
        # Try state_dict format
        for k in sae_ckpt:
            if 'encoder' in k and 'weight' in k:
                sae_enc_w = sae_ckpt[k].float().to(device)
                print(f"Found encoder: {k} → {sae_enc_w.shape}")
        for k in sae_ckpt:
            if 'encoder' in k and 'bias' in k:
                sae_enc_b = sae_ckpt[k].float().to(device)
        SAE_DIM = sae_enc_w.shape[0]

print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ═══════════════════════════════════════════════════════
# Split forward: run n_exit layers, norm, lm_head
# ═══════════════════════════════════════════════════════

def split_forward_14b(model, input_ids, n_exit):
    h = model.model.embed_tokens(input_ids)
    B, T, D = h.shape
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(h, pos)
    pos_emb = (cos, sin)

    for i in range(min(n_exit, N_LAYERS)):
        layer_out = model.model.layers[i](h, position_embeddings=pos_emb)
        h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    h = model.model.norm(h)
    return model.lm_head(h)


def sae_route_14b(h):
    acts = F.relu(h.float() @ sae_enc_w.T + sae_enc_b)
    n_active = (acts > 0).sum().item()
    mean_act = acts[acts > 0].mean().item() if n_active > 0 else 0

    # Route based on activation magnitude (correlates with entropy)
    # Low activation = defined = fewer layers
    # High activation = branching = more layers
    if mean_act < 0.5:
        return max(15, N_LAYERS // 2)  # 20L
    elif mean_act < 1.0:
        return max(25, int(N_LAYERS * 0.7))  # 28L
    elif mean_act < 2.0:
        return max(32, int(N_LAYERS * 0.85))  # 34L
    else:
        return N_LAYERS  # 40L


# ═══════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
    "Once upon a time in a kingdom far away there lived",
]
N_GEN = 40

# Baseline
print(f"\nBASELINE (40 layers):")
for prompt in prompts[:2]:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        logits = split_forward_14b(model, ids, N_LAYERS)
    torch.cuda.synchronize()
    text_tok = logits[0, -1].argmax(-1).item()
    print(f"  {prompt[:40]}... → '{tokenizer.decode([text_tok])}'")

# Fixed exits
print(f"\nFIXED EXIT QUALITY TEST:")
for n_exit in [15, 20, 25, 30, 35, 40]:
    print(f"\n  Exit at {n_exit}L ({n_exit/N_LAYERS*100:.0f}%):")
    for prompt in prompts[:2]:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        # Generate a few tokens
        gen_ids = ids.clone()
        gen_tokens = []
        with torch.no_grad():
            for _ in range(20):
                logits = split_forward_14b(model, gen_ids, n_exit)
                tok = logits[0, -1].argmax(-1).item()
                gen_tokens.append(tok)
                gen_ids = torch.cat([gen_ids, torch.tensor([[tok]], device=device)], dim=-1)
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Compare to baseline
        with torch.no_grad():
            base_ids = ids.clone()
            base_tokens = []
            for _ in range(20):
                logits = split_forward_14b(model, base_ids, N_LAYERS)
                tok = logits[0, -1].argmax(-1).item()
                base_tokens.append(tok)
                base_ids = torch.cat([base_ids, torch.tensor([[tok]], device=device)], dim=-1)

        match = sum(1 for a, b in zip(base_tokens, gen_tokens) if a == b)
        match_pct = match / len(base_tokens) * 100
        print(f"    match={match_pct:.0f}% [{text[:55]}]")

# Dynamic SAE routing
print(f"\nDYNAMIC SAE ROUTING (14B):")
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    gen_ids = ids.clone()
    gen_tokens = []
    layers_used = []

    with torch.no_grad():
        # Get hidden at SAE layer for routing
        h = model.model.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h, pos)
        pos_emb = (cos, sin)
        for i in range(8):  # 8 coarse layers
            layer_out = model.model.layers[i](h, position_embeddings=pos_emb)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        for _ in range(20):
            # Route
            n_exit = sae_route_14b(h[0, -1])
            layers_used.append(n_exit)

            # Forward
            logits = split_forward_14b(model, gen_ids, n_exit)
            tok = logits[0, -1].argmax(-1).item()
            gen_tokens.append(tok)
            gen_ids = torch.cat([gen_ids, torch.tensor([[tok]], device=device)], dim=-1)

            # Update h for next routing (recompute - expensive but correct)
            h = model.model.embed_tokens(gen_ids)
            pos = torch.arange(gen_ids.shape[1], device=device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(h, pos)
            pos_emb = (cos, sin)
            for i in range(8):
                layer_out = model.model.layers[i](h, position_embeddings=pos_emb)
                h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    avg_l = sum(layers_used) / len(layers_used)
    print(f"  L={avg_l:.0f}/{N_LAYERS} [{text[:55]}]")

print(f"\nDone.", flush=True)
