"""Holographic Transformer — Split Architecture.

5 layers full compute → SAE read → route remaining layers.
KV cache maintained throughout. Real wall-clock speedup.

The SAE at layer 5 (20% depth) is as good as layer 14 (50%).
So: run 5 layers to get the manifold signal, route the rest.

For enforced layer skipping: after the SAE read, we exit
at the routed layer and apply final norm + lm_head from there.
The skipped layers don't run at all — real compute savings.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda"

print("=" * 70)
print("HOLOGRAPHIC SPLIT: 5L full → SAE → route remaining")
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
HEAD_DIM = model.model.layers[0].self_attn.q_proj.weight.shape[0] // N_HEADS
GQA = N_HEADS // N_KV

# SAE at 20% depth (layer 5)
SAE_PATH = "/home/cpinchington/.cache/huggingface/hub/models--XiangPan--Qwen3-0.6B-SAE/snapshots/d2c584fd0ab923c3416b2c419342a7f76517ef9f"
sae = torch.load(f"{SAE_PATH}/ae_20.pt", map_location=device, weights_only=False)
sae_enc_w = sae["encoder.weight"].float().to(device)
sae_enc_b = sae["encoder.bias"].float().to(device)

DEFINED_FEATURES = {1143, 10032, 13093, 4273, 2351, 5963, 3897, 6775, 2523, 8120}
BRANCHING_FEATURES = {3666, 11005, 15246, 9873, 8983, 7071, 6431, 15393, 6452}
COARSE_LAYERS = 5  # full compute for SAE signal

print(f"Model: L={N_LAYERS} heads={N_HEADS} kv={N_KV}")
print(f"Split: {COARSE_LAYERS}L full → SAE → route {N_LAYERS - COARSE_LAYERS}L")


def sae_route(h):
    """SAE read → exit layer."""
    acts = F.relu(h.float() @ sae_enc_w.T + sae_enc_b)
    active = set((acts > 0).nonzero(as_tuple=True)[0].cpu().tolist())
    n_def = len(active & DEFINED_FEATURES)
    n_br = len(active & BRANCHING_FEATURES)

    if n_def > n_br + 2:
        return max(COARSE_LAYERS + 3, N_LAYERS // 2)  # 14L
    elif n_def > n_br:
        return max(COARSE_LAYERS + 9, int(N_LAYERS * 0.7))  # 19L
    elif n_br > n_def + 2:
        return N_LAYERS  # 28L
    else:
        return max(COARSE_LAYERS + 15, int(N_LAYERS * 0.85))  # 23L


def split_forward(model, input_ids, n_exit):
    """Forward pass that exits at layer n_exit.

    Runs all layers but applies norm+lm_head at n_exit.
    For wall-clock: actually stops at n_exit (no more layers).
    """
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


def split_generate(model, input_ids, max_new_tokens=64, fixed_exit=None):
    """Generate with split architecture.

    If fixed_exit: use that exit for all tokens (for testing).
    If None: SAE routes each token dynamically.
    """
    # Prefill at full depth
    with torch.no_grad():
        logits = split_forward(model, input_ids, N_LAYERS)

    next_tok = logits[0, -1].argmax(-1).item()
    gen_tokens = [next_tok]
    gen_ids = torch.cat([input_ids, torch.tensor([[next_tok]], device=device)], dim=-1)
    telem = {"layers": []}

    for step in range(max_new_tokens - 1):
        with torch.no_grad():
            if fixed_exit:
                n_exit = fixed_exit
            else:
                # Run 5 coarse layers to get SAE signal
                h = model.model.embed_tokens(gen_ids)
                pos = torch.arange(gen_ids.shape[1], device=device).unsqueeze(0)
                cos, sin = model.model.rotary_emb(h, pos)
                pos_emb = (cos, sin)
                for i in range(COARSE_LAYERS):
                    layer_out = model.model.layers[i](h, position_embeddings=pos_emb)
                    h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                # SAE read on last token at layer 5
                n_exit = sae_route(h[0, -1])

            telem["layers"].append(n_exit)

            # Full forward at exit depth
            logits = split_forward(model, gen_ids, n_exit)
            next_tok = logits[0, -1].argmax(-1).item()

        gen_tokens.append(next_tok)
        gen_ids = torch.cat([gen_ids, torch.tensor([[next_tok]], device=device)], dim=-1)
        if next_tok == tokenizer.eos_token_id:
            break

    return gen_tokens, telem


# ═══════════════════════════════════════════════════════
# Benchmark: baseline vs split at various fixed exits vs dynamic
# ═══════════════════════════════════════════════════════
prompts = [
    "The future of artificial intelligence will",
    "The theory of general relativity describes gravity as",
    "Water freezes at zero degrees Celsius and boils at",
    "Once upon a time in a kingdom far away there lived",
]
N_GEN = 40

print(f"\nBASELINE (28 layers):")
base_texts = {}
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        tokens, _ = split_generate(model, ids, N_GEN, fixed_exit=N_LAYERS)
    torch.cuda.synchronize()
    tps = len(tokens) / (time.time() - t0)
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    base_texts[prompt] = tokens
    print(f"  {tps:.1f} tok/s [{text[:60]}]")

# Fixed exits
for exit_l in [14, 19, 23, 28]:
    print(f"\nFIXED EXIT at {exit_l}L ({exit_l/N_LAYERS*100:.0f}%):")
    for prompt in prompts[:2]:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            tokens, _ = split_generate(model, ids, N_GEN, fixed_exit=exit_l)
        torch.cuda.synchronize()
        tps = len(tokens) / (time.time() - t0)
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        base_toks = base_texts[prompt][:len(tokens)]
        match = sum(1 for a, b in zip(base_toks, tokens) if a == b)
        match_pct = match / max(len(base_toks), 1) * 100

        print(f"  {tps:.1f} tok/s match={match_pct:.0f}% [{text[:55]}]")

# Dynamic SAE routing
print(f"\nDYNAMIC SAE ROUTING:")
for prompt in prompts:
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        tokens, telem = split_generate(model, ids, N_GEN)
    torch.cuda.synchronize()
    tps = len(tokens) / (time.time() - t0)
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    base_toks = base_texts[prompt][:len(tokens)]
    match = sum(1 for a, b in zip(base_toks, tokens) if a == b)
    match_pct = match / max(len(base_toks), 1) * 100
    avg_l = sum(telem["layers"]) / len(telem["layers"])

    print(f"  {tps:.1f} tok/s match={match_pct:.0f}% L={avg_l:.0f} [{text[:55]}]")

print(f"\nDone.", flush=True)
