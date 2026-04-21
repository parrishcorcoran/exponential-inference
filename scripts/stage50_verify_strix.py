"""
Stage 50 — Verify Strix's dynamic-routing claims on Qwen3-0.6B.

Strix's `manifold_inference.py` on 14B reported:
  - 20/40 heads with scale compensation → coherent text
  - Reduced layer count (early exit, residual pass-through) → 1.6-3.0× speedup
  - Combined width × length → coherent text

If these are architecture-level results they should reproduce at 0.6B.
This script replicates the exact approach (zero masked heads, scale
surviving heads by N_HEADS/n_active, exit after n_active_layers) on
Qwen3-0.6B and measures:
  - Token-match vs standard baseline
  - Sample quality (visual inspection)
  - Wall-clock speedup

Head masking protocol (exactly matching Strix's script):
  head_scale[:n_active] = N_HEADS / n_active
  head_scale[n_active:] = 0
  attn_out *= head_scale[None, None, :, None]

Length protocol: after layer n_active_layers, hidden state passes through
unchanged (no more attention, no more MLP).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def apply_rotary(q, k, cos, sin):
    """Qwen3-style RoPE with full-size cos/sin and rotate_half trick."""
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def manifold_forward(model, input_ids, n_active_heads, n_active_layers, device):
    """Custom forward matching Strix's manifold_inference.py."""
    cfg = model.config
    N_LAYERS = cfg.num_hidden_layers
    N_HEADS = cfg.num_attention_heads
    N_KV = cfg.num_key_value_heads
    # Derive head_dim from the actual q_proj shape — more robust than config attr
    q_out = model.model.layers[0].self_attn.q_proj.out_features
    HEAD_DIM = q_out // N_HEADS

    h = model.model.embed_tokens(input_ids)
    B, T, D = h.shape
    pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    pos_emb = model.model.rotary_emb(h, pos_ids)

    # Head scale: [0..n_active) -> N_HEADS/n_active, [n_active..N_HEADS) -> 0
    head_scale = torch.zeros(N_HEADS, device=device, dtype=h.dtype)
    if n_active_heads > 0:
        head_scale[:n_active_heads] = N_HEADS / n_active_heads
    # attn_out has shape [B, H, T, D]; heads are at dim 1
    head_scale = head_scale.view(1, N_HEADS, 1, 1)

    for i in range(N_LAYERS):
        if i >= n_active_layers:
            # Past exit: residual pass-through (just skip the layer)
            continue

        layer = model.model.layers[i]
        attn = layer.self_attn

        residual = h
        h_norm = layer.input_layernorm(h)

        q = attn.q_proj(h_norm).view(B, T, N_HEADS, HEAD_DIM)
        k = attn.k_proj(h_norm).view(B, T, N_KV, HEAD_DIM)
        v = attn.v_proj(h_norm).view(B, T, N_KV, HEAD_DIM)

        if hasattr(attn, 'q_norm') and attn.q_norm is not None:
            q = attn.q_norm(q)
        if hasattr(attn, 'k_norm') and attn.k_norm is not None:
            k = attn.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = pos_emb
        q, k = apply_rotary(q, k, cos, sin)

        n_rep = N_HEADS // N_KV
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # HEAD MASKING + scale
        attn_out = attn_out * head_scale

        # attn_out [B, H, T, HEAD_DIM] -> [B, T, H*HEAD_DIM] for o_proj
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, N_HEADS * HEAD_DIM)
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out

        # MLP (full — bulk preserved)
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))

    return model.lm_head(model.model.norm(h))


def manifold_generate(model, tokenizer, prompt, n_active_heads, n_active_layers,
                       max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out_tokens = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = manifold_forward(model, ids, n_active_heads, n_active_layers, device)
            tok = logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
            ids = torch.cat([ids, tok], dim=-1)
            out_tokens.append(int(tok.item()))
            if int(tok.item()) == tokenizer.eos_token_id:
                break
    return out_tokens


def standard_generate(model, tokenizer, prompt, max_new_tokens, device):
    """KV-cached standard generation for baseline."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage50_verify_strix.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    N_HEADS = model.config.num_attention_heads
    N_LAYERS = model.config.num_hidden_layers
    print(f"  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}")

    print(f"\n=== baseline (standard forward, full width + full length) ===")
    t0 = time.perf_counter()
    base_tokens = standard_generate(model, tokenizer, args.prompt,
                                     args.max_new_tokens, device)
    base_dt = time.perf_counter() - t0
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {base_dt:.2f}s  {args.max_new_tokens/base_dt:.1f} tok/s")
    print(f"  {base_text[:150]}")

    print(f"\n=== sanity: manifold_forward at full width × full length ===")
    t0 = time.perf_counter()
    full_tokens = manifold_generate(model, tokenizer, args.prompt,
                                     N_HEADS, N_LAYERS,
                                     args.max_new_tokens, device)
    full_dt = time.perf_counter() - t0
    full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    match = sum(1 for a, b in zip(base_tokens, full_tokens) if a == b)
    n = min(len(base_tokens), len(full_tokens))
    print(f"  {full_dt:.2f}s  match_vs_baseline {match}/{n}")
    print(f"  {full_text[:150]}")
    print(f"  (should be ~100% match — sanity check of the custom forward)")

    # Sweep: width at full length, then length at full width, then combinations
    configs = [
        ("full (sanity)", N_HEADS, N_LAYERS),
        (f"width=50% ({N_HEADS//2}/{N_HEADS}) len=full", N_HEADS // 2, N_LAYERS),
        (f"width=25% ({N_HEADS//4}/{N_HEADS}) len=full", max(1, N_HEADS // 4), N_LAYERS),
        (f"width=full len=50% ({N_LAYERS//2}/{N_LAYERS})", N_HEADS, N_LAYERS // 2),
        (f"width=full len=25% ({N_LAYERS//4}/{N_LAYERS})", N_HEADS, max(1, N_LAYERS // 4)),
        (f"width=50% len=50%", N_HEADS // 2, N_LAYERS // 2),
    ]

    results = []
    print(f"\n=== sweep ===")
    print(f"  {'config':>32}  {'tok/s':>7}  {'speedup':>8}  {'match':>10}  sample")
    for label, nh, nl in configs:
        t0 = time.perf_counter()
        tokens = manifold_generate(model, tokenizer, args.prompt, nh, nl,
                                    args.max_new_tokens, device)
        dt = time.perf_counter() - t0
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        nmin = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:nmin], tokens[:nmin]) if a == b)
        tps = args.max_new_tokens / dt
        speedup = base_dt / dt
        print(f"  {label:>32}  {tps:>7.1f}  {speedup:>7.2f}x  "
              f"{match}/{nmin:<4}  {text[:60]}")
        results.append({
            "label": label, "n_heads": nh, "n_layers": nl,
            "wall_seconds": dt, "tok_per_sec": tps, "speedup_vs_baseline": speedup,
            "match": match, "total": nmin,
            "sample": text[:300],
        })

    # Honest assessment of each config's coherence
    print(f"\n=== coherence assessment (visual) ===")
    print(f"  baseline: {base_text[:80]}")
    print()
    for r in results:
        print(f"  [{r['label']}]")
        print(f"    sample: {r['sample'][:120]}")
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
            "baseline_tok_per_sec": args.max_new_tokens / base_dt,
            "baseline_sample": base_text[:400],
            "results": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
