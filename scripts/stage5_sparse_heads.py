"""
Stage 5c — Sparse head attention with physically smaller matmuls.

Instead of computing all heads and masking (no savings), or using
HF's head_mask (zeros but still computes), this REPLACES the
attention forward to only compute active heads.

The Q/K/V projections are sliced to only include active heads,
producing genuinely smaller matmuls that run faster on GPU.

Usage:
    python scripts/stage5_sparse_heads.py \
        --model Qwen/Qwen3-0.6B \
        --max-new-tokens 200 \
        --threshold 0.9 \
        --device mps
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


def head_sharpness(attn_weights):
    """[B, n_heads, 1, T_kv] -> [n_heads] sharpness."""
    w = attn_weights[0, :, -1, :]  # [n_heads, T_kv]
    T = w.shape[-1]
    if T <= 1:
        return torch.ones(w.shape[0], device=w.device)
    entropy = -(w * torch.log(w + 1e-10)).sum(dim=-1)
    max_ent = math.log(T)
    return (1.0 - entropy / max_ent) if max_ent > 0 else torch.ones_like(entropy)


def sparse_attention_forward(
    self_attn,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value=None,
    cache_position=None,
    active_heads: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Custom attention forward that only computes active heads.

    If active_heads is None, computes all heads (standard behavior).
    If active_heads is a boolean tensor of [n_heads], only computes
    the True heads — physically smaller Q/K/V projections.
    """
    bsz, q_len, hidden_size = hidden_states.shape
    n_heads = self_attn.config.num_attention_heads
    n_kv_heads = self_attn.config.num_key_value_heads
    head_dim = self_attn.head_dim
    n_kv_groups = n_heads // n_kv_heads

    if active_heads is None or active_heads.all():
        # Standard path — all heads
        query_states = self_attn.q_proj(hidden_states)
        key_states = self_attn.k_proj(hidden_states)
        value_states = self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    else:
        # Sparse path — only active heads
        active_idx = active_heads.nonzero(as_tuple=True)[0]
        n_active = len(active_idx)

        # Determine which KV heads are needed
        active_kv_heads = torch.zeros(n_kv_heads, dtype=torch.bool, device=hidden_states.device)
        for qi in active_idx:
            kv_idx = qi // n_kv_groups
            active_kv_heads[kv_idx] = True
        active_kv_idx = active_kv_heads.nonzero(as_tuple=True)[0]
        n_active_kv = len(active_kv_idx)

        # Slice Q projection weights for active heads only
        q_weight = self_attn.q_proj.weight  # [n_heads * head_dim, hidden]
        q_slices = []
        for hi in active_idx:
            start = hi * head_dim
            q_slices.append(q_weight[start:start + head_dim])
        q_weight_sparse = torch.cat(q_slices, dim=0)  # [n_active * head_dim, hidden]
        query_states = F.linear(hidden_states, q_weight_sparse,
                                 None if self_attn.q_proj.bias is None
                                 else self_attn.q_proj.bias[torch.cat([
                                     torch.arange(hi*head_dim, (hi+1)*head_dim)
                                     for hi in active_idx])])
        query_states = query_states.view(bsz, q_len, n_active, head_dim).transpose(1, 2)

        # Slice K/V projection weights for active KV heads
        k_weight = self_attn.k_proj.weight
        k_slices = []
        for ki in active_kv_idx:
            start = ki * head_dim
            k_slices.append(k_weight[start:start + head_dim])
        k_weight_sparse = torch.cat(k_slices, dim=0)
        key_states = F.linear(hidden_states, k_weight_sparse)
        key_states = key_states.view(bsz, q_len, n_active_kv, head_dim).transpose(1, 2)

        v_weight = self_attn.v_proj.weight
        v_slices = []
        for vi in active_kv_idx:
            start = vi * head_dim
            v_slices.append(v_weight[start:start + head_dim])
        v_weight_sparse = torch.cat(v_slices, dim=0)
        value_states = F.linear(hidden_states, v_weight_sparse)
        value_states = value_states.view(bsz, q_len, n_active_kv, head_dim).transpose(1, 2)

    # Rotary embeddings
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # KV cache
    if past_key_value is not None:
        cache_kwargs = {"cache_position": cache_position}
        if active_heads is not None and not active_heads.all():
            # Store full-size KV in cache, sparse compute only
            # Expand sparse KV back to full for cache compatibility
            full_key = torch.zeros(bsz, n_kv_heads, key_states.shape[2], head_dim,
                                   dtype=key_states.dtype, device=key_states.device)
            full_val = torch.zeros_like(full_key)
            for i, ki in enumerate(active_kv_idx):
                full_key[:, ki] = key_states[:, i]
                full_val[:, ki] = value_states[:, i]
            key_states_cache, value_states_cache = past_key_value.update(
                full_key, full_val, self_attn.layer_idx, cache_kwargs)
            # Pull back only active KV heads from cache
            key_states = key_states_cache[:, active_kv_idx]
            value_states = value_states_cache[:, active_kv_idx]
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self_attn.layer_idx, cache_kwargs)

    # Expand KV for GQA (only active groups)
    if active_heads is not None and not active_heads.all():
        # Map active query heads to their active KV head index
        kv_idx_map = {}
        for i, ki in enumerate(active_kv_idx):
            kv_idx_map[ki.item()] = i

        # Build the repeat pattern for active heads
        kv_indices = []
        for qi in active_idx:
            kv_head = (qi // n_kv_groups).item()
            kv_indices.append(kv_idx_map[kv_head])
        kv_indices = torch.tensor(kv_indices, device=key_states.device)
        key_states = key_states[:, kv_indices]
        value_states = value_states[:, kv_indices]
    else:
        # Standard GQA repeat
        if n_kv_groups > 1:
            key_states = key_states.repeat_interleave(n_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(n_kv_groups, dim=1)

    # Attention computation (only active heads)
    n_compute_heads = query_states.shape[1]
    scale = self_attn.head_dim ** -0.5 if not hasattr(self_attn, 'scaling') else self_attn.scaling
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    # Output projection — only active head slices
    if active_heads is not None and not active_heads.all():
        # Flatten active heads
        attn_flat = attn_output.reshape(bsz, q_len, n_compute_heads * head_dim)
        # Slice O projection: we need columns corresponding to active heads
        o_weight = self_attn.o_proj.weight  # [hidden, n_heads * head_dim]
        o_slices = []
        for hi in active_idx:
            start = hi * head_dim
            o_slices.append(o_weight[:, start:start + head_dim])
        o_weight_sparse = torch.cat(o_slices, dim=1)  # [hidden, n_active * head_dim]
        attn_output = F.linear(attn_flat, o_weight_sparse.T).unsqueeze(0) if bsz == 1 else \
                      torch.matmul(attn_flat, o_weight_sparse.T)
        if self_attn.o_proj.bias is not None:
            attn_output = attn_output + self_attn.o_proj.bias

        # Rescale to compensate for missing heads
        n_total = n_heads
        attn_output = attn_output * (n_total / n_compute_heads)
    else:
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        attn_output = self_attn.o_proj(attn_output)

    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings."""
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SparseHeadGenerator:
    """Generation with physically sparse attention computation."""

    def __init__(self, model, threshold=0.9, min_heads=2, recalibrate_every=20):
        self.model = model
        self.threshold = threshold
        self.min_heads = min_heads
        self.recalibrate_every = recalibrate_every
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads

        # Per-layer active head masks
        self.active_masks = [None] * self.n_layers
        self.step = 0
        self.stats = []
        self._original_forwards = {}

    def _install_sparse_attention(self):
        """Replace attention forward methods with sparse versions."""
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            self._original_forwards[i] = attn.forward
            layer_idx = i

            def make_sparse_forward(attn_module, idx):
                generator = self

                def sparse_forward(hidden_states, **kwargs):
                    active = generator.active_masks[idx]
                    result = sparse_attention_forward(
                        attn_module, hidden_states,
                        active_heads=active, **kwargs)
                    return result

                return sparse_forward

            attn.forward = make_sparse_forward(attn, i)

    def _restore_attention(self):
        """Restore original attention forwards."""
        layers = self.model.model.layers
        for i, orig in self._original_forwards.items():
            layers[i].self_attn.forward = orig
        self._original_forwards.clear()

    def _update_masks(self, attentions):
        """Update per-layer masks from attention weights."""
        n_kept = 0
        n_total = 0
        for li, attn_w in enumerate(attentions):
            if attn_w is None:
                continue
            sharpness = head_sharpness(attn_w)
            keep = sharpness >= self.threshold
            if keep.sum() < self.min_heads:
                topk = sharpness.topk(min(self.min_heads, len(sharpness))).indices
                keep = torch.zeros_like(keep, dtype=torch.bool)
                keep[topk] = True
            self.active_masks[li] = keep
            n_kept += keep.sum().item()
            n_total += len(keep)

        ratio = n_kept / max(n_total, 1)
        self.stats.append(ratio)
        return ratio

    def generate(self, prompt, tokenizer, max_new_tokens, device):
        """Full generation with sparse heads."""
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        times = []
        generated = []

        # Prefill with all heads (need full attention for context)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(input_ids=input_ids, use_cache=True,
                           output_attentions=True)
        times.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

        # Initialize masks from prefill
        if out.attentions:
            self._update_masks(out.attentions)

        # Install sparse forwards
        self._install_sparse_attention()

        try:
            for i in range(max_new_tokens - 1):
                self.step += 1

                # Recalibrate periodically (all heads)
                if self.step % self.recalibrate_every == 0:
                    for li in range(self.n_layers):
                        self.active_masks[li] = None

                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                    )
                dt = time.perf_counter() - t0
                times.append(dt)

                past_key_values = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token.item())

                # Update masks for next step
                if out.attentions:
                    self._update_masks(out.attentions)

                if next_token.item() == tokenizer.eos_token_id:
                    break
        finally:
            self._restore_attention()

        text = tokenizer.decode(generated, skip_special_tokens=True)
        return times, text, generated


def generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    times = []
    generated = []

    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    times.append(time.perf_counter() - t0)
    past_key_values = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token.item())

    for i in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        times.append(time.perf_counter() - t0)
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return times[1:], text, generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--min-heads", type=int, default=2)
    p.add_argument("--recalibrate-every", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\ndevice: {device}")

    print(f"\n=== loading {args.model} ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()

    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    n_layers = model.config.num_hidden_layers
    head_dim = model.config.hidden_size // n_heads
    print(f"  {n_layers} layers, {n_heads} Q heads, {n_kv_heads} KV heads, head_dim={head_dim}")

    # === Baseline ===
    print(f"\n=== baseline ===", flush=True)
    base_times, base_text, base_tokens = generate_baseline(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    avg_base = sum(base_times) / len(base_times) * 1000
    print(f"  {len(base_times)} tokens, {avg_base:.1f}ms/tok")
    print(f"  {base_text[:120]}...")

    # === Sparse heads ===
    print(f"\n=== sparse heads (threshold={args.threshold}) ===", flush=True)
    gen = SparseHeadGenerator(model, threshold=args.threshold,
                              min_heads=args.min_heads,
                              recalibrate_every=args.recalibrate_every)
    sparse_times_raw, sparse_text, sparse_tokens = gen.generate(
        args.prompt, tokenizer, args.max_new_tokens, device)
    sparse_times = sparse_times_raw[1:]  # skip prefill
    avg_sparse = sum(sparse_times) / len(sparse_times) * 1000
    print(f"  {len(sparse_times)} tokens, {avg_sparse:.1f}ms/tok")
    print(f"  {sparse_text[:120]}...")

    ratios = gen.stats
    avg_kept = sum(ratios) / len(ratios) if ratios else 0
    print(f"  avg heads kept: {avg_kept:.1%}")

    # === Results ===
    print(f"\n=== results ===")
    speedup = avg_base / avg_sparse if avg_sparse > 0 else 0
    print(f"  baseline: {avg_base:.1f}ms/tok")
    print(f"  sparse:   {avg_sparse:.1f}ms/tok")
    print(f"  speedup:  {speedup:.2f}x")

    min_len = min(len(base_tokens), len(sparse_tokens))
    match = sum(1 for a, b in zip(base_tokens[:min_len], sparse_tokens[:min_len]) if a == b)
    print(f"  token match: {match}/{min_len} ({match/max(min_len,1):.1%})")

    if len(ratios) >= 20:
        first_10 = sum(ratios[:10]) / 10
        last_10 = sum(ratios[-10:]) / 10
        print(f"  heads kept first 10: {first_10:.1%}")
        print(f"  heads kept last 10:  {last_10:.1%}")

    # Save
    out_path = Path(args.out_dir) / "stage5_sparse_heads.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "threshold": args.threshold,
            "min_heads": args.min_heads,
            "baseline_ms": avg_base,
            "sparse_ms": avg_sparse,
            "speedup": speedup,
            "token_match": f"{match}/{min_len}",
            "avg_kept_ratio": avg_kept,
            "base_text": base_text[:500],
            "sparse_text": sparse_text[:500],
            "per_step_kept_ratio": ratios,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
