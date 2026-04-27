"""Dual-layer sweep — V, K, or Q, closed-form linear regression.

Usage: python <script> [V|K|Q]   (default V)

For each pair (L1, L2) of 28 layers in 0.6B, fit the optimal LINEAR map
W: concat(h[L1], h[L2]) -> {V, K, Q} at target layer 14, offset +1
via least squares (no iterative training).

Q goes through rotary embedding (RoPE), so the linear probe is mapping
into a position-rotated subspace. Lower cosine expected vs K/V.

Math:
  X_train = concat(h_train[L1], h_train[L2])   # [N, 2*d_model]
  Y_train = V_target                            # [N, n_kv*head_dim]
  W*      = (X^T X + λI)^-1 X^T Y               # closed-form ridge regression
  Y_pred  = X_val @ W*
  metric  = mean cosine(Y_pred, Y_val) per head_dim

This replaces 50 gradient steps with one pseudoinverse. Mathematically
optimal for the linear case, no LR/seed/step-count guessing.

Estimated runtime: ~5-15 min total on Mac, ~1-3 min on Strix.
"""
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

AXIS = (sys.argv[1].upper() if len(sys.argv) > 1 else "V")
assert AXIS in ("V", "K", "Q"), f"axis must be V, K, or Q, got {AXIS}"


if torch.cuda.is_available():
    device = "cuda"; dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float32
else:
    device = "cpu"; dtype = torch.float32


CHECKPOINT = "Qwen/Qwen3-0.6B"
SEQ_LEN = 256
TARGET_LAYER = 14
OFFSET = 1
N_TRAIN_BATCHES = 40   # 40 * 256 = 10240 train tokens — 5x feature dim margin
N_VAL_BATCHES = 12     # 12 * 256 = 3072 val tokens
RIDGE_LAMBDA = 1e-3
RESULTS_PATH = Path(f"results/pipeline_kv_medusa_06b_dual_layer_sweep_{AXIS.lower()}_lstsq.json")


def load_owt(tokenizer, max_tokens, skip_tokens=0):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    toks = []; skipped = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        e = tokenizer.encode(t, add_special_tokens=False)
        if skipped < skip_tokens:
            skipped += len(e); continue
        toks.extend(e)
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def iter_batches(tokens, seq_len, device, n=999999):
    nb = (len(tokens) - 1) // seq_len
    idx = list(range(nb)); random.shuffle(idx)
    for i in idx[:n]:
        s = i * seq_len
        w = tokens[s:s + seq_len + 1]
        if len(w) < seq_len + 1: continue
        yield torch.tensor([w], dtype=torch.long, device=device)


def collect_features(model, tokens, n_batches, target_layer, offset, n_layers, axis, captured_Q=None):
    """Run model on n_batches; return per-layer hidden tensors and target axis values,
    flattened over (batch, time) for direct regression."""
    h_per_layer_chunks = [[] for _ in range(n_layers + 1)]
    target_chunks = []
    it = iter_batches(tokens, SEQ_LEN, device, n_batches)
    for batch in it:
        if captured_Q is not None:
            captured_Q.clear()
        with torch.no_grad():
            out = model(batch, use_cache=True, output_hidden_states=True)
            hs = [h.detach().float() for h in out.hidden_states]
            if axis == "Q":
                tgt = captured_Q["q"].detach().float()  # [B, n_attn, T, head_dim]
            else:
                layer_cache = out.past_key_values.layers[target_layer]
                tgt = (layer_cache.values if axis == "V" else layer_cache.keys).detach().float()
        ml = hs[0].shape[1] - offset
        for L in range(n_layers + 1):
            h = hs[L][:, :ml]
            h_per_layer_chunks[L].append(h.reshape(-1, h.shape[-1]))
        t = tgt[:, :, offset:].permute(0, 2, 1, 3)[:, :ml]
        target_chunks.append(t.reshape(-1, t.shape[-2] * t.shape[-1]))
    h_per_layer = [torch.cat(c, dim=0) for c in h_per_layer_chunks]
    Y = torch.cat(target_chunks, dim=0)
    return h_per_layer, Y


def fit_ridge(X, Y, lam):
    """W = (X^T X + lam I)^-1 X^T Y, closed-form ridge regression."""
    XtX = X.T @ X
    XtY = X.T @ Y
    I = torch.eye(XtX.shape[0], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(XtX + lam * I, XtY)
    return W


def main():
    print(f"device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  model loaded in {time.time()-t0:.1f}s")

    d = model.config.hidden_size
    n_kv = model.config.num_key_value_heads
    n_attn = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (d // n_attn)
    n_layers = model.config.num_hidden_layers
    n_target_heads = n_attn if AXIS == "Q" else n_kv

    # Q hook
    captured_Q = None
    if AXIS == "Q":
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        attn_layer = model.model.layers[TARGET_LAYER].self_attn
        captured_Q = {}
        orig_forward = attn_layer.forward
        def capturing_forward(hidden_states, position_embeddings, attention_mask,
                              past_key_values=None, cache_position=None, **kwargs):
            self = attn_layer
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            qs = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            ks = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            cos, sin = position_embeddings
            qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
            captured_Q["q"] = qs.detach()
            return orig_forward(hidden_states, position_embeddings, attention_mask,
                                past_key_values, cache_position, **kwargs)
        attn_layer.forward = capturing_forward

    print(f"Loading tokens + caching features (axis={AXIS})...", flush=True)
    train_tokens = load_owt(tok, max_tokens=SEQ_LEN * (N_TRAIN_BATCHES * 2))
    val_tokens = load_owt(tok, max_tokens=SEQ_LEN * (N_VAL_BATCHES * 2),
                           skip_tokens=SEQ_LEN * (N_TRAIN_BATCHES * 2))

    t0 = time.time()
    h_train, Y_train = collect_features(model, train_tokens, N_TRAIN_BATCHES, TARGET_LAYER, OFFSET, n_layers, AXIS, captured_Q)
    h_val, Y_val = collect_features(model, val_tokens, N_VAL_BATCHES, TARGET_LAYER, OFFSET, n_layers, AXIS, captured_Q)
    print(f"  features cached in {time.time()-t0:.1f}s")
    print(f"  train: N={Y_train.shape[0]} tokens, target dim={Y_train.shape[1]}")
    print(f"  val:   N={Y_val.shape[0]} tokens")
    print(f"  feature dim per layer: {h_train[0].shape[-1]}, pair feature dim: {2*h_train[0].shape[-1]}")

    pairs = [(L1, L2) for L1 in range(n_layers) for L2 in range(L1 + 1, n_layers)]
    print(f"  total pairs: {len(pairs)}")

    results = []
    sweep_t0 = time.time()
    for pair_idx, (L1, L2) in enumerate(pairs):
        X_train = torch.cat([h_train[L1], h_train[L2]], dim=-1).to(torch.float32)
        X_val = torch.cat([h_val[L1], h_val[L2]], dim=-1).to(torch.float32)

        # Closed-form ridge: W shape [2*d, n_kv*head_dim]
        W = fit_ridge(X_train, Y_train.to(torch.float32), RIDGE_LAMBDA)

        # Predict + cosine
        Y_pred = X_val @ W                                         # [N_val, n_kv*head_dim]
        Y_pred_h = Y_pred.view(-1, n_target_heads, head_dim)
        Y_true_h = Y_val.view(-1, n_target_heads, head_dim)
        cos = F.cosine_similarity(Y_pred_h.reshape(-1, head_dim),
                                  Y_true_h.reshape(-1, head_dim), dim=-1).mean().item()

        if pair_idx % 25 == 0 or pair_idx == len(pairs) - 1:
            elapsed = time.time() - sweep_t0
            eta = elapsed / (pair_idx + 1) * (len(pairs) - pair_idx - 1)
            print(f"  [{pair_idx+1:>3}/{len(pairs)}] L1={L1:>2} L2={L2:>2}  cos_{AXIS.lower()}={cos:.4f}  eta={eta:.0f}s", flush=True)
        results.append({"L1": L1, "L2": L2, f"cos_{AXIS.lower()}": round(cos, 4)})

    # Single-layer baseline (in-script, same training regime)
    print(f"\nSingle-layer baseline (linear probe, {AXIS} only)...")
    single_results = []
    for L in range(n_layers + 1):  # include embed layer
        X_train = h_train[L].to(torch.float32)
        X_val = h_val[L].to(torch.float32)
        W = fit_ridge(X_train, Y_train.to(torch.float32), RIDGE_LAMBDA)
        Y_pred = X_val @ W
        Y_pred_h = Y_pred.view(-1, n_target_heads, head_dim)
        Y_true_h = Y_val.view(-1, n_target_heads, head_dim)
        cos = F.cosine_similarity(Y_pred_h.reshape(-1, head_dim),
                                  Y_true_h.reshape(-1, head_dim), dim=-1).mean().item()
        label = "embed" if L == 0 else f"L{L}"
        print(f"  {label:>6}  cos_{AXIS.lower()}={cos:.4f}")
        single_results.append({"layer": L, f"cos_{AXIS.lower()}": round(cos, 4)})

    # Summary
    print(f"\n{'='*60}\nDUAL-LAYER SWEEP ({AXIS}, lstsq) SUMMARY\n{'='*60}")
    cos_key = f"cos_{AXIS.lower()}"
    top10 = sorted(results, key=lambda r: r[cos_key], reverse=True)[:10]
    print(f"  Top 10 PAIRS for {AXIS}:")
    for r in top10:
        print(f"    L1={r['L1']:>2} L2={r['L2']:>2}  {cos_key}={r[cos_key]:.4f}")

    best_pair = top10[0]
    best_single = max(single_results, key=lambda r: r[cos_key])
    delta = best_pair[cos_key] - best_single[cos_key]
    print(f"\n  Best pair:    (L{best_pair['L1']}, L{best_pair['L2']})  {cos_key}={best_pair[cos_key]:.4f}")
    print(f"  Best single:  L{best_single['layer']}                {cos_key}={best_single[cos_key]:.4f}")
    print(f"  Δ (dual − single) = {delta:+.4f}")
    if delta > 0.02:
        print(f"  → Dual layers carry {AXIS} information that no single layer does.")
    elif delta < 0.005:
        print(f"  → Single layer suffices; dual adds no {AXIS} signal.")
    else:
        print("  → Marginal gain.")

    out = {
        "checkpoint": CHECKPOINT, "target_layer": TARGET_LAYER, "offset": OFFSET,
        "axis": AXIS, "method": "closed-form ridge regression",
        "ridge_lambda": RIDGE_LAMBDA,
        "n_train_tokens": Y_train.shape[0], "n_val_tokens": Y_val.shape[0],
        "n_layers": n_layers, "total_pairs": len(pairs),
        "results": results, "single_layer_results": single_results,
        "best_pair": [best_pair["L1"], best_pair["L2"]],
        "best_pair_cos": best_pair[cos_key],
        "best_single_layer": best_single["layer"],
        "best_single_cos": best_single[cos_key],
        "delta_dual_minus_single": round(delta, 4),
    }
    Path(RESULTS_PATH).parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")
    print(f"Total wall time: {time.time()-sweep_t0:.0f}s")


if __name__ == "__main__":
    main()
