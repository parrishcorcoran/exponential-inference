"""
Stage 142 — Wormhole shape on BitNet (1.58-bit ternary weights).

Tests whether the wormhole topology is universal across precision regimes
or specific to FP16/BF16 transformers. If BitNet has the same shape,
the wormhole is an emergent property of training, not a quirk of high
precision.

Predictions:
  - Same bathtub shape (rank-1 throat, wider mouths)
  - Possibly WIDER throat in absolute dims (each ternary weight carries
    less info, so more dims needed to support the same flow)
  - Magnitude pump should still be present

Compares against Qwen3-0.6B baseline.

Runs on CPU so doesn't fight stage 137 on MPS.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def participation_ratio(X):
    """Continuous effective rank."""
    if X.shape[0] == 0: return 0.0
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    return float((s2.sum().pow(2) / s2.pow(2).sum().clamp(min=1e-20)).item())


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


@torch.no_grad()
def measure_wormhole(model, tokens, seq_len, device):
    """Per-layer measurements:
       - Residual PR
       - Residual norm² (magnitude pump)
       - K cache PR
       - V cache PR
    """
    L = model.config.num_hidden_layers
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=True, output_hidden_states=True)

    # Residual stream PR per layer position
    res_pr = []
    res_norm = []
    for h in out.hidden_states:  # L+1 entries
        h0 = h[0].float().cpu()  # [seq, d]
        res_pr.append(participation_ratio(h0))
        res_norm.append(float(h0.pow(2).sum().sqrt().item()))

    # KV cache PR per layer
    kv = out.past_key_values
    if hasattr(kv, "layers") and kv.layers:
        pairs = [(c.keys, c.values) for c in kv.layers]
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)

    K_pr = []
    V_pr = []
    for K, V in pairs:
        K = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        V = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_pr.append(participation_ratio(K))
        V_pr.append(participation_ratio(V))

    return {
        "L": L,
        "d_model": model.config.hidden_size,
        "residual_pr": res_pr,
        "residual_norm": res_norm,
        "K_pr": K_pr,
        "V_pr": V_pr,
    }


def try_load_model(model_id, dtype, device):
    """Try several loading paths. Some BitNet checkpoints need different settings."""
    from transformers import AutoModelForCausalLM
    attempts = [
        {"trust_remote_code": True, "attn_implementation": "eager"},
        {"trust_remote_code": True},
        {"trust_remote_code": False, "attn_implementation": "eager"},
        {},
    ]
    last_err = None
    for kwargs in attempts:
        try:
            print(f"  trying load with {kwargs}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True, **kwargs
            ).to(device).eval()
            return model
        except Exception as e:
            last_err = e
            print(f"  failed: {type(e).__name__}: {str(e)[:120]}")
            continue
    raise RuntimeError(f"all load attempts failed for {model_id}. Last error: {last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bitnet-model", default="1bitLLM/bitnet_b1_58-large",
                   help="BitNet checkpoint to load")
    p.add_argument("--baseline-model", default="Qwen/Qwen3-0.6B",
                   help="Comparison FP16 model")
    p.add_argument("--out", default="results/stage142_bitnet_wormhole.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=256)
    args = p.parse_args()

    print(f"device={args.device}", flush=True)
    dtype = torch.float32

    from transformers import AutoTokenizer

    results = {"models": {}, "comparisons": {}}

    for label, model_id in [("baseline", args.baseline_model),
                              ("bitnet", args.bitnet_model)]:
        print(f"\n{'='*60}\n=== {label}: {model_id} ===\n{'='*60}")
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            print(f"tokenizer load failed: {e}")
            continue

        try:
            model = try_load_model(model_id, dtype, args.device)
        except Exception as e:
            print(f"model load failed: {e}")
            results["models"][label] = {"id": model_id, "error": str(e)}
            continue

        # Get tokens (use tokenizer of this model)
        tokens = load_tokens(tok, args.seq_len * 2, "train")[:args.seq_len]

        print(f"measuring on {args.seq_len} tokens...")
        t0 = time.time()
        m = measure_wormhole(model, tokens, args.seq_len, args.device)
        print(f"  done in {time.time()-t0:.0f}s")
        m["id"] = model_id
        results["models"][label] = m

        L = m["L"]
        print(f"\n  per-layer wormhole shape:")
        print(f"  {'layer':>5s}  {'res_pr':>7s}  {'res_norm':>9s}  {'K_pr':>7s}  {'V_pr':>7s}")
        for l in range(L + 1):
            res_pr_l = m["residual_pr"][l]
            res_norm_l = m["residual_norm"][l]
            k_pr_l = m["K_pr"][l] if l < L else None
            v_pr_l = m["V_pr"][l] if l < L else None
            k_str = f"{k_pr_l:>7.2f}" if k_pr_l is not None else " " * 7
            v_str = f"{v_pr_l:>7.2f}" if v_pr_l is not None else " " * 7
            print(f"  {l:>5d}  {res_pr_l:>7.2f}  {res_norm_l:>9.1f}  {k_str}  {v_str}")

        del model
        import gc; gc.collect()

    # Comparison
    print(f"\n{'='*60}\n=== comparison ===\n{'='*60}")
    if "baseline" in results["models"] and "bitnet" in results["models"] \
        and "error" not in results["models"]["baseline"] \
        and "error" not in results["models"]["bitnet"]:
        b = results["models"]["baseline"]
        bn = results["models"]["bitnet"]
        # Throat finding: minimum residual PR
        throat_b = min(b["residual_pr"])
        throat_bn = min(bn["residual_pr"])
        max_b = max(b["residual_pr"])
        max_bn = max(bn["residual_pr"])
        # Magnitude growth
        max_norm_b = max(b["residual_norm"])
        min_norm_b = min(b["residual_norm"])
        max_norm_bn = max(bn["residual_norm"])
        min_norm_bn = min(bn["residual_norm"])
        print(f"\n  baseline ({b['id']}): d={b['d_model']}, L={b['L']}")
        print(f"    throat PR (min): {throat_b:.2f}")
        print(f"    mouth PR (max):  {max_b:.2f}")
        print(f"    magnitude pump:  {max_norm_b/max(min_norm_b, 1e-6):.0f}×")
        print(f"\n  bitnet  ({bn['id']}): d={bn['d_model']}, L={bn['L']}")
        print(f"    throat PR (min): {throat_bn:.2f}")
        print(f"    mouth PR (max):  {max_bn:.2f}")
        print(f"    magnitude pump:  {max_norm_bn/max(min_norm_bn, 1e-6):.0f}×")

        same_shape = (throat_bn / max_bn) < 0.3 and (throat_b / max_b) < 0.3
        results["comparisons"]["both_have_wormhole"] = same_shape
        results["comparisons"]["throat_baseline"] = throat_b
        results["comparisons"]["throat_bitnet"] = throat_bn
        results["comparisons"]["mouth_baseline"] = max_b
        results["comparisons"]["mouth_bitnet"] = max_bn

        if same_shape:
            print(f"\n  → Both models have wormhole topology (low PR throat, higher PR mouths)")
            if throat_bn > throat_b * 1.5:
                print(f"  → BitNet's throat is WIDER ({throat_bn:.1f} vs {throat_b:.1f})")
                print(f"     consistent with: ternary weights need more dims to support flow")
            elif throat_bn < throat_b * 0.7:
                print(f"  → BitNet's throat is NARROWER ({throat_bn:.1f} vs {throat_b:.1f})")
                print(f"     unexpected — ternary forces tighter compression?")
            else:
                print(f"  → Throat width comparable ({throat_bn:.1f} vs {throat_b:.1f})")
        else:
            print(f"\n  → Wormhole shape ambiguous in one or both models")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
