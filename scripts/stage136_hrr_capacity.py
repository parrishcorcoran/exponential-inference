"""
Stage 136 — HRR capacity of the KV cache: can we compress the cache to
a single superposition vector and still retrieve content?

The KV cache is structurally an HRR superposition: each (K_t, V_t) pair
is a (role, filler) binding, the sum is the holographic store. Standard
attention uses softmax-based retrieval which is approximate unbinding.

This stage tests classical HRR retrieval directly:
  - Build C = Σ_t bind(K_t, V_t) where bind = circular convolution
  - Retrieve V_t_hat = unbind(K_t, C) where unbind = circular correlation
  - Measure cosine similarity to ground truth V_t
  - Find capacity (where retrieval breaks)

HRR theory predicts capacity ≈ d / (4 log d). For d=1024 that's ~37
items. For d=2048 ~70 items.

Runs on CPU so stage 135 can keep using MPS without interference.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def hrr_bind(a, b):
    """Circular convolution via FFT. Returns real part."""
    A = torch.fft.fft(a)
    B = torch.fft.fft(b)
    return torch.fft.ifft(A * B).real


def hrr_unbind(a, c):
    """Circular correlation = convolution with reversed/conjugated."""
    A = torch.fft.fft(a)
    C = torch.fft.fft(c)
    return torch.fft.ifft(A.conj() * C).real


def hrr_normalize(x):
    """Normalize for HRR: zero-mean, unit-norm."""
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-10)


def cosine(a, b):
    a = a / a.norm().clamp(min=1e-10)
    b = b / b.norm().clamp(min=1e-10)
    return (a * b).sum().item()


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
def collect_kv_cpu(model, tokens, seq_len, device):
    """Collect K, V per layer, move to CPU."""
    ids = torch.tensor([tokens[:seq_len]], dtype=torch.long, device=device)
    out = model(ids, use_cache=True)
    kv = out.past_key_values
    if hasattr(kv, "layers") and kv.layers:
        pairs = [(c.keys, c.values) for c in kv.layers]
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)
    K_list, V_list = [], []
    for K, V in pairs:
        K = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        V = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_list.append(K)
        V_list.append(V)
    return K_list, V_list


def hrr_capacity_test(K, V, n_steps, n_probes=10, normalize=True):
    """For increasing cache sizes N, measure retrieval cosine at randomly
       sampled positions.
       Returns: list of (N, mean_cosine, std_cosine, baseline_cosine)
       where baseline is retrieval from a non-target token (control)."""
    seq_len, d = K.shape
    if normalize:
        K = hrr_normalize(K)
        V = hrr_normalize(V)

    # Sample positions to probe
    rng = np.random.RandomState(42)
    probe_positions = rng.choice(seq_len, size=min(n_probes, seq_len), replace=False)

    # Compute incremental superposition
    C = torch.zeros(d)
    bound_per_pos = []  # store bind(K_t, V_t) per position for incremental
    for t in range(seq_len):
        bound_per_pos.append(hrr_bind(K[t], V[t]))

    # Test points
    Ns = np.unique(np.round(np.geomspace(1, seq_len, n_steps)).astype(int))
    Ns = [n for n in Ns if 1 <= n <= seq_len]

    results = []
    for N in Ns:
        # Build superposition of first N tokens
        C = torch.stack([bound_per_pos[t] for t in range(N)]).sum(0)

        target_cos = []
        baseline_cos = []
        for t in probe_positions:
            if t >= N: continue
            V_hat = hrr_unbind(K[t], C)
            target_cos.append(cosine(V_hat, V[t]))
            # Baseline: try to retrieve a non-included token (or wrong slot)
            t_other = (t + N // 2 + 1) % N
            if t_other != t and t_other < N:
                baseline_cos.append(cosine(V_hat, V[t_other]))
        if len(target_cos) > 0:
            results.append({
                "N": int(N),
                "n_samples": len(target_cos),
                "mean_target_cos": float(np.mean(target_cos)),
                "std_target_cos": float(np.std(target_cos)),
                "mean_baseline_cos": float(np.mean(baseline_cos)) if baseline_cos else 0.0,
                "std_baseline_cos": float(np.std(baseline_cos)) if baseline_cos else 0.0,
            })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage136_hrr_capacity.json")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--layers", default="0,7,14,21,27",
                   help="comma-separated layer indices to test")
    p.add_argument("--n-probes", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=15)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    layers = [int(x) for x in args.layers.split(",")]
    print(f"device={device}  layers={layers}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    print(f"L={L}")

    print(f"loading WikiText-2 ({args.seq_len} tokens)...")
    tokens = load_tokens(tokenizer, args.seq_len * 2, "train")[:args.seq_len]

    print("collecting K, V from one forward pass...")
    K_all, V_all = collect_kv_cpu(model, tokens, args.seq_len, device)
    d_kv = K_all[0].shape[1]
    print(f"  d_kv = {d_kv}")
    print(f"  HRR theoretical capacity ≈ d/(4 log d) = "
          f"{d_kv / (4 * np.log(d_kv)):.0f} items")

    # Free model memory — we don't need it anymore
    del model
    if device == "mps":
        torch.mps.empty_cache()

    results = {"model": args.model, "d_kv": d_kv, "seq_len": args.seq_len,
                "theoretical_capacity": d_kv / (4 * np.log(d_kv)),
                "per_layer": {}}

    for l in layers:
        print(f"\n{'=' * 60}\n=== layer {l} ===\n{'=' * 60}")
        t0 = time.time()
        capacity = hrr_capacity_test(K_all[l], V_all[l],
                                       n_steps=args.n_steps,
                                       n_probes=args.n_probes,
                                       normalize=True)
        dur = time.time() - t0
        # Find capacity: largest N where mean_target_cos > 2 × mean_baseline_cos
        cap = None
        for r in capacity:
            if r["mean_target_cos"] > 2 * abs(r["mean_baseline_cos"]) + 0.05:
                cap = r["N"]
        results["per_layer"][str(l)] = {
            "curve": capacity,
            "capacity_estimate": cap,
            "duration_s": dur,
        }
        print(f"  curve (N → mean target cos / baseline cos):")
        for r in capacity:
            sig = "✓" if r["mean_target_cos"] > 2 * abs(r["mean_baseline_cos"]) + 0.05 else " "
            print(f"    N={r['N']:>3d}  target={r['mean_target_cos']:>+.4f} ± {r['std_target_cos']:.4f}  "
                  f"baseline={r['mean_baseline_cos']:>+.4f}  {sig}")
        print(f"  practical capacity at this layer: {cap}")
        print(f"  ({dur:.0f}s)")

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  HRR theoretical capacity: ~{d_kv / (4 * np.log(d_kv)):.0f} items")
    for l in layers:
        cap = results["per_layer"][str(l)]["capacity_estimate"]
        print(f"  L{l}: practical capacity = {cap}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
