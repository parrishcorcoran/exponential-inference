"""
Stage 11 — Entropy-driven dynamic-rank residual projection.

Stage 10b showed that fixed-rank projection breaks generation below rank
~500. But the physics says the required rank should vary with the
system's relaxation state (entropy). When the system is relaxed
(low attention entropy), a small rank suffices. When frustrated
(high entropy, possibly near a saddle), we need more rank.

This stage wires entropy -> rank dynamically, cheaply:

    For free during eager attention:   attn_weights per layer, per step
    Cheap post-hook per layer:         ent = -sum(w * log(w)) averaged over heads
    Pre-hook per layer:                project h into rank-k(ent) basis, back

No training, no online SVD, no new compute besides a log/sum over the
already-materialized attention weights. The calibration bases (stage 10b
PCA) are precomputed at max rank; at inference we slice the top-k columns
determined by current entropy.

Rank schedule:
    k(H) = k_min + (k_max - k_min) * clip(H / log(T_kv), 0, 1)

where T_kv is the current cache length.  H / log(T) is entropy normalized
to [0, 1] (0 = one-hot, 1 = uniform).

Usage:
    python scripts/stage11_entropy_driven.py \\
        --model Qwen/Qwen3-0.6B --k-min 32 --k-max 512 --device mps
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


CALIBRATION_TEXTS = [
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology.",
    "In quantum mechanics, the wave function describes the state of a system and evolves according to the Schrodinger equation.",
    "Protein folding is a process by which a polypeptide chain acquires its three-dimensional structure.",
    "The cosmic microwave background radiation is the thermal afterglow of the Big Bang.",
    "Markov chain Monte Carlo methods sample from complex probability distributions.",
    "The Riemann zeta function encodes deep information about the distribution of prime numbers.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Attention mechanisms in transformers compute weighted averages over token representations.",
    "Plate tectonics describes the movement of Earth lithospheric plates driven by convection in the mantle.",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction but hard to invert.",
    "Neurotransmitters like dopamine and serotonin mediate communication between neurons.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases.",
    "Gravitational waves are ripples in spacetime produced by accelerating masses.",
    "Neural networks are approximators of functions learned from data by gradient descent on a loss.",
    "Evolution by natural selection proceeds through variation, heredity, and differential reproduction.",
    "In topology, a Mobius strip is a surface with only one side and one edge.",
]


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def capture_layer_inputs(model, tokenizer, texts, device, max_len=256):
    n_layers = model.config.num_hidden_layers
    all_inputs = [[] for _ in range(n_layers)]
    model.eval()
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for i in range(n_layers):
                all_inputs[i].append(hs[i][0].to(torch.float32).cpu())
    return [torch.cat(xs, dim=0) for xs in all_inputs]


def pca_basis(X: torch.Tensor, k: int):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)
    return P, mean


class EntropyDrivenProjection:
    """Per-step: attention modules populate entropy_per_layer after their
    forward; decoder-layer pre-hooks on the NEXT step read it to choose rank."""

    def __init__(self, bases, means, k_min, k_max, device):
        self.bases = [P.to(device) for P in bases]         # [d, k_max]
        self.means = [m.to(device) for m in means]         # [d]
        self.k_min = k_min
        self.k_max = k_max
        self.entropy_per_layer = {}        # layer_idx -> normalized entropy [0,1]
        self.ranks_used = []               # list[ (step, layer) -> k ]

    def entropy_hook(self, layer_idx):
        """Post-hook on each attention module. Eager attention returns
        (attn_output, attn_weights) or (attn_output,). Capture normalized
        entropy of the last query's attention distribution, averaged over heads."""
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                return
            w = output[1]
            if w is None:
                return
            # w: [B, H, T_q, T_kv]. Last query only.
            last = w[0, :, -1, :]                           # [H, T_kv]
            T = last.shape[-1]
            if T <= 1:
                self.entropy_per_layer[layer_idx] = 0.0
                return
            ent = -(last * torch.log(last + 1e-10)).sum(dim=-1)  # [H]
            # Normalize by log(T) so entropy ∈ [0, 1] regardless of context length
            norm = math.log(T)
            self.entropy_per_layer[layer_idx] = float(ent.mean().item() / norm)
        return hook

    def projection_pre_hook(self, layer_idx):
        """Pre-hook on each decoder layer. Uses entropy captured at THIS layer
        on the previous step to pick rank for this step."""
        def hook(module, args, kwargs):
            if not args or not isinstance(args[0], torch.Tensor):
                return args, kwargs
            h = args[0]
            # If we don't have an entropy signal yet (first step), pass through at full rank.
            ent = self.entropy_per_layer.get(layer_idx, None)
            if ent is None:
                return args, kwargs

            # Map entropy [0,1] -> rank [k_min, k_max]. Higher entropy = more rank.
            ent_clipped = max(0.0, min(1.0, ent))
            k = int(round(self.k_min + (self.k_max - self.k_min) * ent_clipped))
            k = max(1, min(k, self.bases[layer_idx].shape[1]))
            self.ranks_used.append((layer_idx, k, ent_clipped))

            # Project
            P = self.bases[layer_idx][:, :k].to(h.dtype)       # [d, k]
            mean = self.means[layer_idx].to(h.dtype)           # [d]
            h_centered = h - mean
            c = h_centered @ P                                  # [..., k]
            h_proj = c @ P.T + mean                             # [..., d]
            new_args = list(args)
            new_args[0] = h_proj
            return tuple(new_args), kwargs
        return hook


def install(edp: EntropyDrivenProjection, model):
    handles = []
    for i, layer in enumerate(model.model.layers):
        # Entropy captured from the attention module's output
        handles.append(layer.self_attn.register_forward_hook(edp.entropy_hook(i)))
        # Projection applied at the input of the NEXT decoder layer step
        handles.append(layer.register_forward_pre_hook(
            edp.projection_pre_hook(i), with_kwargs=True))
    return handles


def generate(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    times = []
    for _ in range(max_new_tokens - 1):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return [t * 1000 for t in times], text, generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--k-min", type=int, default=32)
    p.add_argument("--k-max", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
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
    print(f"\ndevice={device}  k_min={args.k_min}  k_max={args.k_max}")

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {n_layers} layers, hidden={d}")

    # === Teacher reference ===
    print(f"\n=== teacher reference ===", flush=True)
    t_times, t_text, t_tokens = generate(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    t_ms = sum(t_times) / len(t_times)
    print(f"  {t_ms:.2f}ms/tok  {t_text[:120]}...")

    # === Calibrate ===
    print(f"\n=== calibrating bases at rank {args.k_max} ===", flush=True)
    t0 = time.perf_counter()
    inputs_per_layer = capture_layer_inputs(
        model, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    print(f"  captured {inputs_per_layer[0].shape[0]} tokens in "
          f"{time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    bases, means = [], []
    for i in range(n_layers):
        P, mean = pca_basis(inputs_per_layer[i], args.k_max)
        bases.append(P)
        means.append(mean)
    print(f"  PCA in {time.perf_counter()-t0:.1f}s")

    # === Entropy-driven decode ===
    print(f"\n=== entropy-driven decode ===", flush=True)
    edp = EntropyDrivenProjection(bases, means, args.k_min, args.k_max, device)
    handles = install(edp, model)
    try:
        g_times, g_text, g_tokens = generate(
            model, tokenizer, args.prompt, args.max_new_tokens, device)
    finally:
        for h in handles:
            h.remove()

    g_ms = sum(g_times) / len(g_times)
    min_len = min(len(t_tokens), len(g_tokens))
    match = sum(1 for a, b in zip(t_tokens[:min_len], g_tokens[:min_len]) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, g_tokens)) if a != b), min_len)

    # Rank statistics per layer over all steps
    rank_stats = {}
    for layer_idx, k, ent in edp.ranks_used:
        rank_stats.setdefault(layer_idx, []).append(k)
    avg_rank_per_layer = {i: sum(ks)/len(ks) for i, ks in rank_stats.items()}
    overall_avg_rank = sum(
        sum(ks) for ks in rank_stats.values()) / max(
        sum(len(ks) for ks in rank_stats.values()), 1)

    print(f"  {g_ms:.2f}ms/tok  match {match}/{min_len}  first div @ {first_div}")
    print(f"  avg rank across (layer, step): {overall_avg_rank:.1f}")
    print(f"  {g_text[:200]}")

    # === Summary ===
    print(f"\n=== summary ===")
    speedup = t_ms / g_ms if g_ms > 0 else 0
    print(f"  teacher:       {t_ms:.2f}ms/tok")
    print(f"  entropy-driven:{g_ms:.2f}ms/tok  ({speedup:.2f}x)")
    print(f"  match: {match}/{min_len}  ({match/max(min_len,1):.1%})")
    print(f"  first divergence: @ {first_div}")
    print(f"  avg rank: {overall_avg_rank:.1f} / {args.k_max}")

    out_path = Path(args.out_dir) / f"stage11_entropy_driven_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "n_layers": n_layers,
            "hidden_size": d,
            "teacher_ms_per_tok": t_ms,
            "entropy_driven_ms_per_tok": g_ms,
            "speedup": speedup,
            "match": match,
            "total": min_len,
            "match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "avg_rank_overall": overall_avg_rank,
            "avg_rank_per_layer": avg_rank_per_layer,
            "teacher_sample": t_text[:400],
            "entropy_sample": g_text[:400],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
