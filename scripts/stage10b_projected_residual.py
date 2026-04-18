"""
Stage 10b — Project the residual stream to rank-k between layers.

Stage 10 was structurally wrong: it tossed the KV cache and made each new
token a function of only itself. Autoregressive LMs can't generate that
way, manifold hypothesis or not.

Stage 10b keeps the full transformer intact — every attention module, every
MLP, full KV cache — but installs a forward hook on each decoder layer that
projects the input hidden state into rank-k coords and back before handing
it to the layer. No training. No distillation. No weight factoring.

    h_{i+1} = Layer_i( P_i · (P_i^T · (h_i - mean_i)) + mean_i )

If the boundary-layer hypothesis is clean, this round-trip is lossless at
k ≈ 9–32 and generation matches teacher. If the residual stream has
tangent-plane rotation that's bigger than rank-k at any position, it breaks.

This is the existing dynamic_rank.py idea, but:
    - On Qwen3 (not BitNet)
    - With fixed rank (not predictor-driven)
    - As a pure measurement: does the projection preserve generation?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


CALIBRATION_TEXTS = [
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology and machine learning.",
    "In quantum mechanics, the wave function describes the state of a system and evolves according to the Schrodinger equation.",
    "Protein folding is a process by which a polypeptide chain acquires its three-dimensional structure.",
    "The cosmic microwave background radiation is the thermal afterglow of the Big Bang, cooled to approximately 2.7 Kelvin.",
    "Markov chain Monte Carlo methods sample from complex probability distributions by constructing a stationary chain.",
    "The Riemann zeta function encodes deep information about the distribution of prime numbers through its non-trivial zeros.",
    "Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct.",
    "Attention mechanisms in transformers compute weighted averages over token representations across a learned subspace.",
    "Plate tectonics describes the movement of Earth lithospheric plates driven by convection currents in the mantle.",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction but hard to invert.",
    "Neurotransmitters like dopamine, serotonin, and glutamate mediate communication between neurons at chemical synapses.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases over time.",
    "Gravitational waves are ripples in the fabric of spacetime produced by accelerating masses of sufficient energy.",
    "Neural networks are approximators of functions learned from data by gradient descent on a differentiable loss.",
    "Evolution by natural selection proceeds through variation, heredity, and differential reproduction in populations.",
    "In topology, a Mobius strip is a surface with only one side and one edge, constructed from a half-twisted band.",
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
    """Return list of per-layer input tensors [N, d] across calibration tokens."""
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
    """Top-k principal directions of X. Returns (P [d, k], mean [d])."""
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)
    return P, mean


def install_projection_hooks(model, bases, means, device):
    """Register forward pre-hooks on each decoder layer that project the input
    hidden state h -> P (P^T (h - mean)) + mean. Returns a list of handles."""
    layers = model.model.layers
    handles = []
    for i, layer in enumerate(layers):
        P = bases[i].to(device)
        mean = means[i].to(device)

        def make_pre_hook(P_, mean_, idx):
            def hook(module, args, kwargs):
                # args may or may not contain hidden_states; check kwargs too
                if args and isinstance(args[0], torch.Tensor):
                    h = args[0]
                    new_args = list(args)
                    # Project: match dtype / device of h
                    P_dev = P_.to(h.device).to(h.dtype)
                    mean_dev = mean_.to(h.device).to(h.dtype)
                    h_centered = h - mean_dev
                    # [B, T, d] @ [d, k] -> [B, T, k]
                    c = h_centered @ P_dev
                    h_proj = c @ P_dev.T + mean_dev
                    new_args[0] = h_proj
                    return tuple(new_args), kwargs
                return args, kwargs
            return hook

        handles.append(layer.register_forward_pre_hook(
            make_pre_hook(P, mean, i), with_kwargs=True))
    return handles


def generate(model, tokenizer, prompt, max_new_tokens, device, warmup=0):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    for _ in range(warmup):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
    times = []
    for _ in range(max_new_tokens - 1 - warmup):
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
    p.add_argument("--ranks", default="8,16,32,64,128,256",
                   help="Comma-separated ranks to sweep")
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
    print(f"\ndevice={device}")

    ranks = [int(x) for x in args.ranks.split(",")]

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

    # === Calibrate bases once across all layers ===
    print(f"\n=== calibrating per-layer bases ===", flush=True)
    t0 = time.perf_counter()
    inputs_per_layer = capture_layer_inputs(
        model, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    n_tokens = inputs_per_layer[0].shape[0]
    print(f"  captured {n_tokens} tokens per layer in {time.perf_counter()-t0:.1f}s")

    max_rank = max(ranks)
    # Fit bases at max rank; we'll slice down per sweep step
    t0 = time.perf_counter()
    bases_full = []
    means_full = []
    for i in range(n_layers):
        P, mean = pca_basis(inputs_per_layer[i], max_rank)
        bases_full.append(P)
        means_full.append(mean)
    print(f"  PCA at rank {max_rank} in {time.perf_counter()-t0:.1f}s")

    # === Sweep ranks ===
    results = []
    for k in ranks:
        print(f"\n=== rank {k} ===", flush=True)
        bases_k = [P[:, :k].contiguous() for P in bases_full]
        means_k = [m for m in means_full]

        handles = install_projection_hooks(model, bases_k, means_k, device)
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

        print(f"  {g_ms:.2f}ms/tok  match {match}/{min_len}  first div @ {first_div}")
        print(f"  {g_text[:120]}...")

        results.append({
            "rank": k,
            "ms_per_tok": g_ms,
            "match": match,
            "total": min_len,
            "match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "sample": g_text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  teacher:  {t_ms:.2f}ms/tok")
    print(f"  {'rank':>5}  {'ms/tok':>8}  {'match':>12}  {'first div':>10}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['ms_per_tok']:>8.2f}  "
              f"{r['match']}/{r['total']:<8}  "
              f"{r['first_divergence']:>10}")

    out_path = Path(args.out_dir) / f"stage10b_projected_residual_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "hidden_size": d,
            "calibration_tokens": n_tokens,
            "teacher_ms_per_tok": t_ms,
            "teacher_sample": t_text[:400],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
