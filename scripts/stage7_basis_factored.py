"""
Stage 7 — Basis-factored decode (manifold-aware).

Stage 6 showed plain SVD(W) destroys quality even at rank 256. That's
because W's top singular directions are NOT the directions that
activations actually occupy. The manifold lives in *activation space*,
not *weight space*.

This stage uses the manifold directly: for each Linear, compute the
top-k principal directions of its ACTUAL INPUT activations on a
calibration pass. Then factor:

    y = W @ x  ≈  W @ P_in @ P_in.T @ x   when x ∈ span(P_in)

which we represent as y = A @ (B @ x) with:
    B = P_in.T       [k, d_in]
    A = W @ P_in     [d_out, k]

The approximation is exact (up to float noise) iff x lies in the
column space of P_in. Per stage1, activations lie on a ~10-dim
manifold, so small k should suffice.

Key difference from Stage 6: P_in comes from data, not from SVD(W).

Usage:
    python scripts/stage7_basis_factored.py \\
        --model Qwen/Qwen3-0.6B \\
        --ranks 8,16,32,64,128,256 \\
        --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


CALIBRATION_TEXTS = [
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology and machine learning. It suggests that both biological and artificial neural systems exploit contextual compression to reduce computational cost.",
    "In quantum mechanics, the wave function describes the state of a system and evolves according to the Schrödinger equation. Measurement collapses the wave function to an eigenstate of the observable.",
    "Protein folding is a process by which a polypeptide chain acquires its three-dimensional structure. Misfolded proteins can aggregate and cause diseases such as Alzheimer's and Parkinson's.",
    "The cosmic microwave background radiation is the thermal afterglow of the Big Bang, cooled to approximately 2.7 Kelvin by the expansion of the universe. Its discovery in 1965 provided strong evidence for cosmological models.",
    "Markov chain Monte Carlo methods sample from complex probability distributions by constructing a chain whose stationary distribution matches the target. Metropolis-Hastings and Gibbs sampling are common variants.",
    "The Riemann zeta function, defined as the analytic continuation of the Dirichlet series over complex numbers, encodes deep information about the distribution of prime numbers through its non-trivial zeros.",
    "Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct. It sustains nearly all life on Earth by forming the base of most food chains.",
    "Attention mechanisms in transformers compute weighted averages over token representations, where the weights reflect contextual relevance learned during training. Multi-head attention allows parallel attention subspaces.",
    "Plate tectonics describes the movement of Earth's lithospheric plates driven by convection in the mantle. Their interactions produce earthquakes, volcanoes, mountain ranges, and ocean trenches.",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction but hard to invert, such as integer factorization or the discrete logarithm problem over elliptic curves.",
    "Neurotransmitters like dopamine, serotonin, and glutamate mediate communication between neurons at chemical synapses. Imbalances are implicated in depression, schizophrenia, and Parkinson's disease.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases. This arrow of time emerges from the statistical behavior of microscopic states.",
]


class BasisFactoredLinear(nn.Module):
    """Linear factored through data-derived input basis: y = A (B x) + b,
    where B = P.T and A = W P for the top-k PCA basis P of input activations."""

    def __init__(self, orig: nn.Linear, P_in: torch.Tensor):
        """P_in: [d_in, k] columns = top-k basis vectors (orthonormal)."""
        super().__init__()
        dtype = orig.weight.dtype
        device = orig.weight.device
        k = P_in.shape[1]

        W = orig.weight.data.to(torch.float32).cpu()       # [d_out, d_in]
        P = P_in.to(torch.float32).cpu()                    # [d_in, k]
        A = (W @ P).to(dtype).to(device)                    # [d_out, k]
        B = P.T.contiguous().to(dtype).to(device)           # [k, d_in]

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)
        if orig.bias is not None:
            self.bias = nn.Parameter(orig.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.rank = k
        self._full_params = orig.in_features * orig.out_features
        self._factored_params = k * (orig.in_features + orig.out_features)

    def forward(self, x):
        return F.linear(F.linear(x, self.B), self.A, self.bias)


def collect_input_covariances(model, tokenizer, texts, device, max_len=256):
    """Forward-hook every target Linear, accumulate C = sum(x x^T) over calibration.
    Covariances accumulated in fp32 on device; moved to CPU at the end."""
    covs = {}
    counts = {}

    target_modules = []
    for name, module in model.named_modules():
        # `name` is the full dotted name; use the last segment to match TARGET_NAMES
        last = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and last in TARGET_NAMES:
            target_modules.append((name, module))

    def make_hook(n, d_in):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            if n not in covs:
                covs[n] = torch.zeros(d_in, d_in, device=device, dtype=torch.float32)
                counts[n] = 0
            covs[n] += x_flat.T @ x_flat
            counts[n] += x_flat.shape[0]
        return hook

    handles = []
    for name, mod in target_modules:
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))

    model.eval()
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    # Move to CPU fp64 for numerically stable eigh (cast after move — MPS has no fp64)
    cpu_covs = {n: c.cpu().to(torch.float64) for n, c in covs.items()}
    return cpu_covs, counts


def top_k_basis_from_cov(cov: torch.Tensor, k: int) -> torch.Tensor:
    """cov: [d, d] PSD (fp64). Return top-k eigenvectors as columns [d, k]."""
    # eigh returns ascending eigenvalues
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # Take the top-k (largest) columns
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:]
    # Flip so column 0 is the largest direction (convention, not required)
    P = P.flip(dims=[1]).contiguous()
    return P


def factorize_with_basis(model, covariances, rank: int):
    """Replace every target Linear with BasisFactoredLinear using top-k
    eigenvectors of its input covariance."""
    stats = {"n_replaced": 0, "full_params": 0, "factored_params": 0}
    # Pre-compute bases once
    bases = {n: top_k_basis_from_cov(c, rank) for n, c in covariances.items()}

    # Walk and replace
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in TARGET_NAMES:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name not in bases:
                continue
            P = bases[full_name].to(torch.float32)
            fact = BasisFactoredLinear(child, P_in=P)
            setattr(module, child_name, fact)
            stats["n_replaced"] += 1
            stats["full_params"] += fact._full_params
            stats["factored_params"] += fact._factored_params
    return stats


def generate(model, tokenizer, prompt, max_new_tokens, device, warmup=2):
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


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="8,16,32,64,128,256")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
    p.add_argument("--calib-max-len", type=int, default=256)
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

    ranks = [int(x) for x in args.ranks.split(",")]

    print(f"\n=== loading baseline {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    print(f"  {n_layers} layers, hidden={hidden}, intermediate={intermediate}")

    # === Baseline decode ===
    print(f"\n=== baseline decode ===", flush=True)
    base_times, base_text, base_tokens = generate(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_ms = sum(base_times) / len(base_times)
    print(f"  {len(base_times)} timed tokens, {base_ms:.2f}ms/tok")
    print(f"  {base_text[:120]}...")

    # === Collect input covariances ONCE ===
    print(f"\n=== collecting calibration covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, counts = collect_input_covariances(
        model, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    calib_time = time.perf_counter() - t0
    total_tokens = next(iter(counts.values())) if counts else 0
    print(f"  {len(covs)} covariance matrices, {total_tokens} tokens/layer, "
          f"{calib_time:.1f}s")

    del model
    if device == "mps":
        torch.mps.empty_cache()

    # === Rank sweep ===
    results = []
    for k in ranks:
        print(f"\n=== rank {k} ===", flush=True)
        model, _ = load_model(args.model, device)

        t0 = time.perf_counter()
        fstats = factorize_with_basis(model, covs, rank=k)
        fact_time = time.perf_counter() - t0
        size_ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
        print(f"  factored {fstats['n_replaced']} linears in {fact_time:.1f}s")
        print(f"  weight params: {fstats['full_params']/1e6:.1f}M -> "
              f"{fstats['factored_params']/1e6:.2f}M ({size_ratio:.2%})")

        rtimes, rtext, rtokens = generate(
            model, tokenizer, args.prompt, args.max_new_tokens, device)
        rms = sum(rtimes) / len(rtimes)

        min_len = min(len(base_tokens), len(rtokens))
        match = sum(1 for a, b in zip(base_tokens[:min_len], rtokens[:min_len]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(
            zip(base_tokens, rtokens)) if a != b), min_len)
        speedup = base_ms / rms if rms > 0 else 0

        print(f"  {len(rtimes)} timed tokens, {rms:.2f}ms/tok  (speedup: {speedup:.2f}x)")
        print(f"  match: {match}/{min_len} ({match/max(min_len,1):.1%}), "
              f"first divergence @ token {first_div}")
        print(f"  {rtext[:120]}...")

        results.append({
            "rank": k,
            "ms_per_tok": rms,
            "speedup_vs_baseline": speedup,
            "token_match": f"{match}/{min_len}",
            "token_match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "weight_params_full_M": fstats["full_params"] / 1e6,
            "weight_params_factored_M": fstats["factored_params"] / 1e6,
            "weight_size_ratio": size_ratio,
            "factorize_time_sec": fact_time,
            "sample_text": rtext[:300],
        })

        del model
        if device == "mps":
            torch.mps.empty_cache()

    # === Summary ===
    print(f"\n=== summary ===")
    print(f"  baseline: {base_ms:.2f}ms/tok")
    print(f"  {'rank':>5}  {'ms/tok':>8}  {'speedup':>8}  {'match':>10}  "
          f"{'weights':>9}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['ms_per_tok']:>8.2f}  "
              f"{r['speedup_vs_baseline']:>7.2f}x  "
              f"{r['token_match']:>10}  "
              f"{r['weight_size_ratio']:>8.2%}")

    out_path = Path(args.out_dir) / f"stage7_basis_factored_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "calibration_tokens_per_layer": total_tokens,
            "calibration_texts": len(CALIBRATION_TEXTS),
            "baseline_ms_per_tok": base_ms,
            "baseline_sample": base_text[:500],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
