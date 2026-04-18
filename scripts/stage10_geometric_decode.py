"""
Stage 10 — Geometric decode (no training, no distillation).

Direct test of the physics framing: if the residual stream lives on a thin
crystallized boundary layer, then each decoder layer's action is a small map
in rank-k coordinates — measurable by least-squares on calibration
trajectories, not predicted by training.

Pipeline:
    1. Run teacher on calibration corpus, capture per-layer hidden-state IO.
    2. Per layer, compute rank-k PCA basis P_i from inputs.
    3. Per layer, fit transport M_i: c_{i+1} ≈ M_i · c_i via least squares
       in k-dim. Report R^2 per layer — if high, the layer IS a small
       geometric operator.
    4. Build a geometric generator: embed -> project -> apply M_0..M_{L-1}
       -> back-project -> lm_head. No attention, no KV cache — just the
       averaged per-layer transport.
    5. Generate tokens from prompt. Compare to teacher.

This is the honest first cut. It ignores attention (treats each layer as a
static map averaged over calibration contexts). If it produces fluent or
teacher-matching text, the boundary-layer-is-geometry claim is clean and
we can add attention as a refinement. If it fails, we know the per-layer
action genuinely depends on cross-token attention structure, and the
geometric form needs a small attention-like operator per layer.

Usage:
    python scripts/stage10_geometric_decode.py \\
        --model Qwen/Qwen3-0.6B --rank 32 --device mps
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
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction.",
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


def capture_layer_io(model, tokenizer, texts, device, max_len=256):
    """Run the model with output_hidden_states=True and return a stacked tensor
    of layer inputs and outputs across all calibration tokens.

    hidden_states from HF is a tuple of length L+1:
        hidden_states[0]  = embedding output (input to layer 0)
        hidden_states[i]  = output of layer i-1 = input of layer i
        hidden_states[L]  = output of last layer (before final norm + lm_head)

    So for layer i, input = hidden_states[i], output = hidden_states[i+1].
    """
    inputs_per_layer = None   # [L, N_total, d]
    outputs_per_layer = None
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size

    all_inputs = [[] for _ in range(n_layers)]
    all_outputs = [[] for _ in range(n_layers)]

    model.eval()
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states  # tuple of L+1 tensors, each [1, T, d]
            for i in range(n_layers):
                all_inputs[i].append(hs[i][0].to(torch.float32).cpu())
                all_outputs[i].append(hs[i + 1][0].to(torch.float32).cpu())

    inputs_per_layer = [torch.cat(xs, dim=0) for xs in all_inputs]    # [N, d] each
    outputs_per_layer = [torch.cat(xs, dim=0) for xs in all_outputs]
    return inputs_per_layer, outputs_per_layer


def pca_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k right singular vectors of centered X. X: [N, d], returns [d, k]."""
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    # torch.linalg.svd can fail on very rectangular matrices; use SVD of covariance
    # for robustness (d <= N typical).
    cov = Xc.T @ Xc  # [d, d], fp32 -> cast fp64 for eigh stability
    cov64 = cov.to(torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(cov64)
    k_eff = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)  # [d, k]
    return P, mean.squeeze(0)  # return mean for centering


def fit_transport(c_in: torch.Tensor, c_out: torch.Tensor):
    """Solve c_out = c_in @ M.T + b by least squares. c_in/c_out: [N, k].
    Returns M [k, k], b [k], R^2."""
    # Augment c_in with a column of ones to fold b into the solve.
    N, k = c_in.shape
    Xa = torch.cat([c_in, torch.ones(N, 1, dtype=c_in.dtype)], dim=1)  # [N, k+1]
    # lstsq: solve Xa · W = c_out, W: [k+1, k]
    sol = torch.linalg.lstsq(Xa, c_out).solution  # [k+1, k]
    M = sol[:k].T.contiguous()      # [k, k]
    b = sol[k]                       # [k]

    # R^2
    pred = c_in @ M.T + b
    ss_res = (c_out - pred).pow(2).sum().item()
    ss_tot = (c_out - c_out.mean(dim=0, keepdim=True)).pow(2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return M, b, r2


def geometric_generate(model, tokenizer, prompt, max_new_tokens, bases, means,
                       Ms, bs, device):
    """Generate greedily using only per-layer geometric transport, no attention.

    For each forward pass:
        1. Tokens -> embedding h_0  (from model.model.embed_tokens)
        2. Project h_0 into rank-k coords via P_0: c_0 = P_0^T · (h_0 - mean_0)
        3. For each layer i: c_{i+1} = M_i · c_i + b_i
        4. Back-project final coords: h_L = P_{L} · c_L + mean_L  (use last layer's
           input basis as output basis, since hidden_states[L] = output of layer L-1)
        5. Apply final layernorm + lm_head to get logits.

    We do NOT build a KV cache. Each new token is decoded independently (no
    cross-token dependence), which is a pure test of the static transport.
    """
    # Get embed + norm + lm_head modules
    embed = model.model.embed_tokens
    # Qwen3 has a final norm in model.model.norm
    final_norm = model.model.norm
    lm_head = model.lm_head

    n_layers = len(Ms)
    # Final output of the last layer = hidden_states[n_layers]. We captured
    # inputs_per_layer[n_layers-1] = hidden_states[n_layers - 1]. The OUTPUT
    # of layer n_layers-1 is outputs_per_layer[n_layers-1]. We need to
    # back-project the final c. We built M_{L-1} to map c_{L-1} -> c_L.
    # But we don't have a basis P_L for the output of the last layer —
    # we only have bases for inputs, i.e., P_0..P_{L-1} corresponding to
    # hidden_states[0..L-1]. We need P_L for hidden_states[L].
    #
    # This is a simplification: for the final back-projection, we approximate
    # the output of layer L-1 in the basis P_{L-1}. This is slightly wrong
    # if hidden_states[L-1] and hidden_states[L] live in rotated tangent
    # planes, but for a first-cut test it's acceptable.

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # We only use the last token's forward — no attention / no KV cache.
    # For the prompt, we just take the last token and treat it as the
    # starting point of the geometric trajectory.
    last_token = input_ids[:, -1:]

    generated = []
    times = []
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        with torch.inference_mode():
            # 1. Embed
            h = embed(last_token)  # [1, 1, d], bfloat16
            h = h[0, 0].to(torch.float32).cpu()  # [d]

            # 2. Project to c_0
            c = bases[0].T @ (h - means[0])  # [k]

            # 3. Apply transports
            for i in range(n_layers):
                c = Ms[i] @ c + bs[i]  # [k]

            # 4. Back-project using last-layer INPUT basis (approximation)
            h_out = bases[-1] @ c + means[-1]  # [d], fp32

            # 5. Final norm + lm_head
            h_out_dev = h_out.to(next(model.parameters()).dtype).to(device).unsqueeze(0).unsqueeze(0)
            h_out_normed = final_norm(h_out_dev)
            logits = lm_head(h_out_normed)[0, 0]  # [vocab]
            next_token = logits.argmax().item()

        times.append(time.perf_counter() - t0)
        generated.append(next_token)
        last_token = torch.tensor([[next_token]], device=device)
        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return [t * 1000 for t in times], text, generated


def teacher_generate(model, tokenizer, prompt, max_new_tokens, device):
    """Standard greedy decode with KV cache (reference)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=50)
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
    print(f"\ndevice={device}  rank={args.rank}")

    print(f"\n=== loading {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {n_layers} layers, hidden={d}")

    # === Teacher reference ===
    print(f"\n=== teacher reference ===", flush=True)
    t_text, t_tokens = teacher_generate(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    print(f"  {t_text[:120]}...")

    # === Capture per-layer IO ===
    print(f"\n=== capturing layer IO on calibration ===", flush=True)
    t0 = time.perf_counter()
    inputs_per_layer, outputs_per_layer = capture_layer_io(
        model, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    n_tokens = inputs_per_layer[0].shape[0]
    print(f"  {n_tokens} tokens across {n_layers} layers in "
          f"{time.perf_counter()-t0:.1f}s")

    # === Per-layer PCA + transport fit ===
    print(f"\n=== fitting bases and transports ===", flush=True)
    bases = []       # P_i: [d, k]
    means = []       # mean_i: [d]
    Ms = []          # M_i: [k, k]
    bs = []          # b_i: [k]
    r2s = []
    t0 = time.perf_counter()
    for i in range(n_layers):
        X_in = inputs_per_layer[i]
        X_out = outputs_per_layer[i]
        P, mean = pca_basis(X_in, args.rank)
        c_in = (X_in - mean) @ P          # [N, k]
        # For the OUTPUT, project using the SAME basis P (since layer residual
        # keeps output close to input — they're both on the boundary layer
        # at this position in the stack).
        c_out = (X_out - mean) @ P
        M, b, r2 = fit_transport(c_in, c_out)
        bases.append(P)
        means.append(mean)
        Ms.append(M)
        bs.append(b)
        r2s.append(r2)
        if i < 5 or i == n_layers - 1 or i % 5 == 0:
            print(f"  layer {i:2d}  R^2={r2:.4f}")
    print(f"  fit in {time.perf_counter()-t0:.1f}s")
    print(f"  R^2 summary: min={min(r2s):.4f}  median={sorted(r2s)[len(r2s)//2]:.4f}  "
          f"max={max(r2s):.4f}")
    print(f"  mean={sum(r2s)/len(r2s):.4f}")

    # === Geometric decode ===
    print(f"\n=== geometric decode (no training, no attention) ===", flush=True)
    g_times, g_text, g_tokens = geometric_generate(
        model, tokenizer, args.prompt, args.max_new_tokens,
        bases, means, Ms, bs, device)
    g_ms = sum(g_times) / len(g_times)
    print(f"  {g_ms:.2f}ms/tok  (note: no kv cache, per-token pure transport)")
    print(f"  {g_text[:200]}")

    min_len = min(len(t_tokens), len(g_tokens))
    match = sum(1 for a, b in zip(t_tokens[:min_len], g_tokens[:min_len]) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, g_tokens)) if a != b), min_len)
    print(f"\n=== comparison ===")
    print(f"  teacher : {t_text[:120]}...")
    print(f"  geometric: {g_text[:120]}...")
    print(f"  match: {match}/{min_len}  first divergence @ {first_div}")

    # Save
    out_path = Path(args.out_dir) / f"stage10_geometric_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "rank": args.rank,
            "n_layers": n_layers,
            "hidden_size": d,
            "calibration_tokens": n_tokens,
            "per_layer_r2": r2s,
            "r2_summary": {
                "min": min(r2s),
                "median": sorted(r2s)[len(r2s)//2],
                "mean": sum(r2s)/len(r2s),
                "max": max(r2s),
            },
            "teacher_sample": t_text[:400],
            "geometric_sample": g_text[:400],
            "match": match,
            "total": min_len,
            "match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "geometric_ms_per_tok": g_ms,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
