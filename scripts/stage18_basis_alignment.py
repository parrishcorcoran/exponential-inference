"""
Stage 18 — Alignment between text-weighted embedding basis and per-layer
activation basis.

Stage 17 showed text-weighted embeddings have TwoNN ≈ 12, matching the
~10 manifold we measure from activations. But similar INTRINSIC DIM
doesn't mean the same SUBSPACE. Two point clouds with TwoNN ≈ 10 could
sit in totally different 10-dim subspaces of the 1024-dim hidden space.

Test: compute rank-k basis from
    (a) text-weighted embeddings (P_embed)
    (b) per-layer activations         (P_act[i], for each layer i)
For each layer i, measure subspace overlap between P_embed and P_act[i]
using principal angles (via Frobenius norm of P_embed.T @ P_act[i] / sqrt(k)).

Overlap interpretation:
    1.0  — P_embed spans the same subspace as P_act[i]  ← substitution works
    ~0.5 — half the directions align
    0.0  — completely orthogonal subspaces

If P_embed overlaps strongly with mid-stack P_act[i], the embedding-
derived basis can substitute for the activation-derived basis in
factored-weight construction.  Enables skipping forward-pass calibration
entirely.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


CALIBRATION_TEXTS = [
    "The cell is the basic structural unit of life. Every organism is composed of one or more cells.",
    "Quantum mechanics describes the behaviour of matter and energy at atomic and subatomic scales.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes and transistors.",
    "Climate change is driven primarily by greenhouse gas emissions from fossil fuel combustion.",
    "Language models learn statistical structure from text by optimizing a next-token prediction objective.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
    "Neural networks consist of layers of parameterised transformations trained by gradient descent.",
    "The Roman Empire expanded through military power, political institutions, and engineering.",
    "Relativity theory links space, time, matter, and energy through Einstein's field equations.",
    "Artificial intelligence studies agents that perceive, reason, learn, and act in complex environments.",
    "Cryptography protects information via operations easy in one direction and hard to invert.",
    "Neurotransmitters mediate communication between neurons at chemical synapses.",
    "Evolution by natural selection operates on heritable variation through differential reproduction.",
    "Protein synthesis begins with transcription of DNA into messenger RNA in ribosomes.",
    "Plate tectonics describes the movement of Earth's lithospheric plates over the mantle.",
    "Graph theory studies mathematical structures of vertices connected by edges.",
    "The Renaissance reshaped art, science, and philosophy through classical revival.",
    "Public-key cryptography relies on asymmetric mathematical operations.",
    "Stars produce energy through nuclear fusion in their cores over billions of years.",
    "In statistics, a normal distribution has mean and standard deviation parameters.",
    "The standard model unifies three fundamental interactions with gauge bosons and fermions.",
    "Homeostasis is the tendency of a biological system to regulate internal conditions.",
    "Thermodynamics describes energy, heat, work, entropy, and macroscopic equilibria.",
    "In linguistics, syntax governs the rules by which words combine into sentences.",
    "Convex optimization solves problems with convex feasible regions and objectives.",
    "Game theory models strategic interactions among rational agents maximizing payoffs.",
    "Ecology studies how organisms interact through food webs, niches, and populations.",
    "Fluid turbulence produces multi-scale vortices transferring energy across scales.",
    "Memory in the brain involves synaptic plasticity and distributed storage patterns.",
    "Operating systems manage hardware, scheduling, and protection among programs.",
]


def pca_basis(X, k):
    """Top-k principal directions. X: [N, d]. Returns (P [d, k], mean [d])."""
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)
    return P, mean


def subspace_overlap(P_a, P_b):
    """Frobenius norm of P_a.T @ P_b divided by sqrt(k).
    Returns scalar in [0, 1]: mean cosine of principal angles between subspaces."""
    k = P_a.shape[1]
    M = P_a.T @ P_b  # [k, k]
    return float(torch.linalg.norm(M, ord="fro").item() / (k ** 0.5))


def principal_angles(P_a, P_b):
    """SVD of P_a.T @ P_b gives singular values = cosines of principal angles."""
    M = P_a.T @ P_b
    _, S, _ = torch.linalg.svd(M)
    return S.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage18_basis_alignment.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}  model={args.model}  rank={args.rank}")

    print(f"\n=== loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers

    # === Tokenize corpus once ===
    all_ids = []
    for text in CALIBRATION_TEXTS:
        ids = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=args.max_len).input_ids
        all_ids.append(ids)
    total_tokens = sum(x.shape[1] for x in all_ids)
    print(f"  corpus: {total_tokens} tokens across {len(all_ids)} texts")

    # === P_embed from text-weighted embeddings ===
    embed = model.model.embed_tokens.weight.detach().cpu().to(torch.float32)
    positions = torch.cat([ids[0] for ids in all_ids], dim=0)
    X_embed = embed[positions]  # [N, d]
    P_embed, _ = pca_basis(X_embed, args.rank)
    print(f"\n=== P_embed computed: [{X_embed.shape[0]}, {X_embed.shape[1]}] -> [{P_embed.shape[0]}, {P_embed.shape[1]}] ===")

    # === P_act[i] from forward-pass activations at each layer ===
    all_inputs = [[] for _ in range(n_layers)]
    model.eval()
    print(f"\n=== running forward for activation bases ===", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for ids in all_ids:
            out = model(input_ids=ids.to(device), use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for i in range(n_layers):
                all_inputs[i].append(hs[i][0].to(torch.float32).cpu())
    inputs_per_layer = [torch.cat(xs, dim=0) for xs in all_inputs]
    print(f"  forward passes: {time.perf_counter()-t0:.1f}s")

    # === Per-layer alignment ===
    print(f"\n=== basis alignment vs P_embed ===")
    print(f"  {'layer':>5}  {'overlap':>8}  {'top-5 principal cos':>28}")
    results = []
    for i in range(n_layers):
        P_act, _ = pca_basis(inputs_per_layer[i], args.rank)
        overlap = subspace_overlap(P_embed, P_act)
        cosines = principal_angles(P_embed, P_act)[:5]
        show = (i < 4) or (i >= n_layers - 4) or (i % 4 == 0)
        if show:
            print(f"  {i:>5}  {overlap:>8.3f}  "
                  f"{' '.join(f'{c:.2f}' for c in cosines)}")
        results.append({
            "layer": i,
            "overlap": overlap,
            "principal_cosines": principal_angles(P_embed, P_act),
        })

    # Summary
    overlaps = [r["overlap"] for r in results]
    mean_overlap = sum(overlaps) / len(overlaps)
    max_overlap = max(overlaps)
    max_idx = overlaps.index(max_overlap)
    print(f"\n=== summary ===")
    print(f"  mean overlap across layers: {mean_overlap:.3f}")
    print(f"  max overlap: {max_overlap:.3f} at layer {max_idx}")
    if mean_overlap > 0.8:
        print(f"  -> STRONG alignment. P_embed can substitute for P_act.")
    elif mean_overlap > 0.5:
        print(f"  -> moderate alignment. P_embed is a reasonable prior but not a substitute.")
    else:
        print(f"  -> weak alignment. P_embed and P_act point at DIFFERENT subspaces.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "rank": args.rank,
            "total_tokens": total_tokens,
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "max_overlap_layer": max_idx,
            "per_layer": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
