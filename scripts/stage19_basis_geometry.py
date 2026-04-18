"""
Stage 19 — Basis geometry: random baseline, rotation profile, corpus invariance.

Three tests on Qwen3-0.6B, all cheap:

(A) Random-subspace baseline:
    How similar are two random k-dim subspaces of 1024-dim? Gives us a
    noise floor. Stage 18 reported mean overlap 0.252 between P_embed
    and per-layer P_act. Is that random-level, or meaningful signal?
    Expected random overlap ≈ sqrt(k/d).

(B) Across-layer rotation profile:
    For each pair (i, j), compute subspace_overlap(P_act[i], P_act[j]).
    Reveals how fast the basis rotates with depth — gradual drift
    (high adjacent, low distant) or abrupt rotation (low everywhere).

(C) Corpus invariance:
    Compute P_act[i] on two DIFFERENT calibration corpora. If the bases
    match, the basis is a model property (training-discovered). If
    they drift, the basis is corpus-specific (artifact of which text
    we sampled).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


CORPUS_A = [
    "The cell is the basic structural unit of life composed of cytoplasm enclosed by a membrane.",
    "Quantum mechanics describes matter and energy at atomic scales through wave-particle duality.",
    "Compilers translate source code into machine code through lexical analysis, parsing, and optimization.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Neural networks approximate functions through layered transformations trained by gradient descent.",
    "Plate tectonics describes the movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into three-dimensional structures determined by their amino acid sequences.",
    "The standard model unifies three fundamental forces with a set of gauge bosons and fermions.",
    "Evolution operates on heritable variation through differential reproduction across generations.",
    "Cryptography protects information using asymmetric mathematical operations.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges, with many practical applications.",
    "The Renaissance marked renewed interest in classical learning in Europe.",
    "Black holes are regions of spacetime where gravity prevents light from escaping.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotides.",
]

CORPUS_B = [
    "Federal monetary policy influences interest rates, inflation, and employment through open market operations.",
    "The orchestra tuned their instruments as the conductor raised her baton to begin the symphony.",
    "Hurricane formation requires warm ocean water, low wind shear, and an initial disturbance in the atmosphere.",
    "Sushi originated in southeast Asia as a method of preserving fish in fermented rice.",
    "The marathon runner paced herself carefully through the final mile despite aching muscles.",
    "Chess combines calculation and intuition as players navigate an exponentially branching game tree.",
    "Impressionist painters captured fleeting light and color rather than precise detail.",
    "Supply-chain disruptions from global events can cascade through manufacturing and retail.",
    "The archaeological dig uncovered pottery fragments dated to the early bronze age.",
    "Olive oil from Mediterranean regions has distinct flavor profiles based on varietal and climate.",
    "Professional soccer teams develop youth academies to identify and train promising players.",
    "The jazz quintet improvised over a standard chord progression, each soloist taking turns.",
    "Mountain climbing expeditions must plan for altitude, weather windows, and supply logistics.",
    "Traditional embroidery techniques vary widely across cultures and historical periods.",
    "The wine sommelier paired each course with a vintage chosen to complement the flavors.",
]


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32), mean


def subspace_overlap(P_a, P_b):
    k = P_a.shape[1]
    M = P_a.T @ P_b
    return float(torch.linalg.norm(M, ord="fro").item() / (k ** 0.5))


def capture_layer_inputs(model, tokenizer, texts, device, max_len=256):
    n_layers = model.config.num_hidden_layers
    all_inputs = [[] for _ in range(n_layers)]
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            for i in range(n_layers):
                all_inputs[i].append(hs[i][0].to(torch.float32).cpu())
    return [torch.cat(xs, dim=0) for xs in all_inputs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage19_basis_geometry.json")
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
    print(f"device={device}  rank={args.rank}")

    print(f"\n=== loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  {n_layers} layers, hidden={d}")

    # === (A) Random-subspace baseline ===
    print(f"\n=== (A) random-subspace baseline, rank {args.rank}, d={d} ===")
    overlaps_random = []
    torch.manual_seed(0)
    for _ in range(30):
        P1 = torch.linalg.qr(torch.randn(d, args.rank))[0]
        P2 = torch.linalg.qr(torch.randn(d, args.rank))[0]
        overlaps_random.append(subspace_overlap(P1, P2))
    mean_r = sum(overlaps_random) / len(overlaps_random)
    min_r = min(overlaps_random)
    max_r = max(overlaps_random)
    # Theoretical expected: sqrt(k/d) for independently-random subspaces
    expected = (args.rank / d) ** 0.5
    print(f"  empirical random overlap: {mean_r:.3f}  (range {min_r:.3f}-{max_r:.3f})")
    print(f"  theoretical sqrt(k/d):    {expected:.3f}")
    print(f"  -> any overlap much above {mean_r:.2f} is meaningful alignment")

    # === (B) Rotation profile across layers ===
    print(f"\n=== (B) activation basis rotation across layers, corpus A ===")
    inputs_A = capture_layer_inputs(model, tokenizer, CORPUS_A, device)
    tokens_A = inputs_A[0].shape[0]
    bases_A = [pca_basis(inputs_A[i], args.rank)[0] for i in range(n_layers)]
    print(f"  corpus A: {tokens_A} tokens")

    # Compute overlap matrix (layer i vs layer j)
    overlap_matrix = torch.zeros(n_layers, n_layers)
    for i in range(n_layers):
        for j in range(n_layers):
            overlap_matrix[i, j] = subspace_overlap(bases_A[i], bases_A[j])

    # Summary: adjacent-layer overlap (i vs i+1) averaged
    adj = [float(overlap_matrix[i, i+1]) for i in range(n_layers-1)]
    mean_adj = sum(adj) / len(adj)
    # Distant-layer overlap (first vs last of mid-stack)
    far = float(overlap_matrix[5, 20])
    very_far = float(overlap_matrix[1, n_layers-1])
    print(f"  adjacent-layer mean:   {mean_adj:.3f}  (range {min(adj):.3f}-{max(adj):.3f})")
    print(f"  layer 5 vs 20:         {far:.3f}")
    print(f"  layer 1 vs last:       {very_far:.3f}")
    print(f"  (reference: random overlap ~{mean_r:.3f})")

    # === (C) Corpus invariance ===
    print(f"\n=== (C) corpus invariance: corpus A vs corpus B on same model ===")
    inputs_B = capture_layer_inputs(model, tokenizer, CORPUS_B, device)
    tokens_B = inputs_B[0].shape[0]
    bases_B = [pca_basis(inputs_B[i], args.rank)[0] for i in range(n_layers)]
    print(f"  corpus B: {tokens_B} tokens")

    corpus_overlaps = [subspace_overlap(bases_A[i], bases_B[i]) for i in range(n_layers)]
    mean_corpus = sum(corpus_overlaps) / len(corpus_overlaps)
    print(f"  layer-matched A-vs-B overlaps:")
    for i in range(0, n_layers, max(1, n_layers // 8)):
        print(f"    layer {i:2d}: {corpus_overlaps[i]:.3f}")
    print(f"  mean across layers:  {mean_corpus:.3f}")
    if mean_corpus > 0.9:
        print(f"  -> basis is highly corpus-invariant (property of the model)")
    elif mean_corpus > 0.6:
        print(f"  -> moderate corpus invariance; basis is mostly model property but corpus matters")
    else:
        print(f"  -> weak corpus invariance; basis depends substantially on corpus")

    # Save
    result = {
        "model": args.model, "rank": args.rank, "hidden": d, "n_layers": n_layers,
        "random_baseline": {
            "mean": mean_r, "min": min_r, "max": max_r,
            "theoretical_sqrt_k_over_d": expected,
        },
        "rotation_profile": {
            "adjacent_layer_overlaps": adj,
            "mean_adjacent": mean_adj,
            "overlap_matrix": overlap_matrix.tolist(),
        },
        "corpus_invariance": {
            "per_layer_overlaps_A_vs_B": corpus_overlaps,
            "mean": mean_corpus,
            "corpus_A_tokens": tokens_A,
            "corpus_B_tokens": tokens_B,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
