"""
Stage 20 — Cross-model basis geometry: phase transitions + rotation curves.

Runs stage 19's rotation-profile analysis across multiple model sizes
and tokenizer families. Answers:

(1) Where is the phase-transition layer in each model? (argmin of
    adjacent-layer overlap). Is it consistent across sizes?

(2) Does the rotation CURVE shape look the same across Qwen family
    sizes? (Normalize layer index to [0, 1], compare curves.)

(3) Does Phi-2 (different tokenizer) have a DIFFERENT curve shape?

The bases live in different ambient spaces per model (different hidden
sizes), so we compare SHAPES not the bases themselves:
    - adjacent-overlap curve as function of normalized depth
    - phase-transition location
    - first-vs-last overlap (total rotation across the stack)
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


CORPUS = [
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
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle.",
    "The auditory system transduces air pressure oscillations into neural signals.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Microbiome research examines the communities of microorganisms living in hosts.",
]


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32)


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


def analyze(model_id, device, rank, corpus):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n=== {model_id} ===", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  loaded in {time.perf_counter()-t0:.1f}s — {n_layers} layers, hidden={d}")

    inputs = capture_layer_inputs(model, tokenizer, corpus, device)
    tokens = inputs[0].shape[0]
    bases = [pca_basis(inputs[i], rank) for i in range(n_layers)]
    adjacent_overlap = [subspace_overlap(bases[i], bases[i+1]) for i in range(n_layers-1)]

    # Phase transition: minimum adjacent overlap
    min_idx = int(torch.tensor(adjacent_overlap).argmin().item())
    min_overlap = adjacent_overlap[min_idx]

    # First-to-last total rotation
    first_last = subspace_overlap(bases[0], bases[-1])

    # Summary stats
    mean_adj = sum(adjacent_overlap) / len(adjacent_overlap)
    max_adj = max(adjacent_overlap)

    result = {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_size": d,
        "rank": rank,
        "tokens_sampled": tokens,
        "adjacent_overlap": adjacent_overlap,
        "mean_adjacent_overlap": mean_adj,
        "min_adjacent_overlap": min_overlap,
        "min_adjacent_pair": [min_idx, min_idx + 1],
        "min_adjacent_fraction_of_depth": min_idx / max(n_layers - 1, 1),
        "max_adjacent_overlap": max_adj,
        "first_last_overlap": first_last,
    }
    print(f"  adjacent overlap: mean={mean_adj:.3f} min={min_overlap:.3f} "
          f"at layer ({min_idx}→{min_idx+1})  frac-depth={min_idx/max(n_layers-1,1):.2f}")
    print(f"  first vs last:    {first_last:.3f}")
    print(f"  max adjacent:     {max_adj:.3f}")

    # Free memory
    del model, tokenizer, inputs, bases
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=(
        "Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,microsoft/phi-2"),
        help="Comma-separated HF model ids")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage20_cross_model_basis.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}  rank={args.rank}")

    models = args.models.split(",")
    all_results = []
    for m in models:
        try:
            r = analyze(m.strip(), device, args.rank, CORPUS)
            all_results.append(r)
        except Exception as e:
            print(f"  ! {m} failed: {e}")
            all_results.append({"model_id": m, "error": str(e)})

    # Cross-model comparison
    print(f"\n=== cross-model summary ===")
    print(f"  {'model':>30} {'layers':>7} {'hidden':>7} {'min@adj':>9} "
          f"{'frac':>6} {'first-last':>12}")
    for r in all_results:
        if "error" in r:
            continue
        print(f"  {r['model_id']:>30} {r['n_layers']:>7} {r['hidden_size']:>7} "
              f"{r['min_adjacent_overlap']:>9.3f} "
              f"{r['min_adjacent_fraction_of_depth']:>6.2f} "
              f"{r['first_last_overlap']:>12.3f}")

    # Is the phase transition at roughly the same fraction of depth?
    fracs = [r["min_adjacent_fraction_of_depth"]
             for r in all_results if "error" not in r]
    if len(fracs) >= 2:
        spread = max(fracs) - min(fracs)
        print(f"\n  phase-transition fraction-of-depth range: {min(fracs):.2f} - {max(fracs):.2f}  (spread {spread:.2f})")
        if spread < 0.1:
            print(f"  -> consistent location across models")
        else:
            print(f"  -> phase transition is MODEL-SPECIFIC in location")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"models": all_results, "corpus_size": len(CORPUS)}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
