"""Generate a tiny corpus.pt for dev-testing the Matryoshka pipeline on Mac."""
import torch
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]

TEXTS = [
    "The cell is the basic structural unit of life, composed of cytoplasm enclosed within a membrane.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
    "Neural networks consist of parameterized layers trained by gradient descent to approximate functions.",
    "Plate tectonics describes the slow movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into complex three-dimensional structures determined by their amino acid sequences.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions.",
    "Evolution by natural selection operates on heritable variation in populations.",
    "Cryptography protects information using mathematical operations that are easy to compute.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems.",
    "Graph theory studies vertices connected by edges across many practical applications.",
    "Black holes are regions of spacetime from which nothing, not even light, can escape.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Bayesian inference updates a prior probability distribution using observed data.",
    "The immune system recognizes pathogens through pattern recognition receptors.",
    "The Riemann zeta function encodes deep information about the distribution of primes.",
]


def main():
    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    sequences = []
    total = 0
    for t in TEXTS:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids
        sequences.append(ids)
        total += ids.shape[1]
    out = {"sequences": sequences, "total_tokens": total, "model": model_id}
    out_path = REPO_ROOT / "machines" / "strix_halo" / "scratch" / "corpora" / "dev_corpus.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"wrote {out_path}  ({len(sequences)} seqs, {total} tokens)")


if __name__ == "__main__":
    main()
