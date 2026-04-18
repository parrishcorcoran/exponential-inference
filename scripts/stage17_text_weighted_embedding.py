"""
Stage 17 — Text-weighted embedding manifold.

Stage 16 showed Qwen3-0.6B's raw embedding matrix has TwoNN ≈ 80 — much
higher than the ~10 we measure from activations. Hypothesis: the 10-dim
manifold emerges from TEXT STATISTICS selecting a frequency-weighted
subset of the 80-dim embedding space.

Test: tokenize a diverse calibration corpus, take the embedding of each
token AT EACH POSITION (so frequent tokens get counted many times,
rare ones rarely). Measure TwoNN on this sample. If it drops to ~10,
the manifold is (embedding geometry × text distribution).

This would confirm that training primarily learns the embedding matrix
and the dynamics; the relevant manifold is a derived quantity we can
extract from tokenizer+embeddings+corpus statistics without running
the trained model at all (beyond loading its embed_tokens).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


CALIBRATION_TEXTS = [
    "The cell is the basic structural unit of life. Every organism is composed of one or more cells, which are the smallest entities exhibiting the characteristics of life.",
    "Quantum mechanics describes the behaviour of matter and energy at atomic and subatomic scales. Particles exhibit wave-like properties, and observation can collapse superpositions of states.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes, transistors, integrated circuits, and the modern silicon processor.",
    "Climate change is driven primarily by greenhouse gas emissions from fossil fuel combustion, deforestation, and industrial agriculture.",
    "Language models learn statistical structure from text corpora by optimizing a next-token prediction objective across billions of parameters.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen, sustaining most life on Earth.",
    "Neural networks consist of layers of parameterised transformations trained by gradient descent on a differentiable loss.",
    "The Roman Empire expanded through a combination of military power, political institutions, and engineering achievements.",
    "Relativity theory links space, time, matter, and energy through Einstein's field equations and their geometric interpretation.",
    "Artificial intelligence as a field studies the design of agents that perceive, reason, learn, and act in complex environments.",
    "Cryptography protects information using mathematical operations that are easy in one direction and hard to invert without a key.",
    "Neurotransmitters mediate communication between neurons at chemical synapses using molecules like dopamine and serotonin.",
    "Evolution by natural selection operates on heritable variation in populations, shifting allele frequencies through differential reproduction.",
    "Protein synthesis begins with transcription of DNA into messenger RNA, followed by translation in ribosomes that assemble amino acids.",
    "Plate tectonics describes the movement of Earth's lithospheric plates, producing earthquakes, volcanoes, and mountain ranges.",
    "Graph theory studies mathematical structures of vertices connected by edges, with applications across many disciplines.",
    "The Renaissance reshaped art, science, and philosophy through classical revival and technological advances like printing.",
    "Public-key cryptography relies on asymmetric operations where a public key encrypts but only a private key decrypts.",
    "Stars produce energy through nuclear fusion in their cores, converting hydrogen into progressively heavier elements over time.",
    "In statistics, a normal distribution has parameters mean and standard deviation, and appears in aggregated quantities by the central limit theorem.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions with a family of fundamental particles.",
    "Homeostasis in biology is the tendency of a system to regulate internal conditions around stable setpoints despite external variation.",
    "Thermodynamics describes energy, heat, work, entropy and the equilibria of macroscopic systems using a few general laws.",
    "In linguistics, syntax governs the rules by which words combine into grammatical sentences within a specific language.",
    "Convex optimization solves problems where feasible region and objective are convex, with efficient algorithms and global optima.",
    "Game theory models strategic interactions among rational agents, each seeking to maximize a payoff given others' actions.",
    "Ecology studies how organisms interact with each other and the environment through food webs, niches, and populations.",
    "Fluid turbulence produces multi-scale vortical motion that transfers energy across scales by non-linear interactions.",
    "Memory in the brain involves synaptic plasticity, distributed storage, and consolidation during sleep and rehearsal.",
    "Operating systems manage hardware resources, process scheduling, file systems, and protection among concurrent programs.",
]


def twonn_dimension(X, sample_limit=3000):
    X = X.to(torch.float64)
    if X.shape[0] > sample_limit:
        idx = torch.randperm(X.shape[0])[:sample_limit]
        X = X[idx]
    N = X.shape[0]
    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    top2, _ = dists.topk(2, dim=1, largest=False)
    r1 = top2[:, 0]
    r2 = top2[:, 1]
    mask = r1 > 1e-10
    if mask.sum() < 10:
        return float("nan")
    mu = (r2[mask] / r1[mask]).clamp_min(1.0 + 1e-10)
    return float(1.0 / torch.log(mu).mean().item())


def rank_coverage(X, fractions=(0.5, 0.9, 0.95, 0.99)):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    eigvals = eigvals.flip(0)
    total = eigvals.sum().clamp_min(1e-12)
    cum = torch.cumsum(eigvals, dim=0) / total
    out = {}
    for f in fractions:
        idx = int((cum >= f).nonzero()[0].item()) + 1 if (cum >= f).any() else len(cum)
        out[f"r{int(f*100)}"] = idx
    return out


def participation_ratio(X):
    Xc = X - X.mean(dim=0, keepdim=True)
    cov = Xc.T @ Xc
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    num = eigvals.sum().pow(2)
    den = eigvals.pow(2).sum().clamp_min(1e-12)
    return float((num / den).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--sample-limit", type=int, default=3000)
    p.add_argument("--out", default="results/stage17_text_weighted_embedding.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"=== {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True)
    embed = model.model.embed_tokens.weight.detach().cpu().to(torch.float32)
    vocab, hidden = embed.shape
    print(f"  embed [{vocab}, {hidden}]")

    # 1. Baseline: uniform-vocab TwoNN (stage 16's result)
    t0 = time.perf_counter()
    t_uniform = twonn_dimension(embed, sample_limit=args.sample_limit)
    print(f"  [uniform over vocab]  TwoNN = {t_uniform:.2f}  ({time.perf_counter()-t0:.1f}s)")

    # 2. Tokenize corpus, take embedding of each position's token. Frequent
    #    tokens appear many times, rare tokens few. This is the text-
    #    distribution-weighted sample of embeddings.
    all_positions = []
    for text in CALIBRATION_TEXTS:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
        all_positions.append(ids)
    positions = torch.cat(all_positions, dim=0)
    N = positions.shape[0]
    print(f"  corpus: {N} token positions")

    # Gather embeddings at those positions
    X_text = embed[positions]  # [N, hidden]

    t0 = time.perf_counter()
    t_text = twonn_dimension(X_text, sample_limit=args.sample_limit)
    pr_text = participation_ratio(X_text)
    rcov_text = rank_coverage(X_text)
    print(f"  [text-weighted]       TwoNN = {t_text:.2f}  "
          f"PR = {pr_text:.1f}  r90 = {rcov_text['r90']}  "
          f"({time.perf_counter()-t0:.1f}s)")

    # 3. Compare to stage 1 measurements at mid-stack for context
    print(f"\n  stage 1 (mid-stack activations): ~10 TwoNN, ~18-80 PR")

    # 4. Also: first-token-only distribution (similar to mid-stack at first position)
    first_ids = torch.tensor([p[0].item() for p in all_positions])
    X_first = embed[first_ids]
    t_first = twonn_dimension(X_first)
    print(f"  [first-token-only]    TwoNN = {t_first:.2f}")

    result = {
        "model": args.model,
        "vocab": int(vocab), "hidden": int(hidden),
        "uniform_vocab_twonn": t_uniform,
        "text_weighted_twonn": t_text,
        "text_weighted_pr": pr_text,
        "text_weighted_rank_coverage": rcov_text,
        "first_token_twonn": t_first,
        "n_corpus_positions": int(N),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
