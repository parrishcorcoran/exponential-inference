"""
Stage 122b — Nested mouth test with per-token states.

Stage 122 was degenerate: only N=40 sentence-mean states vs d=1024/2048,
so CCA trivially returned 1.0 everywhere (rotating one 40-D subspace
onto another in a higher-D ambient is free).

This version re-extracts per-TOKEN hidden states so N ≫ d. Uses longer
passages to guarantee the sample count clears d_large=2048.

Procedure:
  1. Load 0.6B. Run all passages. Save per-token states at 5 norm depths.
     Filter to real tokens via attention mask.
  2. Unload. Load 1.7B. Same.
  3. CCA between the two sets at each position.

With N ~ 6000 tokens and d_large = 2048, CCA is well-posed.
If top 150 correlations plateau at ~1.0 then drop → NESTED mouth.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch


# Longer passages so we get enough tokens. Mix of domains/registers.
PASSAGES = [
    # ~30 short declarative sentences from stage 121
    "The cell is the basic structural unit of life.",
    "Quantum mechanics describes the behavior of matter at small scales.",
    "Neural networks learn from examples through gradient descent.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Plate tectonics explains the movement of continents.",
    "The Riemann hypothesis concerns the zeros of the zeta function.",
    "Cryptography uses mathematical operations hard to reverse.",
    "Proteins fold into three-dimensional structures.",
    "Evolution operates on heritable variation in populations.",
    "Thermodynamics describes energy transfer between systems.",
    "Black holes are regions of extreme gravitational pull.",
    "DNA encodes genetic information in paired bases.",
    "Linear algebra studies vector spaces and transformations.",
    "Bayesian inference updates beliefs with new evidence.",
    "Game theory analyzes strategic decision making.",
    "The immune system recognizes foreign pathogens.",
    "Volcanoes form at tectonic plate boundaries.",
    "Graph theory studies networks of connected nodes.",
    "Quantum entanglement links the states of separated particles.",
    "Statistical mechanics connects microscopic to macroscopic.",
    "The poet walked slowly through the garden at dusk.",
    "She wrote a letter and mailed it the next morning.",
    "A soft rain began to fall as the sun set behind the hills.",
    "The children played with colorful balloons at the party.",
    "He carefully closed the old leather-bound book.",
    "They traveled together for many years across distant lands.",
    "The ancient bridge connected two bustling cities.",
    "Music filled the room as the dancers began to move.",
    "The mountain peak was covered in fresh white snow.",
    "She remembered the summer when they first met.",

    # Long passages (~100-200 tokens each) to push total token count past 6000
    "In condensed matter physics, the emergence of collective behavior from "
    "the interactions of many simple components has long fascinated "
    "researchers. Superconductivity, for instance, arises when electrons "
    "in certain materials cooled below a critical temperature form "
    "bound Cooper pairs that flow without resistance. This phenomenon, "
    "described by BCS theory in 1957, opened pathways to understanding "
    "phase transitions, symmetry breaking, and the macroscopic quantum "
    "states that underlie modern quantum devices ranging from sensitive "
    "magnetometers to the qubits in experimental quantum computers.",

    "The history of evolutionary biology is marked by a series of "
    "syntheses that reconciled Darwin's theory of natural selection with "
    "emerging knowledge of inheritance. The modern synthesis of the 1930s "
    "and 1940s integrated Mendelian genetics with population thinking, "
    "producing a framework in which evolution is change in allele "
    "frequencies over generations. Later work extended this to molecular "
    "data, developmental biology, and ecological interactions, revealing "
    "that organisms evolve through a combination of selection, drift, "
    "mutation, and migration in populations of finite size.",

    "Modern cryptography is built on computational problems believed to be "
    "intractable for classical computers. Public key systems such as RSA "
    "rely on the difficulty of factoring large integers, while elliptic "
    "curve schemes exploit the hardness of the discrete logarithm on "
    "algebraic curves over finite fields. As quantum computers advance, "
    "Shor's algorithm threatens both foundations, motivating the "
    "development of post-quantum cryptographic schemes based on lattice "
    "problems, hash functions, and code-based constructions resistant to "
    "known quantum attacks.",

    "The central dogma of molecular biology describes the flow of "
    "sequence information from DNA through RNA to proteins. Transcription "
    "copies DNA into messenger RNA, which ribosomes translate into chains "
    "of amino acids that fold into functional proteins. Exceptions and "
    "extensions exist: reverse transcriptases copy RNA back into DNA, "
    "non-coding RNAs regulate gene expression without ever being "
    "translated, and prions demonstrate that proteins themselves can "
    "carry heritable conformational information across cell divisions.",

    "In general relativity, mass and energy curve the fabric of spacetime, "
    "and free particles follow geodesics within this curved geometry. "
    "The Einstein field equations relate the stress-energy tensor to the "
    "curvature described by the Ricci tensor and scalar, yielding "
    "solutions ranging from the Schwarzschild metric around a static "
    "black hole to the Friedmann-Lemaitre-Robertson-Walker metric "
    "describing an expanding homogeneous universe. Observational tests "
    "from perihelion precession to gravitational lensing have confirmed "
    "the theory to exquisite precision.",

    "The theory of computation distinguishes problems by the resources "
    "required to solve them. The class P contains problems solvable in "
    "polynomial time on a deterministic Turing machine, while NP contains "
    "those whose solutions can be verified in polynomial time. Whether "
    "these classes are equal remains the central open problem of "
    "complexity theory. Complete problems such as Boolean satisfiability "
    "anchor the hierarchy: if any NP-complete problem had a polynomial "
    "algorithm, every problem in NP would.",

    "Immunology has revealed how vertebrates combine innate and adaptive "
    "defenses to meet a constantly changing landscape of pathogens. "
    "Innate cells such as macrophages and neutrophils provide rapid, "
    "nonspecific responses, while lymphocytes generate highly specific "
    "antibodies and cellular memory through somatic recombination of "
    "receptor genes. Vaccines exploit adaptive memory by priming the "
    "immune system with harmless antigens, establishing a protective "
    "repertoire that can respond rapidly to future exposure.",

    "The development of high-performance neural network models has been "
    "driven by scaling laws that empirically relate compute, data, and "
    "parameters to downstream capability. Transformer architectures, "
    "with self-attention and residual connections, have become the "
    "dominant backbone because they scale favorably and admit efficient "
    "parallel training. Emerging phenomena such as in-context learning, "
    "chain-of-thought reasoning, and tool use appear at certain scale "
    "thresholds, hinting at implicit structures learned from vast and "
    "diverse training corpora.",

    "Plate tectonics unifies a broad range of geological observations by "
    "describing the Earth's lithosphere as a mosaic of rigid plates "
    "moving over a ductile asthenosphere. Boundaries between plates are "
    "sites of seismic and volcanic activity: mid-ocean ridges where new "
    "crust is created, subduction zones where oceanic lithosphere sinks "
    "back into the mantle, and transform faults where plates slide "
    "laterally. The theory explains the distribution of continents, "
    "ocean basins, earthquakes, and mountain belts.",

    "Category theory provides a unifying language across mathematics by "
    "focusing on structure-preserving mappings rather than internal set "
    "details. Categories consist of objects and morphisms composing "
    "associatively with identities; functors map categories to "
    "categories while natural transformations relate functors. This "
    "abstract framework reveals deep analogies between algebra, topology, "
    "and logic, and provides foundational tools for domains ranging "
    "from algebraic geometry to theoretical computer science and "
    "type theory.",

    "Thermodynamics constrains the possible transformations of matter "
    "and energy through a small set of universal laws. The first law "
    "asserts energy conservation, the second law introduces entropy as a "
    "state function that never decreases in isolated systems, and the "
    "third law places limits on the approach to absolute zero. Statistical "
    "mechanics grounds these macroscopic laws in the microscopic behavior "
    "of atoms and molecules, relating equilibrium properties to ensembles "
    "of configurations weighted by their energy.",

    "Bayesian inference treats probability as a measure of belief updated "
    "by evidence. Starting from a prior over hypotheses, the posterior is "
    "obtained by conditioning on observed data through Bayes's rule. This "
    "framework unifies parameter estimation, model comparison, and "
    "prediction, and underpins methods ranging from simple conjugate "
    "updating to modern variational and Markov chain Monte Carlo "
    "algorithms that approximate intractable posteriors in high-"
    "dimensional spaces.",

    "The history of medicine traces a long arc from humoral theory to "
    "molecular biology. Early physicians relied on observation and "
    "empirical remedies, often without understanding underlying causes. "
    "The germ theory revolution of the nineteenth century established "
    "microbes as agents of infectious disease, while twentieth-century "
    "advances in pharmacology, surgery, and molecular genetics transformed "
    "the treatment of everything from bacterial infections to cancer. "
    "Contemporary medicine integrates genomics, imaging, and data "
    "science into increasingly personalized care.",

    "Graph theory provides a flexible language for representing "
    "relationships among discrete entities. Vertices stand for objects "
    "and edges encode connections, with attributes such as weights, "
    "directions, or labels enriching the representation. Algorithms on "
    "graphs solve problems from shortest path and network flow to "
    "matching and coloring, with applications in logistics, "
    "communications, bioinformatics, and social network analysis. The "
    "field continues to grow as new classes of structured data arise "
    "from scientific and industrial systems.",

    "The ocean plays a central role in the Earth's climate system, "
    "storing heat, transporting energy between latitudes, and exchanging "
    "carbon dioxide with the atmosphere. Deep circulation driven by "
    "temperature and salinity gradients couples surface conditions to "
    "abyssal waters over century-long timescales, while surface currents "
    "driven by winds produce the familiar gyres that shape regional "
    "climates. Changes in these circulations are implicated in past "
    "abrupt climate shifts and in projections of future warming.",

    "Narrative film developed a visual grammar over the course of the "
    "twentieth century, learning to compress, expand, and reorder time "
    "through editing. Early directors discovered that cuts between shots "
    "could convey space and causation, while later innovations added "
    "montage, continuity editing, and nonlinear structures. Sound, color, "
    "and digital effects extended the possibilities further, but the "
    "underlying grammar of shots and cuts remained a shared vocabulary "
    "that filmmakers manipulate to guide attention, emotion, and "
    "interpretation.",

    "The study of consciousness remains one of the hardest problems in "
    "contemporary science, straddling neuroscience, philosophy, and "
    "cognitive modeling. Empirical work identifies neural correlates of "
    "perception, attention, and self-reference, while theoretical "
    "frameworks such as global workspace theory and integrated "
    "information theory attempt to specify which physical structures "
    "give rise to subjective experience. Progress is constrained by the "
    "difficulty of operationalizing first-person reports and by deep "
    "uncertainty about the fundamental relationship between computation "
    "and phenomenal awareness.",

    "Architecture responds to climate, material availability, and social "
    "organization as much as to aesthetic preference. Vernacular "
    "traditions developed over centuries produce thick-walled adobe in "
    "arid regions, stilted wooden structures in humid forests, and "
    "stone terraces on mountain slopes, each tuned to local conditions. "
    "Modern architecture breaks with these traditions by using "
    "industrial materials and standardized forms, producing buildings "
    "that are often less adapted to their sites but more uniform across "
    "diverse environments, a tradeoff that informs ongoing debates in "
    "sustainable design.",

    "Linguistics has moved from prescriptive grammar through structuralist "
    "description to generative theories attempting to characterize the "
    "innate faculty of language. Phonology describes the systems of "
    "sound, morphology the internal structure of words, syntax the "
    "combination of words into sentences, and semantics the mapping of "
    "forms to meaning. Typological studies of hundreds of languages "
    "reveal both remarkable diversity and striking universals, "
    "constraining theories of how language is learned by children and "
    "represented in the brain.",
]


@torch.no_grad()
def extract_token_states(model, tokenizer, passages, device, target_layers, max_length=256):
    """Run passages, collect per-token hidden states at target layers.
       Returns dict: layer_idx -> [N_tokens_total, d] tensor."""
    out = {l: [] for l in target_layers}
    model.eval()
    total_tokens = 0
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True,
                        max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()  # [seq]
        output = model(ids, use_cache=False, output_hidden_states=True)
        for l in target_layers:
            if l < len(output.hidden_states):
                h = output.hidden_states[l][0].float()  # [seq, d]
                h = h[mask].cpu()  # keep only real tokens
                out[l].append(h)
        total_tokens += int(mask.sum())
    print(f"  total tokens collected: {total_tokens}", flush=True)
    return {l: torch.cat(vals, dim=0) for l, vals in out.items()}  # [N_total, d]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def cca(X, Y, k=None, ridge=1e-6):
    """Canonical Correlation Analysis between X [N, d1] and Y [N, d2].
       Uses ridge-regularized QR. Returns sorted canonical correlations (descending)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Scale to unit column variance for stability
    X = X / (X.std(axis=0, keepdims=True) + 1e-8)
    Y = Y / (Y.std(axis=0, keepdims=True) + 1e-8)
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    C = Qx.T @ Qy
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    if k:
        S = S[:k]
    return S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", default="Qwen/Qwen3-0.6B")
    p.add_argument("--large", default="Qwen/Qwen3-1.7B")
    p.add_argument("--cache-dir", default="cache/stage122b_states")
    p.add_argument("--out", default="results/stage122b_nested_mouths.json")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)
    print(f"passages: {len(PASSAGES)}  max_length: {args.max_length}", flush=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.small, trust_remote_code=True)

    all_states = {}
    for model_id in [args.small, args.large]:
        safe_id = model_id.replace("/", "_")
        cache_path = cache_dir / f"{safe_id}.pt"
        if cache_path.exists():
            print(f"\nloading cached: {cache_path}", flush=True)
            all_states[model_id] = torch.load(cache_path, map_location="cpu")
            continue

        print(f"\nloading {model_id}...", flush=True)
        t0 = time.time()
        model = load_model(model_id, device)
        L = model.config.num_hidden_layers
        d = model.config.hidden_size
        print(f"  L={L}  d={d}", flush=True)

        norm_positions = [0.10, 0.25, 0.50, 0.75, 0.90]
        target_layers = [max(1, min(L, int(pp * L))) for pp in norm_positions]
        print(f"  target layers @ {norm_positions}: {target_layers}", flush=True)

        states = extract_token_states(model, tokenizer, PASSAGES, device,
                                       target_layers, max_length=args.max_length)
        states_norm = {
            f"{norm_positions[i]:.2f}": states[target_layers[i]]
            for i in range(len(norm_positions))
        }
        torch.save(states_norm, cache_path)
        all_states[model_id] = states_norm
        print(f"  saved {cache_path}  ({time.time()-t0:.0f}s)", flush=True)

        del model; gc.collect()
        if device == "mps": torch.mps.empty_cache()

    s_states = all_states[args.small]
    l_states = all_states[args.large]
    print(f"\npositions: {list(s_states.keys())}")

    results = {}
    for pos in s_states:
        if pos not in l_states:
            continue
        X = s_states[pos].float().numpy()
        Y = l_states[pos].float().numpy()
        # Align lengths (should match exactly since same passages + tokenizer)
        N = min(X.shape[0], Y.shape[0])
        X = X[:N]; Y = Y[:N]
        d_small = X.shape[1]; d_large = Y.shape[1]
        max_corr = min(N, d_small, d_large)

        corrs = cca(X, Y, k=max_corr)

        # Shuffle baseline — permute Y rows and re-run CCA.
        # If shuffled corrs match real corrs, the signal is a dim-counting artifact.
        rng = np.random.default_rng(0)
        perm = rng.permutation(N)
        corrs_shuf = cca(X, Y[perm], k=max_corr)

        # Drop-cliff rank: first index where corr crosses 0.5 (or 0.9)
        def first_below(vals, thresh):
            below = np.where(vals < thresh)[0]
            return int(below[0]) if len(below) else int(len(vals))

        cliff_05 = first_below(corrs, 0.5)
        cliff_09 = first_below(corrs, 0.9)
        cliff_05_shuf = first_below(corrs_shuf, 0.5)

        results[pos] = {
            "d_small": d_small, "d_large": d_large,
            "n_samples": N, "n_corrs": len(corrs),
            "corrs": corrs.tolist(),
            "corrs_shuffled": corrs_shuf.tolist(),
            "top_1": float(corrs[0]),
            "top_5_avg": float(np.mean(corrs[:5])),
            "top_10_avg": float(np.mean(corrs[:10])),
            "top_100_avg": float(np.mean(corrs[:min(100, len(corrs))])),
            "mean_all": float(np.mean(corrs)),
            "mean_all_shuf": float(np.mean(corrs_shuf)),
            "top_10_avg_shuf": float(np.mean(corrs_shuf[:10])),
            "cliff_0.5": cliff_05,
            "cliff_0.9": cliff_09,
            "cliff_0.5_shuf": cliff_05_shuf,
            "frac_above_0_99": float(np.mean(corrs > 0.99)),
            "frac_above_0_9": float(np.mean(corrs > 0.9)),
            "frac_above_0_7": float(np.mean(corrs > 0.7)),
        }
        print(f"\n=== position {pos} ===")
        print(f"  d_small={d_small}  d_large={d_large}  N={N}  n_corrs={len(corrs)}")
        print(f"  top 1:    {corrs[0]:.4f}   (shuf: {corrs_shuf[0]:.4f})")
        print(f"  top 10:   {np.mean(corrs[:10]):.4f}   (shuf: {np.mean(corrs_shuf[:10]):.4f})")
        print(f"  top 100:  {np.mean(corrs[:min(100, len(corrs))]):.4f}   "
              f"(shuf: {np.mean(corrs_shuf[:min(100, len(corrs_shuf))]):.4f})")
        print(f"  mean all: {np.mean(corrs):.4f}   (shuf: {np.mean(corrs_shuf):.4f})")
        print(f"  frac >0.99: {np.mean(corrs > 0.99):.3f}   "
              f">0.9: {np.mean(corrs > 0.9):.3f}   "
              f">0.7: {np.mean(corrs > 0.7):.3f}")
        print(f"  cliff @0.9: rank {cliff_09}   cliff @0.5: rank {cliff_05} (shuf: {cliff_05_shuf})")
        qs = [0, 50, 100, 200, 400, 800, len(corrs)-1]
        qs = [q for q in qs if q < len(corrs)]
        print(f"  corr @ ranks {qs}:      " + " ".join(f"{corrs[q]:.3f}" for q in qs))
        print(f"  shuf @ ranks {qs}:      " + " ".join(f"{corrs_shuf[q]:.3f}" for q in qs))

    # Interpretation
    print(f"\n=== interpretation ===")
    for pos, r in results.items():
        if r["frac_above_0_99"] > 0.3:
            v = f"NESTED — {r['frac_above_0_99']*100:.0f}% of small's directions at corr>0.99 with large"
        elif r["frac_above_0_9"] > 0.3:
            v = f"MOSTLY NESTED — {r['frac_above_0_9']*100:.0f}% of small's directions >0.9"
        elif r["frac_above_0_7"] > 0.3:
            v = f"PARTIAL — {r['frac_above_0_7']*100:.0f}% of small's directions >0.7"
        elif r["top_10_avg"] > 0.8:
            v = "TOP-K ALIGNED — only a few shared directions"
        else:
            v = "PRIVATE — minimal shared structure"
        print(f"  pos {pos}: {v}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"small": args.small, "large": args.large,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
