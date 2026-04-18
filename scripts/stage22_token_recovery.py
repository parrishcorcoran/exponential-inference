"""
Stage 22 — Token recovery from rank-k PCA projection of final hidden states.

The claim (from Z8 analysis): teacher-predicted tokens are ~75%
recoverable from a rank-10 PCA projection of the final-layer hidden
states, validating that prediction-relevant information lives on the
measured manifold.

Protocol:
    1. Run teacher on a calibration corpus, capture final hidden
       states across all positions.
    2. PCA the final-layer hidden states → top-k basis P + mean.
    3. On a HELD-OUT text, at each position:
         a. Take teacher's argmax token ID (ground-truth prediction).
         b. Project position's final hidden state through P and back:
              h_projected = P @ (P.T @ (h - mean)) + mean.
         c. Apply final norm + lm_head to h_projected → logits.
         d. Take argmax. Compare to teacher's argmax.
    4. Report recovery rate at k ∈ {7, 10, 15, 20, 32, 64}.

Why this matters: if 10-15 dims suffice to recover 75% of teacher
predictions via a LINEAR projection, the prediction-relevant manifold
is small enough that direct-manifold routing is feasible. Z8 noted
this underestimates the true recovery because PCA finds variance
directions, not prediction directions — a prediction-aligned basis
(e.g., the unembedding matrix's SVD) should do better.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


CALIB_CORPUS = [
    # Longer paragraphs (more tokens each) for better PCA sampling.
    "The cell is the basic structural unit of life, composed of cytoplasm enclosed within a membrane. Every organism is made of one or more cells, which carry genetic information, perform metabolic reactions, and respond to external stimuli through elaborate signaling pathways involving receptors and second messengers.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales. Particles exhibit wave-like properties, and measurement can collapse superpositions of states. The uncertainty principle places a fundamental limit on how precisely pairs of observables such as position and momentum can be simultaneously known.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes, transistors, integrated circuits, and modern silicon processors. Each generation of technology multiplied computational density while reducing power consumption, enabling applications that were previously intractable at earlier orders of magnitude of compute.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen, sustaining most life on Earth. The process occurs in chloroplasts and involves two coupled reactions: the light-dependent reactions that capture photon energy and the Calvin cycle that fixes carbon into sugars.",
    "Neural networks consist of parameterized layers trained by gradient descent to approximate functions. Depth and width both contribute to capacity, but generalization depends on inductive biases and regularization as much as raw parameter count. Training dynamics have been studied through the lens of statistical mechanics and optimization theory.",
    "Plate tectonics describes the slow movement of Earth's lithospheric plates over the underlying mantle. Their interactions at convergent, divergent, and transform boundaries produce earthquakes, volcanic activity, mountain building, and oceanic trenches that have shaped the planet over billions of years.",
    "Proteins fold into complex three-dimensional structures determined by their amino acid sequences. Misfolding can cause diseases including Alzheimer's, Parkinson's, and prion disorders. Computational prediction of protein structure from sequence has advanced through deep learning methods that consider evolutionary information.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions through a set of gauge bosons mediating forces between fundamental fermions. The Higgs mechanism accounts for particle masses via spontaneous symmetry breaking of the electroweak field.",
    "Evolution by natural selection operates on heritable variation in populations, shifting allele frequencies through differential reproduction across generations. Genetic drift, mutation, migration, and recombination provide the raw material for this process of adaptation and diversification.",
    "Cryptography protects information using mathematical operations that are easy to compute in one direction but hard to invert without knowledge of a secret. Public-key schemes depend on problems such as integer factorization or the discrete logarithm over elliptic curves.",
    "Thermodynamics relates heat, work, energy, and entropy in macroscopic systems. The second law asserts that the entropy of an isolated system never decreases, which establishes the arrow of time and constrains the efficiency of heat engines.",
    "Graph theory studies vertices connected by edges. Applications range from social network analysis to chemistry, routing algorithms, and integrated circuit design. Classical results include Euler's theorem on bridges of Konigsberg and the four-color theorem for planar maps.",
    "The Renaissance marked a period of renewed interest in classical learning in Europe, reshaping art, science, philosophy, and politics. Technological advances like the printing press accelerated the spread of ideas and enabled new forms of communication.",
    "Black holes are regions of spacetime from which nothing, not even light, can escape. They form when massive stars collapse at the end of their lives. General relativity predicts their existence, and gravitational wave observations have directly detected binary black hole mergers.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases. The sequence of bases specifies the sequence of amino acids in proteins through the genetic code, read in triplets by the ribosome during translation of messenger RNA.",
    "Volcanoes form at tectonic plate boundaries and hot spots in Earth's mantle. The composition of the magma determines eruption style, from effusive basaltic flows to explosive silicic events. Volcanic activity has driven atmospheric evolution and mass extinctions throughout geological history.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms. Matrices represent linear transformations, eigendecomposition reveals principal directions of variation, and singular value decomposition underlies dimensionality reduction techniques such as PCA.",
    "Game theory analyzes strategic interactions between rational decision makers, each seeking to maximize their payoff given expectations about others. Equilibrium concepts like Nash equilibrium describe stable outcomes where no player benefits from unilateral deviation.",
    "Neurotransmitters like dopamine, serotonin, glutamate, and GABA mediate communication between neurons at chemical synapses. Imbalances are implicated in depression, schizophrenia, and Parkinson's disease, and psychiatric drugs target specific receptor subtypes to modulate neural circuits.",
    "Relativity theory links space, time, matter, and energy through Einstein's field equations. Special relativity describes the behavior of objects moving at relativistic speeds in flat spacetime, while general relativity extends these ideas to curved spacetime and gravitation.",
    "Algorithms for finding shortest paths in weighted graphs include Dijkstra's algorithm, which works for non-negative edge weights, and Bellman-Ford, which handles negative weights and detects negative cycles. A-star search adds heuristic guidance for efficient pathfinding in explicit state spaces.",
    "The immune system recognizes pathogens through pattern recognition receptors on innate immune cells and through adaptive responses mediated by T and B lymphocytes with diverse antigen receptors generated via somatic recombination of gene segments.",
    "Bayesian inference updates a prior probability distribution over parameters using the likelihood of observed data to produce a posterior distribution consistent with the evidence. Markov chain Monte Carlo methods sample from posteriors that lack closed-form expressions.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases over time. This principle sets the arrow of time and underlies the impossibility of perfect heat engines, with profound consequences for cosmology and information theory.",
    "Satellites orbit Earth because their tangential velocity balances the gravitational pull toward the planet, producing a closed trajectory. Orbital mechanics connects altitude, period, and speed, allowing precise prediction and control of spacecraft trajectories.",
    "The brain is divided into regions such as the cerebral cortex, limbic system, cerebellum, and brainstem, each specialized for different functions including motor control, sensory processing, emotion, and autonomic regulation of bodily systems.",
    "Ohm's law relates voltage, current, and resistance in an electrical circuit through V = IR. Kirchhoff's laws extend this analysis to networks of components, enabling the systematic solution of complex circuits used in electronic devices.",
    "The Fourier transform decomposes a signal into its constituent frequencies. The fast Fourier transform algorithm reduces computation from quadratic to near-linear, enabling practical digital signal processing for audio, images, and communication systems.",
    "Gradient descent minimizes a differentiable loss by iteratively moving parameters in the negative gradient direction. Variants including momentum, AdaGrad, RMSProp, and Adam adapt the step size to improve convergence on stochastic objectives.",
    "The Riemann zeta function extends the Dirichlet series into the complex plane by analytic continuation. Its non-trivial zeros, all conjectured to lie on the critical line, encode deep information about the distribution of prime numbers.",
    "Mitosis divides a cell's nucleus into two genetically identical daughter cells. The process proceeds through prophase, metaphase, anaphase, and telophase, coordinated by cyclins and cyclin-dependent kinases that serve as cell cycle checkpoints.",
    "Public health interventions reduce the spread of infectious disease through vaccination campaigns, sanitation improvements, quarantine measures, and contact tracing. Epidemiology models such as SIR capture transmission dynamics and inform policy decisions.",
    "Wavelets decompose signals across both scale and position, providing time-frequency localization that complements the global view of Fourier analysis. They are widely used in image compression standards like JPEG 2000 and in analyzing non-stationary processes.",
    "Language models learn statistical structure from text corpora by optimizing next-token prediction. The learned representations capture syntactic, semantic, and pragmatic regularities of the training distribution, enabling fluent generation and in-context learning.",
    "Neural circuits in the retina extract features such as edges, motion, and color from light falling on photoreceptors. Signals pass through bipolar and ganglion cells, encoding visual information in spike trains transmitted along the optic nerve to the brain.",
    "The printing press, developed by Johannes Gutenberg in the 15th century, enabled the mass production of books. This dramatically reduced the cost of information dissemination and played a central role in the Reformation and the spread of scientific knowledge.",
    "Molecular dynamics simulations solve Newton's equations of motion for atoms in a system, enabling the study of protein folding, material properties, and chemical reactions at the atomic scale. Force fields parametrize interatomic interactions to reproduce experimental data.",
    "Deep learning models often exhibit double descent in their generalization error as model capacity grows, showing non-monotonic behavior where performance worsens then improves. This phenomenon challenges classical bias-variance intuitions and has spurred new theoretical work.",
    "Weather forecasting uses numerical models that solve the fluid dynamics and thermodynamics equations of the atmosphere, initialized from observational data. Ensemble methods run many simulations with perturbed initial conditions to quantify forecast uncertainty.",
    "Molecular biology studies the structure and function of nucleic acids and proteins. Techniques including polymerase chain reaction, Sanger sequencing, and CRISPR gene editing have transformed biological research and enabled precision medicine.",
    "The kidneys filter blood to produce urine, regulating blood pressure, electrolyte balance, and waste elimination. Nephrons are the functional units, combining a glomerulus for filtration with a tubular system for reabsorption and secretion of solutes.",
    "In topology, a Mobius strip is a one-sided non-orientable surface formed by joining the ends of a rectangle with a half twist. It serves as a classic example of non-orientable manifolds and appears in studies of chirality and symmetry.",
    "The history of mathematics spans thousands of years, from ancient Babylonian and Egyptian calculations to modern abstract algebra and category theory. Each era built upon prior developments, often in response to practical needs from astronomy, commerce, or physics.",
    "Renewable energy sources include solar, wind, hydro, geothermal, and biomass. Transitioning away from fossil fuels requires advances in energy storage, grid management, and economic models that account for intermittency and long-distance transmission losses.",
    "Quantum entanglement occurs when particles share correlated quantum states such that measuring one instantaneously affects the other, regardless of distance. This non-local phenomenon underlies quantum information processing and has been experimentally verified many times.",
    "Statistical mechanics connects microscopic ensembles to macroscopic thermodynamic observables through partition functions and probability distributions. Phase transitions, critical phenomena, and universality classes emerge from the mathematical structure of these ensembles.",
    "Compilers translate high-level source code into executable machine code through stages of lexical analysis, parsing, semantic analysis, optimization, and code generation. Modern compilers apply aggressive optimizations such as inlining, loop unrolling, and vectorization.",
    "The auditory system transduces air pressure oscillations into neural signals through hair cells in the cochlea. The basilar membrane performs a frequency-to-place mapping, so different locations along it respond to different pitches.",
    "Ocean currents transport heat, nutrients, and organisms across the globe. Thermohaline circulation couples surface and deep waters, influencing regional climates. Changes in circulation patterns have been linked to past episodes of rapid climate change.",
    "Distributed systems coordinate multiple computers to perform tasks that exceed the capacity of any single machine. Consensus protocols like Paxos and Raft enable replicated state machines to survive node failures while maintaining consistency.",
    "The Silk Road connected Asia with Europe for centuries, enabling not only the exchange of goods but also ideas, religions, and technologies. Caravans transported silk, spices, and precious metals across thousands of miles through hostile terrain.",
    "Operational amplifiers are high-gain DC-coupled amplifiers widely used in analog circuits. Feedback configurations implement signal processing functions including amplification, filtering, integration, and differentiation with predictable behavior.",
    "The speed of light is constant in all inertial reference frames, a postulate of special relativity. This leads to the relativity of simultaneity, time dilation, and length contraction at speeds approaching that of light.",
    "Artificial intelligence as a field studies the design of agents that perceive, reason, learn, and act in complex environments. Modern systems combine statistical learning with symbolic representations and structured reasoning to solve diverse tasks.",
    "Operating systems manage hardware resources including CPU scheduling, memory allocation, file systems, and device drivers. They provide abstractions that let application programs run without detailed knowledge of underlying hardware.",
]

HELDOUT_CORPUS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres.",
    "Topological insulators behave as insulators in their interior but conduct electricity along their surface.",
    "Recombinant DNA technology emerged in the 1970s with restriction enzymes.",
    "The Antikythera mechanism is an ancient analog computer from the second century BCE.",
    "Edge-triggered flip-flops change state only on the clock edge.",
    "Vector clocks extend Lamport timestamps for distributed event ordering.",
    "Germanium is a metalloid in the carbon group widely used in fiber-optic systems.",
    "The Curie temperature is where a ferromagnetic material loses its permanent magnetism.",
]


def pca_basis(X, k):
    mean = X.mean(dim=0)
    Xc = X - mean
    cov = Xc.T @ Xc
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).to(torch.float32), mean


def capture_final_hidden(model, tokenizer, texts, device, max_len=256, apply_final_norm=False):
    """Return stacked final-layer hidden states and tokenized inputs.
    If apply_final_norm=True, pass hidden states through model.model.norm
    first — this is what the lm_head actually operates on."""
    finals = []
    inputs = []
    final_norm = model.model.norm if apply_final_norm else None
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
            h_final = out.hidden_states[-1][0]
            if final_norm is not None:
                h_final = final_norm(h_final)
            h_final = h_final.to(torch.float32).cpu()
            finals.append(h_final)
            inputs.append(ids.cpu())
    return finals, inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="7,10,15,20,32,64")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage22_token_recovery.json")
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
    print(f"device={device}  model={args.model}")

    ranks = [int(x) for x in args.ranks.split(",")]

    print(f"\n=== loading {args.model} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    d = model.config.hidden_size
    print(f"  hidden={d}")

    # --- Calibration: collect POST-NORM final hidden states ---
    # These are the vectors lm_head actually operates on.
    print(f"\n=== calibration pass ({len(CALIB_CORPUS)} texts) ===")
    t0 = time.perf_counter()
    calib_finals, _ = capture_final_hidden(model, tokenizer, CALIB_CORPUS, device,
                                            apply_final_norm=True)
    X_calib = torch.cat(calib_finals, dim=0)  # [N, d]
    print(f"  collected {X_calib.shape[0]} positions  ({time.perf_counter()-t0:.1f}s)")

    # --- Held-out pass: capture POST-NORM final hidden states ---
    print(f"\n=== held-out pass ({len(HELDOUT_CORPUS)} texts) ===")
    heldout_finals, heldout_ids = capture_final_hidden(
        model, tokenizer, HELDOUT_CORPUS, device, apply_final_norm=True)

    # Teacher argmax computed from the (post-norm) final states going straight to lm_head.
    lm_head = model.lm_head
    teacher_argmax_all = []
    with torch.inference_mode():
        for h, ids in zip(heldout_finals, heldout_ids):
            h_dev = h.to(device).to(torch.bfloat16)
            logits = lm_head(h_dev)
            teacher_argmax_all.append(logits.argmax(dim=-1).cpu())

    # --- Bases to compare ---
    # (1) PCA of post-norm hidden states: variance-aligned.
    # (2) Right-SVD of lm_head weight: prediction-aligned (what matters
    #     for the lm_head output).
    # For a curved manifold, (2) should recover more than (1) at the
    # same rank because lm_head's output is what we're trying to preserve.
    mean_of_calib = X_calib.mean(dim=0)
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)  # [vocab, hidden]
    # Right singular vectors of lm_head are the top-k directions in
    # hidden space that contribute most to logit variance.
    U_lm, S_lm, Vh_lm = torch.linalg.svd(lm_head_weight, full_matrices=False)
    # Vh_lm: [hidden, hidden]; Vh_lm[:k] is the top-k right singular vectors.

    # --- For each rank, try both bases ---
    print(f"\n=== recovery rates (PCA vs lm_head-SVD basis) ===")
    print(f"  {'rank':>5}  {'PCA':>10}  {'lm_head SVD':>12}")
    results = []
    for k in ranks:
        # PCA basis
        P_pca, mean = pca_basis(X_calib, k)  # [d, k], [d]
        # lm_head SVD basis (fixed mean = calib mean)
        P_lm = Vh_lm[:k].T.contiguous()  # [hidden, k]
        # For each held-out position, project and back
        def measure_recovery(P, mean_ref):
            matches = 0
            total = 0
            with torch.inference_mode():
                for h_heldout, teacher_argmax in zip(heldout_finals, teacher_argmax_all):
                    h_centered = h_heldout - mean_ref
                    coords = h_centered @ P
                    h_recon = coords @ P.T + mean_ref
                    h_recon_dev = h_recon.to(device).to(torch.bfloat16)
                    logits = lm_head(h_recon_dev)
                    recon_argmax = logits.argmax(dim=-1).cpu()
                    matches += (recon_argmax == teacher_argmax).sum().item()
                    total += recon_argmax.numel()
            return matches, total

        m_pca, total = measure_recovery(P_pca, mean)
        m_lm, _ = measure_recovery(P_lm, mean_of_calib)
        rate_pca = m_pca / max(total, 1)
        rate_lm = m_lm / max(total, 1)
        print(f"  {k:>5}  {m_pca}/{total:<4} ({rate_pca:5.1%})  "
              f"{m_lm}/{total:<4} ({rate_lm:5.1%})")
        results.append({
            "rank": k,
            "pca_matches": m_pca,
            "lm_svd_matches": m_lm,
            "total": total,
            "pca_recovery": rate_pca,
            "lm_svd_recovery": rate_lm,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "hidden_size": d,
            "calib_positions": int(X_calib.shape[0]),
            "heldout_positions": sum(x.numel() for x in teacher_argmax_all),
            "per_rank": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
