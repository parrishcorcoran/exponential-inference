"""
Stage 13 — Scaled calibration distillation.

Stage 8's 16-text calibration caused the rank-32 student to memorize
calibration rather than match teacher's manifold on held-out text
(ppl 12.5 -> 11,511, 922x gap). RSB reading: too few manifold samples
→ spurious calibration-specific basins.

This stage scales calibration to thousands of diverse text chunks and
re-runs the distillation + distribution eval. If the ppl gap closes on
held-out text, the problem was data-volume. If it doesn't, the problem
is structural (need on-policy training, or different factorization).

Calibration source, in priority order:
    1. HuggingFace `datasets` library if present, loading wikitext-2-raw-v1.
    2. If unavailable / offline, a large hardcoded block of diverse texts.

Usage:
    python scripts/stage13_scaled_calibration.py \\
        --model Qwen/Qwen3-0.6B --rank 32 --steps 3000 --device mps
"""

import argparse
import json
import math
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


# Held-out text, disjoint from any calibration source.
HELDOUT_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres across North America, from Canada to central Mexico, over multiple generations.",
    "Topological insulators are materials that behave as insulators in their interior but conduct electricity along their surface, a consequence of spin-orbit coupling and time-reversal symmetry.",
    "Recombinant DNA technology emerged in the 1970s when researchers discovered restriction enzymes that cut DNA at specific sequences, enabling genes to be inserted into bacterial plasmids.",
    "The Antikythera mechanism, recovered from a Greek shipwreck, is an ancient analog computer dating from the second century BCE that tracked astronomical positions and eclipses.",
    "Edge-triggered flip-flops store a single bit of information and change state only on the rising or falling edge of a clock signal, making them central to digital sequential logic.",
    "The Curie temperature is the point above which a ferromagnetic material loses its permanent magnetic properties as thermal agitation disrupts the alignment of atomic dipoles.",
    "Vector clocks extend Lamport timestamps to detect concurrency relationships between events in distributed systems, making causal ordering observable without global clocks.",
    "Germanium is a lustrous, hard, grayish-white metalloid in the carbon group, chemically similar to its group neighbours silicon and tin, and widely used in fiber-optic systems.",
]


def load_calibration_corpus(min_tokens, max_chunk_len, tokenizer):
    """Return a list of token-id tensors, each of length up to max_chunk_len.
    Aims for at least min_tokens total. Tries datasets/wikitext first; if
    unavailable, uses the large hardcoded block below."""
    chunks = []
    total = 0
    try:
        from datasets import load_dataset
        print("  trying datasets wikitext-2-raw-v1 ...", flush=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                          download_mode="reuse_cache_if_exists")
        buf = ""
        for ex in ds:
            t = (ex.get("text") or "").strip()
            if len(t) < 20:
                continue
            buf += " " + t
            if len(buf) > 4 * max_chunk_len:
                chunks.append(buf.strip())
                buf = ""
                total_est = sum(len(c) for c in chunks)
                if total_est > min_tokens * 6:
                    break
        if buf.strip():
            chunks.append(buf.strip())
        print(f"  wikitext: {len(chunks)} chunks", flush=True)
    except Exception as e:
        print(f"  datasets unavailable / failed ({e!r}); using hardcoded corpus", flush=True)

    if not chunks:
        # Hardcoded large diverse block ~10k+ tokens
        chunks = HARDCODED_CORPUS

    # Tokenize, truncate
    out = []
    for c in chunks:
        ids = tokenizer(c, return_tensors="pt", truncation=True,
                        max_length=max_chunk_len).input_ids
        if ids.shape[1] >= 16:
            out.append(ids)
            total += ids.shape[1]
        if total >= min_tokens * 4:
            break
    print(f"  final calibration: {len(out)} chunks, ~{total} tokens", flush=True)
    return out


# Fallback hardcoded corpus (~120 diverse paragraphs). Kept inline so this
# script works with no internet. Approx 8-12k tokens after tokenization.
HARDCODED_CORPUS = [
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology and machine learning, suggesting that both biological and artificial systems exploit contextual compression.",
    "Quantum mechanics describes the behaviour of matter and energy at the atomic and subatomic scale, where particles exhibit wave-like properties and observations can collapse a superposition of states.",
    "Protein folding is the physical process by which a linear polypeptide chain acquires its three-dimensional structure, guided by a complex free-energy landscape.",
    "The cosmic microwave background is the thermal remnant of the early universe, observed at a temperature of approximately 2.7 kelvin and containing subtle anisotropies that encode cosmological parameters.",
    "Markov chain Monte Carlo methods sample from complex probability distributions by simulating a stochastic chain whose stationary distribution matches the target density.",
    "The Riemann zeta function extends the Dirichlet series into the complex plane by analytic continuation and encodes deep information about the distribution of prime numbers.",
    "Photosynthesis converts light energy into chemical bonds, producing glucose from carbon dioxide and water while releasing molecular oxygen as a byproduct.",
    "Attention in modern neural networks computes weighted averages of token representations, with weights learned to reflect relational structure in the sequence.",
    "Plate tectonics describes the slow movement of the Earth's lithosphere over the mantle, producing earthquakes, volcanic activity, and mountain ranges at plate boundaries.",
    "Public-key cryptography is built on mathematical problems that are easy to compute in one direction but hard to invert without a secret, such as factoring or discrete logarithms.",
    "Neurotransmitters including dopamine, serotonin, glutamate, and GABA mediate communication between neurons at chemical synapses and are common drug targets.",
    "The second law of thermodynamics asserts that the entropy of an isolated system cannot decrease over time, establishing the arrow of time in macroscopic physics.",
    "Gravitational waves are ripples in the fabric of spacetime generated by accelerating masses; LIGO made the first direct detection in 2015 from a binary black hole merger.",
    "Neural networks are parameterised function approximators trained by gradient descent on a differentiable loss, with expressive capacity that grows with depth and width.",
    "Evolution by natural selection operates on heritable variation, causing allele frequencies in a population to shift over generations as some variants reproduce more than others.",
    "In topology, a Mobius strip is a one-sided non-orientable surface that can be constructed by joining the ends of a rectangle with a single half-twist.",
    "Fluid turbulence exhibits multi-scale structure with energy cascading from large to small scales through vortex stretching and dissipating at the Kolmogorov length.",
    "Nuclear magnetic resonance exploits the interaction of nuclear spins with a strong magnetic field and radiofrequency pulses to infer the chemical environment of atoms.",
    "Lattice gauge theory discretises spacetime for numerical simulation of quantum chromodynamics, enabling non-perturbative predictions of hadron properties.",
    "The central dogma of molecular biology describes the flow of information from DNA to RNA to protein, with regulatory mechanisms at each step controlling gene expression.",
    "Formal verification techniques prove mathematical properties of programs or hardware designs, reducing dependence on testing for safety-critical systems.",
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions, predicting a family of fermions and gauge bosons observed in high-energy experiments.",
    "Statistical mechanics connects macroscopic thermodynamic observables to microscopic ensembles of states, deriving phase transitions from partition functions.",
    "The Haber-Bosch process synthesises ammonia from atmospheric nitrogen and hydrogen under high pressure and temperature, underpinning modern agricultural fertilizer.",
    "Graph neural networks generalise convolutional architectures to non-Euclidean domains, learning representations of nodes and edges on arbitrary connectivity.",
    "Convex optimization treats problems where the feasible region is convex and the objective is convex, admitting efficient algorithms and globally optimal solutions.",
    "Superconductivity is the complete loss of electrical resistance below a critical temperature, arising from Cooper pair condensation into a macroscopic quantum state.",
    "The Krebs cycle oxidises acetyl-CoA to carbon dioxide in the mitochondrial matrix, producing NADH and FADH2 that fuel oxidative phosphorylation.",
    "Compiler optimization transforms source code into efficient machine code through techniques such as loop unrolling, inlining, and register allocation.",
    "Homotopy type theory formalises mathematics in the language of dependent types with an equivalence between types treated as paths in a topological space.",
    "The three-body problem in classical mechanics lacks a closed-form general solution and exhibits sensitive dependence on initial conditions, a hallmark of deterministic chaos.",
    "CRISPR-Cas systems are adaptive immune mechanisms in bacteria that recognise viral DNA and have been adapted as programmable gene-editing tools.",
    "Reed-Solomon codes encode data with algebraic redundancy over finite fields, correcting burst errors in optical media and satellite communications.",
    "The Navier-Stokes equations describe the motion of viscous fluid substances, and the question of whether smooth solutions always exist in three dimensions remains open.",
    "Population genetics models how allele frequencies evolve under selection, mutation, migration, and drift, connecting microevolutionary forces to observed diversity.",
    "Reinforcement learning agents interact with an environment to maximise cumulative reward, balancing exploration of new actions against exploitation of known gains.",
    "The Mandelbrot set is the locus of complex parameters for which the iterated quadratic map remains bounded, producing infinitely intricate fractal boundary structure.",
    "Organic chemistry classifies reactions into types such as addition, elimination, substitution, and rearrangement, each with characteristic mechanisms and stereochemical outcomes.",
    "The Schwarzschild metric describes spacetime outside a spherically symmetric non-rotating mass and was the first exact solution of Einstein's field equations.",
    "Linear programming optimises a linear objective over a polyhedral feasible region and is solved efficiently by the simplex method or interior-point algorithms.",
    "Sigmund Freud proposed the division of the psyche into id, ego, and superego, a framework that heavily influenced twentieth-century psychology and culture.",
    "The Metropolis-Hastings algorithm generates samples from a target distribution using an auxiliary proposal and an acceptance probability preserving detailed balance.",
    "Black-body radiation from a heated object follows Planck's law, with spectral peak wavelength inversely proportional to temperature according to Wien's displacement law.",
    "Differential privacy provides mathematical guarantees that an individual's data contributes only a bounded amount to a statistic released from a dataset.",
    "The Higgs field is a scalar field permeating the universe whose non-zero vacuum expectation value gives mass to elementary particles through the Higgs mechanism.",
    "SQL is a declarative query language for relational databases, with operations expressed in terms of selections, projections, joins, and set-theoretic combinations.",
    "Neuronal firing rates encode information in cortical circuits, and populations of neurons represent signals through distributed patterns of activity.",
    "The EPR paradox, formulated by Einstein, Podolsky and Rosen, motivated Bell's inequalities which experimentally rule out local hidden variable theories.",
    "Lagrangian mechanics reformulates Newtonian dynamics in terms of a scalar Lagrangian, whose Euler-Lagrange equations yield the equations of motion.",
    "Heap data structures maintain a partial order on elements to support efficient insertion and extraction of minimum or maximum values, forming the basis of priority queues.",
    "Transformers replaced recurrent networks in sequence modelling by using self-attention to capture long-range dependencies in parallel across positions.",
    "The KL divergence quantifies how one probability distribution diverges from a reference distribution and is minimised by maximum likelihood estimation.",
    "Raman spectroscopy uses inelastic scattering of monochromatic light to probe vibrational modes of molecules, complementing infrared absorption techniques.",
    "The genetic code maps triplets of nucleotides to amino acids, with a few special codons signalling start and stop of protein synthesis.",
    "Bayesian inference updates a prior distribution over parameters by the likelihood of observed data to produce a posterior distribution consistent with evidence.",
    "Cardiac action potentials propagate through the myocardium via gap junctions, coordinating the contraction of atria and ventricles in a regular sequence.",
    "The Chomsky hierarchy classifies formal grammars by generative power, linking language classes to automata models from finite state machines to Turing machines.",
    "Fermat's little theorem states that if p is prime, then a to the power p is congruent to a modulo p, a result central to primality testing.",
    "Catalysis accelerates chemical reactions by providing an alternative pathway with lower activation energy, without being consumed in the overall process.",
    "Hopfield networks are energy-based associative memories where binary neurons evolve to low-energy attractors encoding stored patterns.",
    "The Wiener filter estimates a signal by minimising the mean square error between the filter output and a desired process under stationarity assumptions.",
    "The Pauli exclusion principle forbids two identical fermions from occupying the same quantum state, underlying the structure of atomic shells and chemistry.",
    "Category theory abstracts mathematics through objects and morphisms, studying structure-preserving maps between categories and the compositionality of mathematics itself.",
    "Coastal upwelling brings nutrient-rich deep water to the surface, supporting high biological productivity and major fisheries along continental margins.",
    "Kalman filters combine noisy measurements with a dynamical model to estimate the state of a linear system under Gaussian noise assumptions.",
    "The James Webb Space Telescope observes in the infrared from the second Lagrange point, revealing early galaxies, exoplanet atmospheres, and dusty star-forming regions.",
    "Boltzmann's entropy S equals k log W, connecting macroscopic entropy to the number of microstates consistent with a given macroscopic state.",
    "Elliptic curve cryptography offers security comparable to RSA at smaller key sizes by exploiting the difficulty of the discrete logarithm on elliptic curves.",
    "Dendrochronology uses tree-ring patterns to reconstruct past climates and to date archaeological timbers with annual precision.",
    "Operator algebras generalise matrix algebras to infinite-dimensional spaces and provide the mathematical scaffolding of quantum mechanics and quantum field theory.",
    "Glycolysis converts glucose to pyruvate in the cytoplasm, generating ATP and NADH that feed downstream energy-producing pathways.",
    "The transformer's multi-head attention mechanism splits representations across heads so that different relations can be attended to in parallel.",
    "Ricci flow smooths the metric of a manifold by evolving it along its Ricci curvature, and was used by Perelman to prove the Poincare conjecture.",
    "Wavelets decompose signals across scale and location, providing time-frequency localisation that complements the global view of Fourier analysis.",
    "The immune system distinguishes self from non-self using a combinatorial repertoire of receptors generated by genetic recombination and affinity maturation.",
    "Algebraic topology studies topological spaces via invariants such as homotopy and homology groups, which encode holes of different dimensions.",
    "The Berry phase arises when a quantum state is transported along a closed path in parameter space and depends only on the geometry of the path.",
    "Bit error rate characterises the quality of a digital communication channel and decreases as modulation, coding, and signal-to-noise ratio improve.",
    "Allosteric regulation of proteins involves a conformational change at one site that propagates to affect function at a distant site, enabling cellular control.",
    "The Fourier uncertainty principle limits how localised a signal can be simultaneously in time and frequency, with a lower bound on the product of their spreads.",
    "Solar cells convert sunlight into electricity through the photovoltaic effect, with modern silicon modules reaching conversion efficiencies above 20 percent.",
    "Gradient boosting builds ensembles of decision trees by iteratively fitting residuals, producing strong predictive models on tabular data.",
    "The Gibbs free energy determines the spontaneity of a chemical process at constant temperature and pressure, combining enthalpy and entropy into a single criterion.",
    "Satellite navigation systems trilaterate receiver positions from signals broadcast by a constellation of atomic-clock-equipped satellites.",
    "Cilia and flagella are microtubule-based cellular appendages that generate motion through coordinated bending powered by dynein motor proteins.",
    "Quantum error correction encodes logical qubits redundantly across many physical qubits so that local errors can be detected and reversed without measuring the state.",
    "The Hubble tension refers to a disagreement between local and cosmological measurements of the expansion rate of the universe that has resisted explanation.",
    "Stochastic differential equations extend ordinary differential equations with a random term and are used to model asset prices, thermal noise, and biological processes.",
    "Metabolic flux analysis infers the rates of intracellular reactions from measurements of isotope labelling patterns, illuminating cellular physiology.",
    "Hash functions map arbitrary-length inputs to fixed-length outputs, and cryptographic hash functions additionally resist preimage and collision attacks.",
    "The Meissner effect is the expulsion of magnetic flux from a superconductor as it is cooled below its critical temperature, distinguishing superconductors from mere perfect conductors.",
    "Reinforcement signals in the basal ganglia modulate synaptic plasticity in the striatum, supporting trial-and-error learning of motor and cognitive skills.",
    "The Fast Fourier Transform reduces discrete Fourier transform complexity from quadratic to nearly linear, enabling modern digital signal processing.",
    "Prions are misfolded proteins that can template the misfolding of other proteins of the same type, causing neurodegenerative diseases such as Creutzfeldt-Jakob.",
    "Tensor network methods efficiently represent quantum many-body states with limited entanglement, making some otherwise intractable problems tractable.",
    "The Boltzmann machine is a generative stochastic neural network whose learning rule descends the gradient of the log-likelihood under a Gibbs distribution.",
    "Epigenetic modifications such as DNA methylation and histone acetylation modulate gene expression without altering the underlying DNA sequence.",
    "The speed of light in a medium is reduced relative to vacuum by the refractive index, which varies with wavelength, producing dispersion and chromatic aberration.",
    "Lagrange multipliers enforce constraints in optimisation by augmenting the objective with a term proportional to each constraint, with the multiplier as dual variable.",
    "The Gamma function extends the factorial to complex arguments, satisfying the recurrence Gamma(n+1) = n Gamma(n) and with a pole at every non-positive integer.",
    "Bird flight involves lift generation by wing shape, thrust by flapping, and fine control by tail movements, adapted across species to hovering, soaring, or diving.",
    "The PageRank algorithm models a random surfer visiting pages in proportion to link structure, producing an eigenvector that scores each node's importance.",
]


class BasisFactoredLinear(nn.Module):
    def __init__(self, orig: nn.Linear, P_in: torch.Tensor, trainable: bool = True):
        super().__init__()
        k = P_in.shape[1]
        device = orig.weight.device
        W = orig.weight.data.to(torch.float32).cpu()
        P = P_in.to(torch.float32).cpu()
        A = (W @ P).contiguous().to(device).to(torch.float32)
        B = P.T.contiguous().to(device).to(torch.float32)
        self.A = nn.Parameter(A, requires_grad=trainable)
        self.B = nn.Parameter(B, requires_grad=trainable)
        if orig.bias is not None:
            self.bias = nn.Parameter(
                orig.bias.data.to(torch.float32).to(device),
                requires_grad=trainable)
        else:
            self.register_parameter("bias", None)
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.rank = k
        self._full_params = orig.in_features * orig.out_features
        self._factored_params = k * (orig.in_features + orig.out_features)

    def forward(self, x):
        dt = x.dtype
        x32 = x.to(torch.float32)
        out = F.linear(F.linear(x32, self.B), self.A, self.bias)
        return out.to(dt)


def collect_input_covariances(model, tokenizer, batches, device):
    covs = {}
    counts = {}
    target_modules = []
    for name, module in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and last in TARGET_NAMES:
            target_modules.append((name, module))

    def make_hook(n, d_in):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            if n not in covs:
                covs[n] = torch.zeros(d_in, d_in, dtype=torch.float32)
                counts[n] = 0
            covs[n] += x_flat.T @ x_flat
            counts[n] += x_flat.shape[0]
        return hook

    handles = []
    for name, mod in target_modules:
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))

    model.eval()
    with torch.inference_mode():
        for ids in batches:
            ids = ids.to(device)
            model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()
    return {n: c.to(torch.float64) for n, c in covs.items()}, counts


def top_k_basis_from_cov(cov: torch.Tensor, k: int) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).contiguous()


def factorize_with_basis(model, covariances, rank: int, trainable: bool = True):
    stats = {"n_replaced": 0, "full_params": 0, "factored_params": 0}
    bases = {n: top_k_basis_from_cov(c, rank) for n, c in covariances.items()}
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
            fact = BasisFactoredLinear(child, P_in=P, trainable=trainable)
            setattr(module, child_name, fact)
            stats["n_replaced"] += 1
            stats["full_params"] += fact._full_params
            stats["factored_params"] += fact._factored_params
    return stats


def freeze_non_factored(model):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_params = 0
    for mod in model.modules():
        if isinstance(mod, BasisFactoredLinear):
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)
            trainable_params += mod.A.numel() + mod.B.numel()
            if mod.bias is not None:
                trainable_params += mod.bias.numel()
    return trainable_params


def distill(teacher, student, batches, steps, lr, device, log_every=100, warmup=100):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        progress = (s - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    student.train()
    teacher.eval()
    history = []
    t0 = time.perf_counter()
    step = 0
    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = batch.to(device)
            with torch.inference_mode():
                t_out = teacher(input_ids=batch, use_cache=False,
                                output_hidden_states=True)
            t_logits = t_out.logits.detach()
            t_hidden = [h.detach() for h in t_out.hidden_states]

            s_out = student(input_ids=batch, use_cache=False,
                            output_hidden_states=True)
            s_logits = s_out.logits
            s_hidden = s_out.hidden_states

            h_loss = 0.0
            n_layers = len(t_hidden)
            for th, sh in zip(t_hidden, s_hidden):
                num = (sh.float() - th.float()).pow(2).mean()
                denom = th.float().pow(2).mean().clamp_min(1e-8)
                h_loss = h_loss + num / denom
            h_loss = h_loss / n_layers

            s_logp = F.log_softmax(s_logits.float(), dim=-1)
            t_p = F.softmax(t_logits.float(), dim=-1)
            kl = F.kl_div(s_logp, t_p, reduction="batchmean")

            loss = h_loss + kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()

            if (step % log_every == 0) or (step == steps - 1):
                elapsed = time.perf_counter() - t0
                print(f"  step {step:5d}  loss={loss.item():.4f}  "
                      f"(h={h_loss.item():.4f} kl={kl.item():.4f})  "
                      f"lr={lr_at(step):.2e}  ({elapsed:.1f}s)", flush=True)
                history.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "kl": float(kl.item()),
                    "h_loss": float(h_loss.item()),
                    "elapsed_sec": elapsed,
                })
            step += 1
    return history


def distribution_eval(teacher, student, tokenizer, texts, device):
    teacher.eval()
    student.eval()
    res = {"teacher_ppl": [], "student_ppl": [], "position_kl": [],
           "top1_agree": [], "top5_agree": []}
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=256).input_ids.to(device)
            if ids.shape[1] < 4:
                continue
            t_out = teacher(input_ids=ids, use_cache=False)
            s_out = student(input_ids=ids, use_cache=False)
            t_logits = t_out.logits[0, :-1].float()
            s_logits = s_out.logits[0, :-1].float()
            targets = ids[0, 1:]

            t_nll = -F.log_softmax(t_logits, -1).gather(1, targets.unsqueeze(1)).mean()
            s_nll = -F.log_softmax(s_logits, -1).gather(1, targets.unsqueeze(1)).mean()
            res["teacher_ppl"].append(float(t_nll.exp().item()))
            res["student_ppl"].append(float(s_nll.exp().item()))

            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            res["position_kl"].append(float(kl.item()))

            t_top1 = t_logits.argmax(-1)
            s_top1 = s_logits.argmax(-1)
            res["top1_agree"].append(float((t_top1 == s_top1).float().mean().item()))
            t_top5 = t_logits.topk(5, dim=-1).indices
            in_top5 = (s_top1.unsqueeze(-1) == t_top5).any(-1).float().mean()
            res["top5_agree"].append(float(in_top5.item()))

    out = {k: sum(v) / max(len(v), 1) for k, v in res.items()}
    out["ppl_ratio"] = out["student_ppl"] / max(out["teacher_ppl"], 1e-9)
    return out


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--calib-target-tokens", type=int, default=8000,
                   help="Approximate number of tokens per calibration chunk to aim for")
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
    print(f"\ndevice={device}  rank={args.rank}  steps={args.steps}")

    print(f"\n=== loading teacher {args.model} ===", flush=True)
    teacher, tokenizer = load_model(args.model, device)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print(f"\n=== building calibration corpus ===", flush=True)
    batches = load_calibration_corpus(args.calib_target_tokens, args.calib_max_len, tokenizer)
    total_calib_tokens = sum(b.shape[1] for b in batches)
    print(f"  {len(batches)} chunks, {total_calib_tokens} total tokens")

    print(f"\n=== collecting covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, counts = collect_input_covariances(teacher, tokenizer, batches, device)
    print(f"  {len(covs)} covs, {next(iter(counts.values()))} toks/layer, "
          f"{time.perf_counter()-t0:.1f}s")

    print(f"\n=== factorizing student ===", flush=True)
    student, _ = load_model(args.model, device)
    fstats = factorize_with_basis(student, covs, rank=args.rank, trainable=True)
    ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
    print(f"  factored {fstats['n_replaced']} linears, "
          f"{fstats['factored_params']/1e6:.2f}M params ({ratio:.2%})")
    trainable = freeze_non_factored(student)

    print(f"\n=== distribution eval PRE-training ===", flush=True)
    pre = distribution_eval(teacher, student, tokenizer, HELDOUT_TEXTS, device)
    print(f"  ppl_ratio={pre['ppl_ratio']:.2f}  kl={pre['position_kl']:.3f}  "
          f"top1={pre['top1_agree']:.1%}  top5={pre['top5_agree']:.1%}")

    print(f"\n=== distilling ===", flush=True)
    history = distill(teacher, student, batches, args.steps, args.lr, device)

    print(f"\n=== distribution eval POST-training ===", flush=True)
    post = distribution_eval(teacher, student, tokenizer, HELDOUT_TEXTS, device)
    print(f"  teacher ppl: {post['teacher_ppl']:.3f}")
    print(f"  student ppl: {post['student_ppl']:.3f}")
    print(f"  ppl ratio:   {post['ppl_ratio']:.3f}  (1.0 = perfect)")
    print(f"  mean KL: {post['position_kl']:.4f}")
    print(f"  top-1 agreement: {post['top1_agree']:.1%}")
    print(f"  top-5 agreement: {post['top5_agree']:.1%}")

    out_path = Path(args.out_dir) / f"stage13_scaled_calibration_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "rank": args.rank,
            "steps": args.steps,
            "calibration_chunks": len(batches),
            "calibration_tokens": total_calib_tokens,
            "weight_size_ratio": ratio,
            "trainable_params_M": trainable / 1e6,
            "distribution_eval_pre": pre,
            "distribution_eval_post": post,
            "loss_history": history,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
