"""
Stage 124b — Fine-grained rank anneal at throat to find the working floor.

Stage 124 jumped from k=1024 to 256 to 64 — and had two bugs:
  - hook indexing off-by-one (projected wrong layer's input)
  - N_calib=450 < d=1024 so even k=1024 was projected onto rank-450

Fix both, then slowly step k down: 1024 → 900 → 768 → … → 1. Identify
the rank at which NLL falls off a cliff.

This replaces stage 124's sparse sweep with a dense one.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# More + longer passages to get N_calib >> d_model=1024
CALIB_SENTS = [
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
    "describing an expanding homogeneous universe.",
    "The theory of computation distinguishes problems by the resources "
    "required to solve them. The class P contains problems solvable in "
    "polynomial time on a deterministic Turing machine, while NP contains "
    "those whose solutions can be verified in polynomial time. Whether "
    "these classes are equal remains the central open problem of "
    "complexity theory. Complete problems such as Boolean satisfiability "
    "anchor the hierarchy.",
    "Immunology has revealed how vertebrates combine innate and adaptive "
    "defenses to meet a constantly changing landscape of pathogens. "
    "Innate cells such as macrophages and neutrophils provide rapid, "
    "nonspecific responses, while lymphocytes generate highly specific "
    "antibodies and cellular memory through somatic recombination of "
    "receptor genes.",
    "The development of high-performance neural network models has been "
    "driven by scaling laws that empirically relate compute, data, and "
    "parameters to downstream capability. Transformer architectures, "
    "with self-attention and residual connections, have become the "
    "dominant backbone because they scale favorably and admit efficient "
    "parallel training.",
    "Plate tectonics unifies a broad range of geological observations by "
    "describing the Earth's lithosphere as a mosaic of rigid plates "
    "moving over a ductile asthenosphere. Boundaries between plates are "
    "sites of seismic and volcanic activity: mid-ocean ridges where new "
    "crust is created, subduction zones where oceanic lithosphere sinks "
    "back into the mantle, and transform faults where plates slide "
    "laterally.",
    "Category theory provides a unifying language across mathematics by "
    "focusing on structure-preserving mappings rather than internal set "
    "details. Categories consist of objects and morphisms composing "
    "associatively with identities; functors map categories to "
    "categories while natural transformations relate functors.",
    "Thermodynamics constrains the possible transformations of matter "
    "and energy through a small set of universal laws. The first law "
    "asserts energy conservation, the second law introduces entropy as a "
    "state function that never decreases in isolated systems, and the "
    "third law places limits on the approach to absolute zero.",
]

TEST_SENTS = [
    "Superconductors carry electricity without any resistance below a critical temperature.",
    "Coral reefs support an extraordinary diversity of marine organisms.",
    "The theorem states that every continuous function on a compact set is bounded.",
    "Ancient astronomers tracked the motion of planets through the night sky.",
    "Enzymes accelerate chemical reactions by lowering the activation energy barrier.",
    "The novelist wrote each morning before the sun came up over the hills.",
    "Computer scientists study the limits of what algorithms can compute efficiently.",
    "The river carved a deep canyon through the soft sandstone over millions of years.",
    "Bacteria reproduce rapidly when conditions of temperature and moisture are favorable.",
    "Poets have long used metaphor to compress complex feelings into small phrases.",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


@torch.no_grad()
def collect_layer_inputs(model, tokenizer, passages, layer_indices, device, max_length=512):
    """hidden_states[l] = input to layer l. Collect those for l in layer_indices."""
    out = {l: [] for l in layer_indices}
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        for l in layer_indices:
            h = output.hidden_states[l][0].float()
            out[l].append(h[mask].cpu())
    return {l: torch.cat(v, dim=0) for l, v in out.items()}


def compute_pcs(X, k_max):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    return V[:, :k_max], S[:k_max], X.mean(0)


@torch.no_grad()
def compute_nll(model, tokenizer, passages, device, hooks=None, max_length=512):
    handles = []
    if hooks:
        for module, hook in hooks:
            handles.append(module.register_forward_pre_hook(hook))
    total_nll = 0.0
    total_toks = 0
    try:
        for sent in passages:
            enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask[0].bool().to(device)
            logits = model(ids, use_cache=False).logits[0]
            shift_logits = logits[:-1]
            shift_labels = ids[0, 1:]
            shift_mask   = mask[1:]
            logp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            nll = nll[shift_mask]
            total_nll += nll.sum().item()
            total_toks += int(shift_mask.sum())
    finally:
        for h in handles:
            h.remove()
    return total_nll / max(1, total_toks), total_toks


def make_projection_pre_hook(U_k_np, mean_np, device):
    """Project the INPUT of this decoder layer to rank-k:
       new_x = (x - μ) @ U_k U_k^T + μ"""
    U_k = torch.from_numpy(U_k_np).to(device).to(torch.float32)        # [d, k]
    mu  = torch.from_numpy(mean_np).to(device).to(torch.float32)       # [d]
    P   = (U_k @ U_k.T).to(torch.float32)                              # [d, d]

    def hook(module, inputs):
        x = inputs[0]
        orig_dtype = x.dtype
        xc = x.float() - mu
        new_x = (xc @ P + mu).to(orig_dtype)
        return (new_x,) + inputs[1:]
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage124b_rank_anneal.json")
    p.add_argument("--device", default=None)
    p.add_argument("--throat-start", type=float, default=0.10)
    p.add_argument("--throat-end", type=float, default=0.75)
    p.add_argument("--ks",
                   default="1024,900,768,640,512,448,384,320,256,224,192,160,128,96,64,48,32,24,16,12,8,6,4,3,2,1")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    ks = [int(x) for x in args.ks.split(",")]
    print(f"device={device}  ks={ks}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model = load_model(args.model, device)
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    throat_first = max(1, int(args.throat_start * L))
    throat_last  = max(throat_first, int(args.throat_end * L))
    throat_layers = list(range(throat_first, throat_last + 1))
    print(f"L={L}  d={d}  throat layers (hidden_states indices): {throat_layers}")

    # Calibration — count tokens first
    tok_counts = []
    for s in CALIB_SENTS:
        tok_counts.append(len(tokenizer(s, truncation=True, max_length=512).input_ids))
    print(f"calib sentences: {len(CALIB_SENTS)}  total tokens: {sum(tok_counts)}")

    t0 = time.time()
    layer_inputs = collect_layer_inputs(model, tokenizer, CALIB_SENTS, throat_layers, device, max_length=512)
    N_calib = layer_inputs[throat_layers[0]].shape[0]
    print(f"  collected N={N_calib} tokens per layer in {time.time()-t0:.0f}s  (d={d}, N/d ratio={N_calib/d:.2f})")
    if N_calib < 2 * d:
        print(f"  WARNING: N_calib={N_calib} < 2*d={2*d}; calibration span may be under-spanned for high k")

    # Fit PCs per layer to k_max
    k_max = max(ks)
    k_max = min(k_max, d, N_calib - 1)
    pcs = {}
    for l in throat_layers:
        X = layer_inputs[l].numpy()
        U, S, mean = compute_pcs(X, k_max)
        pcs[l] = (U, S, mean)

    # baseline
    nll_base, n_toks = compute_nll(model, tokenizer, TEST_SENTS, device)
    print(f"\nbaseline: NLL={nll_base:.4f}  PPL={np.exp(nll_base):.2f}  toks={n_toks}")

    # sweep k (high → low). Print compact one-line per k.
    print(f"\n=== anneal k from {max(ks)} down to {min(ks)} ===")
    results = {"baseline_nll": nll_base, "baseline_ppl": float(np.exp(nll_base)),
               "throat_layers": throat_layers, "d_model": d, "N_calib": N_calib,
               "per_k": {}}
    prev_nll = nll_base
    for k in ks:
        if k > k_max:
            # Can't faithfully project to k > k_max given SVD rank
            continue
        hooks = []
        for l in throat_layers:
            U, S, mean = pcs[l]
            U_k = U[:, :k].astype(np.float32)
            pre_hook = make_projection_pre_hook(U_k, mean.astype(np.float32), device)
            if l < len(model.model.layers):
                module = model.model.layers[l]
                hooks.append((module, pre_hook))
        nll, _ = compute_nll(model, tokenizer, TEST_SENTS, device, hooks=hooks)
        delta = nll - nll_base
        jump = nll - prev_nll
        throat_frac = len(throat_layers) / L
        per_matmul_speedup = d / k * 0.5  # rough — matmul dominated by bigger dim
        amdahl = 1.0 / ((1 - throat_frac) + throat_frac / per_matmul_speedup)
        results["per_k"][str(k)] = {
            "nll": nll, "ppl": float(np.exp(nll)),
            "delta_nll": delta, "jump_from_prev": jump,
            "per_matmul_speedup": float(per_matmul_speedup),
            "model_wide_speedup": float(amdahl),
        }
        marker = "   "
        if delta < 0.05:        marker = " ✓ "
        elif delta < 0.2:       marker = " ~ "
        elif delta < 1.0:       marker = " ! "
        else:                   marker = "XXX"
        print(f"  k={k:5d}  NLL={nll:7.4f}  Δ={delta:+7.4f}  step={jump:+7.4f}  "
              f"speedup×={amdahl:5.2f}  {marker}")
        prev_nll = nll

    # find floor: smallest k with Δ<0.05 and smallest with Δ<0.2
    print(f"\n=== floor analysis ===")
    for thresh, label in [(0.05, "tight (Δ<0.05)"), (0.2, "loose (Δ<0.2)"), (1.0, "survival (Δ<1.0)")]:
        good = sorted((int(k) for k, r in results["per_k"].items() if r["delta_nll"] < thresh))
        if good:
            floor = min(good)
            r = results["per_k"][str(floor)]
            print(f"  {label} floor: k={floor}  (speedup {r['model_wide_speedup']:.2f}×)")
        else:
            print(f"  {label} floor: NONE — no k achieves this quality")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
