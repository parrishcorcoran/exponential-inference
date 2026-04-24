"""
Stage 124 — Rank-k activation factorization at throat layers.

If throat activations live on ~k principal directions, each throat
matmul `y = x @ W` can be factored as:
  y ≈ (x @ U_k) @ (U_k^T @ W)
  total flops: seq×d×k + seq×k×d_out
  baseline:   seq×d×d_out
  speedup:    d_out / k  (for k << d)

Stage 111 claims throat is rank-1 in VARIANCE but stage 119 showed
rank-1 projection destroys NLL. So the TRUE working rank is higher.
This stage measures it directly: sweep k, report NLL.

Procedure:
  1. Calibrate: run CALIB_SENTS through 0.6B, collect activations at
     each throat layer.
  2. For each layer, compute top-k PCs (U_k) from calibration.
  3. Install activation-projection hook at each throat layer's INPUT:
     replace x with (x @ U_k) @ U_k^T (still d-dimensional, but rank-k).
     This is equivalent in output to the factored matmul.
  4. Measure NLL on TEST_SENTS for k ∈ {1, 4, 16, 64, 256, 1024}.
  5. Compute theoretical wall-clock speedup per k.

Output: working rank floor + speedup curve.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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
def collect_layer_inputs(model, tokenizer, passages, layer_indices, device, max_length=256):
    """For each layer in layer_indices, collect its INPUT activation
       (i.e. the residual state at that layer's entry)."""
    # hidden_states[l] = input to layer l (or embedding if l=0)
    out = {l: [] for l in layer_indices}
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        for l in layer_indices:
            h = output.hidden_states[l][0].float()  # [seq, d]
            out[l].append(h[mask].cpu())
    return {l: torch.cat(v, dim=0) for l, v in out.items()}  # [N_tokens, d]


def compute_pcs(X, k_max):
    """Return top-k PCs U [d, k_max] from centered X."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    return V[:, :k_max], S[:k_max], X.mean(0)


@torch.no_grad()
def compute_nll(model, tokenizer, passages, device, hooks=None, max_length=256):
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


def make_projection_pre_hook(U_k, mean_vec, device):
    """Pre-forward hook on a decoder layer: replaces input x with
       projection (x - μ) @ U_k @ U_k^T + μ (i.e. rank-k approximation
       of the centered activation, then re-add mean)."""
    U_k_t = torch.from_numpy(U_k).to(device).to(torch.float32)         # [d, k]
    mean_t = torch.from_numpy(mean_vec).to(device).to(torch.float32)   # [d]
    P = (U_k_t @ U_k_t.T).to(torch.float32)                            # [d, d]

    def hook(module, inputs):
        x = inputs[0]  # [1, seq, d]
        orig_dtype = x.dtype
        xc = x.float() - mean_t
        new_x = (xc @ P + mean_t).to(orig_dtype)
        return (new_x,) + inputs[1:]
    return hook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage124_rank_factored.json")
    p.add_argument("--device", default=None)
    p.add_argument("--throat-start", type=float, default=0.10)
    p.add_argument("--throat-end", type=float, default=0.75)
    p.add_argument("--ks", default="1,4,16,64,256,1024")
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

    print(f"\nloading {args.model}...")
    model = load_model(args.model, device)
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    throat_first = max(1, int(args.throat_start * L))
    throat_last  = max(throat_first, int(args.throat_end * L))
    throat_layers = list(range(throat_first, throat_last + 1))
    print(f"L={L}  d={d}  throat layers: {throat_layers} ({len(throat_layers)} layers)")

    k_max = max(ks)
    k_max = min(k_max, d)

    # === calibrate: get top-k PCs per throat layer ===
    print(f"\ncalibrating on {len(CALIB_SENTS)} sentences...")
    t0 = time.time()
    layer_inputs = collect_layer_inputs(model, tokenizer, CALIB_SENTS, throat_layers, device)
    print(f"  collected in {time.time()-t0:.0f}s")

    pcs = {}  # layer -> (U[d, k_max], S, mean)
    for l in throat_layers:
        X = layer_inputs[l].numpy()
        U, S, mean = compute_pcs(X, k_max)
        pcs[l] = (U, S, mean)
        # Report EVR at select k
        evr = (S**2).cumsum() / (S**2).sum()
        evr_at = {k: float(evr[min(k-1, len(evr)-1)]) for k in ks if k <= len(evr)}
        print(f"  L{l}: EVR at k " + "  ".join(f"k={k}:{v:.3f}" for k,v in evr_at.items()))

    # === baseline ===
    print(f"\nbaseline (no projection)...")
    nll_base, n_toks = compute_nll(model, tokenizer, TEST_SENTS, device)
    print(f"  NLL={nll_base:.4f}  PPL={np.exp(nll_base):.2f}  toks={n_toks}")

    # === sweep k ===
    print(f"\nsweep k ∈ {ks}...")
    results = {"baseline_nll": nll_base, "baseline_ppl": float(np.exp(nll_base)),
               "throat_layers": throat_layers, "d_model": d, "per_k": {}}
    for k in ks:
        if k > d:
            print(f"  k={k} > d={d}, skip")
            continue
        # Build hooks: one per throat layer
        hooks = []
        for l in throat_layers:
            U, S, mean = pcs[l]
            U_k = U[:, :k].astype(np.float32)
            pre_hook = make_projection_pre_hook(U_k, mean.astype(np.float32), device)
            module = model.model.layers[l - 1] if l > 0 else model.model.embed_tokens
            hooks.append((module, pre_hook))
        nll, _ = compute_nll(model, tokenizer, TEST_SENTS, device, hooks=hooks)
        ppl = float(np.exp(nll))
        # Theoretical FLOP speedup in throat matmuls only
        # Each matmul: seq × d × d_out  vs  seq × (d + d_out) × k
        # Qwen uses d_ffn = 3 * d for gate/up/down roughly.
        d_out_avg = d  # Q/K/V/O are d×d; gate/up/down are d×d_ffn
        speedup = d_out_avg / (k * (1 + d_out_avg / d))  # rough per-matmul
        total_layer_flops_base = len(throat_layers)
        # Proportion of full-model FLOPs saved (throat layers / total)
        throat_frac = len(throat_layers) / L
        # Amdahl: 1 / ((1 - throat_frac) + throat_frac / speedup)
        amdahl = 1.0 / ((1 - throat_frac) + throat_frac / speedup)
        results["per_k"][str(k)] = {
            "nll": nll, "ppl": ppl,
            "delta_nll": nll - nll_base,
            "per_matmul_speedup": float(speedup),
            "model_wide_speedup_amdahl": float(amdahl),
            "throat_fraction": throat_frac,
        }
        print(f"  k={k:5d}:  NLL={nll:.4f}  Δ={nll-nll_base:+.4f}  "
              f"per-matmul={speedup:.1f}×  model-wide={amdahl:.2f}×")

    # verdict: find floor
    print(f"\n=== verdict ===")
    print(f"  baseline:        NLL={nll_base:.4f}  PPL={np.exp(nll_base):.2f}")
    acceptable = [(int(k), r) for k, r in results["per_k"].items() if r["delta_nll"] < 0.05]
    if acceptable:
        acceptable.sort()
        k_min, r_min = acceptable[0]
        print(f"  smallest k with Δ<0.05 nat:  k={k_min}  "
              f"model-wide speedup={r_min['model_wide_speedup_amdahl']:.2f}×")
    else:
        print(f"  no k achieves Δ<0.05 nat — throat needs >{ks[-1]} dims")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
