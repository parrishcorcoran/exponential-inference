"""
Stage 123 — Throat transplant: inject large model's throat state into
small model's forward pass. Does small's mouth 2 decode large's denser
superposition?

Hypothesis path:
  - Stage 121: throat coords align linearly across models (R²=0.94 at
    pos 0.25). So there exists A: large_throat → small_throat_basis.
  - Stage 120: throat is rank-1 universal.
  - If the throat is a universal holographic channel, large packs MORE
    superimposed information into the same rank-1 axis. Small's mouth 2
    was trained to unbind from its own throat, but if the basis aligns,
    it should still decode a richer throat state and give better logits.

Procedure:
  1. Split sentences into train (fit A) and test (measure perplexity).
  2. For train sentences:
       - Run large up to throat → large_throat_state [N, d_large]
       - Run small up to throat → small_throat_state [N, d_small]
     Fit A (least-squares): Y ≈ X @ A  where X=large, Y=small basis.
  3. For test sentences, evaluate three conditions via teacher-forced
     forward passes, measuring mean next-token NLL:
       baseline_small: small model unmodified
       baseline_large: large model unmodified (upper-bound reference)
       transplant:     small model, throat layer output replaced with
                       A @ large_throat_state for α ∈ {0, 0.25, 0.5, 0.75, 1.0}

  α=0 should exactly match baseline_small (sanity). α=1 tests the
  hypothesis. Intermediate points show the transition curve.

Outputs mean NLL + perplexity per condition per α.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PASSAGES_TRAIN = [
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

PASSAGES_TEST = [
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
def extract_throat_tokens(model, tokenizer, passages, throat_layer, device, max_length=256):
    """Return per-token hidden states at throat_layer: [N_tokens, d]."""
    out = []
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        h = output.hidden_states[throat_layer][0].float()  # [seq, d]
        out.append(h[mask].cpu())
    return torch.cat(out, dim=0)  # [N_total, d]


@torch.no_grad()
def compute_nll(model, tokenizer, passages, device, max_length=256,
                transplant_hook=None, throat_layer=None):
    """Mean next-token NLL across passages (teacher-forced).
       If transplant_hook is not None, installs a forward hook on
       model.model.layers[throat_layer-1] that replaces its output."""
    handle = None
    if transplant_hook is not None:
        # hidden_states[l] is the OUTPUT of layer l-1 (or embedding if l=0).
        # So to replace hidden_states at index throat_layer, we hook layer index (throat_layer-1).
        target_module = model.model.layers[throat_layer - 1]
        handle = target_module.register_forward_hook(transplant_hook)

    total_nll = 0.0
    total_toks = 0
    try:
        for sent in passages:
            enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask[0].bool().to(device)
            logits = model(ids, use_cache=False).logits[0]  # [seq, V]
            # teacher-forced NLL: for position t, predict ids[t+1] from logits[t]
            shift_logits = logits[:-1]           # [seq-1, V]
            shift_labels = ids[0, 1:]            # [seq-1]
            shift_mask   = mask[1:]              # [seq-1]
            logp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [seq-1]
            nll = nll[shift_mask]
            total_nll += nll.sum().item()
            total_toks += int(shift_mask.sum())
    finally:
        if handle is not None:
            handle.remove()

    mean_nll = total_nll / max(1, total_toks)
    return mean_nll, float(np.exp(mean_nll)), total_toks


def make_transplant_hook(replacement_states_per_sentence, alpha, state_iter):
    """Return a forward hook that replaces the module output's first tensor
       (hidden states) with a mix: α * transplant + (1-α) * original.

       state_iter is a list of replacement tensors, one per sentence, each
       [seq_len, d]. Consumed in order of calls."""
    call_idx = {"i": 0}

    def hook(module, inputs, output):
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx >= len(replacement_states_per_sentence):
            return output
        # Handle both tensor-output and tuple-output layer APIs
        if isinstance(output, tuple):
            hs = output[0]; rest = output[1:]
        else:
            hs = output; rest = None
        # hs: [1, seq, d]
        replacement = replacement_states_per_sentence[idx].to(hs.device).to(hs.dtype)
        L = min(hs.shape[1], replacement.shape[0])
        new_hs = hs.clone()
        new_hs[0, :L] = alpha * replacement[:L] + (1.0 - alpha) * hs[0, :L]
        if rest is None:
            return new_hs
        return (new_hs,) + rest

    return hook, call_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", default="Qwen/Qwen3-0.6B")
    p.add_argument("--large", default="Qwen/Qwen3-1.7B")
    p.add_argument("--throat-frac", type=float, default=0.50,
                   help="Normalized depth for throat layer")
    p.add_argument("--out", default="results/stage123_throat_transplant.json")
    p.add_argument("--device", default=None)
    p.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}  throat_frac={args.throat_frac}", flush=True)

    alphas = [float(a) for a in args.alphas.split(",")]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.small, trust_remote_code=True)

    # === Step 1: run LARGE, collect throat states on train + test ===
    print(f"\nloading {args.large}...", flush=True); t0 = time.time()
    large = load_model(args.large, device)
    L_large = large.config.num_hidden_layers
    throat_l = max(1, min(L_large, int(args.throat_frac * L_large)))
    print(f"  L={L_large}  throat_layer={throat_l}  d={large.config.hidden_size}")

    train_large_throat = extract_throat_tokens(large, tokenizer, PASSAGES_TRAIN, throat_l, device)
    # For test, store throat state PER SENTENCE (to hook one-by-one)
    test_large_throat_per_sent = []
    for sent in PASSAGES_TEST:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        with torch.no_grad():
            output = large(ids, use_cache=False, output_hidden_states=True)
        h = output.hidden_states[throat_l][0].float().cpu()  # [seq, d_large]
        test_large_throat_per_sent.append(h)

    baseline_large_nll, baseline_large_ppl, _ = compute_nll(large, tokenizer, PASSAGES_TEST, device)
    print(f"  baseline large: NLL={baseline_large_nll:.4f}  PPL={baseline_large_ppl:.2f}")
    print(f"  extract large took {time.time()-t0:.0f}s")

    del large; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # === Step 2: run SMALL, collect throat states on train; compute baseline NLL ===
    print(f"\nloading {args.small}...", flush=True); t0 = time.time()
    small = load_model(args.small, device)
    L_small = small.config.num_hidden_layers
    throat_s = max(1, min(L_small, int(args.throat_frac * L_small)))
    print(f"  L={L_small}  throat_layer={throat_s}  d={small.config.hidden_size}")

    train_small_throat = extract_throat_tokens(small, tokenizer, PASSAGES_TRAIN, throat_s, device)
    baseline_small_nll, baseline_small_ppl, n_toks = compute_nll(
        small, tokenizer, PASSAGES_TEST, device)
    print(f"  baseline small: NLL={baseline_small_nll:.4f}  PPL={baseline_small_ppl:.2f}  toks={n_toks}")

    # === Step 3: fit A such that train_small_throat ≈ train_large_throat @ A ===
    X = train_large_throat.numpy()  # [N, d_large]
    Y = train_small_throat.numpy()  # [N, d_small]
    N_train = min(X.shape[0], Y.shape[0])
    X = X[:N_train]; Y = Y[:N_train]
    # Center
    xm = X.mean(axis=0, keepdims=True); ym = Y.mean(axis=0, keepdims=True)
    Xc = X - xm; Yc = Y - ym
    A, *_ = np.linalg.lstsq(Xc, Yc, rcond=None)
    Y_pred = Xc @ A
    ss_res = ((Yc - Y_pred) ** 2).sum()
    ss_tot = ((Yc - Yc.mean(axis=0)) ** 2).sum()
    r2_train = 1 - ss_res / ss_tot
    print(f"\n  trained A: shape {A.shape}  train R²={r2_train:.3f}")

    # Project test large throat states into small basis: Yhat = (Xc) @ A + ym
    test_replacements = []
    for h_large in test_large_throat_per_sent:
        Xc_test = h_large.numpy() - xm
        Yhat = Xc_test @ A + ym
        test_replacements.append(torch.from_numpy(Yhat).float())

    # === Step 4: sweep alpha, compute NLL with transplant hook ===
    print("\n=== alpha sweep (transplant into small at throat) ===")
    sweep = {}
    for a in alphas:
        hook_fn, call_idx = make_transplant_hook(test_replacements, a, None)
        nll, ppl, _ = compute_nll(small, tokenizer, PASSAGES_TEST, device,
                                   transplant_hook=hook_fn, throat_layer=throat_s)
        sweep[f"{a:.2f}"] = {"nll": nll, "ppl": ppl, "calls": call_idx["i"]}
        print(f"  α={a:.2f}:  NLL={nll:.4f}  PPL={ppl:.2f}   (hook fired {call_idx['i']}x)")

    # === Step 5: report ===
    print(f"\n=== summary ===")
    print(f"  baseline small:  NLL={baseline_small_nll:.4f}  PPL={baseline_small_ppl:.2f}")
    print(f"  baseline large:  NLL={baseline_large_nll:.4f}  PPL={baseline_large_ppl:.2f}")
    for a, r in sweep.items():
        delta = r["nll"] - baseline_small_nll
        gain = " (↓ better)" if delta < 0 else " (↑ worse)"
        print(f"  α={a}: NLL={r['nll']:.4f}  PPL={r['ppl']:.2f}  Δ vs small={delta:+.4f}{gain}")

    out = {
        "args": vars(args),
        "throat_layer_small": throat_s,
        "throat_layer_large": throat_l,
        "train_R2": float(r2_train),
        "baseline_small": {"nll": baseline_small_nll, "ppl": baseline_small_ppl},
        "baseline_large": {"nll": baseline_large_nll, "ppl": baseline_large_ppl},
        "sweep": sweep,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
