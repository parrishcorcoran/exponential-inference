"""
Stage 123b — PCA-subspace throat transplant.

Stage 123 failed because the full-rank linear map A (2048×1024) was
underdetermined and didn't generalize. Stage 122c showed only ~100 of
small's top-200 PCs align with large's at corr>0.9.

So the correct transplant swaps ONLY the aligned subspace. Procedure:

  1. PCA train throat states → V_L (d_L × k), V_S (d_S × k), k≈100.
  2. Fit small (k×k) map A: large_PC ≈ small_PC via linear regression
     in PC coords. Much smaller → generalizes.
  3. At inference, hook small's throat layer. Decompose own throat:
       h_S = shared_S + private_S   (shared via V_S V_S^T projection)
     Compute large's shared contribution mapped to small's basis:
       shared_from_L = (h_L - xm) V_L A V_S^T + ym_shared
     Replace shared component:
       new_throat = α * shared_from_L + (1-α) * shared_S + private_S
  4. Sweep α. Report NLL.

This isolates the question: "does aligned-subspace information in
large's throat help small's mouth 2?"

We also evaluate:
  - test R² of A in PC space (generalization diagnostic)
  - sweep over k ∈ {50, 100, 200} to find the right shared dimension
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
    out = []
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        h = output.hidden_states[throat_layer][0].float()
        out.append(h[mask].cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def extract_throat_per_sentence(model, tokenizer, passages, throat_layer, device, max_length=256):
    """Per-sentence list of [seq, d] tensors (real tokens only)."""
    out = []
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        h = output.hidden_states[throat_layer][0].float()
        out.append(h[mask].cpu())
    return out


@torch.no_grad()
def compute_nll(model, tokenizer, passages, device, max_length=256,
                transplant_hook=None, throat_layer=None):
    handle = None
    if transplant_hook is not None:
        target_module = model.model.layers[throat_layer - 1]
        handle = target_module.register_forward_hook(transplant_hook)
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
        if handle is not None:
            handle.remove()
    mean_nll = total_nll / max(1, total_toks)
    return mean_nll, float(np.exp(mean_nll)), total_toks


def pca_basis(X, k):
    """Return PCs [d, k] from centered data [N, d]. X already centered."""
    # SVD on X: X = U S V^T. V's columns are PCs.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    return V[:, :k], S[:k]


def make_pca_transplant_hook(replacement_large_per_sent, xm, V_L, V_S, A, ym_small, alpha,
                              mask_per_sent):
    """Hook that implements the PCA-subspace transplant.

    For each call idx:
      h_S = small's own throat output        [seq, d_S]
      h_S_c = h_S - ym_small
      shared_S = h_S_c @ (V_S @ V_S^T)       [seq, d_S]
      private_S = h_S_c - shared_S
      h_L = replacement_large_per_sent[idx]  [real_seq, d_L]
      shared_from_L = (h_L - xm) @ V_L @ A @ V_S^T  [real_seq, d_S]
      new_shared = α * shared_from_L + (1-α) * shared_S[:real_seq]
      new_throat = new_shared + private_S[:real_seq] + ym_small
    """
    VSVS = V_S @ V_S.T  # [d_S, d_S]
    map_full = V_L @ A @ V_S.T  # [d_L, d_S]
    call_idx = {"i": 0}

    def hook(module, inputs, output):
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx >= len(replacement_large_per_sent):
            return output
        if isinstance(output, tuple):
            hs = output[0]; rest = output[1:]
        else:
            hs = output; rest = None
        # hs: [1, seq, d_S]
        h_S = hs[0].float().cpu().numpy()  # [seq, d_S]
        h_S_c = h_S - ym_small
        shared_S = h_S_c @ VSVS
        private_S = h_S_c - shared_S

        h_L = replacement_large_per_sent[idx].numpy()  # [real_seq, d_L]
        h_L_c = h_L - xm
        shared_from_L = h_L_c @ map_full  # [real_seq, d_S]

        L = min(h_L.shape[0], h_S.shape[0])
        new_shared = alpha * shared_from_L[:L] + (1.0 - alpha) * shared_S[:L]
        new_throat = new_shared + private_S[:L] + ym_small  # [L, d_S]

        new_hs_np = h_S.copy()
        new_hs_np[:L] = new_throat
        new_hs = torch.from_numpy(new_hs_np).to(hs.device).to(hs.dtype).unsqueeze(0)
        if rest is None:
            return new_hs
        return (new_hs,) + rest

    return hook, call_idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--small", default="Qwen/Qwen3-0.6B")
    p.add_argument("--large", default="Qwen/Qwen3-1.7B")
    p.add_argument("--throat-frac", type=float, default=0.50)
    p.add_argument("--out", default="results/stage123b_pca_transplant.json")
    p.add_argument("--device", default=None)
    p.add_argument("--ks", default="50,100,200")
    p.add_argument("--alphas", default="0.0,0.5,1.0")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    ks = [int(x) for x in args.ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    print(f"device={device}  throat_frac={args.throat_frac}  ks={ks}  alphas={alphas}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.small, trust_remote_code=True)

    # === LARGE ===
    print(f"\nloading {args.large}...", flush=True); t0 = time.time()
    large = load_model(args.large, device)
    L_large = large.config.num_hidden_layers
    throat_l = max(1, min(L_large, int(args.throat_frac * L_large)))
    d_L = large.config.hidden_size
    print(f"  L={L_large}  throat_layer={throat_l}  d={d_L}")
    train_large_throat = extract_throat_tokens(large, tokenizer, PASSAGES_TRAIN, throat_l, device)
    test_large_throat_per_sent = extract_throat_per_sentence(
        large, tokenizer, PASSAGES_TEST, throat_l, device)
    # Also collect per-token test for R² check
    test_large_throat_concat = torch.cat(test_large_throat_per_sent, dim=0)
    baseline_large_nll, baseline_large_ppl, _ = compute_nll(large, tokenizer, PASSAGES_TEST, device)
    print(f"  baseline large: NLL={baseline_large_nll:.4f}  PPL={baseline_large_ppl:.2f}")
    print(f"  train_N={train_large_throat.shape[0]}  test_N={test_large_throat_concat.shape[0]}")
    del large; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # === SMALL ===
    print(f"\nloading {args.small}...", flush=True)
    small = load_model(args.small, device)
    L_small = small.config.num_hidden_layers
    throat_s = max(1, min(L_small, int(args.throat_frac * L_small)))
    d_S = small.config.hidden_size
    print(f"  L={L_small}  throat_layer={throat_s}  d={d_S}")
    train_small_throat = extract_throat_tokens(small, tokenizer, PASSAGES_TRAIN, throat_s, device)
    test_small_throat_concat = extract_throat_tokens(small, tokenizer, PASSAGES_TEST, throat_s, device)
    baseline_small_nll, baseline_small_ppl, n_toks = compute_nll(small, tokenizer, PASSAGES_TEST, device)
    print(f"  baseline small: NLL={baseline_small_nll:.4f}  PPL={baseline_small_ppl:.2f}  toks={n_toks}")

    # === PCA + A per k ===
    X_train = train_large_throat.numpy()
    Y_train = train_small_throat.numpy()
    X_test  = test_large_throat_concat.numpy()
    Y_test  = test_small_throat_concat.numpy()
    N_train = min(X_train.shape[0], Y_train.shape[0])
    X_train = X_train[:N_train]; Y_train = Y_train[:N_train]
    N_test = min(X_test.shape[0], Y_test.shape[0])
    X_test  = X_test[:N_test];   Y_test  = Y_test[:N_test]

    xm = X_train.mean(axis=0, keepdims=True)  # [1, d_L]
    ym = Y_train.mean(axis=0, keepdims=True)  # [1, d_S]
    Xc = X_train - xm; Yc = Y_train - ym
    Xc_te = X_test - xm; Yc_te = Y_test - ym

    # Compute large's & small's full PCA bases once (k_max)
    k_max = max(ks)
    V_L_full, S_L = pca_basis(Xc, k_max)
    V_S_full, S_S = pca_basis(Yc, k_max)

    results = {
        "baseline_small": {"nll": baseline_small_nll, "ppl": baseline_small_ppl},
        "baseline_large": {"nll": baseline_large_nll, "ppl": baseline_large_ppl},
        "throat_layers": {"small": throat_s, "large": throat_l},
        "per_k": {},
    }

    print("\n=== PCA-subspace transplant sweep ===")
    for k in ks:
        V_L = V_L_full[:, :k]
        V_S = V_S_full[:, :k]
        # Project both train & test into PC spaces
        Xp_tr = Xc @ V_L       # [N_tr, k]
        Yp_tr = Yc @ V_S       # [N_tr, k]
        Xp_te = Xc_te @ V_L    # [N_te, k]
        Yp_te = Yc_te @ V_S    # [N_te, k]
        # Fit A in PC space: Yp ≈ Xp @ A
        A, *_ = np.linalg.lstsq(Xp_tr, Yp_tr, rcond=None)
        # R² train + test
        Yp_tr_pred = Xp_tr @ A
        r2_tr = 1 - ((Yp_tr - Yp_tr_pred)**2).sum() / ((Yp_tr - Yp_tr.mean(0))**2).sum()
        Yp_te_pred = Xp_te @ A
        r2_te = 1 - ((Yp_te - Yp_te_pred)**2).sum() / ((Yp_te - Yp_te.mean(0))**2).sum()
        evr_s = (S_S[:k]**2).sum() / (S_S**2).sum()
        evr_l = (S_L[:k]**2).sum() / (S_L**2).sum()
        print(f"\n  k={k}: evr_s={evr_s:.3f}  evr_l={evr_l:.3f}  "
              f"A shape {A.shape}  R²_train={r2_tr:.3f}  R²_test={r2_te:.3f}")

        per_alpha = {}
        for alpha in alphas:
            hook_fn, call_idx = make_pca_transplant_hook(
                test_large_throat_per_sent, xm, V_L, V_S, A, ym, alpha, None)
            nll, ppl, _ = compute_nll(small, tokenizer, PASSAGES_TEST, device,
                                       transplant_hook=hook_fn, throat_layer=throat_s)
            per_alpha[f"{alpha:.2f}"] = {"nll": nll, "ppl": ppl}
            delta = nll - baseline_small_nll
            direction = "↓ better" if delta < 0 else "↑ worse"
            print(f"    α={alpha:.2f}:  NLL={nll:.4f}  PPL={ppl:.2f}   Δ={delta:+.4f}  {direction}")
        results["per_k"][str(k)] = {
            "evr_small": float(evr_s), "evr_large": float(evr_l),
            "R2_train": float(r2_tr), "R2_test": float(r2_te),
            "alphas": per_alpha,
        }

    # === verdict ===
    print(f"\n=== summary ===")
    print(f"  baseline small:  NLL={baseline_small_nll:.4f}  PPL={baseline_small_ppl:.2f}")
    print(f"  baseline large:  NLL={baseline_large_nll:.4f}  PPL={baseline_large_ppl:.2f}")
    best_k, best_nll, best_alpha = None, baseline_small_nll, None
    for k_str, kr in results["per_k"].items():
        for a_str, ar in kr["alphas"].items():
            if ar["nll"] < best_nll:
                best_nll = ar["nll"]; best_k = k_str; best_alpha = a_str
    if best_k is not None:
        print(f"  best transplant: k={best_k}  α={best_alpha}  NLL={best_nll:.4f}  "
              f"(Δ vs small {best_nll - baseline_small_nll:+.4f})")
    else:
        print(f"  no transplant configuration improved on small baseline")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
