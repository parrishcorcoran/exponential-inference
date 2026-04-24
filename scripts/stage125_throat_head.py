"""
Stage 125 — Throat head: can a tiny decoder at L14 match full mouth 2?

Hypothesis: the throat state at L14 already encodes the next-token
distribution (and possibly t+2, t+3). Mouth 2 (layers 15-28) mostly
unbinds into vocabulary-space. If a small learned head can match
mouth 2's output from L14 state, then:
  - Half the layers are redundant for single-step decode
  - Medusa-style parallel unbind becomes: K heads at L14, zero extra
    sequential work per future token
  - Speculative decoding acceptance ceiling is whatever this head can
    approach

Tests in escalating order:

  (A) linear projection: h_14 @ A → predicted h_27 (input to LM head)
      If this works, mouth 2 is LINEARLY predictable from throat.
  (B) 2-layer MLP: same target, more capacity
  (C) multi-step: add a head that predicts h_14(t+1) from h_14(t) —
      i.e. future throat state. Then apply mouth 2 or linear head to
      decode t+2.

For each, measure NLL on held-out vs full-forward baseline.

Tiny training loop: MSE on h_27 residual from training sentences.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_SENTS = [
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
    "bound Cooper pairs that flow without resistance.",
    "Modern cryptography is built on computational problems believed to be "
    "intractable for classical computers. Public key systems such as RSA "
    "rely on the difficulty of factoring large integers, while elliptic "
    "curve schemes exploit the hardness of the discrete logarithm.",
    "The central dogma of molecular biology describes the flow of "
    "sequence information from DNA through RNA to proteins.",
    "The development of high-performance neural network models has been "
    "driven by scaling laws that empirically relate compute, data, and "
    "parameters to downstream capability.",
    "Category theory provides a unifying language across mathematics by "
    "focusing on structure-preserving mappings rather than internal set "
    "details.",
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
def collect_state_pairs(model, tokenizer, passages, l_throat, l_final, device, max_length=512):
    """Collect (h_throat, h_final) per-token pairs.
       h_final is taken AFTER the model's final RMSNorm — i.e. the input to the LM head.
       That way our predicted h_final can be piped directly into model.lm_head."""
    Xs, Ys, Toks = [], [], []
    # The final hidden state available via output_hidden_states is BEFORE the final
    # norm. We need AFTER the final norm to match LM head input. Apply norm manually.
    final_norm = model.model.norm
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        mask = enc.attention_mask[0].bool()
        output = model(ids, use_cache=False, output_hidden_states=True)
        h_throat = output.hidden_states[l_throat][0].float()      # [seq, d]
        h_prefinal = output.hidden_states[l_final][0]             # bf16 [seq, d]
        h_final = final_norm(h_prefinal).float()                  # [seq, d]
        Xs.append(h_throat[mask].cpu())
        Ys.append(h_final[mask].cpu())
        # shifted targets for next-token prediction check — we'll recompute in eval
        Toks.append(ids[0].cpu())
    X = torch.cat(Xs, dim=0)   # [N, d]
    Y = torch.cat(Ys, dim=0)   # [N, d]
    return X, Y, Toks


class LinearHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=True)
    def forward(self, x):
        return self.lin(x)


class MLPHead(nn.Module):
    def __init__(self, d, d_hidden=None):
        super().__init__()
        if d_hidden is None:
            d_hidden = 2 * d
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d),
        )
    def forward(self, x):
        return x + self.net(x)   # residual: predict a delta on top of input


def train_head(head, X, Y, device, epochs=80, lr=1e-3, batch=1024, val_frac=0.15):
    n = X.shape[0]
    idx = torch.randperm(n)
    n_val = int(val_frac * n)
    X_val, Y_val = X[idx[:n_val]].to(device), Y[idx[:n_val]].to(device)
    X_tr, Y_tr = X[idx[n_val:]].to(device), Y[idx[n_val:]].to(device)
    head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n_tr = X_tr.shape[0]
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        perm = torch.randperm(n_tr)
        X_tr_p, Y_tr_p = X_tr[perm], Y_tr[perm]
        total = 0.0
        for start in range(0, n_tr, batch):
            xb = X_tr_p[start:start+batch]
            yb = Y_tr_p[start:start+batch]
            pred = head(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.shape[0]
        tr_mse = total / n_tr
        with torch.no_grad():
            val_mse = F.mse_loss(head(X_val), Y_val).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d}:  train MSE={tr_mse:.5f}  val MSE={val_mse:.5f}")
    head.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def baseline_nll(model, tokenizer, passages, device, max_length=512):
    total_nll = 0.0; total_toks = 0
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
    return total_nll / max(1, total_toks), total_toks


@torch.no_grad()
def head_nll(model, tokenizer, passages, l_throat, head, device, max_length=512):
    """Run model forward, capture h_throat via hook, apply head, push
       through model.lm_head, compute NLL."""
    captured = {"h": None}
    def cap_pre(module, inputs):
        # Called as pre-hook on layers[l_throat]: inputs[0] is h_throat
        captured["h"] = inputs[0]
        return None
    # The target module is layers[l_throat] (its input is hidden_states[l_throat])
    handle = model.model.layers[l_throat].register_forward_pre_hook(cap_pre)

    total_nll = 0.0; total_toks = 0
    try:
        for sent in passages:
            enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc.input_ids.to(device)
            mask = enc.attention_mask[0].bool().to(device)
            captured["h"] = None
            try:
                _ = model(ids, use_cache=False)
            except Exception:
                pass
            if captured["h"] is None:
                continue
            h_throat = captured["h"][0].float()  # [seq, d]
            # Apply head → predicted h_final (normed)
            h_pred = head(h_throat)
            logits = model.lm_head(h_pred.to(model.lm_head.weight.dtype)).float()  # [seq, V]
            shift_logits = logits[:-1]
            shift_labels = ids[0, 1:]
            shift_mask   = mask[1:]
            logp = F.log_softmax(shift_logits, dim=-1)
            nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            nll = nll[shift_mask]
            total_nll += nll.sum().item()
            total_toks += int(shift_mask.sum())
    finally:
        handle.remove()
    return total_nll / max(1, total_toks), total_toks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage125_throat_head.json")
    p.add_argument("--device", default=None)
    p.add_argument("--throat-frac", type=float, default=0.50)
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\nloading {args.model}...")
    model = load_model(args.model, device)
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    l_throat = max(1, min(L - 1, int(args.throat_frac * L)))
    l_final = L  # hidden_states[L] is output of last decoder layer, before final norm
    print(f"L={L}  d={d}  throat layer (hidden_states idx): {l_throat}")
    print(f"final hidden idx (pre-final-norm): {l_final}")

    # Collect training pairs
    print(f"\ncollecting training pairs from {len(TRAIN_SENTS)} sentences...")
    t0 = time.time()
    X_tr, Y_tr, _ = collect_state_pairs(model, tokenizer, TRAIN_SENTS, l_throat, l_final, device)
    print(f"  X shape {X_tr.shape}  Y shape {Y_tr.shape}  in {time.time()-t0:.0f}s")

    # Baseline NLL
    print(f"\nbaseline full-model NLL:")
    nll_base, n_toks = baseline_nll(model, tokenizer, TEST_SENTS, device)
    print(f"  NLL={nll_base:.4f}  PPL={np.exp(nll_base):.2f}  toks={n_toks}")

    results = {"baseline_nll": nll_base, "baseline_ppl": float(np.exp(nll_base)),
               "throat_layer": l_throat, "d": d, "n_train": X_tr.shape[0],
               "heads": {}}

    # (A) Linear head
    print(f"\n=== head A: linear (1024 → 1024) ===")
    head_A = LinearHead(d).to(device)
    n_params = sum(p.numel() for p in head_A.parameters())
    print(f"  params: {n_params:,}")
    best_val = train_head(head_A, X_tr, Y_tr, device, epochs=args.epochs, lr=1e-3)
    nll_A, _ = head_nll(model, tokenizer, TEST_SENTS, l_throat, head_A, device)
    print(f"  eval: NLL={nll_A:.4f}  PPL={np.exp(nll_A):.2f}  Δ={nll_A - nll_base:+.4f}")
    results["heads"]["linear"] = {
        "nll": nll_A, "ppl": float(np.exp(nll_A)), "delta_nll": nll_A - nll_base,
        "val_mse": best_val, "n_params": n_params,
    }

    # (B) MLP head
    print(f"\n=== head B: residual MLP (1024 → 2048 → 1024) ===")
    head_B = MLPHead(d, d_hidden=2*d).to(device)
    n_params = sum(p.numel() for p in head_B.parameters())
    print(f"  params: {n_params:,}")
    best_val = train_head(head_B, X_tr, Y_tr, device, epochs=args.epochs, lr=1e-3)
    nll_B, _ = head_nll(model, tokenizer, TEST_SENTS, l_throat, head_B, device)
    print(f"  eval: NLL={nll_B:.4f}  PPL={np.exp(nll_B):.2f}  Δ={nll_B - nll_base:+.4f}")
    results["heads"]["mlp"] = {
        "nll": nll_B, "ppl": float(np.exp(nll_B)), "delta_nll": nll_B - nll_base,
        "val_mse": best_val, "n_params": n_params,
    }

    # Summary
    print(f"\n=== summary ===")
    print(f"  baseline (full mouth 2, ~{28-l_throat} layers + LM head): "
          f"NLL={nll_base:.4f}  PPL={np.exp(nll_base):.2f}")
    for name, r in results["heads"].items():
        verdict = "SUPER UNLOCK" if r["delta_nll"] < 0.3 else \
                  "usable for speculative" if r["delta_nll"] < 1.5 else \
                  "too lossy"
        print(f"  {name:10s} ({r['n_params']:>12,} params)  "
              f"NLL={r['nll']:.4f}  Δ={r['delta_nll']:+.4f}  [{verdict}]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
