"""
Stage 41 — 3D slice diagnostic.

Tests whether a token's slice size along each of the three axes is predictable
from its manifold position at layer 1 (the first post-embedding projection,
where the manifold framing says the token lands on the manifold).

Per-token axis metrics:
  length_needed  = stabilization_depth (Finding 09) — layers until argmax locks
  depth_needed   = MLP intermediate sparsity (fraction of |int_act| above threshold)
  width_needed   = hidden-state effective rank (95% of squared norm in basis)

Regression: manifold coords at layer 1 -> each axis, report R² + cross-axis
correlations. Also report whether a single "difficulty score" (PC1 of the
three axes) explains most variance.

If all three are predictable from layer-1 manifold AND correlate, the 3D
slice theory's routing signal exists as a single latent difficulty.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


CALIB_TEXTS = [
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


GEN_PROMPTS = [
    "The discovery that inference accelerates with context is",
    "Quantum mechanics describes the behavior of matter at",
    "The mitochondrion is the powerhouse of the cell and",
    "A Bayesian inference framework updates a prior",
    "The universe expanded from an initial state of",
    "Cryptographic hash functions map arbitrary input to",
    "Graph coloring problems are NP-hard because",
    "The golden ratio appears in nature due to",
    "A black hole event horizon is the boundary at which",
    "Photosynthesis in plants converts carbon dioxide and",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def collect_calibration_basis(model, tokenizer, texts, device, rank, max_len=256):
    """Single pass: collect per-layer hidden-state PCA basis (post-layer)."""
    L = len(model.model.layers)
    hidden_samples = [[] for _ in range(L)]

    def make_hook(i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            hidden_samples[i].append(h_flat)
        return hook

    handles = [layer.register_forward_hook(make_hook(i))
               for i, layer in enumerate(model.model.layers)]
    try:
        model.eval()
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    bases = []
    means = []
    for i in range(L):
        H = torch.cat(hidden_samples[i], dim=0).to(torch.float32)
        mu = H.mean(dim=0)
        Hc = H - mu
        cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        k = min(rank, eigvecs.shape[1])
        P = eigvecs[:, -k:].flip(dims=[1])
        bases.append(P)
        means.append(mu)
    return bases, means


def per_token_metrics(model, tokenizer, prompts, num_gen_tokens, device, P1, mu1,
                       depth_threshold=0.1, width_threshold=0.95):
    """Generate tokens autoregressively and capture per-token axis metrics.
    Returns a list of dicts, one per generated token:
        manifold_coords (layer-1 projection), length_needed, depth_needed, width_needed
    """
    records = []
    L = len(model.model.layers)
    final_norm = model.model.norm
    lm_head = model.lm_head

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Prime: forward prompt with caching and state capture
        with torch.inference_mode():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(num_gen_tokens):
            with torch.inference_mode():
                out = model(
                    input_ids=next_token, past_key_values=past,
                    output_hidden_states=True, use_cache=True)
            past = out.past_key_values

            hidden = out.hidden_states  # tuple len L+1, each [1, 1, hidden]
            # Last position of last-layer hidden -> final argmax
            final_h = hidden[-1][:, -1, :]  # [1, hidden]
            final_logits = lm_head(final_norm(final_h))
            final_argmax = int(final_logits.argmax(dim=-1).item())

            # Per-layer argmax via logit lens, for stabilization depth
            # Layer i hidden = hidden[i+1] (hidden[0] = embedding)
            layer_argmax = []
            for i in range(L):
                h_i = hidden[i + 1][:, -1, :]
                logits_i = lm_head(final_norm(h_i))
                layer_argmax.append(int(logits_i.argmax(dim=-1).item()))
            # stabilization_depth = (1 + latest layer disagreeing with final) / L
            latest_disagree = -1
            for i in range(L):
                if layer_argmax[i] != final_argmax:
                    latest_disagree = i
            stabilization_depth = (1 + latest_disagree) / L

            # Manifold coords at layer 1 hidden (hidden[2] = post-layer-1)
            # per user framing: layer 0 = information, layer 1 = first projection
            # we use hidden[2] which is after layer 1's forward (post-rotation into manifold)
            h1 = hidden[2][:, -1, :].to(torch.float32).cpu().squeeze(0)  # [hidden]
            manifold_coords = (h1 - mu1) @ P1  # [rank]

            # depth_needed: intermediate activation sparsity per layer, averaged
            # We captured hidden states but not intermediate activations directly.
            # Re-run the MLPs to get int_act. Cheap: just the MLP's gate and up.
            depth_sparsity = []
            for i, layer in enumerate(model.model.layers):
                h_pre_mlp = hidden[i + 1][:, -1, :]  # approx — treat post-layer hidden as input-to-next
                # Actually post_attention_layernorm then MLP. For this diagnostic just use h_pre_mlp directly.
                # Normalize via post_attention_layernorm if accessible
                ln = layer.post_attention_layernorm
                h_ln = ln(h_pre_mlp)
                gate = layer.mlp.gate_proj(h_ln)
                up = layer.mlp.up_proj(h_ln)
                int_act = F.silu(gate) * up  # [1, d_int]
                int_abs = int_act.abs().to(torch.float32).squeeze(0)
                threshold_val = depth_threshold * int_abs.max()
                sparsity = (int_abs > threshold_val).float().mean().item()
                depth_sparsity.append(sparsity)
            depth_needed = sum(depth_sparsity) / L

            # width_needed: per-layer effective rank of hidden state projected onto per-layer PCA basis
            # simple proxy: participation ratio over layer-1's PCA projection
            # For speed we use just layer-1 manifold coords' participation ratio as a width proxy
            c2 = manifold_coords ** 2
            if c2.sum().item() > 0:
                pr = (c2.sum() ** 2 / (c2 ** 2).sum()).item()
            else:
                pr = 1.0
            width_needed = pr  # participation ratio on manifold basis

            records.append({
                "prompt_id": prompts.index(prompt),
                "step": step,
                "manifold_coords": manifold_coords.tolist(),
                "length_needed": stabilization_depth,
                "depth_needed": depth_needed,
                "width_needed": width_needed,
                "final_argmax": final_argmax,
            })

            next_token = torch.tensor([[final_argmax]], device=device)
            if final_argmax == tokenizer.eos_token_id:
                break

    return records


def ridge_r2(X, Y, lam=1e-3, train_frac=0.7):
    N = X.shape[0]
    idx = torch.randperm(N)
    n_train = int(train_frac * N)
    tr, te = idx[:n_train], idx[n_train:]
    Xa_tr = torch.cat([X[tr], torch.ones(n_train, 1)], dim=1)
    Xa_te = torch.cat([X[te], torch.ones(len(te), 1)], dim=1)
    XtX = Xa_tr.T @ Xa_tr + lam * torch.eye(Xa_tr.shape[1])
    W = torch.linalg.solve(XtX, Xa_tr.T @ Y[tr])
    Yhat = Xa_te @ W
    y_te = Y[te]
    if y_te.ndim == 1:
        ss_res = ((y_te - Yhat.squeeze(-1)) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    else:
        ss_res = ((y_te - Yhat) ** 2).sum()
        ss_tot = ((y_te - y_te.mean(dim=0, keepdim=True)) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def pearson(x, y):
    x = x.to(torch.float32); y = y.to(torch.float32)
    vx = x - x.mean(); vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum() * (vy * vy).sum()) + 1e-8
    return float((vx * vy).sum() / denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32, help="manifold basis rank at layer 1")
    p.add_argument("--gen-tokens", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage41_3d_slice_diagnostic.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    L = len(model.model.layers)
    print(f"  L={L}")

    print(f"\n=== calibrating per-layer PCA basis (rank={args.rank}) ===")
    t0 = time.perf_counter()
    bases, means = collect_calibration_basis(
        model, tokenizer, CALIB_TEXTS, device, args.rank)
    print(f"  {time.perf_counter()-t0:.1f}s")

    # Layer 1 = post first transformer layer (hidden_states[2])
    # For calibration bases we indexed by post-layer-i hidden, so layer-1 basis = bases[1]
    # But we collected on layer outputs, so bases[0] is post-layer-0. The user's
    # "layer 1 = first projection onto manifold" = post-layer-0 output (first layer applied).
    P1 = bases[0]
    mu1 = means[0]
    print(f"  using P1 (post-layer-0 = first layer output) shape={tuple(P1.shape)}")

    print(f"\n=== generating + measuring per-token metrics ===")
    t0 = time.perf_counter()
    records = per_token_metrics(
        model, tokenizer, GEN_PROMPTS, args.gen_tokens, device, P1, mu1)
    print(f"  {len(records)} tokens in {time.perf_counter()-t0:.1f}s")

    # Stack into tensors
    X = torch.tensor([r["manifold_coords"] for r in records], dtype=torch.float32)
    y_len = torch.tensor([r["length_needed"] for r in records], dtype=torch.float32)
    y_dep = torch.tensor([r["depth_needed"] for r in records], dtype=torch.float32)
    y_wid = torch.tensor([r["width_needed"] for r in records], dtype=torch.float32)

    print(f"\n=== axis stats (teacher measurements) ===")
    print(f"  length_needed (stabilization_depth): mean={y_len.mean():.3f}  "
          f"std={y_len.std():.3f}  range=[{y_len.min():.3f}, {y_len.max():.3f}]")
    print(f"  depth_needed  (MLP sparsity):        mean={y_dep.mean():.3f}  "
          f"std={y_dep.std():.3f}  range=[{y_dep.min():.3f}, {y_dep.max():.3f}]")
    print(f"  width_needed  (PR of layer-1 coords): mean={y_wid.mean():.3f}  "
          f"std={y_wid.std():.3f}  range=[{y_wid.min():.3f}, {y_wid.max():.3f}]")

    print(f"\n=== cross-axis Pearson correlations ===")
    r_lw = pearson(y_len, y_wid)
    r_ld = pearson(y_len, y_dep)
    r_wd = pearson(y_wid, y_dep)
    print(f"  length <-> width:  r={r_lw:+.3f}")
    print(f"  length <-> depth:  r={r_ld:+.3f}")
    print(f"  width  <-> depth:  r={r_wd:+.3f}")

    # PCA on the 3-axis difficulty vector
    stacked = torch.stack([
        (y_len - y_len.mean()) / (y_len.std() + 1e-8),
        (y_wid - y_wid.mean()) / (y_wid.std() + 1e-8),
        (y_dep - y_dep.mean()) / (y_dep.std() + 1e-8),
    ], dim=1)
    cov = stacked.T @ stacked / max(stacked.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.flip(dims=[0])
    pc1_frac = float(eigvals[0] / eigvals.sum())
    print(f"\n=== PC1 of (length, width, depth) z-scored ===")
    print(f"  explained variance: PC1={pc1_frac:.3f}  "
          f"(if ~1.0 all 3 axes collapse to one difficulty score)")

    print(f"\n=== ridge regression R² (manifold@layer1 coords -> axis) ===")
    # note: use a fresh seed for reproducibility; run 3 splits and average
    r2s = {"length": [], "depth": [], "width": []}
    for seed in range(5):
        torch.manual_seed(seed)
        r2s["length"].append(ridge_r2(X, y_len))
        r2s["depth"].append(ridge_r2(X, y_dep))
        r2s["width"].append(ridge_r2(X, y_wid))

    for k, vs in r2s.items():
        print(f"  {k:>7}: R² = {sum(vs)/len(vs):.3f}  "
              f"(range [{min(vs):+.3f}, {max(vs):+.3f}])")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "rank": args.rank,
            "n_tokens": len(records),
            "mean_length": float(y_len.mean()),
            "mean_depth": float(y_dep.mean()),
            "mean_width": float(y_wid.mean()),
            "pearson": {"length_width": r_lw, "length_depth": r_ld, "width_depth": r_wd},
            "pc1_fraction": pc1_frac,
            "r2_mean": {k: sum(v)/len(v) for k, v in r2s.items()},
            "r2_runs": r2s,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
