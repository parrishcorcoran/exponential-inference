"""
Stage 111 — Fractal test: does the bathtub shape appear at three scales?

Three measurements of participation ratio (PR) across layers:

  Scale 1 (bulk): PR of hidden states across MANY inputs and positions.
                  Already have this in Qwen_Qwen3-0.6B_manifold.json from
                  build_manifold_map.py. Load and plot.

  Scale 2 (weights): PR of singular values of each layer's weight
                     matrices. Per-layer, pick canonical weights
                     (q_proj, k_proj, v_proj, o_proj, gate, up, down).

  Scale 3 (atomic): PR of hidden states across token POSITIONS of
                    a SINGLE fixed input sequence.

If bathtub shows at all three → fractal / unified manifold.
If only one or two → hierarchical but not self-similar.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def participation_ratio(vec):
    """PR = (sum x)^2 / sum x^2. Normalized soft rank."""
    vec = vec.float()
    total = vec.sum()
    sqr = (vec * vec).sum()
    if sqr.item() < 1e-12:
        return 0.0
    return float((total * total) / sqr)


def pr_of_hidden(H):
    """PR of [seq, d] hidden state across positions — uses eigenvalue spectrum
       of H.T @ H as the variance distribution."""
    H = H.float()
    # H is [seq, d]; compute the singular values of H
    # Variance per direction = S^2 / N
    U, S, V = torch.linalg.svd(H, full_matrices=False)
    var = S * S  # variance per direction
    return participation_ratio(var)


def pr_of_weight(W):
    """PR of singular-value spectrum of a weight matrix — measures effective
       rank of the weight's variance distribution."""
    U, S, V = torch.linalg.svd(W.float(), full_matrices=False)
    return participation_ratio(S * S)


@torch.no_grad()
def scale_3_trajectory(model, tokenizer, text, device):
    """PR of hidden states at each layer for ONE input sequence."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    out = model(ids, use_cache=False, output_hidden_states=True)
    # hidden_states[l] is [1, seq, d]
    L_plus_1 = len(out.hidden_states)
    per_layer = []
    for l in range(L_plus_1):
        H = out.hidden_states[l][0]  # [seq, d]
        pr = pr_of_hidden(H)
        norm = float(H.norm(dim=-1).mean().item())
        per_layer.append({"layer_index": l, "pr": pr, "norm": norm})
    return per_layer


@torch.no_grad()
def scale_2_weights(model):
    """PR of weight singular values per layer, per matrix type."""
    per_layer = []
    for i, layer in enumerate(model.model.layers):
        row = {"layer_index": i, "matrices": {}}
        for name, module in [("q_proj", layer.self_attn.q_proj),
                             ("k_proj", layer.self_attn.k_proj),
                             ("v_proj", layer.self_attn.v_proj),
                             ("o_proj", layer.self_attn.o_proj),
                             ("gate_proj", layer.mlp.gate_proj),
                             ("up_proj", layer.mlp.up_proj),
                             ("down_proj", layer.mlp.down_proj)]:
            W = module.weight.data
            row["matrices"][name] = pr_of_weight(W)
        per_layer.append(row)
    return per_layer


def scale_1_from_manifold_json(path):
    """Load existing bulk manifold measurement."""
    if not Path(path).exists():
        return None
    d = json.load(open(path))
    # per_layer contains 'pr' for each layer of hidden state PR across bulk data
    return [{"layer_index": l["layer_index"], "pr": l["pr"]} for l in d["per_layer"]]


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--manifold-json", default="results/Qwen_Qwen3-0.6B_manifold.json")
    p.add_argument("--test-text", default=(
        "The discovery that inference accelerates with context is a significant "
        "finding in the field of cognitive psychology. This is because the context "
        "provides a rich and complex environment that allows for more sophisticated "
        "and nuanced understanding. The study aimed to investigate the nature of "
        "these cognitive processes by examining how subjects respond to varying "
        "levels of contextual information presented during inference tasks."
    ))
    p.add_argument("--out", default="results/stage111_fractal.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_fresh(args.model, device)
    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"model={args.model}  L={L}  d={d_model}", flush=True)

    # Scale 1 — load existing bulk manifold
    print("\n--- scale 1: bulk hidden-state manifold (from existing measurement) ---", flush=True)
    scale_1 = scale_1_from_manifold_json(args.manifold_json)
    if scale_1:
        print(f"  loaded {len(scale_1)} layers from {args.manifold_json}", flush=True)
        print(f"  {'layer':>5}  {'pr':>8}")
        for r in scale_1:
            print(f"  {r['layer_index']:>5}  {r['pr']:>8.2f}")
    else:
        print(f"  NOT FOUND — skipping", flush=True)

    # Scale 2 — weight SVD
    print("\n--- scale 2: per-layer weight PR ---", flush=True)
    t0 = time.time()
    scale_2 = scale_2_weights(model)
    print(f"  computed in {time.time()-t0:.0f}s", flush=True)
    print(f"  {'layer':>5}  {'q':>7}  {'k':>7}  {'v':>7}  {'o':>7}  {'gate':>7}  {'up':>7}  {'down':>7}")
    for r in scale_2:
        m = r["matrices"]
        print(f"  {r['layer_index']:>5}  {m['q_proj']:>7.2f}  {m['k_proj']:>7.2f}  "
              f"{m['v_proj']:>7.2f}  {m['o_proj']:>7.2f}  {m['gate_proj']:>7.2f}  "
              f"{m['up_proj']:>7.2f}  {m['down_proj']:>7.2f}")

    # Scale 3 — single sequence trajectory
    print("\n--- scale 3: single-sequence hidden-state PR per layer ---", flush=True)
    t0 = time.time()
    scale_3 = scale_3_trajectory(model, tokenizer, args.test_text, device)
    print(f"  computed in {time.time()-t0:.0f}s", flush=True)
    print(f"  {'layer':>5}  {'pr':>8}  {'||h||':>10}")
    for r in scale_3:
        print(f"  {r['layer_index']:>5}  {r['pr']:>8.2f}  {r['norm']:>10.2f}")

    # Summary — are the three curves similar?
    print(f"\n=== shape comparison ===")
    if scale_1:
        # Use v_proj as weight baseline (one of the KV-side matrices)
        s1_pr = np.array([r["pr"] for r in scale_1])
        s2_pr = np.array([r["matrices"]["v_proj"] for r in scale_2])
        s3_pr = np.array([r["pr"] for r in scale_3[1:]])  # drop embedding layer (0) for alignment

        # Align lengths
        min_len = min(len(s1_pr), len(s2_pr), len(s3_pr))
        a = s1_pr[:min_len]; b = s2_pr[:min_len]; c = s3_pr[:min_len]
        # Normalize each to [0, 1] for shape comparison
        def zscore(x):
            if x.std() < 1e-12: return x - x.mean()
            return (x - x.mean()) / x.std()
        za, zb, zc = zscore(a), zscore(b), zscore(c)
        # Pearson correlations
        corr_12 = float(np.corrcoef(za, zb)[0, 1])
        corr_13 = float(np.corrcoef(za, zc)[0, 1])
        corr_23 = float(np.corrcoef(zb, zc)[0, 1])
        print(f"  Pearson r (scale1 vs scale2, v_proj): {corr_12:+.3f}")
        print(f"  Pearson r (scale1 vs scale3):         {corr_13:+.3f}")
        print(f"  Pearson r (scale2 vs scale3):         {corr_23:+.3f}")
        if min(corr_12, corr_13, corr_23) > 0.5:
            print(f"  verdict: shapes CORRELATED at all three scales → fractal-like")
        elif max(corr_12, corr_13, corr_23) > 0.5:
            print(f"  verdict: some pairs correlated, some not → partial self-similarity")
        else:
            print(f"  verdict: shapes DISTINCT → not fractal")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "L": L, "d_model": d_model,
                   "test_text_chars": len(args.test_text),
                   "scale_1_bulk_pr": scale_1,
                   "scale_2_weight_pr": scale_2,
                   "scale_3_sequence_pr": scale_3}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
