"""
Stage 37 — Parallel-layer decomposition test.

Under the "layer = rotation-view, not step" framing, each layer's residual
delta should depend only weakly on the accumulated state of prior layers.
If this is right, the transformer can be rewritten as L parallel functions
of x_0 (plus basis rotations), collapsing serial depth to parallel breadth.

Two measurements:

(1) Delta-similarity curve. For each layer i, compare:
      - Δ_i(actual) = f_i(h_{i-1}) - h_{i-1}   (sequential, from real forward)
      - Δ_i(from j) = f_i(h_j) - h_j            (hypothetical, from earlier state)
    for j in {0, i-8, i-4, i-2, i-1}. Report mean cosine similarity per
    (layer, gap). If Δ_i(from j=0) ≈ Δ_i(actual) across layers, layers are
    strongly parallel. Curve vs gap tells us the memory depth needed.

(2) Parallel-forward simulation. Compute all Δ_i from h_0, sum them onto h_0,
    apply final norm + lm_head, measure top-1 token match vs the real forward.
    If match is high, the parallel model is a viable drop-in.

Protocol on Qwen3-0.6B across a handful of prompts (we use the teacher-input
positions of the prompt, not generation, for clean aggregation).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


PROMPTS = [
    "The discovery that inference accelerates with context is",
    "Proteins fold into complex three-dimensional structures determined by their",
    "Quantum mechanics describes the behavior of matter and energy at",
    "Linear algebra provides the mathematical foundation for many",
    "Evolution by natural selection operates on heritable variation in",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def forward_with_hidden(model, input_ids):
    with torch.inference_mode():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple of (L+1) tensors, each [1, seq, hidden]
    return out.hidden_states, out.logits


def run_layer(layer, hidden, position_ids, position_embeddings, attention_mask=None):
    """Run a single decoder layer on provided hidden state and return output."""
    with torch.inference_mode():
        out = layer(
            hidden_states=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    if isinstance(out, tuple):
        return out[0]
    return out


def cosine_mean(a, b):
    """Mean cosine similarity across positions. a,b shape: [1, seq, hidden]"""
    a = a.reshape(-1, a.shape[-1]).to(torch.float32)
    b = b.reshape(-1, b.shape[-1]).to(torch.float32)
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    dot = (a * b).sum(dim=-1)
    cos = dot / (na * nb + 1e-8)
    mask = (na > 1e-6) & (nb > 1e-6)
    if mask.sum() == 0:
        return 0.0
    return float(cos[mask].mean().cpu())


def relative_mse(actual, approx):
    """|| actual - approx ||^2 / || actual ||^2 (position-averaged)."""
    a = actual.reshape(-1, actual.shape[-1]).to(torch.float32)
    b = approx.reshape(-1, approx.shape[-1]).to(torch.float32)
    num = ((a - b) ** 2).sum(dim=-1)
    den = (a ** 2).sum(dim=-1)
    mask = den > 1e-6
    if mask.sum() == 0:
        return float("inf")
    return float((num[mask] / den[mask]).mean().cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gaps", default="0,2,4,8,16",
                   help="j-values: feed h_{i-gap} to layer i (0 = h_0, 1 = prev)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage37_parallel_layers.json")
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
    print(f"  L = {L}")

    gap_choices = [int(x) for x in args.gaps.split(",")]

    cos_sum  = {g: [0.0] * L for g in gap_choices}
    rmse_sum = {g: [0.0] * L for g in gap_choices}
    count    = {g: [0]   * L for g in gap_choices}

    parallel_match_totals = {"top1_match": 0, "total": 0}

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n=== prompt {prompt_idx+1}/{len(PROMPTS)}: {prompt[:50]}... ===")
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # (1) sequential forward with all hidden states captured
        hidden, logits = forward_with_hidden(model, ids)
        # compute rotary embeddings once at h_0 (they depend only on position_ids + dtype/device)
        with torch.inference_mode():
            position_embeddings = model.model.rotary_emb(hidden[0], position_ids)
        # hidden has L+1 entries: h_0 is embedding output, h_i is after layer i
        h = hidden  # tuple len L+1

        for i in range(1, L + 1):
            actual_delta = h[i] - h[i-1]
            for g in gap_choices:
                if g == 0:
                    j = 0
                else:
                    j = max(0, i - 1 - g)
                if j == i - 1:
                    # trivially equal: feeding h_{i-1} to layer_i IS the sequential delta
                    cos_sum[g][i-1] += 1.0
                    rmse_sum[g][i-1] += 0.0
                    count[g][i-1] += 1
                    continue
                out = run_layer(model.model.layers[i-1], h[j], position_ids, position_embeddings)
                approx_delta = out - h[j]
                cs = cosine_mean(actual_delta, approx_delta)
                rm = relative_mse(actual_delta, approx_delta)
                cos_sum[g][i-1] += cs
                rmse_sum[g][i-1] += rm
                count[g][i-1] += 1

        # (2) parallel-forward simulation from h_0
        with torch.inference_mode():
            h_parallel = h[0].clone()
            for i in range(1, L + 1):
                out_i = run_layer(model.model.layers[i-1], h[0], position_ids, position_embeddings)
                delta_i = out_i - h[0]
                h_parallel = h_parallel + delta_i
            # final norm
            if hasattr(model.model, "norm"):
                h_final = model.model.norm(h_parallel)
            else:
                h_final = h_parallel
            parallel_logits = model.lm_head(h_final)

        # Top-1 match at every position
        teacher_top1 = logits.argmax(dim=-1)[0]          # [seq]
        parallel_top1 = parallel_logits.argmax(dim=-1)[0]
        matches = (teacher_top1 == parallel_top1).sum().item()
        total = teacher_top1.shape[0]
        parallel_match_totals["top1_match"] += matches
        parallel_match_totals["total"] += total
        print(f"  parallel-forward top-1 match: {matches}/{total} = {matches/total:.2%}")

    cos_mean = {g: [cos_sum[g][i] / max(count[g][i], 1) for i in range(L)] for g in gap_choices}
    rmse_mean = {g: [rmse_sum[g][i] / max(count[g][i], 1) for i in range(L)] for g in gap_choices}

    print(f"\n=== per-layer cosine similarity (actual delta vs delta from h_{{i-1-gap}}) ===")
    print(f"  gap=0 means feed h_0; gap=∞ means sequential (equals actual)")
    header = "  " + "layer".rjust(5) + "  " + "  ".join(f"g={g:>2}".rjust(7) for g in gap_choices)
    print(header)
    for i in range(0, L, max(1, L // 14)):
        row = f"  {i+1:>5}  " + "  ".join(f"{cos_mean[g][i]:>7.3f}" for g in gap_choices)
        print(row)

    overall_cos = {g: sum(cos_mean[g]) / L for g in gap_choices}
    overall_rmse = {g: sum(rmse_mean[g]) / L for g in gap_choices}

    print(f"\n=== overall mean cosine vs sequential delta ===")
    for g in gap_choices:
        print(f"  gap={g:>2}  mean_cos={overall_cos[g]:.3f}  mean_rel_mse={overall_rmse[g]:.3f}")

    total_match = parallel_match_totals["top1_match"]
    total_tokens = parallel_match_totals["total"]
    print(f"\n=== parallel-forward simulation ===")
    print(f"  {total_match}/{total_tokens} top-1 tokens match teacher ({total_match/total_tokens:.2%})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "gaps": gap_choices,
            "per_layer_cos": cos_mean,
            "per_layer_rmse": rmse_mean,
            "overall_cos": overall_cos,
            "overall_rmse": overall_rmse,
            "parallel_top1_match": total_match,
            "parallel_top1_total": total_tokens,
            "parallel_top1_ratio": total_match / total_tokens if total_tokens else 0.0,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
