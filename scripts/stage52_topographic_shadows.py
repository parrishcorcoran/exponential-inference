"""
Stage 52 — Test theory #5: entropy profiles are topographic shadows of
geodesic paths through the manifold landscape.

Claim: Finding 06's four canonical entropy descent profiles during
generation (monotone decline, bell curve, plateau, mid-gen spike) are
projections of different path geometries on the manifold's topography.

  - Monotone = smooth downhill traversal
  - Bell = single saddle crossing
  - Plateau = valley-floor / ridge-top walk
  - Spike = cliff climb

Test: for each generation, compute per-token
  1. output_entropy (the observed Finding 06 signal)
  2. stabilization_depth (Finding 09's geodesic-length proxy)
  3. step_norm = ||h_final(t) - h_final(t-1)|| (path speed)
  4. residual_magnitude = ||Δh|| summed across the layer stack (path length)

If the SHAPE of (1) across tokens matches the SHAPE of (2)/(3)/(4),
then output_entropy is a shadow of the same underlying topography the
geometric features are sampling. Supports Theory #5.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Diverse prompts intended to elicit different profile types.
PROMPTS = [
    "The discovery that inference accelerates with context is",
    "To solve a quadratic equation of the form ax^2 + bx + c = 0,",
    "A first-order logic proof requires",
    "The mitochondrion is the powerhouse of the cell and",
    "Black holes form when",
    "If a train leaves station A at 3 PM traveling 60 mph and another train",
    "Photosynthesis in plants converts",
    "The causes of the French Revolution include",
    "In graph theory, a Hamiltonian cycle is",
    "DNA encodes genetic information by",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def generate_with_features(model, tokenizer, prompt, max_new_tokens, device):
    """Autoregressive generation capturing per-token geometric features."""
    L = len(model.model.layers)
    final_norm = model.model.norm
    lm_head = model.lm_head

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=True)
    past = out.past_key_values
    prev_final_h = out.hidden_states[-1][:, -1, :].to(torch.float32).cpu().squeeze(0)
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    features = {
        "tokens": [],
        "output_entropy": [],
        "stabilization_depth": [],
        "step_norm": [],
        "residual_path_length": [],
    }

    for step in range(max_new_tokens):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past,
                        output_hidden_states=True, use_cache=True)
        past = out.past_key_values
        hidden = out.hidden_states  # tuple len L+1
        final_h = hidden[-1][:, -1, :]
        final_logits = lm_head(final_norm(final_h))[0]

        # 1. output entropy
        probs = F.softmax(final_logits.float(), dim=-1)
        ent = float(-(probs.clamp_min(1e-12).log() * probs).sum().item())
        features["output_entropy"].append(ent)

        # 2. stabilization_depth (Finding 09)
        final_argmax = int(final_logits.argmax(dim=-1).item())
        latest_disagree = -1
        for i in range(L):
            h_i = hidden[i + 1][:, -1, :]
            logits_i = lm_head(final_norm(h_i))[0]
            if int(logits_i.argmax(dim=-1).item()) != final_argmax:
                latest_disagree = i
        stab = (1 + latest_disagree) / L
        features["stabilization_depth"].append(stab)

        # 3. step_norm: how far the final-state moved since previous token
        cur_final = final_h.to(torch.float32).cpu().squeeze(0)
        step = float((cur_final - prev_final_h).norm().item())
        features["step_norm"].append(step)
        prev_final_h = cur_final

        # 4. residual_path_length: sum of ||h_i - h_{i-1}|| over layers (path length through stack)
        path_len = 0.0
        for i in range(1, L + 1):
            h_cur = hidden[i][:, -1, :].to(torch.float32).cpu().squeeze(0)
            h_prev = hidden[i - 1][:, -1, :].to(torch.float32).cpu().squeeze(0)
            path_len += float((h_cur - h_prev).norm().item())
        features["residual_path_length"].append(path_len)

        features["tokens"].append(final_argmax)
        next_token = torch.tensor([[final_argmax]], device=device)
        if final_argmax == tokenizer.eos_token_id:
            break

    return features


def classify_profile(series):
    """Classify a 1D sequence into one of {monotone_decline, bell, plateau, spike}.
    Heuristic classifier.
    """
    x = torch.tensor(series, dtype=torch.float32)
    n = len(x)
    if n < 5:
        return "short"
    xmin, xmax = float(x.min()), float(x.max())
    xrange = xmax - xmin
    if xrange < 1e-6:
        return "plateau"
    # Normalize to [0, 1]
    xn = (x - xmin) / xrange
    # Find peak (max) location
    peak_idx = int(xn.argmax())
    trough_idx = int(xn.argmin())
    # Monotone decline: values mostly decreasing
    diffs = xn[1:] - xn[:-1]
    n_dec = int((diffs < -0.02).sum().item())
    n_inc = int((diffs > 0.02).sum().item())
    # Flatness
    std = float(xn.std())
    if std < 0.12:
        return "plateau"
    # Pure monotone decline (relative to range)
    if n_dec > 2 * n_inc and peak_idx < n // 3:
        return "monotone_decline"
    # Mid-gen spike: peak in middle and narrow
    if 0.2 * n < peak_idx < 0.8 * n and xn[peak_idx] > 0.85:
        # Narrow peak check: drop-off
        width = 1
        for j in range(peak_idx + 1, n):
            if xn[j] < 0.5 * xn[peak_idx]:
                width = j - peak_idx
                break
        if width < n // 4:
            return "spike"
    # Bell: peak in middle, broad
    if 0.2 * n < peak_idx < 0.8 * n:
        return "bell"
    return "other"


def profile_similarity(a, b):
    """Pearson correlation between two sequences after z-score normalization."""
    xa = torch.tensor(a, dtype=torch.float32)
    xb = torch.tensor(b, dtype=torch.float32)
    if len(xa) != len(xb) or len(xa) < 3:
        return float("nan")
    xa = (xa - xa.mean()) / (xa.std().clamp_min(1e-8))
    xb = (xb - xb.mean()) / (xb.std().clamp_min(1e-8))
    return float((xa * xb).mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage52_topographic_shadows.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    print(f"  L={len(model.model.layers)}")

    print(f"\n=== generating + measuring features ===")
    all_records = []
    for i, prompt in enumerate(PROMPTS):
        t0 = time.perf_counter()
        f = generate_with_features(model, tokenizer, prompt,
                                    args.max_new_tokens, device)
        dt = time.perf_counter() - t0
        print(f"  [{i+1}/{len(PROMPTS)}] {len(f['tokens'])} tokens in {dt:.1f}s "
              f"— {prompt[:50]}")
        all_records.append({"prompt": prompt, "features": f})

    # Classify entropy profile + measure cross-feature correlation per prompt
    print(f"\n=== per-prompt profile classification + feature correlations ===")
    print(f"  {'#':>2}  {'entropy type':>16}  "
          f"{'H~stab':>8}  {'H~step':>8}  {'H~path':>8}  prompt")

    rows = []
    for i, rec in enumerate(all_records):
        f = rec["features"]
        etype = classify_profile(f["output_entropy"])
        r_stab = profile_similarity(f["output_entropy"], f["stabilization_depth"])
        r_step = profile_similarity(f["output_entropy"], f["step_norm"])
        r_path = profile_similarity(f["output_entropy"], f["residual_path_length"])
        print(f"  {i+1:>2}  {etype:>16}  "
              f"{r_stab:>+8.3f}  {r_step:>+8.3f}  {r_path:>+8.3f}  {rec['prompt'][:50]}")
        rows.append({
            "prompt": rec["prompt"],
            "entropy_profile_type": etype,
            "r_entropy_vs_stabilization": r_stab,
            "r_entropy_vs_stepnorm": r_step,
            "r_entropy_vs_pathlen": r_path,
            "entropy_series": f["output_entropy"],
            "stabilization_series": f["stabilization_depth"],
            "stepnorm_series": f["step_norm"],
            "pathlen_series": f["residual_path_length"],
        })

    # Global stats
    r_stab_all = [r["r_entropy_vs_stabilization"] for r in rows if not math.isnan(r["r_entropy_vs_stabilization"])]
    r_step_all = [r["r_entropy_vs_stepnorm"] for r in rows if not math.isnan(r["r_entropy_vs_stepnorm"])]
    r_path_all = [r["r_entropy_vs_pathlen"] for r in rows if not math.isnan(r["r_entropy_vs_pathlen"])]

    print(f"\n=== aggregate across {len(rows)} prompts ===")
    print(f"  entropy ↔ stabilization_depth: mean r = {sum(r_stab_all)/len(r_stab_all):+.3f}  "
          f"(range [{min(r_stab_all):+.3f}, {max(r_stab_all):+.3f}])")
    print(f"  entropy ↔ step_norm:          mean r = {sum(r_step_all)/len(r_step_all):+.3f}  "
          f"(range [{min(r_step_all):+.3f}, {max(r_step_all):+.3f}])")
    print(f"  entropy ↔ path_length:        mean r = {sum(r_path_all)/len(r_path_all):+.3f}  "
          f"(range [{min(r_path_all):+.3f}, {max(r_path_all):+.3f}])")

    # Verdict
    print(f"\n=== verdict ===")
    strong = max(abs(sum(r_stab_all)/len(r_stab_all)),
                 abs(sum(r_step_all)/len(r_step_all)),
                 abs(sum(r_path_all)/len(r_path_all)))
    if strong > 0.5:
        print(f"  STRONG shape correlation: entropy profile tracks a geometric path feature.")
        print(f"  Supports Theory #5 — entropy profiles are topographic shadows.")
    elif strong > 0.3:
        print(f"  MODERATE correlation — some path feature tracks entropy but not dominantly.")
    else:
        print(f"  WEAK correlation — entropy is largely independent of these path features.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_prompts": len(rows),
            "max_new_tokens": args.max_new_tokens,
            "rows": rows,
            "aggregate": {
                "r_entropy_stab_mean": sum(r_stab_all) / len(r_stab_all),
                "r_entropy_step_mean": sum(r_step_all) / len(r_step_all),
                "r_entropy_path_mean": sum(r_path_all) / len(r_path_all),
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
