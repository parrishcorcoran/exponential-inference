"""
Stage 116 — Position-aware KV rank compression on Qwen3-0.6B.

Strix found at 14B that middle layers IMPROVE with KV compression
(FREE_WIN territory). Stage 38 showed that UNIFORM KV rank reduction
on 0.6B post-hoc breaks argmax at every rank. But uniform mixes edges
and middle. Does KV compression in MIDDLE-ONLY on 0.6B help?

Test: post-hoc KV rank projection applied ONLY to layers L3-24 (middle
dead zone). Edge layers L0-2 and L25-27 keep full-rank KV.

Variants:
  0. Baseline (no compression)
  1. Mid-only KV rank 512 (sanity — half of d_kv=1024)
  2. Mid-only KV rank 256
  3. Mid-only KV rank 128
  4. Mid-only KV rank 64
  5. Mid-only KV rank 32
  6. Mid-only KV rank 16

Compare to:
  - Stage 38 UNIFORM KV (all broken at all ranks on 0.6B)
  - Strix 14B where middle KV compression was FREE_WIN

Prediction (from bathtub + scaling):
  - If bathtub holds at 0.6B: middle-only KV 256 or 128 should hold
    quality, possibly improve
  - If scaling matters: middle-only may still break because 0.6B is
    manifold-floor
  - The specific break rank for mid-only vs uniform tells us how much
    of stage 38's failure was "edges broke" vs "middle broke"
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
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
]


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    n = (len(tokens) - 1) // seq_len
    for i in range(min(max_batches, n)):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        ids = torch.tensor([window], dtype=torch.long, device=device)
        inp = ids[:, :-1]; tgt = ids[:, 1:]
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
    return total / max(count, 1)


def find_kv_projs_per_layer(model):
    """Return dict: layer_idx -> [(name, module)] for that layer's kv projections."""
    by_layer = {}
    for i, layer in enumerate(model.model.layers):
        by_layer[i] = [
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
        ]
    return by_layer


@torch.no_grad()
def collect_output_covariances(model, tokenizer, texts, device):
    """Hook every k_proj and v_proj's output, accumulate y^T y per (layer, name)."""
    covs = {}  # (layer, name) -> cov matrix
    handles = []
    for i, layer in enumerate(model.model.layers):
        for name, mod in [("k_proj", layer.self_attn.k_proj), ("v_proj", layer.self_attn.v_proj)]:
            key = (i, name)
            covs[key] = None
            def make_hook(k):
                def hook(mod, inputs, output):
                    y = output.detach().reshape(-1, output.shape[-1]).float().cpu()
                    if covs[k] is None:
                        covs[k] = torch.zeros(y.shape[1], y.shape[1])
                    covs[k] += y.T @ y
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))
    try:
        model.eval()
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()
    return covs


def top_k_projector(cov, k):
    """Compute top-k eigenvector projection P P^T (d x d, rank k)."""
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    P = eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)   # [d, k]
    return (P @ P.T).contiguous()


def install_mid_only_kv_projectors(model, covs, rank_mid, edge_width):
    """Install rank-k KV output projection on middle layers only.
       Edge layers (first `edge_width` and last `edge_width`) stay full-rank."""
    L = len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        if i < edge_width or i >= L - edge_width:
            continue   # skip edges
        for name, mod in [("k_proj", layer.self_attn.k_proj), ("v_proj", layer.self_attn.v_proj)]:
            ppt = top_k_projector(covs[(i, name)], rank_mid).to(mod.weight.dtype).to(mod.weight.device)
            def make_hook(projector):
                def hook(mod, inputs, output):
                    return output @ projector
                return hook
            handles.append(mod.register_forward_hook(make_hook(ppt)))
    return handles


def remove_hooks(handles):
    for h in handles: h.remove()


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--edge-width", type=int, default=3,
                   help="Edge layers on each side kept at full KV rank")
    p.add_argument("--ranks", default="1024,512,256,128,64,32,16,8",
                   help="Mid-only KV rank values to test")
    p.add_argument("--out", default="results/stage116_position_aware_kv_06b.json")
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
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 30, split="validation")

    # Teacher baseline
    print("loading model (teacher baseline)...", flush=True)
    model = load_fresh(args.model, device)
    L = model.config.num_hidden_layers
    d_kv = model.model.layers[0].self_attn.k_proj.out_features
    print(f"  L={L}  d_kv={d_kv}  edge_width={args.edge_width}", flush=True)
    print(f"  edges: L0..L{args.edge_width-1}  and L{L-args.edge_width}..L{L-1}", flush=True)
    print(f"  middle: L{args.edge_width}..L{L-args.edge_width-1}  ({L - 2*args.edge_width} layers)", flush=True)

    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}", flush=True)

    # Calibrate covariances ONCE on teacher
    print(f"\ncalibrating KV output covariances on {len(CALIB_TEXTS)} texts...", flush=True)
    covs = collect_output_covariances(model, tokenizer, CALIB_TEXTS, device)
    print(f"  {len(covs)} covariance matrices", flush=True)

    ranks = [int(x) for x in args.ranks.split(",")]
    tests = []
    for r in ranks:
        print(f"\n--- mid-only KV rank {r} (edges full) ---", flush=True)
        t0 = time.time()
        handles = install_mid_only_kv_projectors(model, covs, r, args.edge_width)
        try:
            ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
            ppl = math.exp(ce)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            ce = float('inf'); ppl = float('inf')
        finally:
            remove_hooks(handles)
        delta = ppl - teacher_ppl if math.isfinite(ppl) else float('inf')
        compression = d_kv / r
        bucket = ("FREE_WIN" if delta < -0.1 else
                  "free" if abs(delta) < 0.5 else
                  "cheap" if delta < 2 else
                  "moderate" if delta < 10 else
                  "expensive" if delta < 100 else
                  "broken")
        print(f"  {compression:.1f}× compression  val_ppl={ppl:.3f}  Δ={delta:+.3f}  [{bucket}]  ({time.time()-t0:.0f}s)",
              flush=True)
        tests.append({
            "rank": r, "compression": compression,
            "val_ce": ce, "val_ppl": ppl, "delta_ppl": delta, "cost": bucket,
        })

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"teacher val_ppl: {teacher_ppl:.3f}")
    print(f"edge width: {args.edge_width} each side → middle = {L - 2*args.edge_width}/{L} layers")
    print(f"\n{'rank':>5}  {'compression':>11}  {'val_ppl':>10}  {'delta':>8}  bucket")
    for t in tests:
        print(f"  {t['rank']:>5}  {t['compression']:>10.1f}×  {t['val_ppl']:>10.3f}  {t['delta_ppl']:>+8.3f}  {t['cost']}")

    # Compare to stage 38 (uniform KV compression on 0.6B broke at all ranks)
    print(f"\n=== comparison to stage 38 (UNIFORM KV, all layers) ===", flush=True)
    print(f"  Stage 38 uniform rank 512 (2×):   2/80 token match, first_div @ token 1")
    print(f"  Stage 38 uniform rank 128 (8×):   2/80 token match, first_div @ token 1")
    print(f"  Stage 38 uniform rank 16 (64×):   broken output")
    print(f"  (stage 38 used argmax match — different metric than val_ppl)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "L": L, "d_kv": d_kv,
                   "edge_width": args.edge_width,
                   "teacher_val_ce": teacher_ce, "teacher_val_ppl": teacher_ppl,
                   "tests": tests}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
