"""
Z8G4 — Manifold measurement on large models.

Loads any HF causal LM, runs Stage-1-style measurement (PR, TwoNN, r50/
r90/r95/r99 per layer). Designed for models > 72B that only Z8G4 can
fit in RAM.

Streams forward passes one chunk at a time, accumulating hidden states
only for the layers we sample (to keep memory manageable even at huge
model scale). Per-layer covariance accumulated in fp32 on CPU, eigh run
at the end in fp64.

Usage:
    python machines/z8g4/scripts/measure_manifold_large.py \\
        --model meta-llama/Meta-Llama-3-70B \\
        --calib-tokens 4000 \\
        --out machines/z8g4/results/manifold_Llama-3-70B.json

Big models (70B+) take hours per forward pass on CPU. Start small and
scale up. Use nohup or tmux to survive disconnects.

Tuning:
    numactl -N 0 -m 0 python ...          # pin to one socket (small model)
    OMP_NUM_THREADS=<cores> python ...    # adjust thread count
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_CALIBRATION_TEXTS = [
    # Short, diverse paragraphs. Not critical — the manifold emerges from
    # any broad text distribution.
    "The cell is the basic structural unit of life. Every organism is composed of one or more cells, which are the smallest entities exhibiting the characteristics of life.",
    "Quantum mechanics describes the behaviour of matter and energy at atomic and subatomic scales. Particles exhibit wave-like properties, and observation can collapse superpositions of states.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes, transistors, integrated circuits, and the modern silicon processor, each step multiplying computational density.",
    "Climate change is driven primarily by greenhouse gas emissions from fossil fuel combustion, deforestation, and industrial agriculture. The accumulated warming has measurable consequences for oceans and ecosystems.",
    "Language models learn statistical structure from text corpora by optimizing a next-token prediction objective. The learned representations capture syntactic, semantic, and pragmatic regularities of the training distribution.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen. The process sustains nearly all life on Earth and underlies the long-term carbon cycle.",
    "Neural networks consist of layers of parameterised transformations trained by gradient descent. Depth and width both contribute to their capacity, and regularisation controls how that capacity is used.",
    "The Roman Empire expanded through a combination of military power, political institutions, and engineering. Its decline was driven by economic strain, political fragmentation, and external pressures.",
    "Relativity theory links space, time, matter, and energy through a small set of principles. Its predictions have been confirmed by precise measurements of gravitational waves, black holes, and cosmological expansion.",
    "Artificial intelligence as a field studies the design of agents that perceive, reason, learn, and act. Modern systems combine statistical learning with symbolic structures to solve tasks that were once considered definitional of intelligence.",
    "Cryptography protects information using mathematical operations that are easy to compute in one direction and hard to invert. Public-key schemes rely on problems like integer factorization or elliptic-curve discrete logarithm.",
    "Neurotransmitters mediate communication between neurons at chemical synapses. Dopamine, serotonin, glutamate, and GABA each play distinct roles and serve as targets for many psychiatric medications.",
    "Evolution by natural selection operates on heritable variation in populations. Over time, differential reproduction shifts allele frequencies, and accumulated changes can give rise to new species.",
    "Protein synthesis begins with the transcription of DNA into messenger RNA, followed by translation in ribosomes that assemble amino acids into folded proteins according to the genetic code.",
    "Plate tectonics describes the movement of Earth's lithospheric plates over the mantle. Interactions at plate boundaries produce earthquakes, volcanoes, mountain ranges, and ocean trenches.",
    "Graph theory studies mathematical structures used to model pairwise relations between objects. Applications range from social-network analysis to routing algorithms, chemistry, and the design of integrated circuits.",
    "The Renaissance was a period of renewed interest in classical learning that reshaped art, science, and philosophy in Europe. Technological advances like the printing press accelerated the spread of ideas.",
    "Public-key cryptography relies on asymmetric mathematical operations. A public key allows anyone to encrypt a message, but only the holder of the matching private key can decrypt it.",
    "Stars produce energy through nuclear fusion in their cores, converting hydrogen into helium and heavier elements. Stellar evolution depends on mass, with very massive stars ending as supernovae or black holes.",
    "In statistics, a normal distribution is characterised by its mean and standard deviation. The central limit theorem explains why many aggregated quantities approximate normality despite underlying heterogeneity.",
]


def participation_ratio(cov: torch.Tensor) -> float:
    """(sum_i lambda_i)^2 / sum_i lambda_i^2 — effective linear rank."""
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    num = eigvals.sum().pow(2)
    den = eigvals.pow(2).sum().clamp_min(1e-12)
    return float((num / den).item())


def rank_coverage(cov: torch.Tensor, fractions=(0.5, 0.9, 0.95, 0.99)) -> dict:
    eigvals = torch.linalg.eigvalsh(cov.to(torch.float64)).clamp_min(0)
    eigvals = eigvals.flip(0)
    total = eigvals.sum().clamp_min(1e-12)
    cum = torch.cumsum(eigvals, dim=0) / total
    out = {}
    for f in fractions:
        idx = int((cum >= f).nonzero()[0].item()) + 1 if (cum >= f).any() else len(cum)
        out[f"r{int(f*100)}"] = idx
    return out


def twonn_dimension(X: torch.Tensor, sample_limit: int = 3000) -> float:
    """TwoNN estimator. X: [N, d]."""
    X = X.to(torch.float64)
    if X.shape[0] > sample_limit:
        idx = torch.randperm(X.shape[0])[:sample_limit]
        X = X[idx]
    N = X.shape[0]
    if N < 4:
        return float("nan")
    dists = torch.cdist(X, X)
    dists.fill_diagonal_(float("inf"))
    top2, _ = dists.topk(2, dim=1, largest=False)
    r1 = top2[:, 0]
    r2 = top2[:, 1]
    mask = r1 > 1e-10
    if mask.sum() < 10:
        return float("nan")
    mu = (r2[mask] / r1[mask]).clamp_min(1.0 + 1e-10)
    return float(1.0 / torch.log(mu).mean().item())


def load_model(model_id, dtype_str, device_map):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
                 "int8": torch.bfloat16, "int4": torch.bfloat16}
    kwargs = dict(
        torch_dtype=dtype_map.get(dtype_str, torch.bfloat16),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    )
    if dtype_str == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif dtype_str == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--calib-tokens", type=int, default=2000,
                   help="Target total calibration tokens across all texts")
    p.add_argument("--max-len", type=int, default=512,
                   help="Max sequence length per calibration chunk")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "int8", "int4"])
    p.add_argument("--device-map", default="cpu",
                   help="'cpu', 'auto', or a dict mapping layer names to devices")
    p.add_argument("--calib-file", default=None,
                   help="Optional plain-text file, one chunk per line. Falls back to built-in corpus.")
    p.add_argument("--twonn-sample-limit", type=int, default=3000)
    p.add_argument("--out", required=True, help="Path to output JSON (usually under machines/z8g4/results/)")
    args = p.parse_args()

    print(f"=== model: {args.model}  dtype: {args.dtype}  device_map: {args.device_map} ===")

    print(f"=== loading ===", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load_model(args.model, args.dtype, args.device_map)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    print(f"  n_layers={n_layers}  hidden={hidden}")

    # Build calibration text pool
    if args.calib_file:
        with open(args.calib_file) as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = DEFAULT_CALIBRATION_TEXTS
    print(f"  {len(texts)} calibration texts available")

    # Per-layer sample buffer (flat [N, d] of hidden states)
    # For very large models we can't keep ALL tokens across ALL layers in RAM
    # simultaneously — but Z8G4 with 700 GB can hold, say, 8k tokens across
    # 80 layers of hidden 8192: 8000 * 80 * 8192 * 4 bytes = 20 GB. Fine.
    layer_samples = [[] for _ in range(n_layers)]
    total_tokens = 0

    print(f"\n=== forward passes (target {args.calib_tokens} tokens) ===", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for text in texts:
            if total_tokens >= args.calib_tokens:
                break
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=args.max_len).input_ids
            print(f"  running on {ids.shape[1]} tokens... ", end="", flush=True)
            t_start = time.perf_counter()
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple [L+1] of [1, T, d]
            for i in range(n_layers):
                layer_samples[i].append(hs[i][0].to(torch.float32).cpu())
            total_tokens += ids.shape[1]
            # Free
            del out
            gc.collect()
            print(f"  {time.perf_counter()-t_start:.1f}s  (cum {total_tokens} tok)", flush=True)

    print(f"  forward passes total: {time.perf_counter()-t0:.1f}s")
    print(f"  collected {total_tokens} tokens across {n_layers} layers")

    # Compute stats per layer
    print(f"\n=== per-layer analysis ===", flush=True)
    per_layer = []
    t0 = time.perf_counter()
    for i in range(n_layers):
        X = torch.cat(layer_samples[i], dim=0)  # [N, d]
        Xc = X - X.mean(dim=0, keepdim=True)
        cov = Xc.T @ Xc
        pr = participation_ratio(cov)
        rank_cov = rank_coverage(cov)
        twonn = twonn_dimension(X, sample_limit=args.twonn_sample_limit)
        per_layer.append({
            "layer_index": i,
            "pr": pr,
            "twonn": twonn,
            "rank_coverage": rank_cov,
        })
        # Free the sample buffer for this layer once stats are computed
        layer_samples[i] = None
        if i % max(1, n_layers // 10) == 0 or i == n_layers - 1:
            print(f"  layer {i:3d}  PR={pr:.2f}  TwoNN={twonn:.2f}  "
                  f"r90={rank_cov['r90']}", flush=True)
        del X, Xc, cov
        gc.collect()
    print(f"  analysis total: {time.perf_counter()-t0:.1f}s")

    result = {
        "model_id": args.model,
        "dtype": args.dtype,
        "total_tokens": total_tokens,
        "chunk_size": args.max_len,
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "per_layer": per_layer,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
