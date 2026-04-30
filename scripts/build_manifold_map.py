"""
Build a unified manifold.pt artifact from one or more teachers.

This is the precomputed "map" that a small student model will consume
at training and inference time. Contains everything about the
tokenizer-family's hidden-state geometry that the student needs to
know, so the student's weights only have to encode TRAVERSAL POLICY.

Artifact format (torch.save of a dict, version 1):
  model_info:           dict — teacher(s) used, tokenizer, timestamp
  L:                    int — num layers (L+1 bases including embedding)
  d_model:              int
  vocab_size:           int
  rank:                 int — PCA rank used for per-layer bases
  per_layer_mean:       [L+1, d_model]
  per_layer_basis:      [L+1, d_model, rank]  orthonormal columns
  rotation_operators:   [L, rank, rank]      R_{i→i+1} = P_{i+1}^T @ P_i
  rotation_curve:       [L]                  mean angle per transition
  carry_indices:        list[L] of tensors   which columns in rotation_op are carry-mode
  flip_indices:         list[L] of tensors   flip-mode columns
  mid_indices:          list[L] of tensors   mid-rotation columns
  twonn_per_layer:      [L+1]                bootstrap-mean TwoNN dim
  twonn_std_per_layer:  [L+1]                bootstrap std
  stabilization_dirs:   [L, rank]            per-layer vector pointing toward the attractor in the manifold basis
  calibration_corpus_hash: str               hash of the tokenized corpus used

Usage:
  # Single teacher, default calibration corpus
  python scripts/build_manifold_map.py \\
      --teacher Qwen/Qwen3-0.6B \\
      --rank 64 \\
      --out artifacts/manifold_qwen3_0p6b.pt

  # Multi-teacher ensemble (averaged PCA bases; teachers must share
  # tokenizer AND architecture — same L, same d_model)
  python scripts/build_manifold_map.py \\
      --teachers Qwen/Qwen3-0.6B Qwen/Qwen3-0.6B-Base \\
      --rank 64 \\
      --out artifacts/manifold_qwen3_family.pt

Future extensions (not in this version):
  - attention_geometry per head
  - cross-architecture ensemble (depth-fraction interpolation)
  - per-token local tangent basis (beyond global PCA)
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_VERSION = 1


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


def load_model(model_id, device, dtype_str="bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_str, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def corpus_hash(texts):
    blob = "\n".join(texts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_calibration_texts(calib_dataset, calib_config, calib_split, max_tokens, tokenizer):
    """Load a real calibration corpus from HF datasets, yielding text chunks
    whose concatenated token count reaches max_tokens. Falls back to CALIB_TEXTS
    if calib_dataset is None or "default"."""
    if calib_dataset in (None, "default"):
        return CALIB_TEXTS
    from datasets import load_dataset
    ds = load_dataset(calib_dataset, calib_config, split=calib_split)
    texts = []
    total = 0
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        # Cheap tokenization just for counting
        n = len(tokenizer.encode(t, add_special_tokens=False))
        if n == 0: continue
        texts.append(t)
        total += n
        if total >= max_tokens: break
    print(f"  loaded {len(texts)} calibration texts (~{total} tokens) from {calib_dataset}/{calib_config}:{calib_split}")
    return texts


def collect_per_layer_hiddens(model, tokenizer, texts, device, max_len=256):
    """Return dict i -> [N, d_model] of hidden states per layer (i=0 is
    embedding, i=L is after last transformer layer)."""
    L = len(model.model.layers)
    samples = [[] for _ in range(L + 1)]

    def make_hook(i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            samples[i].append(h_flat)
        return hook

    def embed_hook(mod, inputs, output):
        h = output
        h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
        samples[0].append(h_flat)

    h0 = model.model.embed_tokens.register_forward_hook(embed_hook)
    handles = [h0] + [model.model.layers[i].register_forward_hook(make_hook(i + 1))
                       for i in range(L)]
    try:
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()
    return {i: torch.cat(samples[i], dim=0) for i in range(L + 1)}


def pca_basis_with_mean(H, k):
    """H: [N, d]. Returns basis [d, k] (orthonormal columns, descending
    variance) and mean [d]."""
    mu = H.mean(dim=0)
    Hc = H - mu.unsqueeze(0)
    cov = Hc.T @ Hc / max(Hc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1])
    return P, mu


def bootstrap_twonn(H, n_boot=20, subsample_frac=0.8, seed=0, max_n_sub=4000):
    """TwoNN mean + std via bootstrap subsamples.
    Cap subsample size at max_n_sub to keep distance matrix memory/compute
    bounded even when N is huge. 4000 gives ±0.2 dim precision on 20 boots."""
    N = H.shape[0]
    n_sub = min(max_n_sub, max(10, int(subsample_frac * N)))
    estimates = []
    for b in range(n_boot):
        torch.manual_seed(seed + b)
        idx = torch.randperm(N)[:n_sub]
        X = H[idx].to(torch.float32)
        D = torch.cdist(X, X)
        D.fill_diagonal_(float("inf"))
        sorted_d, _ = D.sort(dim=1)
        r1, r2 = sorted_d[:, 0], sorted_d[:, 1]
        mask = (r1 > 1e-8) & (r2 > r1 + 1e-10)
        if mask.sum() < 10: continue
        mu = r2[mask] / r1[mask]
        d_est = float(mask.sum().item() / torch.log(mu).sum().item())
        if math.isfinite(d_est): estimates.append(d_est)
    if not estimates:
        return float("nan"), float("nan")
    t = torch.tensor(estimates, dtype=torch.float32)
    return float(t.mean()), float(t.std())


def partition_modes(R, carry_tol=0.3, flip_tol=0.3):
    """Eigendecompose rotation R [k, k]; return indices into columns of R's
    eigenvector matrix for carry / flip / mid modes by angle.
    Also return per-column rotation angle."""
    R = R.to(torch.float32)
    eigvals, eigvecs = torch.linalg.eig(R)
    angles = torch.atan2(eigvals.imag, eigvals.real).abs()  # [k]
    carry_idx = torch.where(angles < carry_tol)[0]
    flip_idx = torch.where(angles > math.pi - flip_tol)[0]
    mid_idx = torch.where(
        (angles >= carry_tol) & (angles <= math.pi - flip_tol))[0]
    return {
        "carry_idx": carry_idx, "flip_idx": flip_idx, "mid_idx": mid_idx,
        "angles": angles, "eigvals": eigvals, "eigvecs": eigvecs,
    }


def compute_stabilization_directions(model, tokenizer, texts, device, P_list, mu_list, max_len=256):
    """For each layer, find the direction in its manifold basis that points
    toward the final-layer attractor. Specifically: project each layer's
    hidden states onto its PCA basis, then find the direction in the
    projected space that maximizes correlation with the final-layer argmax.
    Simpler approach: project mean-final-hidden through each layer's basis."""
    # Simpler: for each layer i, compute the mean of tokens' projected
    # coords (they should concentrate toward a direction at late layers)
    stabilization_dirs = []
    with torch.inference_mode():
        # Collect per-layer hiddens (we have them already — reuse)
        # But to keep this function self-contained, compute from scratch:
        L = len(model.model.layers)
        samples = [[] for _ in range(L + 1)]

        def make_hook(i):
            def hook(mod, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
                samples[i].append(h_flat)
            return hook

        def embed_hook(mod, inputs, output):
            h = output
            h_flat = h.detach().reshape(-1, h.shape[-1]).to(torch.float32).cpu()
            samples[0].append(h_flat)

        h0 = model.model.embed_tokens.register_forward_hook(embed_hook)
        handles = [h0] + [model.model.layers[i].register_forward_hook(make_hook(i + 1))
                           for i in range(L)]
        try:
            for text in texts:
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
        finally:
            for h in handles: h.remove()

        for i in range(L + 1):
            H = torch.cat(samples[i], dim=0)
            mu = mu_list[i]
            P = P_list[i]
            projected = (H - mu.unsqueeze(0)) @ P              # [N, rank]
            mean_direction = projected.mean(dim=0)             # [rank]
            norm = mean_direction.norm().clamp_min(1e-8)
            stabilization_dirs.append(mean_direction / norm)

    return torch.stack(stabilization_dirs)  # [L+1, rank]


def build_map_for_teacher(model_id, rank, device, dtype, calib_dataset, calib_config, calib_split, calib_max_tokens):
    print(f"\n{'='*60}")
    print(f"building map for: {model_id}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_id, device, dtype)
    L = len(model.model.layers)
    d_model = model.config.hidden_size
    vocab_size = model.config.vocab_size
    print(f"  L={L}  d_model={d_model}  vocab_size={vocab_size}")

    calib_texts = load_calibration_texts(calib_dataset, calib_config, calib_split, calib_max_tokens, tokenizer)

    print(f"  collecting hidden states...")
    t0 = time.perf_counter()
    hiddens = collect_per_layer_hiddens(model, tokenizer, calib_texts, device)
    n_tokens = hiddens[0].shape[0]
    oversample = n_tokens / d_model
    print(f"  {time.perf_counter()-t0:.1f}s  ({n_tokens} tokens, {oversample:.1f}x d_model)")
    if oversample < 4.0:
        print(f"  WARNING: oversample ratio {oversample:.1f}x is low — PCA and TwoNN may be noisy.")
        print(f"           recommend ≥10x for stable estimates. Increase --calib-max-tokens.")

    print(f"  computing PCA bases and bootstrap TwoNN per layer (rank {rank})...")
    t0 = time.perf_counter()
    P_list = []; mu_list = []
    twonn_mean = []; twonn_std = []
    for i in range(L + 1):
        P, mu = pca_basis_with_mean(hiddens[i], rank)
        P_list.append(P)
        mu_list.append(mu)
        m, s = bootstrap_twonn(hiddens[i])
        twonn_mean.append(m)
        twonn_std.append(s)
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"  computing rotation operators + mode partitions...")
    t0 = time.perf_counter()
    rotation_operators = []
    rotation_curve = []
    carry_indices = []
    flip_indices = []
    mid_indices = []
    for i in range(L):
        R = P_list[i + 1].T @ P_list[i]                         # [rank, rank]
        rotation_operators.append(R.to(torch.float32))
        parts = partition_modes(R)
        carry_indices.append(parts["carry_idx"])
        flip_indices.append(parts["flip_idx"])
        mid_indices.append(parts["mid_idx"])
        rotation_curve.append(float(parts["angles"].mean()))
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"  computing stabilization directions per layer...")
    t0 = time.perf_counter()
    stab_dirs = compute_stabilization_directions(
        model, tokenizer, calib_texts, device, P_list, mu_list)
    print(f"  {time.perf_counter()-t0:.1f}s")

    # Pack per-teacher artifact
    artifact = {
        "version": ARTIFACT_VERSION,
        "model_info": {
            "teacher": model_id,
            "tokenizer": tokenizer.name_or_path,
            "vocab_size": vocab_size,
            "L": L,
            "d_model": d_model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rank": rank,
        "per_layer_mean": torch.stack(mu_list),                # [L+1, d_model]
        "per_layer_basis": torch.stack(P_list),                # [L+1, d_model, rank]
        "rotation_operators": torch.stack(rotation_operators), # [L, rank, rank]
        "rotation_curve": torch.tensor(rotation_curve, dtype=torch.float32),
        "carry_indices": carry_indices,
        "flip_indices": flip_indices,
        "mid_indices": mid_indices,
        "twonn_per_layer": torch.tensor(twonn_mean, dtype=torch.float32),
        "twonn_std_per_layer": torch.tensor(twonn_std, dtype=torch.float32),
        "stabilization_dirs": stab_dirs,                       # [L+1, rank]
        "calibration_corpus_hash": corpus_hash(CALIB_TEXTS),
    }

    # Report
    print(f"\n  per-layer summary (TwoNN mean ± std, mean rotation angle):")
    for i in [0, 1, L // 2, L - 1, L]:
        if i <= L:
            rot = rotation_curve[i - 1] if i > 0 and i <= L else float("nan")
            print(f"    layer {i:>2}: TwoNN {twonn_mean[i]:.2f} ± {twonn_std[i]:.2f}  "
                  f"rotation(i-1→i) = {rot:.3f} rad")

    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return artifact


def ensemble_artifacts(artifacts):
    """Given a list of per-teacher artifacts, return a single ensembled
    artifact. Teachers must share tokenizer AND (L, d_model, rank).
    Averages bases, rotation operators, stabilization directions."""
    if len(artifacts) == 1:
        out = dict(artifacts[0])
        out["model_info"] = {
            "teachers": [artifacts[0]["model_info"]["teacher"]],
            "tokenizer": artifacts[0]["model_info"]["tokenizer"],
            "ensemble": False,
            **{k: v for k, v in artifacts[0]["model_info"].items()
               if k not in ("teacher",)},
        }
        return out

    # Verify compatibility
    L = artifacts[0]["model_info"]["L"]
    d_model = artifacts[0]["model_info"]["d_model"]
    rank = artifacts[0]["rank"]
    for a in artifacts[1:]:
        assert a["model_info"]["L"] == L, f"L mismatch: {L} vs {a['model_info']['L']}"
        assert a["model_info"]["d_model"] == d_model, f"d_model mismatch"
        assert a["rank"] == rank, f"rank mismatch"

    # Averaging bases is tricky (PCA bases have sign ambiguity). Simple
    # approach: align signs by inner product with first teacher's basis.
    base_ref = artifacts[0]["per_layer_basis"]  # [L+1, d_model, rank]
    aligned_bases = [base_ref]
    for a in artifacts[1:]:
        P = a["per_layer_basis"]                 # [L+1, d_model, rank]
        # For each (layer, column), flip sign if inner product with ref is negative
        inner = (base_ref * P).sum(dim=1)        # [L+1, rank]
        signs = torch.sign(inner).unsqueeze(1)   # [L+1, 1, rank]
        signs[signs == 0] = 1
        aligned_bases.append(P * signs)

    mean_basis = torch.stack(aligned_bases).mean(dim=0)       # [L+1, d_model, rank]

    # Re-orthonormalize via QR (averaging destroys orthonormality)
    ortho_basis = torch.zeros_like(mean_basis)
    for i in range(L + 1):
        Q, _ = torch.linalg.qr(mean_basis[i].to(torch.float32))
        ortho_basis[i] = Q

    mean_mu = torch.stack([a["per_layer_mean"] for a in artifacts]).mean(dim=0)
    mean_stab = torch.stack([a["stabilization_dirs"] for a in artifacts]).mean(dim=0)
    mean_stab = mean_stab / mean_stab.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Recompute rotation ops from ensembled basis
    rotation_operators = []
    rotation_curve = []
    for i in range(L):
        R = ortho_basis[i + 1].T @ ortho_basis[i]
        rotation_operators.append(R)
        parts = partition_modes(R)
        rotation_curve.append(float(parts["angles"].mean()))

    # TwoNN: average across teachers
    twonn_mean = torch.stack([a["twonn_per_layer"] for a in artifacts]).mean(dim=0)
    twonn_std = torch.stack([a["twonn_std_per_layer"] for a in artifacts]).mean(dim=0)

    # Recompute mode indices from ensembled rotations
    carry_indices = []
    flip_indices = []
    mid_indices = []
    for i in range(L):
        parts = partition_modes(rotation_operators[i])
        carry_indices.append(parts["carry_idx"])
        flip_indices.append(parts["flip_idx"])
        mid_indices.append(parts["mid_idx"])

    return {
        "version": ARTIFACT_VERSION,
        "model_info": {
            "teachers": [a["model_info"]["teacher"] for a in artifacts],
            "tokenizer": artifacts[0]["model_info"]["tokenizer"],
            "vocab_size": artifacts[0]["model_info"]["vocab_size"],
            "L": L, "d_model": d_model,
            "ensemble": True,
            "n_teachers": len(artifacts),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rank": rank,
        "per_layer_mean": mean_mu,
        "per_layer_basis": ortho_basis,
        "rotation_operators": torch.stack(rotation_operators),
        "rotation_curve": torch.tensor(rotation_curve, dtype=torch.float32),
        "carry_indices": carry_indices,
        "flip_indices": flip_indices,
        "mid_indices": mid_indices,
        "twonn_per_layer": twonn_mean,
        "twonn_std_per_layer": twonn_std,
        "stabilization_dirs": mean_stab,
        "calibration_corpus_hash": corpus_hash(CALIB_TEXTS),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", help="single teacher (convenience)")
    p.add_argument("--teachers", nargs="+", help="multiple teachers for ensemble")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--calib-dataset", default="default",
                   help="HF dataset name for calibration (e.g. 'wikitext'); 'default' uses built-in short texts")
    p.add_argument("--calib-config", default="wikitext-2-raw-v1")
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-max-tokens", type=int, default=20000,
                   help="cap on calibration tokens. For d_model=5120, use ≥50000 for stable PCA")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}  rank={args.rank}  dtype={args.dtype}")

    teachers = []
    if args.teacher:
        teachers.append(args.teacher)
    if args.teachers:
        teachers.extend(args.teachers)
    if not teachers:
        raise SystemExit("specify --teacher or --teachers")

    print(f"teachers: {teachers}")

    artifacts = []
    for tid in teachers:
        artifacts.append(build_map_for_teacher(
            tid, args.rank, device, args.dtype,
            args.calib_dataset, args.calib_config, args.calib_split, args.calib_max_tokens))

    print(f"\n=== ensembling {len(artifacts)} teacher(s) ===")
    final = ensemble_artifacts(artifacts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final, out_path)

    # Also write a small JSON sidecar describing the artifact
    sidecar = out_path.with_suffix(".json")
    sidecar_info = {
        "version": ARTIFACT_VERSION,
        "model_info": final["model_info"],
        "rank": final["rank"],
        "calibration_corpus_hash": final["calibration_corpus_hash"],
        "twonn_per_layer_mean": final["twonn_per_layer"].tolist(),
        "twonn_per_layer_std": final["twonn_std_per_layer"].tolist(),
        "rotation_curve": final["rotation_curve"].tolist(),
        "n_carry_per_transition": [len(x) for x in final["carry_indices"]],
        "n_flip_per_transition": [len(x) for x in final["flip_indices"]],
        "n_mid_per_transition": [len(x) for x in final["mid_indices"]],
    }
    with open(sidecar, "w") as f:
        json.dump(sidecar_info, f, indent=2)

    print(f"\n=== wrote {out_path} ===")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  artifact size: {size_mb:.1f} MB")
    print(f"  sidecar: {sidecar}")
    print(f"\nusage (student):")
    print(f"  m = torch.load('{args.out}', map_location='cpu')")
    print(f"  # m['per_layer_basis']: [L+1, d_model, rank]")
    print(f"  # m['rotation_operators']: [L, rank, rank]")
    print(f"  # m['carry_indices'][i], m['flip_indices'][i], m['mid_indices'][i]")


if __name__ == "__main__":
    main()
