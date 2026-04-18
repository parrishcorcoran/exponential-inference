"""
Stage 6 — Factored-weight decode.

Cashing in the manifold measurement: if hidden states live on a ~10-dim
manifold (TwoNN), and every Linear maps manifold -> manifold, then each
Linear's effective rank is bounded by that manifold dimension.

Offline SVD every attention + MLP Linear into W ≈ A · B with rank k, and
replace the modules so forward literally runs fewer FLOPs and loads less
weight memory (the actual bottleneck at batch=1 decode on MPS).

Savings at rank k, per Linear of shape [d_out, d_in]:
    FLOPs:       2·d_out·d_in  ->  2·k·(d_out + d_in)
    Weight bytes: d_out·d_in   ->  k·(d_out + d_in)

For Qwen3-0.6B (H=1024, I=3072):
    MLP gate/up/down full: 1024·3072 = 3.1M params each
    At k=16:               16·(1024+3072) = 65K params each  -> 48x smaller
    Attention q/o full:    1024·1024 = 1M
    At k=16:               16·2048 = 33K                      -> 32x smaller

Wall-clock win at decode scales with the *weight-bytes* ratio, not just
FLOPs, because batch=1 decode is memory-bandwidth bound.

Usage:
    python scripts/stage6_factored_decode.py \\
        --model Qwen/Qwen3-0.6B \\
        --ranks 8,16,32,64,128,256 \\
        --max-new-tokens 200 \\
        --device mps
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

from src.common.model_loader import describe_backend


class FactoredLinear(nn.Module):
    """W ≈ A @ B via truncated SVD. F.linear(x, W) = F.linear(F.linear(x, B), A)."""

    def __init__(self, orig: nn.Linear, rank: int):
        super().__init__()
        W = orig.weight.data.to(torch.float32).cpu()  # [d_out, d_in]
        # Truncated SVD: W = U S Vh. Absorb S into A so forward is A(Bx).
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, S.numel())
        A = (U[:, :k] * S[:k]).to(orig.weight.dtype).to(orig.weight.device)
        B = Vh[:k].contiguous().to(orig.weight.dtype).to(orig.weight.device)

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)
        if orig.bias is not None:
            self.bias = nn.Parameter(orig.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.rank = k
        # Tag for external sanity checks / memory counting
        self._factored_params = k * (orig.in_features + orig.out_features)
        self._full_params = orig.in_features * orig.out_features

    def forward(self, x):
        return F.linear(F.linear(x, self.B), self.A, self.bias)


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",        # attention
    "gate_proj", "up_proj", "down_proj",           # MLP
)


def factorize_model(model, rank: int, skip_embed_and_head: bool = True):
    """Replace each target nn.Linear with FactoredLinear at given rank.

    Returns stats dict. lm_head and embed are left full-rank by default —
    they touch vocab (large output space) and distortions there hit
    next-token probabilities directly.
    """
    stats = {
        "n_replaced": 0,
        "full_params": 0,
        "factored_params": 0,
        "per_type": {},
    }
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in TARGET_NAMES:
                continue
            if skip_embed_and_head and child_name in ("lm_head",):
                continue
            fact = FactoredLinear(child, rank=rank)
            setattr(module, child_name, fact)
            stats["n_replaced"] += 1
            stats["full_params"] += fact._full_params
            stats["factored_params"] += fact._factored_params
            stats["per_type"].setdefault(child_name, 0)
            stats["per_type"][child_name] += 1
    return stats


def generate(model, tokenizer, prompt, max_new_tokens, device, warmup=2):
    """Greedy decode with per-step timing. Returns (decode_times_ms, text, tokens)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Prefill (not timed)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    # Warmup decode steps (not timed) — MPS JIT / kernel caching
    for _ in range(warmup):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

    # Timed decode
    times = []
    for _ in range(max_new_tokens - 1 - warmup):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return [t * 1000 for t in times], text, generated


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="8,16,32,64,128,256",
                   help="Comma-separated rank values to sweep")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\ndevice: {device}")

    ranks = [int(x) for x in args.ranks.split(",")]

    # === Baseline (full rank, unmodified) ===
    print(f"\n=== loading baseline {args.model} ===", flush=True)
    model, tokenizer = load_model(args.model, device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    print(f"  {n_layers} layers, hidden={hidden}, intermediate={intermediate}")

    print(f"\n=== baseline decode ===", flush=True)
    base_times, base_text, base_tokens = generate(
        model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_ms = sum(base_times) / len(base_times)
    print(f"  {len(base_times)} timed tokens, {base_ms:.2f}ms/tok")
    print(f"  {base_text[:120]}...")

    # Free baseline
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    # === Rank sweep ===
    results = []
    for k in ranks:
        print(f"\n=== rank {k} ===", flush=True)
        model, _ = load_model(args.model, device)

        t0 = time.perf_counter()
        fstats = factorize_model(model, rank=k)
        svd_time = time.perf_counter() - t0
        size_ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
        print(f"  factored {fstats['n_replaced']} linears in {svd_time:.1f}s")
        print(f"  weight params: {fstats['full_params']/1e6:.1f}M -> "
              f"{fstats['factored_params']/1e6:.2f}M "
              f"({size_ratio:.2%} of full)")

        rtimes, rtext, rtokens = generate(
            model, tokenizer, args.prompt, args.max_new_tokens, device)
        rms = sum(rtimes) / len(rtimes)

        min_len = min(len(base_tokens), len(rtokens))
        match = sum(1 for a, b in zip(base_tokens[:min_len], rtokens[:min_len]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(
            zip(base_tokens, rtokens)) if a != b), min_len)

        speedup = base_ms / rms if rms > 0 else 0
        print(f"  {len(rtimes)} timed tokens, {rms:.2f}ms/tok  (speedup: {speedup:.2f}x)")
        print(f"  match: {match}/{min_len} ({match/max(min_len,1):.1%}), "
              f"first divergence @ token {first_div}")
        print(f"  {rtext[:120]}...")

        results.append({
            "rank": k,
            "ms_per_tok": rms,
            "speedup_vs_baseline": speedup,
            "token_match": f"{match}/{min_len}",
            "token_match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "weight_params_full_M": fstats["full_params"] / 1e6,
            "weight_params_factored_M": fstats["factored_params"] / 1e6,
            "weight_size_ratio": size_ratio,
            "svd_time_sec": svd_time,
            "sample_text": rtext[:300],
        })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    # === Summary ===
    print(f"\n=== summary ===")
    print(f"  baseline: {base_ms:.2f}ms/tok")
    print(f"  {'rank':>5}  {'ms/tok':>8}  {'speedup':>8}  {'match':>10}  "
          f"{'weights':>9}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['ms_per_tok']:>8.2f}  "
              f"{r['speedup_vs_baseline']:>7.2f}x  "
              f"{r['token_match']:>10}  "
              f"{r['weight_size_ratio']:>8.2%}")

    out_path = Path(args.out_dir) / f"stage6_factored_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "n_layers": n_layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "baseline_ms_per_tok": base_ms,
            "baseline_sample": base_text[:500],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
