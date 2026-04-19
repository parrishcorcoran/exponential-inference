"""
Stage 42 — Oracle depth-pruning test.

Stage 41 showed MLP intermediate sparsity averages ~5% (only 1 in 20 positions
active per token). If non-active positions can be zeroed without quality loss,
depth compression is 20× on the MLP intermediate axis.

This stage is an ORACLE test: for each token at each layer, we compute
int_act = silu(gate) * up, then keep only the top-k positions by |int_act|
and zero the rest. k is chosen per token from the teacher's own computation.
This is the upper-bound quality — what's possible if we had perfect depth
routing. If it works, we then build a predictor.

Sweep fraction kept: {1%, 2%, 5%, 10%, 25%, 50%, 100%} of d_intermediate.
Report token-match, first-divergence, and generation sample vs teacher.

Prediction: at fraction ≈ 5% (matching measured mean sparsity), match_ratio
should be high. Below 5%, quality degrades. Above 5%, no gain.
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


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


class PrunedMLP(nn.Module):
    """SwiGLU MLP that zeros all but top-k intermediate positions per token."""

    def __init__(self, orig, keep_k):
        super().__init__()
        self.gate_proj = orig.gate_proj
        self.up_proj = orig.up_proj
        self.down_proj = orig.down_proj
        self.act_fn = orig.act_fn if hasattr(orig, "act_fn") else F.silu
        self.keep_k = keep_k
        self.d_int = orig.gate_proj.out_features

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        int_act = self.act_fn(gate) * up            # [..., d_int]
        if self.keep_k >= self.d_int:
            return self.down_proj(int_act)
        abs_act = int_act.abs()
        topk_idx = abs_act.topk(self.keep_k, dim=-1).indices
        mask = torch.zeros_like(abs_act)
        mask.scatter_(-1, topk_idx, 1.0)
        pruned = int_act * mask.to(int_act.dtype)
        return self.down_proj(pruned)


def install_pruned_mlps(model, keep_k):
    n = 0
    originals = []
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            if cname == "mlp" and hasattr(child, "gate_proj") and hasattr(child, "down_proj"):
                new_mlp = PrunedMLP(child, keep_k)
                setattr(module, cname, new_mlp)
                originals.append((module, cname, child))
                n += 1
    return n, originals


def restore_mlps(originals):
    for module, cname, child in originals:
        setattr(module, cname, child)


def generate(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--fractions", default="0.01,0.02,0.05,0.10,0.25,0.50,1.0")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage42_depth_pruning.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    d_int = model.config.intermediate_size
    print(f"  d_intermediate = {d_int}")

    print(f"\n=== baseline ===")
    t0 = time.perf_counter()
    base_tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
    base_text = tokenizer.decode(base_tokens, skip_special_tokens=True)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  {base_text[:150]}")

    fractions = [float(x) for x in args.fractions.split(",")]
    results = []
    for frac in fractions:
        keep_k = max(1, int(frac * d_int))
        print(f"\n=== fraction {frac:.3f}  keep_k={keep_k}/{d_int}  "
              f"({d_int/keep_k:.1f}x depth compression) ===")
        _, originals = install_pruned_mlps(model, keep_k)
        try:
            t0 = time.perf_counter()
            tokens = generate(model, tokenizer, args.prompt, args.max_new_tokens, device)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        finally:
            restore_mlps(originals)
        n = min(len(base_tokens), len(tokens))
        match = sum(1 for a, b in zip(base_tokens[:n], tokens[:n]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(base_tokens, tokens)) if a != b), n)
        print(f"  {time.perf_counter()-t0:.1f}s  match {match}/{n}  first_div @ {first_div}")
        print(f"  {text[:150]}")
        results.append({
            "fraction": frac,
            "keep_k": keep_k,
            "compression": d_int / keep_k,
            "match": match, "total": n,
            "match_ratio": match / max(n, 1),
            "first_divergence": first_div,
            "sample": text[:300],
        })

    print(f"\n=== summary ===")
    print(f"  d_int={d_int}  baseline: {base_text[:70]}")
    print(f"  {'frac':>6}  {'keep_k':>7}  {'compress':>8}  {'match':>10}  {'first_div':>9}")
    for r in results:
        print(f"  {r['fraction']:>6.3f}  {r['keep_k']:>7}  "
              f"{r['compression']:>7.1f}x  "
              f"{r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "d_intermediate": d_int,
            "baseline_sample": base_text[:400],
            "fractions": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
