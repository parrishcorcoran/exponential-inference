"""
Stage 35 — Direct-construct student from teacher's per-layer rotations, no training.

Tests the "train instantly" claim: if we have
  - tokenizer (fixed),
  - per-layer calibration basis P_i (from teacher's activations),
  - teacher's per-layer weights,
then we can construct a student's rank-k MLP weights directly by
projecting teacher's MLP weights through each layer's PCA basis.
Attention is kept intact (teacher's full Q/K/V retained) so cross-token
coupling isn't lost. No gradient descent.

Compare to:
  (a) full teacher (baseline)
  (b) stage-7-style all-weights-factored (no training, no attention retention)
  (c) this stage: MLP-only factored, attention intact (no training)

If (c) produces coherent output, the claim "train instantly" is
validated for the MLP factorization with attention retained.

Implementation: for each MLP's gate_proj, up_proj, down_proj:
  - Compute P_in from calibration inputs to that Linear (cov + eigh).
  - W_factored = W @ P_in [d_out, k], B = P_in.T [k, d_in].
  - Replace with F.linear(F.linear(x, B), A).
Attention linears (q_proj, k_proj, v_proj, o_proj) left intact.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# Only factor MLP linears; leave attention intact.
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


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


class FactoredLinear(nn.Module):
    """y = A @ (B @ x) + bias. Direct construction from teacher's W and basis P_in.
    W: [d_out, d_in]; P_in: [d_in, k]; A = W @ P_in [d_out, k]; B = P_in.T [k, d_in]."""

    def __init__(self, orig: nn.Linear, P_in: torch.Tensor):
        super().__init__()
        W = orig.weight.data.to(torch.float32).cpu()
        P = P_in.to(torch.float32).cpu()
        A = (W @ P).contiguous().to(orig.weight.dtype).to(orig.weight.device)
        B = P.T.contiguous().to(orig.weight.dtype).to(orig.weight.device)
        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)
        self.bias = nn.Parameter(orig.bias.data.clone(), requires_grad=False) if orig.bias is not None else None
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.rank = P.shape[1]

    def forward(self, x):
        return F.linear(F.linear(x, self.B), self.A, self.bias)


def collect_mlp_input_covariances(model, tokenizer, texts, device, max_len=256):
    """Forward-hook each MLP Linear and accumulate X^T X on CPU (fp32)."""
    covs = {}
    target_modules = []
    for name, module in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and last in MLP_TARGETS:
            target_modules.append((name, module))

    def make_hook(n, d_in):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            if n not in covs:
                covs[n] = torch.zeros(d_in, d_in, dtype=torch.float32)
            covs[n] += x_flat.T @ x_flat
        return hook

    handles = [mod.register_forward_hook(make_hook(name, mod.in_features))
                for name, mod in target_modules]
    try:
        model.eval()
        with torch.inference_mode():
            for text in texts:
                ids = tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=max_len).input_ids.to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {n: c.to(torch.float64) for n, c in covs.items()}


def top_k_basis(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)


def factor_mlp_weights(model, covariances, rank):
    """Replace every MLP Linear with a FactoredLinear using calibration basis."""
    n_replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in MLP_TARGETS:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name not in covariances:
                continue
            P = top_k_basis(covariances[full_name], rank)
            fact = FactoredLinear(child, P_in=P)
            setattr(module, child_name, fact)
            n_replaced += 1
    return n_replaced


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


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
    p.add_argument("--ranks", default="32,64,128,256,512",
                   help="Comma-separated MLP ranks to sweep")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage35_direct_construct.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading teacher {args.model} ===")
    teacher, tokenizer = load_model(args.model, device)

    print(f"\n=== teacher baseline ===")
    t0 = time.perf_counter()
    teacher_tokens = generate(teacher, tokenizer, args.prompt, args.max_new_tokens, device)
    teacher_text = tokenizer.decode(teacher_tokens, skip_special_tokens=True)
    print(f"  generated in {time.perf_counter()-t0:.1f}s")
    print(f"  {teacher_text[:150]}...")

    print(f"\n=== collecting calibration covariances (MLP inputs) ===")
    t0 = time.perf_counter()
    covs = collect_mlp_input_covariances(teacher, tokenizer, CALIB_TEXTS, device)
    print(f"  {len(covs)} MLP linears, {time.perf_counter()-t0:.1f}s")

    # Free teacher before building student (shared memory on MPS)
    del teacher
    if device == "mps":
        torch.mps.empty_cache()

    ranks = [int(x) for x in args.ranks.split(",")]
    results = []
    for k in ranks:
        print(f"\n=== rank {k} direct-construct student ===")
        t0 = time.perf_counter()
        student, _ = load_model(args.model, device)
        n_replaced = factor_mlp_weights(student, covs, k)
        print(f"  factored {n_replaced} MLP linears in {time.perf_counter()-t0:.1f}s "
              f"(NO training)")

        t0 = time.perf_counter()
        student_tokens = generate(student, tokenizer, args.prompt, args.max_new_tokens, device)
        student_text = tokenizer.decode(student_tokens, skip_special_tokens=True)

        min_len = min(len(teacher_tokens), len(student_tokens))
        match = sum(1 for a, b in zip(teacher_tokens[:min_len], student_tokens[:min_len]) if a == b)
        first_div = next((i for i, (a, b) in enumerate(zip(teacher_tokens, student_tokens)) if a != b), min_len)
        print(f"  generated in {time.perf_counter()-t0:.1f}s")
        print(f"  match: {match}/{min_len}  first divergence @ {first_div}")
        print(f"  {student_text[:150]}...")

        results.append({
            "rank": k, "n_mlp_factored": n_replaced,
            "match": match, "total": min_len,
            "match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "sample": student_text[:300],
        })

        del student
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\n=== summary ===")
    print(f"  teacher: {teacher_text[:80]}")
    print(f"  {'rank':>5}  {'match':>10}  {'first div':>10}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['match']}/{r['total']:<4}  {r['first_divergence']:>10}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "teacher_sample": teacher_text[:400],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
