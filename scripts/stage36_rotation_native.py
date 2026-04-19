"""
Stage 36 — Rotation-native student architecture (program step 2 + 3).

Unlike stage 35 (factor teacher's full MLP through PCA basis, attention intact),
this stage rebuilds the MLP as rank-k native: intermediate dim = k, not the
teacher's d_intermediate. The MLP lives natively in per-layer PCA space.

Per MLP:
  x_residual [d_model]
    -> B_in = P_in.T                      [k, d_model]
    -> x_k [k]
    -> gate_k = silu(W_g_s @ x_k)         W_g_s init = P_pre_down.T @ W_g @ P_in
    -> up_k   = W_u_s @ x_k               W_u_s init = P_pre_down.T @ W_u @ P_in
    -> hidden_k = gate_k * up_k           [k]
    -> out = A_d @ hidden_k               A_d init = W_d @ P_pre_down
    -> residual_out [d_model]

Attention is left intact. Step 4 (distillation fine-tune) is NOT run here —
this measures the geometric-init-alone baseline that stage 35 measured for the
project-in-place version. Stage 36b will add fine-tune.

Two calibration bases per MLP:
  P_in        [d_model, k]            from MLP input cov
  P_pre_down  [d_intermediate, k]     from pre-down activation cov (post silu*up)
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


class RotationNativeMLP(nn.Module):
    """Rank-k native SwiGLU MLP. Intermediate dim = k. Init from teacher projection."""

    def __init__(self, teacher_mlp, P_in, P_pre_down):
        super().__init__()
        dtype = teacher_mlp.gate_proj.weight.dtype
        device = teacher_mlp.gate_proj.weight.device

        W_g = teacher_mlp.gate_proj.weight.data.to(torch.float32).cpu()   # [d_int, d_model]
        W_u = teacher_mlp.up_proj.weight.data.to(torch.float32).cpu()     # [d_int, d_model]
        W_d = teacher_mlp.down_proj.weight.data.to(torch.float32).cpu()   # [d_model, d_int]

        P_in_32 = P_in.to(torch.float32).cpu()                 # [d_model, k]
        P_pd_32 = P_pre_down.to(torch.float32).cpu()           # [d_int, k]

        B_in = P_in_32.T.contiguous()                          # [k, d_model]
        W_g_s = (P_pd_32.T @ W_g @ P_in_32).contiguous()       # [k, k]
        W_u_s = (P_pd_32.T @ W_u @ P_in_32).contiguous()       # [k, k]
        A_d   = (W_d @ P_pd_32).contiguous()                   # [d_model, k]

        self.B_in  = nn.Parameter(B_in.to(dtype).to(device),  requires_grad=False)
        self.W_g_s = nn.Parameter(W_g_s.to(dtype).to(device), requires_grad=False)
        self.W_u_s = nn.Parameter(W_u_s.to(dtype).to(device), requires_grad=False)
        self.A_d   = nn.Parameter(A_d.to(dtype).to(device),   requires_grad=False)

        self.rank = P_in_32.shape[1]
        self.d_model = P_in_32.shape[0]
        self.d_int = P_pd_32.shape[0]

    def forward(self, x):
        x_k = F.linear(x, self.B_in)                      # [..., k]
        gate_k = F.silu(F.linear(x_k, self.W_g_s))        # [..., k]
        up_k = F.linear(x_k, self.W_u_s)                  # [..., k]
        hidden_k = gate_k * up_k                          # [..., k]
        return F.linear(hidden_k, self.A_d)               # [..., d_model]


def collect_mlp_bases(model, tokenizer, texts, device, max_len=256):
    """Forward-hook each MLP to collect input cov and pre-down-input cov."""
    in_covs = {}
    pre_down_covs = {}
    mlp_modules = []

    for name, module in model.named_modules():
        if name.endswith(".mlp") and hasattr(module, "gate_proj") and hasattr(module, "down_proj"):
            mlp_modules.append((name, module))

    def make_gate_hook(n, d_in):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            if n not in in_covs:
                in_covs[n] = torch.zeros(d_in, d_in, dtype=torch.float32)
            in_covs[n] += x_flat.T @ x_flat
        return hook

    def make_down_hook(n, d_int):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            if n not in pre_down_covs:
                pre_down_covs[n] = torch.zeros(d_int, d_int, dtype=torch.float32)
            pre_down_covs[n] += x_flat.T @ x_flat
        return hook

    handles = []
    for name, mlp in mlp_modules:
        handles.append(mlp.gate_proj.register_forward_hook(
            make_gate_hook(name, mlp.gate_proj.in_features)))
        handles.append(mlp.down_proj.register_forward_hook(
            make_down_hook(name, mlp.down_proj.in_features)))

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

    in_covs = {n: c.to(torch.float64) for n, c in in_covs.items()}
    pre_down_covs = {n: c.to(torch.float64) for n, c in pre_down_covs.items()}
    return in_covs, pre_down_covs


def top_k_basis(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)


def replace_mlps(model, in_covs, pre_down_covs, rank):
    n_replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name != "mlp":
                continue
            if not (hasattr(child, "gate_proj") and hasattr(child, "down_proj")):
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name not in in_covs or full_name not in pre_down_covs:
                continue
            P_in = top_k_basis(in_covs[full_name], rank)
            P_pd = top_k_basis(pre_down_covs[full_name], rank)
            new_mlp = RotationNativeMLP(child, P_in, P_pd)
            setattr(module, child_name, new_mlp)
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


def count_mlp_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="32,64,128,256,512")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage36_rotation_native.json")
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
    teacher_mlp_params = count_mlp_params(teacher)

    print(f"\n=== teacher baseline ===")
    t0 = time.perf_counter()
    teacher_tokens = generate(teacher, tokenizer, args.prompt, args.max_new_tokens, device)
    teacher_text = tokenizer.decode(teacher_tokens, skip_special_tokens=True)
    print(f"  generated in {time.perf_counter()-t0:.1f}s")
    print(f"  {teacher_text[:150]}...")
    print(f"  teacher MLP params: {teacher_mlp_params/1e6:.1f}M")

    print(f"\n=== collecting calibration bases ===")
    t0 = time.perf_counter()
    in_covs, pre_down_covs = collect_mlp_bases(teacher, tokenizer, CALIB_TEXTS, device)
    print(f"  {len(in_covs)} MLPs, {time.perf_counter()-t0:.1f}s")

    del teacher
    if device == "mps":
        torch.mps.empty_cache()

    ranks = [int(x) for x in args.ranks.split(",")]
    results = []
    for k in ranks:
        print(f"\n=== rank {k} rotation-native student ===")
        t0 = time.perf_counter()
        student, _ = load_model(args.model, device)
        n = replace_mlps(student, in_covs, pre_down_covs, k)
        student_mlp_params = count_mlp_params(student)
        print(f"  built {n} rotation-native MLPs in {time.perf_counter()-t0:.1f}s")
        print(f"  student MLP params: {student_mlp_params/1e6:.2f}M "
              f"({student_mlp_params/teacher_mlp_params:.2%} of teacher)")

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
            "rank": k,
            "n_mlp_replaced": n,
            "student_mlp_params_M": student_mlp_params / 1e6,
            "compression": student_mlp_params / teacher_mlp_params,
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
    print(f"  {'rank':>5}  {'MLP params':>12}  {'compress':>8}  {'match':>10}  {'first div':>9}")
    for r in results:
        print(f"  {r['rank']:>5}  {r['student_mlp_params_M']:>10.2f}M  "
              f"{r['compression']:>7.1%}  {r['match']}/{r['total']:<4}  {r['first_divergence']:>9}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "teacher_mlp_params_M": teacher_mlp_params / 1e6,
            "teacher_sample": teacher_text[:400],
            "ranks": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
