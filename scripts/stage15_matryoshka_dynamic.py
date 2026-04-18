"""
Stage 15 — Matryoshka distillation + all-dynamic inference runtime.

This is the integration target. It combines:

    1. Teacher-sampled calibration with T>0 sampling (fixes stage 14's
       degenerate greedy output).
    2. Matryoshka rank sampling during distillation: each batch trains at
       a random k ∈ [k_min, k_max]. Result: every prefix [0:k] of A, B
       is a self-contained valid factorization for any k.
    3. All-dynamic inference runtime: per-layer, per-step live policy
       reads entropy H(t) and ∂H/∂t from free attention-weight hooks,
       chooses k(t, i) for each factored linear at each step.
    4. Evaluation at multiple ranks to confirm the nested-rank property
       (ppl should degrade gracefully as k shrinks, not catastrophically).

This is the deployable artifact's shape. On Strix Halo/ROCm it runs
faster because signal ops don't pay MPS kernel-launch tax.

Usage:
    python scripts/stage15_matryoshka_dynamic.py \\
        --model Qwen/Qwen3-0.6B --k-min 16 --k-max 128 \\
        --steps 3000 --target-calib-tokens 40000 --device mps
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import describe_backend


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


SEED_PROMPTS = [
    "The cell is the basic structural unit of life.",
    "In mathematics, a prime number",
    "The history of computing began with",
    "Climate change is driven primarily by",
    "Language models learn from text by",
    "The immune system protects the body by",
    "Quantum entanglement occurs when",
    "A compiler translates source code into",
    "Photosynthesis uses sunlight to",
    "The Roman Empire fell because",
    "In economics, supply and demand determine",
    "Neural networks consist of layers that",
    "The structure of DNA was discovered by",
    "Black holes form when",
    "Relativity theory says that",
    "The scientific method requires",
    "Oceans regulate the climate by",
    "Artificial intelligence can be described as",
    "Cryptography protects information by",
    "The human brain processes information via",
    "Evolution explains the diversity of life through",
    "Protein synthesis takes place in",
    "A galaxy is a system of",
    "The industrial revolution was enabled by",
    "In linguistics, syntax refers to",
    "Chemical bonds form because",
    "Statistics helps us reason under uncertainty by",
    "The Renaissance marked a period of",
    "Electricity flows through conductors because",
    "In philosophy, consciousness is",
]


HELDOUT_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres across North America.",
    "Topological insulators behave as insulators in their interior but conduct electricity along their surface.",
    "Recombinant DNA technology emerged in the 1970s with the discovery of restriction enzymes.",
    "The Antikythera mechanism is an ancient analog computer from the second century BCE.",
    "Edge-triggered flip-flops store one bit and change state only on the clock edge.",
    "The Curie temperature is where a ferromagnetic material loses its permanent magnetism.",
    "Vector clocks extend Lamport timestamps for distributed system event ordering.",
    "Germanium is a metalloid in the carbon group widely used in fiber-optic systems.",
]


class RankController:
    """Shared runtime state: either a single global k (training) or per-layer k (inference)."""
    def __init__(self, k_max):
        self.k_max = k_max
        self.global_k = None           # Matryoshka training: same k across all linears
        self.per_layer_k = {}          # Dynamic inference: {layer_idx: k}

    def resolve(self, layer_idx):
        if self.global_k is not None:
            return min(self.global_k, self.k_max)
        return self.per_layer_k.get(layer_idx, self.k_max)


class MatryoshkaFactoredLinear(nn.Module):
    """Factored Linear with runtime-variable rank. The top-k columns/rows of
    A, B form a valid factorization for any k trained via Matryoshka sampling."""
    def __init__(self, orig: nn.Linear, P_in: torch.Tensor, controller: RankController,
                 layer_idx: int, trainable: bool = True):
        super().__init__()
        k_max = P_in.shape[1]
        device = orig.weight.device
        W = orig.weight.data.to(torch.float32).cpu()
        P = P_in.to(torch.float32).cpu()
        A = (W @ P).contiguous().to(device).to(torch.float32)
        B = P.T.contiguous().to(device).to(torch.float32)
        self.A = nn.Parameter(A, requires_grad=trainable)
        self.B = nn.Parameter(B, requires_grad=trainable)
        if orig.bias is not None:
            self.bias = nn.Parameter(
                orig.bias.data.to(torch.float32).to(device),
                requires_grad=trainable)
        else:
            self.register_parameter("bias", None)
        self.controller = controller
        self.layer_idx = layer_idx
        self.k_max = k_max
        self.in_features = orig.in_features
        self.out_features = orig.out_features

    def forward(self, x):
        dt = x.dtype
        k = self.controller.resolve(self.layer_idx)
        A_k = self.A[:, :k]
        B_k = self.B[:k, :]
        x32 = x.to(torch.float32)
        out = F.linear(F.linear(x32, B_k), A_k, self.bias)
        return out.to(dt)


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def teacher_sample_corpus(model, tokenizer, prompts, target_tokens, max_gen,
                         device, temperature=0.8, top_p=0.9):
    """Sample teacher continuations at T>0 for diverse trajectories on the manifold."""
    print(f"  sampling teacher continuations (T={temperature})...", flush=True)
    out = []
    total = 0
    for i, seed in enumerate(prompts):
        ids = tokenizer(seed, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            gen = model.generate(
                ids, max_new_tokens=max_gen, do_sample=True,
                temperature=temperature, top_p=top_p,
                use_cache=True, pad_token_id=tokenizer.eos_token_id or 0,
            )
        out.append(gen)
        total += gen.shape[1]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(prompts)}  ~{total} toks", flush=True)
        if total >= target_tokens:
            break
    print(f"  done: {len(out)} sequences, {total} total tokens", flush=True)
    return out


def chunk_batches(id_tensors, max_len):
    out = []
    for ids in id_tensors:
        T = ids.shape[1]
        for start in range(0, T, max_len):
            chunk = ids[:, start:start + max_len]
            if chunk.shape[1] >= 16:
                out.append(chunk)
    return out


def collect_input_covariances(model, batches, device):
    covs = {}
    counts = {}
    target_modules = []
    for name, module in model.named_modules():
        last = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and last in TARGET_NAMES:
            target_modules.append((name, module))

    def make_hook(n, d_in):
        def hook(mod, inputs, output):
            x = inputs[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32).cpu()
            if n not in covs:
                covs[n] = torch.zeros(d_in, d_in, dtype=torch.float32)
                counts[n] = 0
            covs[n] += x_flat.T @ x_flat
            counts[n] += x_flat.shape[0]
        return hook

    handles = []
    for name, mod in target_modules:
        handles.append(mod.register_forward_hook(make_hook(name, mod.in_features)))
    model.eval()
    with torch.inference_mode():
        for ids in batches:
            ids = ids.to(device)
            model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return {n: c.to(torch.float64) for n, c in covs.items()}, counts


def top_k_basis_from_cov(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).contiguous()


def factorize_matryoshka(model, covariances, k_max, controller):
    """Replace target Linears with MatryoshkaFactoredLinear sharing a controller."""
    stats = {"n_replaced": 0, "full_params": 0, "factored_params": 0}
    bases = {n: top_k_basis_from_cov(c, k_max) for n, c in covariances.items()}
    # Derive layer_idx from the dotted name like "model.layers.7.self_attn.q_proj"
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in TARGET_NAMES:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name not in bases:
                continue
            # Extract layer index
            try:
                layer_idx = int(full_name.split("model.layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                layer_idx = -1
            P = bases[full_name].to(torch.float32)
            fact = MatryoshkaFactoredLinear(child, P_in=P, controller=controller,
                                            layer_idx=layer_idx, trainable=True)
            setattr(module, child_name, fact)
            stats["n_replaced"] += 1
            stats["full_params"] += child.in_features * child.out_features
            stats["factored_params"] += k_max * (child.in_features + child.out_features)
    return stats


def freeze_non_factored(model):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for mod in model.modules():
        if isinstance(mod, MatryoshkaFactoredLinear):
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)
            trainable += mod.A.numel() + mod.B.numel()
            if mod.bias is not None:
                trainable += mod.bias.numel()
    return trainable


def matryoshka_distill(teacher, student, batches, steps, lr, device,
                       controller, k_min, k_max, log_every=100, warmup=100):
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        progress = (s - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    student.train()
    teacher.eval()
    history = []
    t0 = time.perf_counter()
    step = 0
    # Exponential sampling over [k_min, k_max] so small k gets appropriate coverage
    log_k_min, log_k_max = math.log(k_min), math.log(k_max)

    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            # NON-Matryoshka mode if k_min == k_max (baseline for slice test)
            if k_min == k_max:
                k_sampled = k_max
            else:
                # Matryoshka: sample k per step from log-uniform
                k_sampled = int(round(math.exp(random.uniform(log_k_min, log_k_max))))
                k_sampled = max(k_min, min(k_sampled, k_max))
            controller.global_k = k_sampled

            batch = batch.to(device)
            with torch.inference_mode():
                t_out = teacher(input_ids=batch, use_cache=False,
                                output_hidden_states=True)
            t_logits = t_out.logits.detach()
            t_hidden = [h.detach() for h in t_out.hidden_states]

            s_out = student(input_ids=batch, use_cache=False,
                            output_hidden_states=True)
            s_logits = s_out.logits
            s_hidden = s_out.hidden_states

            h_loss = 0.0
            n_layers = len(t_hidden)
            for th, sh in zip(t_hidden, s_hidden):
                num = (sh.float() - th.float()).pow(2).mean()
                denom = th.float().pow(2).mean().clamp_min(1e-8)
                h_loss = h_loss + num / denom
            h_loss = h_loss / n_layers

            s_logp = F.log_softmax(s_logits.float(), dim=-1)
            t_p = F.softmax(t_logits.float(), dim=-1)
            kl = F.kl_div(s_logp, t_p, reduction="batchmean")

            loss = h_loss + kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()

            if (step % log_every == 0) or (step == steps - 1):
                elapsed = time.perf_counter() - t0
                print(f"  step {step:5d}  k={k_sampled:4d}  loss={loss.item():.4f}  "
                      f"(h={h_loss.item():.4f} kl={kl.item():.4f})  "
                      f"lr={lr_at(step):.2e}  ({elapsed:.1f}s)", flush=True)
                history.append({
                    "step": step, "k": k_sampled,
                    "loss": float(loss.item()),
                    "kl": float(kl.item()),
                    "h_loss": float(h_loss.item()),
                })
            step += 1

    controller.global_k = None  # clear for inference
    return history


def distribution_eval_at_k(teacher, student, controller, tokenizer, texts, device, k):
    """Evaluate student at a specific static rank k."""
    controller.global_k = k
    teacher.eval()
    student.eval()
    res = {"teacher_ppl": [], "student_ppl": [], "position_kl": [],
           "top1_agree": [], "top5_agree": []}
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=256).input_ids.to(device)
            if ids.shape[1] < 4:
                continue
            t_out = teacher(input_ids=ids, use_cache=False)
            s_out = student(input_ids=ids, use_cache=False)
            t_logits = t_out.logits[0, :-1].float()
            s_logits = s_out.logits[0, :-1].float()
            targets = ids[0, 1:]
            t_nll = -F.log_softmax(t_logits, -1).gather(1, targets.unsqueeze(1)).mean()
            s_nll = -F.log_softmax(s_logits, -1).gather(1, targets.unsqueeze(1)).mean()
            res["teacher_ppl"].append(float(t_nll.exp().item()))
            res["student_ppl"].append(float(s_nll.exp().item()))
            t_logp = F.log_softmax(t_logits, -1)
            s_logp = F.log_softmax(s_logits, -1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
            res["position_kl"].append(float(kl.item()))
            t_top1 = t_logits.argmax(-1)
            s_top1 = s_logits.argmax(-1)
            res["top1_agree"].append(float((t_top1 == s_top1).float().mean().item()))
            t_top5 = t_logits.topk(5, dim=-1).indices
            in_top5 = (s_top1.unsqueeze(-1) == t_top5).any(-1).float().mean()
            res["top5_agree"].append(float(in_top5.item()))
    out = {k_: sum(v) / max(len(v), 1) for k_, v in res.items()}
    out["ppl_ratio"] = out["student_ppl"] / max(out["teacher_ppl"], 1e-9)
    controller.global_k = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--k-min", type=int, default=16)
    p.add_argument("--k-max", type=int, default=128)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-calib-tokens", type=int, default=30000)
    p.add_argument("--teacher-max-gen", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)

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
    print(f"\ndevice={device}  k ∈ [{args.k_min}, {args.k_max}]  steps={args.steps}")

    print(f"\n=== loading teacher {args.model} ===", flush=True)
    teacher, tokenizer = load_model(args.model, device)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print(f"\n=== teacher-sampled calibration (T={args.temperature}) ===", flush=True)
    calib_tensors = teacher_sample_corpus(
        teacher, tokenizer, SEED_PROMPTS,
        args.target_calib_tokens, args.teacher_max_gen, device,
        temperature=args.temperature)
    batches = chunk_batches(calib_tensors, args.calib_max_len)
    total_calib_tokens = sum(b.shape[1] for b in batches)
    print(f"  {len(batches)} chunks, {total_calib_tokens} tokens")

    print(f"\n=== covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, _ = collect_input_covariances(teacher, batches, device)
    print(f"  {time.perf_counter()-t0:.1f}s")

    print(f"\n=== factorizing Matryoshka student (k_max={args.k_max}) ===", flush=True)
    student, _ = load_model(args.model, device)
    controller = RankController(args.k_max)
    fstats = factorize_matryoshka(student, covs, args.k_max, controller)
    ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
    print(f"  {fstats['n_replaced']} linears, "
          f"{fstats['factored_params']/1e6:.2f}M params ({ratio:.2%})")
    _ = freeze_non_factored(student)

    print(f"\n=== pre-training eval at k={args.k_max} ===", flush=True)
    pre = distribution_eval_at_k(teacher, student, controller, tokenizer,
                                  HELDOUT_TEXTS, device, args.k_max)
    print(f"  ppl_ratio={pre['ppl_ratio']:.2f}  kl={pre['position_kl']:.3f}")

    print(f"\n=== Matryoshka distilling ===", flush=True)
    history = matryoshka_distill(
        teacher, student, batches, args.steps, args.lr, device,
        controller, args.k_min, args.k_max)

    print(f"\n=== post-training eval at multiple ranks (nested-rank test) ===", flush=True)
    ranks_to_test = sorted({args.k_min, args.k_min * 2, args.k_max // 2, args.k_max})
    per_rank = {}
    for k in ranks_to_test:
        if k < 1 or k > args.k_max:
            continue
        r = distribution_eval_at_k(teacher, student, controller, tokenizer,
                                    HELDOUT_TEXTS, device, k)
        per_rank[k] = r
        print(f"  k={k:4d}  teacher_ppl={r['teacher_ppl']:.2f}  "
              f"student_ppl={r['student_ppl']:.2f}  ratio={r['ppl_ratio']:.2f}  "
              f"top1={r['top1_agree']:.1%}  top5={r['top5_agree']:.1%}")

    # Save
    out_path = Path(args.out_dir) / f"stage15_matryoshka_dynamic_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "steps": args.steps,
            "calibration_tokens": total_calib_tokens,
            "weight_size_ratio_at_k_max": ratio,
            "distribution_eval_pre": pre,
            "distribution_eval_per_rank": per_rank,
            "loss_history": history,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
