"""
Stage 14 — Teacher-sampled calibration + TwoNN structural check.

Stage 13 established that scaling calibration from 733 to 2875 tokens
improves held-out ppl ratio 9x (922 -> 99). The manifold-copying frame
predicts this continues: each calibration sample is one angle of the
hologram; reconstruction quality scales with angular density.

This stage:
    1. Seeds the teacher with diverse short prompts and lets IT generate
       continuations. The concatenated teacher-output text becomes the
       calibration corpus. This samples the teacher's actual manifold at
       the points it naturally traverses (on-policy sampling).
    2. Targets ~100k calibration tokens (~35x stage 13).
    3. Distills a rank-k student and evaluates distribution metrics on a
       DISJOINT held-out text set.
    4. Measures per-layer TwoNN intrinsic dimension of student vs teacher
       to check the topological invariant. If TwoNN drifts, the
       compression broke the manifold; if it stays put, we've preserved
       the topological shape.

Usage:
    python scripts/stage14_teacher_sampled.py \\
        --model Qwen/Qwen3-0.6B --rank 32 --target-calib-tokens 60000 \\
        --steps 5000 --device mps
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

from src.common.model_loader import describe_backend


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


# Diverse seed prompts spanning many domains. Teacher continues each.
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
    "Democracy is a system in which",
    "The periodic table organizes elements by",
    "Music theory describes relationships between",
    "Epidemics spread through populations via",
    "The nervous system transmits signals using",
    "Renewable energy includes sources like",
    "Genetics is the study of",
    "The speed of light is constant because",
    "Bacteria differ from viruses in that",
    "Plate tectonics causes geological activity by",
    "The atom was discovered to be divisible when",
    "Thermodynamics governs how",
    "Game theory analyzes situations where",
    "The digestive system breaks down food by",
    "Stars produce energy through",
    "Memory in the brain relies on",
    "Operating systems manage resources by",
    "In ecology, food webs describe",
    "The Roman alphabet derives from",
    "Orbital mechanics describes the motion of",
    "Mitosis is the process by which cells",
    "Electric motors convert energy by",
    "In psychology, cognition refers to",
    "The theory of evolution was shaped by",
    "Photons are quantum particles of",
    "Democracy and freedom of the press are connected because",
    "Water molecules are polar because",
    "The internet works by",
    "Astronomy studies celestial objects by",
    "Chemical reactions can be categorized as",
    "Sleep is essential for health because",
    "The Silk Road connected",
    "Mathematics proves theorems by",
    "Neurons communicate across synapses using",
    "Glaciers form when",
    "Shakespeare wrote in an era when",
    "Programming languages are designed to",
    "Earthquakes occur because",
    "The Enlightenment was characterized by",
    "Chemistry explains how matter",
    "The wheel changed human society because",
    "In biology, homeostasis means",
    "Robots can perform tasks by",
    "Newton's laws describe",
    "Viruses reproduce by",
    "The French Revolution was triggered by",
    "In statistics, a normal distribution",
    "The moon affects Earth by",
    "Music genres evolved because",
    "The digestive tract absorbs nutrients at",
    "Black-body radiation is",
    "Satellites orbit Earth because",
    "The ocean floor has features like",
    "In physics, entropy measures",
    "Agriculture began when",
    "Electromagnetic waves include",
    "Yeast is used in baking because",
    "Volcanoes form at",
    "The brain is divided into regions such as",
    "Public-key cryptography relies on",
    "Mountains form through",
    "The immune system recognizes pathogens by",
    "Vaccines work by",
    "Genetic mutations can arise from",
    "In chemistry, an acid is",
    "Renewable resources include",
    "A democracy depends on",
    "Fossil fuels formed over",
    "In geometry, a circle is",
    "Computers solve problems by",
    "The seasons are caused by",
    "Migration of species happens when",
    "Caves form through",
    "Hurricanes gain strength over",
    "In linguistics, phonemes are",
    "The auditory system detects sound by",
]


HELDOUT_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres across North America, from Canada to central Mexico, over multiple generations.",
    "Topological insulators are materials that behave as insulators in their interior but conduct electricity along their surface, a consequence of spin-orbit coupling and time-reversal symmetry.",
    "Recombinant DNA technology emerged in the 1970s when researchers discovered restriction enzymes that cut DNA at specific sequences, enabling genes to be inserted into bacterial plasmids.",
    "The Antikythera mechanism, recovered from a Greek shipwreck, is an ancient analog computer dating from the second century BCE that tracked astronomical positions and eclipses.",
    "Edge-triggered flip-flops store a single bit of information and change state only on the rising or falling edge of a clock signal, making them central to digital sequential logic.",
    "The Curie temperature is the point above which a ferromagnetic material loses its permanent magnetic properties as thermal agitation disrupts the alignment of atomic dipoles.",
    "Vector clocks extend Lamport timestamps to detect concurrency relationships between events in distributed systems, making causal ordering observable without global clocks.",
    "Germanium is a lustrous, hard, grayish-white metalloid in the carbon group, chemically similar to its group neighbours silicon and tin, and widely used in fiber-optic systems.",
]


def teacher_sample_corpus(model, tokenizer, prompts, target_tokens, max_gen, device):
    """Run teacher on each seed prompt, generate max_gen tokens greedy.
    Return list of token-id tensors, aggregated until target_tokens reached."""
    print(f"  generating teacher continuations (target {target_tokens} toks)...",
          flush=True)
    out = []
    total = 0
    for i, seed in enumerate(prompts):
        ids = tokenizer(seed, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            gen = model.generate(
                ids, max_new_tokens=max_gen, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.eos_token_id or 0,
            )
        # gen shape: [1, prompt+new]
        out.append(gen)
        total += gen.shape[1]
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(prompts)}  ~{total} toks", flush=True)
        if total >= target_tokens:
            break
    print(f"  done: {len(out)} sequences, {total} total tokens", flush=True)
    return out


def chunk_batches(id_tensors, max_len):
    """Split long id tensors into chunks of up to max_len."""
    out = []
    for ids in id_tensors:
        T = ids.shape[1]
        for start in range(0, T, max_len):
            chunk = ids[:, start:start + max_len]
            if chunk.shape[1] >= 16:
                out.append(chunk)
    return out


# ----- BasisFactoredLinear, covariance collection, distillation -----
# (Copied from stage 13 with minor tweaks)

class BasisFactoredLinear(nn.Module):
    def __init__(self, orig: nn.Linear, P_in: torch.Tensor, trainable: bool = True):
        super().__init__()
        k = P_in.shape[1]
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
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.rank = k
        self._full_params = orig.in_features * orig.out_features
        self._factored_params = k * (orig.in_features + orig.out_features)

    def forward(self, x):
        dt = x.dtype
        x32 = x.to(torch.float32)
        out = F.linear(F.linear(x32, self.B), self.A, self.bias)
        return out.to(dt)


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


def top_k_basis_from_cov(cov: torch.Tensor, k: int) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k_eff = min(k, eigvecs.shape[1])
    return eigvecs[:, -k_eff:].flip(dims=[1]).contiguous()


def factorize_with_basis(model, covariances, rank: int, trainable: bool = True):
    stats = {"n_replaced": 0, "full_params": 0, "factored_params": 0}
    bases = {n: top_k_basis_from_cov(c, rank) for n, c in covariances.items()}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in TARGET_NAMES:
                continue
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name not in bases:
                continue
            P = bases[full_name].to(torch.float32)
            fact = BasisFactoredLinear(child, P_in=P, trainable=trainable)
            setattr(module, child_name, fact)
            stats["n_replaced"] += 1
            stats["full_params"] += fact._full_params
            stats["factored_params"] += fact._factored_params
    return stats


def freeze_non_factored(model):
    for p in model.parameters():
        p.requires_grad_(False)
    trainable_params = 0
    for mod in model.modules():
        if isinstance(mod, BasisFactoredLinear):
            mod.A.requires_grad_(True)
            mod.B.requires_grad_(True)
            if mod.bias is not None:
                mod.bias.requires_grad_(True)
            trainable_params += mod.A.numel() + mod.B.numel()
            if mod.bias is not None:
                trainable_params += mod.bias.numel()
    return trainable_params


def distill(teacher, student, batches, steps, lr, device, log_every=100, warmup=100):
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
    while step < steps:
        for batch in batches:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
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
                print(f"  step {step:5d}  loss={loss.item():.4f}  "
                      f"(h={h_loss.item():.4f} kl={kl.item():.4f})  "
                      f"lr={lr_at(step):.2e}  ({elapsed:.1f}s)", flush=True)
                history.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "kl": float(kl.item()),
                    "h_loss": float(h_loss.item()),
                    "elapsed_sec": elapsed,
                })
            step += 1
    return history


def twonn_dimension(X: torch.Tensor) -> float:
    """TwoNN estimator (Facco et al. 2017). X: [N, d] fp32. Returns scalar."""
    X = X.to(torch.float64)
    # pairwise distances
    dists = torch.cdist(X, X)
    # set self-distance to inf
    dists.fill_diagonal_(float("inf"))
    # first two nearest neighbors per point
    top2, _ = dists.topk(2, dim=1, largest=False)
    r1 = top2[:, 0]
    r2 = top2[:, 1]
    mask = r1 > 1e-10
    if mask.sum() < 10:
        return float("nan")
    mu = (r2[mask] / r1[mask]).clamp_min(1.0 + 1e-10)
    log_mu = torch.log(mu)
    d_hat = 1.0 / log_mu.mean().item()
    return d_hat


def twonn_per_layer(model, batches, device, sample_limit=2000):
    """Run model on batches with output_hidden_states=True, collect activations
    per layer, compute TwoNN on a random sample of up to sample_limit points."""
    model.eval()
    collected = None
    total = 0
    with torch.inference_mode():
        for batch in batches:
            batch = batch.to(device)
            out = model(input_ids=batch, use_cache=False, output_hidden_states=True)
            hs = out.hidden_states  # tuple [L+1] of [1, T, d]
            if collected is None:
                collected = [[] for _ in range(len(hs))]
            for i, h in enumerate(hs):
                collected[i].append(h[0].to(torch.float32).cpu())
            total += batch.shape[1]
            if total >= sample_limit:
                break
    results = []
    for i, xs in enumerate(collected):
        X = torch.cat(xs, dim=0)
        # Subsample uniformly to reduce O(N^2) cost of cdist
        if X.shape[0] > sample_limit:
            idx = torch.randperm(X.shape[0])[:sample_limit]
            X = X[idx]
        d = twonn_dimension(X)
        results.append(d)
    return results


def distribution_eval(teacher, student, tokenizer, texts, device):
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

    out = {k: sum(v) / max(len(v), 1) for k, v in res.items()}
    out["ppl_ratio"] = out["student_ppl"] / max(out["teacher_ppl"], 1e-9)
    return out


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default=None)
    p.add_argument("--target-calib-tokens", type=int, default=60000)
    p.add_argument("--teacher-max-gen", type=int, default=300,
                   help="New tokens teacher generates per seed prompt")
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--twonn-sample-limit", type=int, default=2000)
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
    print(f"\ndevice={device}  rank={args.rank}  steps={args.steps}")

    print(f"\n=== loading teacher {args.model} ===", flush=True)
    teacher, tokenizer = load_model(args.model, device)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    print(f"\n=== teacher-sampled calibration ===", flush=True)
    calib_tensors = teacher_sample_corpus(
        teacher, tokenizer, SEED_PROMPTS,
        args.target_calib_tokens, args.teacher_max_gen, device)
    batches = chunk_batches(calib_tensors, args.calib_max_len)
    total_calib_tokens = sum(b.shape[1] for b in batches)
    print(f"  {len(batches)} chunks, {total_calib_tokens} tokens total")

    print(f"\n=== teacher TwoNN per layer (reference) ===", flush=True)
    t0 = time.perf_counter()
    teacher_twonn = twonn_per_layer(teacher, batches, device, args.twonn_sample_limit)
    print(f"  {time.perf_counter()-t0:.1f}s")
    print(f"  layer  0: {teacher_twonn[0]:.2f}")
    mid = len(teacher_twonn) // 2
    print(f"  layer {mid}: {teacher_twonn[mid]:.2f}")
    print(f"  layer {len(teacher_twonn)-1}: {teacher_twonn[-1]:.2f}")

    print(f"\n=== collecting covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, counts = collect_input_covariances(teacher, batches, device)
    print(f"  {len(covs)} covs, {next(iter(counts.values()))} toks/layer, "
          f"{time.perf_counter()-t0:.1f}s")

    print(f"\n=== factorizing student ===", flush=True)
    student, _ = load_model(args.model, device)
    fstats = factorize_with_basis(student, covs, rank=args.rank, trainable=True)
    ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
    print(f"  factored {fstats['n_replaced']} linears, "
          f"{fstats['factored_params']/1e6:.2f}M params ({ratio:.2%})")
    trainable = freeze_non_factored(student)

    print(f"\n=== distribution eval PRE-training ===", flush=True)
    pre = distribution_eval(teacher, student, tokenizer, HELDOUT_TEXTS, device)
    print(f"  ppl_ratio={pre['ppl_ratio']:.2f}  kl={pre['position_kl']:.3f}  "
          f"top1={pre['top1_agree']:.1%}")

    print(f"\n=== distilling ===", flush=True)
    history = distill(teacher, student, batches, args.steps, args.lr, device)

    print(f"\n=== distribution eval POST-training ===", flush=True)
    post = distribution_eval(teacher, student, tokenizer, HELDOUT_TEXTS, device)
    print(f"  teacher ppl: {post['teacher_ppl']:.3f}")
    print(f"  student ppl: {post['student_ppl']:.3f}")
    print(f"  ppl ratio:   {post['ppl_ratio']:.3f}  (1.0 = perfect)")
    print(f"  mean KL: {post['position_kl']:.4f}")
    print(f"  top-1 agreement: {post['top1_agree']:.1%}")
    print(f"  top-5 agreement: {post['top5_agree']:.1%}")

    print(f"\n=== student TwoNN per layer POST-training ===", flush=True)
    t0 = time.perf_counter()
    student_twonn = twonn_per_layer(student, batches, device, args.twonn_sample_limit)
    print(f"  {time.perf_counter()-t0:.1f}s")
    # Compare
    print(f"  {'layer':>5} {'teacher':>8} {'student':>8} {'ratio':>6}")
    for i in range(0, len(teacher_twonn), max(1, len(teacher_twonn)//10)):
        print(f"  {i:>5} {teacher_twonn[i]:>8.2f} {student_twonn[i]:>8.2f} "
              f"{student_twonn[i]/max(teacher_twonn[i], 1e-6):>6.2f}")

    # Manifold-preservation metric: mean abs difference in TwoNN
    diffs = [abs(a - b) for a, b in zip(teacher_twonn, student_twonn)]
    mean_twonn_diff = sum(diffs) / len(diffs)
    print(f"  mean |Δ TwoNN|: {mean_twonn_diff:.3f}")

    out_path = Path(args.out_dir) / f"stage14_teacher_sampled_{args.model.replace('/', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "rank": args.rank,
            "steps": args.steps,
            "calibration_chunks": len(batches),
            "calibration_tokens": total_calib_tokens,
            "weight_size_ratio": ratio,
            "trainable_params_M": trainable / 1e6,
            "teacher_twonn": teacher_twonn,
            "student_twonn": student_twonn,
            "mean_twonn_abs_diff": mean_twonn_diff,
            "distribution_eval_pre": pre,
            "distribution_eval_post": post,
            "loss_history": history,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
