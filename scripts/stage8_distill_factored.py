"""
Stage 8 — Distill a factored student from a frozen teacher.

Stage 7 showed naive basis-factoring collapses even at rank 256 because
intrinsic manifold dim (TwoNN=10) is not the same as linear rank (r90~470).
But the *model* doesn't have to factor through the linear variance basis —
we only need the factored forward to reproduce teacher behavior.

Setup:
    - Teacher: full Qwen3-0.6B, frozen, bf16.
    - Student: same model with every attention/MLP Linear replaced by a
      rank-k factored Linear (A · B), trainable in fp32.
    - Initialize A,B from PCA of teacher input activations (Stage 7).
    - Loss: KL(teacher || student) on logits over a calibration corpus.
    - Freeze embeddings, layernorms, lm_head.

If the manifold claim holds, the student should drive loss ~0 at small k
even though a non-trained factoring fails. That's the real test.

Usage:
    python scripts/stage8_distill_factored.py \\
        --model Qwen/Qwen3-0.6B \\
        --rank 32 \\
        --steps 400 \\
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


TARGET_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


CALIBRATION_TEXTS = [
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology and machine learning. It suggests that both biological and artificial neural systems exploit contextual compression to reduce computational cost, and this observation has implications for how we design future inference systems.",
    "In quantum mechanics, the wave function describes the state of a system and evolves according to the Schrodinger equation. Measurement collapses the wave function to an eigenstate of the observable, and the probabilities of outcomes are given by the squared magnitude of the amplitude.",
    "Protein folding is a process by which a polypeptide chain acquires its three-dimensional structure. Misfolded proteins can aggregate and cause diseases such as Alzheimer and Parkinson. Understanding the energy landscape of folding is a central problem in computational biology.",
    "The cosmic microwave background radiation is the thermal afterglow of the Big Bang, cooled to approximately 2.7 Kelvin by the expansion of the universe. Its discovery in 1965 provided strong evidence for cosmological models of the early universe.",
    "Markov chain Monte Carlo methods sample from complex probability distributions by constructing a chain whose stationary distribution matches the target. Metropolis-Hastings and Gibbs sampling are common variants that work for high-dimensional posteriors in Bayesian inference.",
    "The Riemann zeta function, defined as the analytic continuation of the Dirichlet series over complex numbers, encodes deep information about the distribution of prime numbers through its non-trivial zeros along the critical line.",
    "Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct. It sustains nearly all life on Earth by forming the base of most food chains, and it drives the carbon cycle.",
    "Attention mechanisms in transformers compute weighted averages over token representations, where the weights reflect contextual relevance learned during training. Multi-head attention allows parallel attention subspaces to capture different relational patterns.",
    "Plate tectonics describes the movement of Earth lithospheric plates driven by convection in the mantle. Their interactions produce earthquakes, volcanoes, mountain ranges, and ocean trenches, and they have reshaped the surface of Earth over billions of years.",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction but hard to invert, such as integer factorization or the discrete logarithm problem over elliptic curves. Modern secure communication depends on this asymmetry.",
    "Neurotransmitters like dopamine, serotonin, and glutamate mediate communication between neurons at chemical synapses. Imbalances are implicated in depression, schizophrenia, and Parkinson disease, and psychiatric drugs often target specific receptor subtypes.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases. This arrow of time emerges from the statistical behavior of microscopic states, and it constrains every physical and biological process we know.",
    "Gravitational waves are ripples in the fabric of spacetime produced by accelerating masses, predicted by general relativity and first directly detected in 2015 by LIGO. They open a new observational window on compact binary systems.",
    "Neural networks are approximators of functions learned from data by gradient descent on a loss. Their expressive power scales with depth and width, but generalization depends on inductive biases and regularization as much as raw capacity.",
    "Evolution by natural selection proceeds through variation, heredity, and differential reproduction. Genetic drift, mutation, migration, and recombination all contribute to the dynamics of allele frequencies in populations over time.",
    "In topology, a Mobius strip is a surface with only one side and one edge, constructed by joining the ends of a rectangle with a half twist. It is a classic example of a non-orientable surface and illustrates the subtleties of orientability.",
]


class BasisFactoredLinear(nn.Module):
    """y = A(Bx) + b. A: [d_out, k], B: [k, d_in]. Trainable in fp32."""

    def __init__(self, orig: nn.Linear, P_in: torch.Tensor, trainable: bool = True):
        super().__init__()
        k = P_in.shape[1]
        device = orig.weight.device

        W = orig.weight.data.to(torch.float32).cpu()
        P = P_in.to(torch.float32).cpu()
        A = (W @ P).contiguous().to(device).to(torch.float32)   # [d_out, k]
        B = P.T.contiguous().to(device).to(torch.float32)        # [k, d_in]

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
        # Cast to fp32 for stability while training, then back to input dtype
        x_dtype = x.dtype
        x32 = x.to(torch.float32)
        out = F.linear(F.linear(x32, self.B), self.A, self.bias)
        return out.to(x_dtype)


def collect_input_covariances(model, tokenizer, texts, device, max_len=256):
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
            # Accumulate on CPU — covariance for d_in=9728 is 380MB per layer,
            # which overflows MPS unified memory budget at 4B scale.
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
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()
    cpu_covs = {n: c.cpu().to(torch.float64) for n, c in covs.items()}
    return cpu_covs, counts


def top_k_basis_from_cov(cov: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k eigenvectors of PSD covariance. Full eigh on fp64 CPU."""
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
    """Everything not in a BasisFactoredLinear is frozen."""
    # First freeze all
    for p in model.parameters():
        p.requires_grad_(False)
    # Then unfreeze A, B, bias of each BasisFactoredLinear
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


HELDOUT_TEXTS = [
    "The migratory patterns of monarch butterflies span thousands of kilometres across North America, from Canada to central Mexico, over multiple generations.",
    "Topological insulators are materials that behave as insulators in their interior but conduct electricity along their surface, a consequence of spin-orbit coupling and time-reversal symmetry.",
    "Recombinant DNA technology emerged in the 1970s when researchers discovered restriction enzymes that cut DNA at specific sequences, enabling genes to be inserted into bacterial plasmids.",
    "The Antikythera mechanism, recovered from a Greek shipwreck, is an ancient analog computer dating from the second century BCE that tracked astronomical positions and eclipses.",
    "Edge-triggered flip-flops store a single bit of information and change state only on the rising or falling edge of a clock signal, making them central to digital sequential logic.",
]


def distribution_eval(teacher, student, tokenizer, texts, device):
    """Measure how close student's output distribution is to teacher's on
    held-out text, position by position (teacher-forced).

    Returns dict with:
      teacher_ppl, student_ppl              — perplexity under each model
      position_kl                            — mean KL(teacher || student) over positions
      top1_agree                             — fraction of positions where argmax matches
      top5_agree                             — fraction where student's top-1 is in teacher's top-5
      ppl_ratio                              — student_ppl / teacher_ppl (should be ~1.0 for a good match)
    """
    teacher.eval()
    student.eval()
    results = {
        "teacher_ppl": [], "student_ppl": [],
        "position_kl": [], "top1_agree": [], "top5_agree": [],
    }
    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=256).input_ids.to(device)
            if ids.shape[1] < 4:
                continue
            t_out = teacher(input_ids=ids, use_cache=False)
            s_out = student(input_ids=ids, use_cache=False)
            t_logits = t_out.logits[0, :-1].float()  # [T-1, V]
            s_logits = s_out.logits[0, :-1].float()
            targets = ids[0, 1:]  # [T-1]

            # Perplexity (exp of mean neg-log-likelihood of next token under each model)
            t_nll = -F.log_softmax(t_logits, dim=-1).gather(1, targets.unsqueeze(1)).mean()
            s_nll = -F.log_softmax(s_logits, dim=-1).gather(1, targets.unsqueeze(1)).mean()
            results["teacher_ppl"].append(float(t_nll.exp().item()))
            results["student_ppl"].append(float(s_nll.exp().item()))

            # KL(teacher || student) averaged over positions
            t_logp = F.log_softmax(t_logits, dim=-1)
            s_logp = F.log_softmax(s_logits, dim=-1)
            t_p = t_logp.exp()
            kl = (t_p * (t_logp - s_logp)).sum(dim=-1).mean()
            results["position_kl"].append(float(kl.item()))

            # Top-1 agreement
            t_top1 = t_logits.argmax(dim=-1)
            s_top1 = s_logits.argmax(dim=-1)
            results["top1_agree"].append(float((t_top1 == s_top1).float().mean().item()))

            # Student top-1 in teacher top-5
            t_top5 = t_logits.topk(5, dim=-1).indices  # [T-1, 5]
            in_top5 = (s_top1.unsqueeze(-1) == t_top5).any(dim=-1).float().mean()
            results["top5_agree"].append(float(in_top5.item()))

    summary = {k: sum(v) / max(len(v), 1) for k, v in results.items()}
    summary["ppl_ratio"] = summary["student_ppl"] / max(summary["teacher_ppl"], 1e-9)
    summary["per_text"] = results
    return summary


def generate(model, tokenizer, prompt, max_new_tokens, device, warmup=2):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token.item()]

    for _ in range(warmup):
        with torch.inference_mode():
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token.item())

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


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


def tokenize_corpus(tokenizer, texts, max_len, device):
    """Return a list of tensors [1, T] of token IDs."""
    out = []
    for t in texts:
        ids = tokenizer(t, return_tensors="pt",
                        truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] >= 16:
            out.append(ids)
    return out


def distill(teacher, student, batches, steps, lr, device, log_every=25,
            hidden_weight=1.0, kl_weight=1.0, warmup=50,
            eval_every=0, eval_fn=None, early_exit_ratio=None):
    """Train student to match teacher. Loss = KL(logits) + MSE(hidden_states)
    across all layers (layer-wise distillation gives much denser gradient)."""
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    # Cosine LR with linear warmup
    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        import math
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

            with torch.inference_mode():
                t_out = teacher(input_ids=batch, use_cache=False,
                                output_hidden_states=True)
            t_logits = t_out.logits.detach()
            t_hidden = [h.detach() for h in t_out.hidden_states]  # list[L+1] of [1,T,H]

            s_out = student(input_ids=batch, use_cache=False,
                            output_hidden_states=True)
            s_logits = s_out.logits
            s_hidden = s_out.hidden_states

            # Relative MSE per layer: (||s-t||^2 / ||t||^2). Scale-invariant,
            # so layer magnitudes can't dominate the gradient.
            h_loss = 0.0
            n_layers = len(t_hidden)
            for th, sh in zip(t_hidden, s_hidden):
                th32 = th.float()
                sh32 = sh.float()
                num = (sh32 - th32).pow(2).mean()
                denom = th32.pow(2).mean().clamp_min(1e-8)
                h_loss = h_loss + num / denom
            h_loss = h_loss / n_layers

            # KL on final logits
            Tt = 1.0
            s_logp = F.log_softmax(s_logits.float() / Tt, dim=-1)
            t_p = F.softmax(t_logits.float() / Tt, dim=-1)
            kl = F.kl_div(s_logp, t_p, reduction="batchmean") * (Tt * Tt)

            loss = hidden_weight * h_loss + kl_weight * kl

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
                    "hidden_loss": float(h_loss.item()),
                    "kl": float(kl.item()),
                    "lr": lr_at(step),
                    "elapsed_sec": elapsed,
                })

            if eval_every > 0 and eval_fn is not None and step > 0 and \
               (step % eval_every == 0 or step == steps - 1):
                student.eval()
                match, total, sample = eval_fn()
                student.train()
                ratio = match / max(total, 1)
                print(f"    [eval @ step {step}] match {match}/{total} "
                      f"= {ratio:.1%}   {sample[:80]!r}", flush=True)
                history.append({
                    "step": step,
                    "eval_match": match,
                    "eval_total": total,
                    "eval_ratio": ratio,
                    "eval_sample": sample[:200],
                })
                if early_exit_ratio is not None and ratio >= early_exit_ratio:
                    print(f"    [early exit] match {ratio:.1%} >= target "
                          f"{early_exit_ratio:.0%}", flush=True)
                    return history

            step += 1

    return history


def convert_student_to_bf16(student):
    """After training, cast A/B/bias of each BasisFactoredLinear to bf16
    for memory-bandwidth-bound decode. Also switch forward to keep bf16 in-flight."""
    for mod in student.modules():
        if isinstance(mod, BasisFactoredLinear):
            mod.A.data = mod.A.data.to(torch.bfloat16)
            mod.B.data = mod.B.data.to(torch.bfloat16)
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.to(torch.bfloat16)
            # Flip forward behavior: skip the fp32 cast for eval
            mod._bf16_eval = True

    # Replace forward with a bf16-preserving version
    def bf16_forward(self, x):
        # Keep input dtype, no fp32 upcast
        return F.linear(F.linear(x, self.B), self.A, self.bias)

    import types
    for mod in student.modules():
        if isinstance(mod, BasisFactoredLinear):
            mod.forward = types.MethodType(bf16_forward, mod)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
    p.add_argument("--early-exit", type=float, default=0.96,
                   help="Stop training when mid-eval match ratio >= this (0-1)")
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
    print(f"\ndevice: {device}   rank: {args.rank}   steps: {args.steps}")

    # === Teacher ===
    print(f"\n=== loading teacher {args.model} (bf16, frozen) ===", flush=True)
    teacher, tokenizer = load_model(args.model, device, dtype=torch.bfloat16)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    n_layers = teacher.config.num_hidden_layers
    hidden = teacher.config.hidden_size
    intermediate = teacher.config.intermediate_size

    # === Teacher baseline decode (reference) ===
    print(f"\n=== teacher decode (reference) ===", flush=True)
    t_times, t_text, t_tokens = generate(
        teacher, tokenizer, args.prompt, args.max_new_tokens, device)
    t_ms = sum(t_times) / len(t_times)
    print(f"  {len(t_times)} tokens, {t_ms:.2f}ms/tok")
    print(f"  {t_text[:120]}...")

    # === Covariance collection ===
    print(f"\n=== collecting covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, counts = collect_input_covariances(
        teacher, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    print(f"  {len(covs)} covs, {next(iter(counts.values()))} toks/layer, "
          f"{time.perf_counter()-t0:.1f}s")

    # === Student ===
    print(f"\n=== loading student, factorizing at rank {args.rank} ===", flush=True)
    student, _ = load_model(args.model, device, dtype=torch.bfloat16)
    fstats = factorize_with_basis(student, covs, rank=args.rank, trainable=True)
    size_ratio = fstats["factored_params"] / max(fstats["full_params"], 1)
    print(f"  factored {fstats['n_replaced']} linears, "
          f"{fstats['factored_params']/1e6:.2f}M params "
          f"({size_ratio:.2%} of full)")

    trainable = freeze_non_factored(student)
    print(f"  trainable params: {trainable/1e6:.2f}M")

    # === Pre-training student eval ===
    print(f"\n=== student decode (pre-training) ===", flush=True)
    student.eval()
    s0_times, s0_text, s0_tokens = generate(
        student, tokenizer, args.prompt, args.max_new_tokens, device)
    s0_ms = sum(s0_times) / len(s0_times)
    min_len = min(len(t_tokens), len(s0_tokens))
    s0_match = sum(1 for a, b in zip(t_tokens[:min_len], s0_tokens[:min_len]) if a == b)
    print(f"  {s0_ms:.2f}ms/tok, match={s0_match}/{min_len}")
    print(f"  {s0_text[:120]}...")

    # === Distill ===
    print(f"\n=== distilling ({args.steps} steps, lr={args.lr}) ===", flush=True)
    batches = tokenize_corpus(tokenizer, CALIBRATION_TEXTS, args.calib_max_len, device)
    print(f"  {len(batches)} batches")

    # Mid-training eval closure: generate 80 tokens, compare to teacher tokens
    eval_max = 80
    teacher_eval_tokens = t_tokens[:eval_max]
    def eval_fn():
        _, s_text, s_tokens = generate(student, tokenizer, args.prompt,
                                        eval_max, device, warmup=0)
        ml = min(len(teacher_eval_tokens), len(s_tokens))
        m = sum(1 for a, b in zip(teacher_eval_tokens[:ml], s_tokens[:ml]) if a == b)
        return m, ml, s_text

    history = distill(teacher, student, batches, args.steps, args.lr, device,
                      eval_every=max(args.steps // 6, 500), eval_fn=eval_fn,
                      log_every=max(args.steps // 60, 25),
                      early_exit_ratio=args.early_exit)

    # === Distribution-based eval (physics-correct metric, pre-bf16-convert) ===
    print(f"\n=== distribution eval on held-out text (fp32 student) ===", flush=True)
    dist_eval = distribution_eval(teacher, student, tokenizer, HELDOUT_TEXTS, device)
    print(f"  teacher ppl: {dist_eval['teacher_ppl']:.3f}")
    print(f"  student ppl: {dist_eval['student_ppl']:.3f}")
    print(f"  ppl ratio:   {dist_eval['ppl_ratio']:.3f}  (1.0 = perfect)")
    print(f"  mean KL(T||S): {dist_eval['position_kl']:.4f}")
    print(f"  top-1 agreement: {dist_eval['top1_agree']:.1%}")
    print(f"  student top-1 in teacher top-5: {dist_eval['top5_agree']:.1%}")

    # === Post-training student eval (fp32 A/B) ===
    print(f"\n=== student decode (post-training, fp32) ===", flush=True)
    student.eval()
    _ = generate(student, tokenizer, args.prompt, 20, device)  # warmup

    # Convert student weights to bf16 for bandwidth-bound decode
    print(f"\n=== converting student to bf16 for deployment ===", flush=True)
    convert_student_to_bf16(student)

    print(f"\n=== student decode (post-training, bf16) ===", flush=True)
    s1_times, s1_text, s1_tokens = generate(
        student, tokenizer, args.prompt, args.max_new_tokens, device)
    s1_ms = sum(s1_times) / len(s1_times)
    min_len = min(len(t_tokens), len(s1_tokens))
    s1_match = sum(1 for a, b in zip(t_tokens[:min_len], s1_tokens[:min_len]) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, s1_tokens)) if a != b), min_len)
    speedup = t_ms / s1_ms if s1_ms > 0 else 0
    print(f"  {s1_ms:.2f}ms/tok, match={s1_match}/{min_len} "
          f"(first divergence @ {first_div})")
    print(f"  speedup vs teacher: {speedup:.2f}x")
    print(f"  {s1_text[:200]}...")

    # === Save ===
    out_path = Path(args.out_dir) / (
        f"stage8_distill_r{args.rank}_{args.model.replace('/', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "rank": args.rank,
            "steps": args.steps,
            "lr": args.lr,
            "n_layers": n_layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "weight_params_full_M": fstats["full_params"] / 1e6,
            "weight_params_factored_M": fstats["factored_params"] / 1e6,
            "weight_size_ratio": size_ratio,
            "trainable_params_M": trainable / 1e6,
            "teacher_ms_per_tok": t_ms,
            "teacher_sample": t_text[:400],
            "student_pre_ms_per_tok": s0_ms,
            "student_pre_match": f"{s0_match}/{min_len}",
            "student_pre_sample": s0_text[:400],
            "student_post_ms_per_tok": s1_ms,
            "student_post_match": f"{s1_match}/{min_len}",
            "student_post_first_divergence": first_div,
            "student_post_speedup_vs_teacher": speedup,
            "student_post_sample": s1_text[:400],
            "loss_history": history,
            "distribution_eval": {
                k: v for k, v in dist_eval.items() if k != "per_text"
            },
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
