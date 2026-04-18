"""
Stage 9 — Layer-by-layer distillation of a factored student.

Stage 8 (end-to-end) struggles because backprop through 28 layers of
untrained factored weights gives a noisy, compounding gradient. This
stage trains each layer independently:

    input  = teacher's hidden state entering layer i
    target = teacher's hidden state exiting layer i
    loss   = MSE(student_layer_i(input), target)

Each layer is a well-posed local regression. Fast, stable, per-layer
loss is diagnostic: if layer 5 won't fit at rank 32 but others do, we
know to bump rank for that layer.

After all layers are trained, assemble into a full student and
evaluate end-to-end generation + wall-clock.

Usage:
    python scripts/stage9_layerwise_distill.py \\
        --model Qwen/Qwen3-0.6B \\
        --rank 32 \\
        --steps-per-layer 400 \\
        --device mps
"""

import argparse
import copy
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
    "The discovery that inference accelerates with context is a significant finding in cognitive psychology and machine learning. It suggests that both biological and artificial neural systems exploit contextual compression to reduce computational cost.",
    "In quantum mechanics, the wave function describes the state of a system and evolves according to the Schrodinger equation. Measurement collapses the wave function to an eigenstate of the observable.",
    "Protein folding is a process by which a polypeptide chain acquires its three-dimensional structure. Misfolded proteins can aggregate and cause diseases such as Alzheimer and Parkinson.",
    "The cosmic microwave background radiation is the thermal afterglow of the Big Bang, cooled to approximately 2.7 Kelvin by the expansion of the universe.",
    "Markov chain Monte Carlo methods sample from complex probability distributions by constructing a chain whose stationary distribution matches the target.",
    "The Riemann zeta function encodes deep information about the distribution of prime numbers through its non-trivial zeros along the critical line.",
    "Photosynthesis converts light energy into chemical energy stored in glucose, releasing oxygen as a byproduct. It sustains nearly all life on Earth.",
    "Attention mechanisms in transformers compute weighted averages over token representations, where the weights reflect contextual relevance learned during training.",
    "Plate tectonics describes the movement of Earth lithospheric plates driven by convection in the mantle. Their interactions produce earthquakes and volcanoes.",
    "Public-key cryptography relies on mathematical problems that are easy to compute in one direction but hard to invert, such as integer factorization or elliptic-curve discrete logarithms.",
    "Neurotransmitters like dopamine, serotonin, and glutamate mediate communication between neurons at chemical synapses, and imbalances are implicated in several disorders.",
    "The second law of thermodynamics states that the entropy of an isolated system never decreases. This arrow of time emerges from the statistical behavior of microscopic states.",
    "Gravitational waves are ripples in spacetime produced by accelerating masses, predicted by general relativity and first directly detected in 2015 by LIGO.",
    "Neural networks are approximators of functions learned from data by gradient descent on a loss. Their expressive power scales with depth and width.",
    "Evolution by natural selection proceeds through variation, heredity, and differential reproduction, and genetic drift adds stochastic change to allele frequencies.",
    "In topology, a Mobius strip is a surface with only one side and one edge, constructed by joining the ends of a rectangle with a half twist.",
]


class BasisFactoredLinear(nn.Module):
    """y = A(Bx) + b. A: [d_out, k], B: [k, d_in]. fp32 during training."""

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


def make_bf16_forward(mod):
    """Once trained, swap forward to preserve input dtype (no fp32 upcast)."""
    import types
    def bf16_forward(self, x):
        return F.linear(F.linear(x, self.B), self.A, self.bias)
    mod.forward = types.MethodType(bf16_forward, mod)


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
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
            if n not in covs:
                covs[n] = torch.zeros(d_in, d_in, device=device, dtype=torch.float32)
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
    return {n: c.cpu().to(torch.float64) for n, c in covs.items()}, counts


def top_k_basis_from_cov(cov: torch.Tensor, k: int) -> torch.Tensor:
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    return eigvecs[:, -k:].flip(dims=[1]).contiguous()


def factorize_layer(layer, covariances, rank, layer_name_prefix):
    """Replace each target Linear inside this layer with a BasisFactoredLinear."""
    bases_used = 0
    for name, module in list(layer.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if child_name not in TARGET_NAMES:
                continue
            local_path = f"{name}.{child_name}" if name else child_name
            full_name = f"{layer_name_prefix}.{local_path}"
            if full_name not in covariances:
                continue
            P = top_k_basis_from_cov(covariances[full_name], rank).to(torch.float32)
            fact = BasisFactoredLinear(child, P_in=P, trainable=True)
            setattr(module, child_name, fact)
            bases_used += 1
    return bases_used


def capture_per_layer_io(teacher, tokenizer, texts, device, max_len=256):
    """Run teacher, capture (input, output, position_embeddings, attention_mask,
    cache_position) for each decoder layer, per calibration batch.
    Returns list of dicts, one per batch, with per-layer tensors stacked."""
    layers = teacher.model.layers
    rope = teacher.model.rotary_emb

    captured = []  # list of dicts per batch

    # Use module hooks to capture per-layer IO
    per_layer_inputs = {i: None for i in range(len(layers))}
    per_layer_outputs = {i: None for i in range(len(layers))}
    per_layer_pos_emb = {i: None for i in range(len(layers))}
    per_layer_attn_mask = {i: None for i in range(len(layers))}
    per_layer_cache_pos = {i: None for i in range(len(layers))}

    def make_pre_hook(i):
        def hook(module, args, kwargs):
            # Decoder layer forward signature in transformers Qwen3:
            # (hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)
            # kwargs possibly mixed.
            h = args[0] if args else kwargs.get("hidden_states")
            pos_emb = kwargs.get("position_embeddings")
            if pos_emb is None and len(args) > 1:
                pos_emb = args[1]
            attn_mask = kwargs.get("attention_mask")
            if attn_mask is None and len(args) > 2:
                attn_mask = args[2]
            cache_pos = kwargs.get("cache_position")
            if cache_pos is None and len(args) > 4:
                cache_pos = args[4]
            per_layer_inputs[i] = h.detach().clone()
            per_layer_pos_emb[i] = tuple(p.detach().clone() for p in pos_emb) if pos_emb is not None else None
            per_layer_attn_mask[i] = attn_mask.detach().clone() if attn_mask is not None else None
            per_layer_cache_pos[i] = cache_pos.detach().clone() if cache_pos is not None else None
        return hook

    def make_hook(i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            per_layer_outputs[i] = h.detach().clone()
        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(make_hook(i)))

    with torch.inference_mode():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=max_len).input_ids.to(device)
            teacher(input_ids=ids, use_cache=False)
            batch = {
                "input_ids": ids,
                "inputs": {i: per_layer_inputs[i].clone() for i in range(len(layers))},
                "outputs": {i: per_layer_outputs[i].clone() for i in range(len(layers))},
                "pos_emb": {i: (per_layer_pos_emb[i][0].clone(),
                                per_layer_pos_emb[i][1].clone())
                            if per_layer_pos_emb[i] else None
                            for i in range(len(layers))},
                "attn_mask": {i: per_layer_attn_mask[i].clone() if per_layer_attn_mask[i] is not None else None
                              for i in range(len(layers))},
                "cache_pos": {i: per_layer_cache_pos[i].clone() if per_layer_cache_pos[i] is not None else None
                              for i in range(len(layers))},
            }
            captured.append(batch)

    for h in handles:
        h.remove()
    return captured


def train_one_layer(layer, batches_for_layer, steps, lr, device, log_every=50):
    """Train trainable params inside `layer` to minimize MSE between its output
    and the stored target, given stored inputs and position embeddings."""
    params = [p for p in layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    import math
    warmup = max(10, steps // 20)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        progress = (s - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    history = []
    step = 0
    losses = []
    while step < steps:
        for batch in batches_for_layer:
            if step >= steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            inp = batch["input"]
            tgt = batch["target"]
            pos_emb = batch["pos_emb"]
            attn_mask = batch["attn_mask"]
            cache_pos = batch["cache_pos"]

            # Layer forward
            out = layer(
                inp,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                past_key_values=None,
                cache_position=cache_pos,
            )
            out = out[0] if isinstance(out, tuple) else out

            # Relative MSE to keep scale consistent
            num = (out.float() - tgt.float()).pow(2).mean()
            denom = tgt.float().pow(2).mean().clamp_min(1e-8)
            loss = num / denom

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            losses.append(float(loss.item()))
            step += 1

    final = sum(losses[-10:]) / max(len(losses[-10:]), 1)
    initial = losses[0] if losses else float("nan")
    return {"initial_rel_mse": initial, "final_rel_mse": final, "n_steps": len(losses)}


def load_model(model_id, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager",
    ).to(device).eval()
    return model, tokenizer


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps-per-layer", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--calib-max-len", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--prompt",
                   default="The discovery that inference accelerates with context is")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2), flush=True)

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"\ndevice={device}  rank={args.rank}  steps/layer={args.steps_per_layer}")

    # === Teacher ===
    print(f"\n=== teacher {args.model} ===", flush=True)
    teacher, tokenizer = load_model(args.model, device, dtype=torch.bfloat16)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    n_layers = teacher.config.num_hidden_layers

    # Reference decode
    t_times, t_text, t_tokens = generate(
        teacher, tokenizer, args.prompt, args.max_new_tokens, device)
    t_ms = sum(t_times) / len(t_times)
    print(f"  teacher decode: {t_ms:.2f}ms/tok")
    print(f"  {t_text[:120]}...")

    # === Covariances (for basis-PCA init) ===
    print(f"\n=== covariances ===", flush=True)
    t0 = time.perf_counter()
    covs, counts = collect_input_covariances(
        teacher, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    print(f"  {len(covs)} covs in {time.perf_counter()-t0:.1f}s")

    # === Per-layer IO capture ===
    print(f"\n=== capturing per-layer IO ===", flush=True)
    t0 = time.perf_counter()
    captured = capture_per_layer_io(
        teacher, tokenizer, CALIBRATION_TEXTS, device, max_len=args.calib_max_len)
    print(f"  {len(captured)} batches in {time.perf_counter()-t0:.1f}s")

    # === Student: build by copying teacher then replacing each layer in-place ===
    print(f"\n=== building student (rank {args.rank}) ===", flush=True)
    student, _ = load_model(args.model, device, dtype=torch.bfloat16)
    # Freeze everything in student by default; trainable is managed per factored module.
    for p_ in student.parameters():
        p_.requires_grad_(False)

    # === Train each layer ===
    print(f"\n=== training {n_layers} layers ===", flush=True)
    per_layer_stats = []
    total_factored = 0
    for i in range(n_layers):
        s_layer = student.model.layers[i]
        used = factorize_layer(s_layer, covs, args.rank, layer_name_prefix=f"model.layers.{i}")
        total_factored += used
        # Re-enable gradients on factored params for this layer
        trainable = 0
        for mod in s_layer.modules():
            if isinstance(mod, BasisFactoredLinear):
                mod.A.requires_grad_(True)
                mod.B.requires_grad_(True)
                if mod.bias is not None:
                    mod.bias.requires_grad_(True)
                trainable += mod.A.numel() + mod.B.numel()
                if mod.bias is not None:
                    trainable += mod.bias.numel()

        # Assemble per-layer batches from captured teacher IO
        batches_for_layer = []
        for b in captured:
            batches_for_layer.append({
                "input": b["inputs"][i],
                "target": b["outputs"][i],
                "pos_emb": b["pos_emb"][i],
                "attn_mask": b["attn_mask"][i],
                "cache_pos": b["cache_pos"][i],
            })

        t0 = time.perf_counter()
        stats = train_one_layer(
            s_layer, batches_for_layer, args.steps_per_layer, args.lr, device)
        dt = time.perf_counter() - t0
        print(f"  layer {i:2d}  rel_mse {stats['initial_rel_mse']:.4f} -> "
              f"{stats['final_rel_mse']:.4f}  ({dt:.1f}s, {trainable/1e6:.2f}M params)",
              flush=True)

        # Freeze this layer's factored params now
        for mod in s_layer.modules():
            if isinstance(mod, BasisFactoredLinear):
                mod.A.requires_grad_(False)
                mod.B.requires_grad_(False)
                if mod.bias is not None:
                    mod.bias.requires_grad_(False)

        per_layer_stats.append({
            "layer": i,
            "initial_rel_mse": stats["initial_rel_mse"],
            "final_rel_mse": stats["final_rel_mse"],
            "train_seconds": dt,
            "trainable_params": trainable,
        })

    print(f"  total factored linears: {total_factored}")

    # === Convert all factored weights to bf16 for deployment ===
    print(f"\n=== converting student to bf16 ===", flush=True)
    for mod in student.modules():
        if isinstance(mod, BasisFactoredLinear):
            mod.A.data = mod.A.data.to(torch.bfloat16)
            mod.B.data = mod.B.data.to(torch.bfloat16)
            if mod.bias is not None:
                mod.bias.data = mod.bias.data.to(torch.bfloat16)
            make_bf16_forward(mod)

    # === Student eval ===
    print(f"\n=== student decode (post-training, bf16) ===", flush=True)
    student.eval()
    s_times, s_text, s_tokens = generate(
        student, tokenizer, args.prompt, args.max_new_tokens, device)
    s_ms = sum(s_times) / len(s_times)
    min_len = min(len(t_tokens), len(s_tokens))
    match = sum(1 for a, b in zip(t_tokens[:min_len], s_tokens[:min_len]) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(t_tokens, s_tokens)) if a != b), min_len)
    speedup = t_ms / s_ms if s_ms > 0 else 0
    print(f"  {s_ms:.2f}ms/tok   speedup {speedup:.2f}x   match {match}/{min_len}  "
          f"(first divergence @ {first_div})")
    print(f"  {s_text[:200]}...")

    # === Weight totals ===
    full_params = 0
    fact_params = 0
    for mod in student.modules():
        if isinstance(mod, BasisFactoredLinear):
            full_params += mod._full_params
            fact_params += mod._factored_params

    out_path = Path(args.out_dir) / (
        f"stage9_layerwise_r{args.rank}_{args.model.replace('/', '_')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "device": device,
            "rank": args.rank,
            "steps_per_layer": args.steps_per_layer,
            "lr": args.lr,
            "n_layers": n_layers,
            "teacher_ms_per_tok": t_ms,
            "student_ms_per_tok": s_ms,
            "speedup_vs_teacher": speedup,
            "token_match": f"{match}/{min_len}",
            "token_match_ratio": match / max(min_len, 1),
            "first_divergence": first_div,
            "teacher_sample": t_text[:400],
            "student_sample": s_text[:400],
            "weight_params_full_M": full_params / 1e6,
            "weight_params_factored_M": fact_params / 1e6,
            "weight_size_ratio": fact_params / max(full_params, 1),
            "per_layer": per_layer_stats,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
