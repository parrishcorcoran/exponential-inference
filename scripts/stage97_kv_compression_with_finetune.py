"""
Stage 97 — KV compression with fine-tune compensation.

Stage 96 showed per-head scalar α recovers nothing (8× KV compression still
matches 4/80 tokens). The compensation channel was too coarse.

This stage installs the same rank-K KV projection permanently and then
fine-tunes the full model so the remaining weights (q_proj, o_proj,
MLPs, etc.) can adapt to produce correct attention outputs given the
compressed K and V.

Protocol:
  1. Load teacher Qwen3-0.6B. Capture baseline next-token predictions
     on a held-out prompt (stage 38 prompt).
  2. Calibrate rank-K SVD bases for each k_proj and v_proj output.
  3. Install FIXED rank-K projection hooks (no learnable α — just
     project y → y @ P @ P.T).
  4. Fine-tune full model on wikitext-2 train for N steps. Standard
     next-token CE on train tokens.
  5. At every M steps, evaluate match vs teacher baseline. Report
     curve.

Measures: did fine-tuning recover teacher-equivalent prediction under
the compression constraint, and how many steps it took.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CALIB_TEXTS = [
    "The cell is the basic structural unit of life, composed of cytoplasm enclosed within a membrane.",
    "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales.",
    "The history of computing began with mechanical calculators and evolved through vacuum tubes.",
    "Photosynthesis uses sunlight to convert carbon dioxide and water into glucose and oxygen.",
    "Neural networks consist of parameterized layers trained by gradient descent to approximate functions.",
    "Plate tectonics describes the slow movement of Earth's lithospheric plates over the mantle.",
    "Proteins fold into complex three-dimensional structures determined by their amino acid sequences.",
    "Evolution by natural selection operates on heritable variation in populations.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
    "Linear algebra provides the mathematical foundation for many machine learning algorithms.",
    "Bayesian inference updates a prior probability distribution using observed data.",
    "The Riemann zeta function encodes deep information about the distribution of primes.",
]


def load_tokens(tokenizer, max_tokens, split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def find_kv_projs(model):
    result = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if last in ("k_proj", "v_proj"):
            result.append((name, mod))
    return result


@torch.no_grad()
def collect_output_covariances(model, tokenizer, texts, modules, device, max_len=256):
    covs = {name: None for name, _ in modules}
    def make_hook(n):
        def hook(mod, inputs, output):
            y = output.detach()
            y_flat = y.reshape(-1, y.shape[-1]).to(torch.float32).cpu()
            if covs[n] is None:
                covs[n] = torch.zeros(y_flat.shape[1], y_flat.shape[1], dtype=torch.float32)
            covs[n] += y_flat.T @ y_flat
        return hook
    handles = [mod.register_forward_hook(make_hook(name)) for name, mod in modules]
    try:
        model.eval()
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to(device)
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles: h.remove()
    return {n: c.to(torch.float64) for n, c in covs.items()}


def top_k_basis(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)
    return P


def install_projection_hooks(modules, bases, dtype, device):
    """Install fixed rank-k projection hooks: y -> y @ P @ P.T."""
    handles = []
    for name, mod in modules:
        P = bases[name].to(dtype).to(device)
        PPt = (P @ P.T).contiguous()
        def make_hook(projector):
            def hook(mod, inputs, output):
                return output @ projector
            return hook
        handles.append(mod.register_forward_hook(make_hook(PPt)))
    return handles


@torch.no_grad()
def generate_match(model, tokenizer, prompt, max_new_tokens, device, teacher_tokens):
    """Generate from prompt greedily, compare to teacher_tokens.
       Returns (match_count, first_div)."""
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = [next_token.item()]
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
    n = min(len(tokens), len(teacher_tokens))
    match = sum(1 for a, b in zip(tokens[:n], teacher_tokens[:n]) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(tokens, teacher_tokens)) if a != b), n)
    model.train()
    return match, first_div, n, tokens


def iter_batches(tokens, seq_len, batch_size, device):
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rank", type=int, default=128,
                   help="Rank-k KV projection (stage 38's 8x was rank 128)")
    p.add_argument("--prompt", default="The discovery that inference accelerates with context is")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage97_kv_finetune.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}  rank={args.rank}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device)
    modules = find_kv_projs(model)
    d_kv = modules[0][1].out_features
    print(f"  d_kv={d_kv}, {len(modules)} kv projections")

    # 1. Teacher baseline — no compression
    print("\nteacher baseline generation...")
    model.eval()
    with torch.no_grad():
        ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        teacher_tokens = [next_token.item()]
        for _ in range(args.max_new_tokens - 1):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            teacher_tokens.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id: break
    print(f"  teacher generated {len(teacher_tokens)} tokens")
    print(f"  {tokenizer.decode(teacher_tokens, skip_special_tokens=True)[:150]}")

    # 2. Calibrate SVD bases
    print(f"\ncalibrating SVD bases at rank {args.rank}...")
    covs = collect_output_covariances(model, tokenizer, CALIB_TEXTS, modules, device)
    bases = {n: top_k_basis(covs[n], args.rank) for n, _ in modules}
    dtype = modules[0][1].weight.dtype

    # 3. Install fixed projection hooks
    print(f"\ninstalling rank-{args.rank} projection hooks...")
    handles = install_projection_hooks(modules, bases, dtype, device)

    # 4. Pre-finetune eval (= stage 38 result at this rank)
    match, first_div, n, _ = generate_match(
        model, tokenizer, args.prompt, args.max_new_tokens, device, teacher_tokens)
    print(f"\npre-finetune (rank {args.rank}): match {match}/{n}  first_div @ {first_div}")
    history = [{"step": 0, "match": match, "total": n,
                "match_ratio": match / max(n, 1), "first_div": first_div}]

    # 5. Fine-tune
    print(f"\nfine-tuning {args.steps} steps at lr={args.lr}...")
    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 300, split="train")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    step = 0; t0 = time.time(); running = []
    while step < args.steps:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.steps: break
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(loss.item()); step += 1
            if step % args.eval_every == 0:
                tr = float(np.mean(running[-args.eval_every:]))
                match, first_div, n, _ = generate_match(
                    model, tokenizer, args.prompt, args.max_new_tokens, device, teacher_tokens)
                history.append({"step": step, "train_ce": tr,
                               "match": match, "total": n,
                               "match_ratio": match / max(n, 1),
                               "first_div": first_div,
                               "elapsed": time.time() - t0})
                print(f"  step {step}/{args.steps}  train_ce={tr:.4f}  "
                      f"match={match}/{n}  first_div@{first_div}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # 6. Final decode
    for h in handles: h.remove()
    print("\nhooks removed. summary:")
    for h in history:
        print(f"  step {h['step']:>5}  match {h['match']}/{h['total']}  "
              f"({h['match_ratio']*100:.1f}%)  first_div@{h['first_div']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "d_kv": d_kv,
                   "teacher_decode": tokenizer.decode(teacher_tokens, skip_special_tokens=True)[:400],
                   "history": history}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
