"""
Stage 118 — Slow progressive mid-only KV compression on Qwen3-0.6B.

User's theory: 0.6B has a narrow compression budget in middle layers.
Only slow annealing (small rank steps, fine-tune between each) can
access it. Big jumps (stage 116/117 with 50% cuts) overshoot the budget.

Test: descend rank in small steps (64 units = ~6% per step), fine-tune
between each step, observe where quality starts degrading.

Schedule:
  rank 1024 → 960 → 896 → 832 → 768 → 704 → 640 → 576 → 512

Between each step:
  - SVD re-computed from current model's k_proj/v_proj outputs
  - projector installed at new rank (mid-only, edges full)
  - fine-tune N steps
  - eval val_ppl

Model persists across steps (not reloaded). Each step inherits adapted
weights from the previous rank. This is the proper slow-annealing test.
"""

import argparse
import json
import math
import time
import gc
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
    "The standard model of particle physics unifies electromagnetic, weak, and strong interactions.",
    "Evolution by natural selection operates on heritable variation in populations.",
    "DNA encodes genetic information in a double-helix structure of paired nucleotide bases.",
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


def iter_batches(tokens, seq_len, batch_size, device, shuffle=True):
    import random
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n))
    if shuffle: random.shuffle(idx)
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


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=10):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def collect_covs(model, tokenizer, texts, device):
    covs = {}
    handles = []
    for i, layer in enumerate(model.model.layers):
        for name, mod in [("k_proj", layer.self_attn.k_proj), ("v_proj", layer.self_attn.v_proj)]:
            key = (i, name)
            covs[key] = None
            def make_hook(k):
                def hook(mod, inputs, output):
                    y = output.detach().reshape(-1, output.shape[-1]).float().cpu()
                    if covs[k] is None:
                        covs[k] = torch.zeros(y.shape[1], y.shape[1])
                    covs[k] += y.T @ y
                return hook
            handles.append(mod.register_forward_hook(make_hook(key)))
    model.eval()
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(device)
        model(input_ids=ids, use_cache=False)
    for h in handles: h.remove()
    return covs


def top_k_projector(cov, k):
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float64))
    P = eigvecs[:, -k:].flip(dims=[1]).to(torch.float32)
    return (P @ P.T).contiguous()


def install_mid_kv_projectors(model, covs, rank, edge_width):
    L = len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        if i < edge_width or i >= L - edge_width:
            continue
        for name, mod in [("k_proj", layer.self_attn.k_proj), ("v_proj", layer.self_attn.v_proj)]:
            ppt = top_k_projector(covs[(i, name)], rank).to(mod.weight.dtype).to(mod.weight.device)
            def make_hook(projector):
                def hook(mod, inputs, output):
                    return output @ projector
                return hook
            handles.append(mod.register_forward_hook(make_hook(ppt)))
    return handles


def remove_hooks(handles):
    for h in handles: h.remove()


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)


def fine_tune_steps(model, train_tokens, val_tokens, seq_len, batch_size, steps, lr, device,
                    eval_every=50, active_handles=None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    trajectory = []
    step = 0; t0 = time.time(); running = []
    while step < steps:
        for inp, tgt in iter_batches(train_tokens, seq_len, batch_size, device, shuffle=True):
            if step >= steps: break
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(float(loss.item())); step += 1
            if step % eval_every == 0:
                val_ce = eval_ppl(model, val_tokens, seq_len, device, max_batches=4)
                val_ppl = math.exp(val_ce)
                trajectory.append({"step": step, "val_ppl": val_ppl,
                                   "train_ce": float(np.mean(running[-eval_every:])),
                                   "elapsed": time.time() - t0})
                print(f"      fine-tune step {step}/{steps}  val_ppl={val_ppl:.3f}",
                      flush=True)
    return trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps-per-phase", type=int, default=150)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--edge-width", type=int, default=3)
    p.add_argument("--schedule", default="1024,960,896,832,768,704,640,576,512",
                   help="Rank schedule (slow descent)")
    p.add_argument("--out", default="results/stage118_slow_progressive_kv.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")
    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 200, split="train")

    print("loading teacher + measuring baseline...", flush=True)
    model = load_fresh(args.model, device)
    L = model.config.num_hidden_layers
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=10)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}  L={L}  edge_width={args.edge_width}", flush=True)
    print(f"  middle layers: {L - 2*args.edge_width} of {L}", flush=True)

    schedule = [int(r) for r in args.schedule.split(",")]
    print(f"\nschedule: {schedule}  ({args.steps_per_phase} fine-tune steps per phase)", flush=True)

    phases = []
    current_handles = []

    for phase_idx, rank in enumerate(schedule):
        t0 = time.time()
        print(f"\n=== phase {phase_idx+1}/{len(schedule)}: rank {rank} (mid-only) ===", flush=True)

        # Remove previous projectors
        remove_hooks(current_handles)
        current_handles = []

        # Recalibrate covariances on current (possibly fine-tuned) model
        print(f"  recalibrating covs...", flush=True)
        covs = collect_covs(model, tokenizer, CALIB_TEXTS, device)

        # Install new rank
        current_handles = install_mid_kv_projectors(model, covs, rank, args.edge_width)

        # Pre-tune eval
        pre_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=4)
        pre_ppl = math.exp(pre_ce)
        print(f"  pre-tune val_ppl={pre_ppl:.3f}  Δ={pre_ppl-teacher_ppl:+.2f}", flush=True)

        # Fine-tune (skip if rank 1024 which is no-compression sanity; short tune to warm up)
        if phase_idx == 0 and rank >= 1024:
            # Minimal tune, just warmup
            ft_steps = min(50, args.steps_per_phase)
        else:
            ft_steps = args.steps_per_phase

        trajectory = fine_tune_steps(model, train_tokens, val_tokens, args.seq_len,
                                     args.batch_size, ft_steps, args.lr, device,
                                     eval_every=args.eval_every)

        # Final eval for this phase
        post_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=10)
        post_ppl = math.exp(post_ce)
        delta = post_ppl - teacher_ppl
        bucket = ("FREE_WIN" if delta < -0.1 else
                  "free" if abs(delta) < 0.5 else
                  "cheap" if delta < 2 else
                  "moderate" if delta < 10 else
                  "expensive" if delta < 100 else
                  "broken")
        print(f"  POST-TUNE val_ppl={post_ppl:.3f}  Δ={delta:+.2f}  [{bucket}]  "
              f"({time.time()-t0:.0f}s)", flush=True)

        phases.append({
            "phase": phase_idx + 1,
            "rank": rank,
            "compression": 1024.0 / rank,
            "pre_tune_ppl": pre_ppl,
            "post_tune_ppl": post_ppl,
            "delta_from_teacher": delta,
            "cost": bucket,
            "tune_steps": ft_steps,
            "trajectory": trajectory,
        })

    # Summary
    print(f"\n=== SUMMARY — slow progressive mid-only KV on 0.6B ===", flush=True)
    print(f"teacher val_ppl: {teacher_ppl:.3f}")
    print(f"{'rank':>5}  {'compression':>11}  {'pre_tune':>10}  {'post_tune':>10}  {'delta':>8}  bucket")
    for ph in phases:
        print(f"  {ph['rank']:>5}  {ph['compression']:>10.2f}×  {ph['pre_tune_ppl']:>10.3f}  "
              f"{ph['post_tune_ppl']:>10.3f}  {ph['delta_from_teacher']:>+8.2f}  {ph['cost']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "L": L, "edge_width": args.edge_width,
                   "teacher_val_ce": teacher_ce, "teacher_val_ppl": teacher_ppl,
                   "args": vars(args),
                   "phases": phases}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
