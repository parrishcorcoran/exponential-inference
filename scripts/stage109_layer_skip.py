"""
Stage 109 — Layer skip sweep on 0.6B.

Post-hoc test: bypass specific layers (output = input for that layer).
Our manifold data shows L3-22 is dead-zone (pr ≈ 1, rank-1 residual).
If that measurement is right, skipping dead-zone layers should be
cheap. Skipping active layers (L0-2, L23-28) should be expensive.

Variants:
  A. skip every 3rd: L2,5,8,11,14,17,20,23,26  (naive, 9 skipped)
  B. skip dead-zone chunk 1: L3-10  (8 layers)
  C. skip dead-zone chunk 2: L10-17 (8 layers)
  D. skip contiguous middle: L11-20 (10 layers)
  E. skip dead-zone wide:  L3-22    (20 layers, nearly all dead)
  F. skip L0 only          (calibration: active layer — should hurt)
  G. skip L27 only         (calibration: active final — should hurt)
  H. no skip (baseline)

Expected outcomes:
  - H = teacher
  - B, C, D = cheap-ish (rank-1 layers doing little)
  - E = moderate-to-expensive (cumulative)
  - F, G = expensive (active layer needed)
  - A = mixed (partially skipping active layers too)
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


def iter_batches(tokens, seq_len, device):
    n = (len(tokens) - 1) // seq_len
    for i in range(n):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        t = torch.tensor([window], dtype=torch.long, device=device)
        yield t[:, :-1], t[:, 1:]


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, max_batches=20):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    return total / max(count, 1)


def install_layer_skip_hooks(model, skip_indices):
    """For each layer idx in skip_indices, install a forward post-hook that
       replaces the layer's output with a copy of the input residual stream
       (effectively bypassing the layer). Uses a post-hook so we don't
       have to match Qwen3's exact return signature."""
    handles = []
    for idx in skip_indices:
        layer = model.model.layers[idx]

        def make_hook():
            def hook(mod, args, kwargs, output):
                # args[0] or kwargs['hidden_states'] is the input residual
                if len(args) > 0:
                    h_in = args[0]
                else:
                    h_in = kwargs.get("hidden_states")
                # output is typically a tuple (hidden_states, ...) or a tensor
                if isinstance(output, tuple):
                    return (h_in,) + output[1:]
                return h_in
            return hook

        h = layer.register_forward_hook(make_hook(), with_kwargs=True)
        handles.append(h)
    return handles


def uninstall_layer_skip_hooks(handles):
    for h in handles:
        h.remove()


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--out", default="results/stage109_layer_skip.json")
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
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 25, split="validation")

    # Teacher baseline
    print("teacher baseline...", flush=True)
    model = load_fresh(args.model, device)
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}", flush=True)

    L = model.config.num_hidden_layers
    print(f"  L={L}", flush=True)

    # Variants
    variants = {
        "H_baseline_no_skip": [],
        "A_every_3rd":         [i for i in range(L) if i % 3 == 2],  # 2,5,8,...
        "B_deadzone_3_10":     list(range(3, 11)),
        "C_deadzone_10_17":    list(range(10, 18)),
        "D_middle_11_20":      list(range(11, 21)),
        "E_deadzone_wide_3_22": list(range(3, 23)),
        "F_skip_L0_only":      [0],
        "G_skip_final_only":   [L - 1],
        "I_skip_alternate_dead": [i for i in range(3, 23) if i % 2 == 0],  # even in dead zone
        "J_skip_first_3":      [0, 1, 2],
        "K_skip_last_3":       [L-3, L-2, L-1],
    }

    tests = []
    for label, skip_idx in variants.items():
        t0 = time.time()
        handles = install_layer_skip_hooks(model, skip_idx)
        try:
            ce = eval_ppl(model, val_tokens, args.seq_len, device, args.eval_batches)
            ppl = math.exp(ce)
        except Exception as e:
            print(f"  ERROR {label}: {e}", flush=True)
            ce = float('inf'); ppl = float('inf')
        finally:
            uninstall_layer_skip_hooks(handles)

        delta = ppl - teacher_ppl if math.isfinite(ppl) else float('inf')
        n_skipped = len(skip_idx)
        elapsed = time.time() - t0
        tests.append({
            "label": label,
            "skip_indices": skip_idx,
            "n_skipped": n_skipped,
            "val_ce": ce,
            "val_ppl": ppl,
            "delta_ppl": delta,
            "elapsed_sec": elapsed,
        })
        bucket = ("FREE_WIN" if delta < -0.5 else
                  "free" if abs(delta) < 0.5 else
                  "cheap" if delta < 2 else
                  "moderate" if delta < 10 else
                  "expensive" if delta < 100 else
                  "broken")
        print(f"  {label:>28}  skipped={n_skipped:>2}  val_ppl={ppl:>10.3f}  "
              f"delta={delta:>+8.3f}  [{bucket}]", flush=True)

    print(f"\n=== summary ===", flush=True)
    print(f"teacher val_ppl: {teacher_ppl:.3f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "L": L,
                   "teacher_val_ce": teacher_ce, "teacher_val_ppl": teacher_ppl,
                   "tests": tests}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
