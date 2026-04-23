"""
Stage 102 — Medusa speculative-decode heads on Qwen3-14B (or any Qwen3).

Add K Medusa heads on top of the base model's final hidden state. Each
predicts the token at offset +k (k=1..K). Used at inference in speculative
decoding: draft K tokens from Medusa heads, verify with main model in
parallel, commit the longest matching prefix.

Per the user's instruction, heads are added and trained ONE AT A TIME.
Use --num-heads K to train K heads; each subsequent run adds the next.
If --load-prev is given, load existing heads, freeze them, and train one
new head on top.

Why this coexists with our compression stack:
Medusa heads are standard Linear modules. Same GPU tensor-core path as
the base forward. BitNet's ternary weights run on CPU; Medusa needs GPU
parallelism — they don't compose. Our shared-basis / MLA / int4 body
compression keeps GPU path, so Medusa works on top naturally.

Training:
  - Base model frozen.
  - Each Medusa head is 2-layer MLP + shared LM head.
  - Loss for head k = CE(head_k(h_final_t), token_{t+k}).
  - Teacher-forcing: h_final comes from base model on the same sequence.
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


class MedusaHead(nn.Module):
    """One Medusa head: ResNet-style MLP on top of final hidden state + shared LM head."""
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.mlp_layers = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)
        ])
    def forward(self, h, lm_head_weight):
        # h: [B, T, d_model]. Residual SiLU MLP, then share the LM head.
        for layer in self.mlp_layers:
            h = h + F.silu(layer(h))
        return F.linear(h.to(lm_head_weight.dtype), lm_head_weight)


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


def iter_batches(tokens, seq_len, batch_size, device, offset_max):
    """Yield (input, targets_per_head) where targets_per_head is a list of
       shifted targets for offsets 1..offset_max."""
    import random
    window_len = seq_len + offset_max
    n = (len(tokens) - 1) // window_len
    idx = list(range(n)); random.shuffle(idx)
    batch = []
    for i in idx:
        start = i * window_len
        window = tokens[start:start + window_len + 1]
        if len(window) < seq_len + 2: continue
        batch.append(window)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            inp = t[:, :seq_len]
            targets = [t[:, k+1:k+1+seq_len] for k in range(offset_max)]
            yield inp, targets
            batch = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--num-heads", type=int, default=1,
                   help="Total number of Medusa heads. Existing heads loaded from --load-prev and frozen.")
    p.add_argument("--load-prev", default=None, help="Path to previous heads state_dict")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-mlp-layers", type=int, default=1)
    p.add_argument("--save-heads", default="medusa_heads.pt")
    p.add_argument("--out", default="results/stage102_medusa.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}  num_heads={args.num_heads}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    d_model = model.config.hidden_size
    vocab = model.config.vocab_size
    print(f"  d_model={d_model}  vocab={vocab}", flush=True)

    for p_ in model.parameters(): p_.requires_grad = False
    lm_head_weight = model.lm_head.weight

    heads = nn.ModuleList([
        MedusaHead(d_model, n_layers=args.head_mlp_layers) for _ in range(args.num_heads)
    ]).to(device).to(torch.float32)

    # Load previous heads if provided, freeze them
    n_frozen = 0
    if args.load_prev:
        state = torch.load(args.load_prev, map_location=device)
        prev_count = sum(1 for k in state if k.startswith("0.") or ".mlp_layers" in k)
        # Simpler: count top-level head indices
        n_loaded = 1 + max([int(k.split(".")[0]) for k in state.keys() if k.split(".")[0].isdigit()], default=-1)
        print(f"  loading {n_loaded} previous heads from {args.load_prev}")
        # Load into the first n_loaded heads
        for i in range(min(n_loaded, args.num_heads)):
            sub_state = {k[len(f"{i}."):]: v for k, v in state.items() if k.startswith(f"{i}.")}
            heads[i].load_state_dict(sub_state)
            for p_ in heads[i].parameters():
                p_.requires_grad = False
            n_frozen += 1
    print(f"  {n_frozen} frozen heads, {args.num_heads - n_frozen} trainable")

    train_params = [p for p in heads.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")

    step = 0; t0 = time.time(); running = []
    history = []
    while step < args.steps:
        for inp, targets in iter_batches(train_tokens, args.seq_len, args.batch_size, device, args.num_heads):
            if step >= args.steps: break
            opt.zero_grad()
            with torch.no_grad():
                out = model(inp, use_cache=False, output_hidden_states=True)
                h_final = out.hidden_states[-1].detach()
            total = 0.0
            per_head = []
            for k, head in enumerate(heads):
                if not any(p.requires_grad for p in head.parameters()):
                    continue
                logits = head(h_final.to(torch.float32), lm_head_weight.to(torch.float32))
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[k].reshape(-1))
                total = total + loss
                per_head.append((k, float(loss.item())))
            if isinstance(total, float): continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            running.append(float(total.item())); step += 1
            if step % args.eval_every == 0:
                tr = float(np.mean(running[-args.eval_every:]))
                # Val per head accuracy
                heads.eval()
                acc_per_head = {k: 0.0 for k, _ in per_head}
                vcount = 0
                with torch.no_grad():
                    for v_inp, v_tgt in iter_batches(val_tokens, args.seq_len, args.batch_size, device, args.num_heads):
                        out = model(v_inp, use_cache=False, output_hidden_states=True)
                        h_final = out.hidden_states[-1]
                        for k in acc_per_head:
                            logits = heads[k](h_final.to(torch.float32), lm_head_weight.to(torch.float32))
                            preds = logits.argmax(-1)
                            acc = (preds == v_tgt[k]).float().mean().item()
                            acc_per_head[k] += acc
                        vcount += 1
                        if vcount >= 5: break
                acc_per_head = {k: v / max(vcount, 1) for k, v in acc_per_head.items()}
                heads.train()
                history.append({"step": step, "train_total": tr,
                               "val_acc_per_head": acc_per_head,
                               "elapsed": time.time()-t0})
                acc_str = "  ".join(f"head{k}_acc={a:.3f}" for k, a in acc_per_head.items())
                print(f"  step {step}/{args.steps}  tot={tr:.4f}  {acc_str}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    torch.save(heads.state_dict(), args.save_heads)
    print(f"\nsaved heads to {args.save_heads}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "d_model": d_model,
                   "history": history}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
