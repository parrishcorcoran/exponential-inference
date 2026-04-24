"""
Stage 129 — Holographic probe: does the throat at L14 encode multiple
future tokens?

For each k in {1, 2, 3, 5, 7, 10, 15, 20}:
  Train a linear probe from h_t (throat state at L14) to logits for
  token at position t+k. Use the model's own frozen LM head as the
  vocabulary projection and learn just a (d × d) adapter.
  Measure top-1 and top-5 accuracy on held-out 20%.

Decay profile tells us:
  cliff at k=2          → no hologram, only next-token
  graceful decay        → hologram, ~5 tokens decodable
  plateau out to k>>1   → long-range encoding, many tokens decodable

Takes ~25 min on MPS.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeHead(nn.Module):
    """Linear adapter (d × d) + frozen RMSNorm + frozen LM head.
       Trains the adapter only."""
    def __init__(self, d, norm, lm_head):
        super().__init__()
        self.adapter = nn.Linear(d, d, bias=True)
        # Init close to identity to make training easier
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(d))
            self.adapter.bias.zero_()
        self.norm = norm
        self.lm_head = lm_head
        for p in self.norm.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

    def forward(self, h):
        # h: [..., d]
        z = self.adapter(h)
        z = self.norm(z.to(self.norm.weight.dtype))
        logits = self.lm_head(z)
        return logits.float()


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


@torch.no_grad()
def collect_throat_states(model, tokens, throat_layer, device, seq_len=256):
    """Run tokens through the model, grab hidden_states[throat_layer] at
       every position. Returns [N_total_positions, d] tensor."""
    states = []
    n = len(tokens) // seq_len
    for i in range(n):
        window = tokens[i*seq_len:(i+1)*seq_len]
        if len(window) < 2: continue
        ids = torch.tensor([window], dtype=torch.long, device=device)
        out = model(ids, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[throat_layer][0].float().cpu()  # [seq, d]
        states.append(h)
    return torch.cat(states, dim=0)  # [N, d]


def build_pairs(states, tokens, seq_len, k):
    """Build (h_t, token_{t+k}) pairs. States are laid out in chunks of
       seq_len; within a chunk, only positions 0..seq_len-1-k are valid."""
    X_list, Y_list = [], []
    n_chunks = len(states) // seq_len
    for c in range(n_chunks):
        start = c * seq_len
        # Valid positions in this chunk:
        for pos in range(seq_len - k):
            global_t = c * seq_len + pos   # position in original token stream
            target_t = global_t + k
            if target_t >= len(tokens): break
            X_list.append(states[start + pos])
            Y_list.append(tokens[target_t])
    X = torch.stack(X_list)
    Y = torch.tensor(Y_list, dtype=torch.long)
    return X, Y


def train_probe(head, X_tr, Y_tr, X_val, Y_val, device, epochs=5,
                 batch=128, lr=5e-4):
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    n = X_tr.shape[0]
    best_top1 = 0.0
    best_top5 = 0.0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start+batch]
            xb = X_tr[idx].to(device)
            yb = Y_tr[idx].to(device)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.shape[0]

        # Eval
        head.eval()
        with torch.no_grad():
            correct1 = 0; correct5 = 0; total = 0; loss_val = 0.0
            for start in range(0, X_val.shape[0], 256):
                xb = X_val[start:start+256].to(device)
                yb = Y_val[start:start+256].to(device)
                logits = head(xb)
                loss_val += F.cross_entropy(logits, yb, reduction="sum").item()
                top1 = logits.argmax(dim=-1)
                correct1 += (top1 == yb).sum().item()
                _, top5 = logits.topk(5, dim=-1)
                correct5 += (top5 == yb.unsqueeze(-1)).any(dim=-1).sum().item()
                total += yb.shape[0]
            t1 = correct1 / total
            t5 = correct5 / total
            avg_loss_val = loss_val / total
        if t1 > best_top1:
            best_top1 = t1
            best_top5 = t5
        if ep == 0 or ep == epochs - 1:
            print(f"    ep {ep:2d}:  train loss={total_loss/n:.3f}  "
                  f"val loss={avg_loss_val:.3f}  top1={t1:.3f}  top5={t5:.3f}")
    return {"top1": best_top1, "top5": best_top5, "val_loss": avg_loss_val}


def random_baseline(vocab_size):
    """Expected top-1 and top-5 acc if predicting uniformly."""
    return {"top1": 1.0 / vocab_size, "top5": 5.0 / vocab_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage129_holographic_probe.json")
    p.add_argument("--device", default=None)
    p.add_argument("--throat-layer", type=int, default=14,
                   help="hidden_states index; 14 for 0.6B deep throat")
    p.add_argument("--ks", default="1,2,3,5,7,10,15,20")
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=256)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    ks = [int(x) for x in args.ks.split(",")]
    print(f"device={device}  dtype={dtype}  throat_layer={args.throat_layer}  ks={ks}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    d = model.config.hidden_size
    V = model.config.vocab_size
    print(f"d={d}  vocab={V}")

    # Tokens
    print(f"loading WikiText-2 tokens (max {args.max_tokens})...")
    tokens = load_tokens(tok, args.max_tokens, "train")
    print(f"  got {len(tokens)} tokens")

    # Collect throat states
    print(f"collecting throat states at layer {args.throat_layer}...")
    t0 = time.time()
    states = collect_throat_states(model, tokens, args.throat_layer,
                                     device, seq_len=args.seq_len)
    print(f"  states: {states.shape}  in {time.time()-t0:.0f}s")

    # Truncate tokens to match states
    n_chunks = states.shape[0] // args.seq_len
    tokens_used = tokens[:n_chunks * args.seq_len]

    rand = random_baseline(V)
    print(f"  random baseline: top1={rand['top1']:.2e}  top5={rand['top5']:.2e}")

    results = {
        "model": args.model,
        "throat_layer": args.throat_layer,
        "d": d, "vocab": V,
        "max_tokens": args.max_tokens,
        "random_baseline": rand,
        "per_k": {},
    }

    for k in ks:
        print(f"\n=== k = {k} (predict token at t+{k}) ===")
        X, Y = build_pairs(states, tokens_used, args.seq_len, k)
        n = X.shape[0]
        n_val = n // 5
        perm = torch.randperm(n)
        X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
        X_tr, Y_tr = X[perm[n_val:]], Y[perm[n_val:]]
        print(f"  train pairs: {X_tr.shape[0]}  val: {X_val.shape[0]}")

        head = ProbeHead(d, model.model.norm, model.lm_head)
        res = train_probe(head, X_tr, Y_tr, X_val, Y_val, device,
                          epochs=args.epochs)
        res["n_train"] = int(X_tr.shape[0])
        res["n_val"] = int(X_val.shape[0])
        lift_top1 = res["top1"] / rand["top1"]
        lift_top5 = res["top5"] / rand["top5"]
        res["lift_over_random_top1"] = lift_top1
        res["lift_over_random_top5"] = lift_top5
        results["per_k"][str(k)] = res
        print(f"  best:  top1={res['top1']:.4f}  top5={res['top5']:.4f}  "
              f"lift over random: {lift_top1:.0f}× / {lift_top5:.0f}×")

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Interpretation
    print(f"\n{'=' * 60}\n=== decay profile ===\n{'=' * 60}")
    print(f"  {'k':>4}  {'top1':>8}  {'top5':>8}  {'lift t1':>10}  {'lift t5':>10}")
    for k in ks:
        r = results["per_k"][str(k)]
        print(f"  {k:>4}  {r['top1']:>8.4f}  {r['top5']:>8.4f}  "
              f"{r['lift_over_random_top1']:>10.1f}×  "
              f"{r['lift_over_random_top5']:>10.1f}×")

    # Verdict
    top1_k1 = results["per_k"][str(ks[0])]["top1"]
    top1_k_last = results["per_k"][str(ks[-1])]["top1"]
    cliff_k = None
    for k in ks[1:]:
        if results["per_k"][str(k)]["top1"] < 2 * rand["top1"]:
            cliff_k = k
            break

    print(f"\n  k=1 top1: {top1_k1:.3f}")
    print(f"  k={ks[-1]} top1: {top1_k_last:.3f}")
    if cliff_k is not None:
        print(f"  cliff at k={cliff_k} (signal disappears)")
    else:
        print(f"  NO cliff — signal persists to k={ks[-1]} (HOLOGRAM)")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
