"""
Stage 130 — Layer × k head grid: do throat heads beat final-layer heads?

Standard Medusa places heads at the final hidden state. Our hypothesis
(from findings 14 + 15 + stage 129) says the throat encodes multi-token
information in a more-extractable form than the final layer, because
the final layer has already "committed" to predicting t+1.

Test: train linear-probe heads (1024×1024 adapter + frozen model norm
+ frozen LM head) at multiple layers and multiple k, report accuracy
matrix.

Layers:
  L5  — entry wall (post mouth 1)
  L14 — deep throat (stage 129 ran here)
  L21 — exit wall
  L28 — final hidden state (standard Medusa position)

k values: {1, 2, 3, 5}

Prediction: at k=2 and k=3, L14 should beat L28 because L14's state is
less committed. That would be a concrete argument for throat-Medusa.

Also computes speculative-decoding acceptance estimate:
  expected tokens accepted = sum over k of P(accept head at k)
  where P(accept) ≈ top-1 accuracy (approximate)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbeHead(nn.Module):
    """Adapter + frozen model-side norm + frozen LM head."""
    def __init__(self, d, norm, lm_head):
        super().__init__()
        self.adapter = nn.Linear(d, d, bias=True)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(d))
            self.adapter.bias.zero_()
        self.norm = norm
        self.lm_head = lm_head
        for p in self.norm.parameters(): p.requires_grad = False
        for p in self.lm_head.parameters(): p.requires_grad = False

    def forward(self, h):
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
def collect_all_states(model, tokens, layer_indices, device, seq_len=256,
                        apply_final_norm_at=None):
    """Run tokens, collect hidden_states at each layer_idx.
       If apply_final_norm_at is set (e.g., L=28), apply model.model.norm
       to that one's output (that's what the LM head would see)."""
    per_layer = {l: [] for l in layer_indices}
    n = len(tokens) // seq_len
    for i in range(n):
        window = tokens[i*seq_len:(i+1)*seq_len]
        if len(window) < 2: continue
        ids = torch.tensor([window], dtype=torch.long, device=device)
        out = model(ids, use_cache=False, output_hidden_states=True)
        for l in layer_indices:
            h = out.hidden_states[l][0]  # [seq, d]
            if apply_final_norm_at is not None and l == apply_final_norm_at:
                h = model.model.norm(h)
            per_layer[l].append(h.float().cpu())
    return {l: torch.cat(v, dim=0) for l, v in per_layer.items()}


def build_pairs(states, tokens, seq_len, k):
    X_list, Y_list = [], []
    n_chunks = len(states) // seq_len
    for c in range(n_chunks):
        start = c * seq_len
        for pos in range(seq_len - k):
            global_t = c * seq_len + pos
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
        head.eval()
        with torch.no_grad():
            correct1 = 0; correct5 = 0; total = 0
            for start in range(0, X_val.shape[0], 256):
                xb = X_val[start:start+256].to(device)
                yb = Y_val[start:start+256].to(device)
                logits = head(xb)
                top1 = logits.argmax(dim=-1)
                correct1 += (top1 == yb).sum().item()
                _, top5 = logits.topk(5, dim=-1)
                correct5 += (top5 == yb.unsqueeze(-1)).any(dim=-1).sum().item()
                total += yb.shape[0]
            t1 = correct1 / total
            t5 = correct5 / total
        if t1 > best_top1:
            best_top1 = t1
            best_top5 = t5
    return {"top1": best_top1, "top5": best_top5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage130_layer_head_grid.json")
    p.add_argument("--device", default=None)
    p.add_argument("--layers", default="5,14,21,28",
                   help="hidden_states indices to probe")
    p.add_argument("--ks", default="1,2,3,5")
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=256)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    layers = [int(x) for x in args.layers.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    print(f"device={device}  layers={layers}  ks={ks}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    d = model.config.hidden_size
    V = model.config.vocab_size
    L = model.config.num_hidden_layers
    print(f"L={L}  d={d}  V={V}")

    # Tokens
    print(f"loading WikiText-2 tokens (max {args.max_tokens})...")
    tokens = load_tokens(tok, args.max_tokens, "train")
    print(f"  got {len(tokens)} tokens")

    # Final hidden state needs norm applied; others don't.
    final_idx = max(layers) if max(layers) == L else None
    if final_idx is None and L in layers:
        final_idx = L

    print(f"collecting hidden states at {layers}...")
    t0 = time.time()
    states = collect_all_states(model, tokens, layers, device,
                                  seq_len=args.seq_len,
                                  apply_final_norm_at=final_idx)
    for l in layers:
        print(f"  L{l}: shape {states[l].shape}"
              + (" (post-norm)" if l == final_idx else ""))
    print(f"  collected in {time.time()-t0:.0f}s")

    n_chunks = states[layers[0]].shape[0] // args.seq_len
    tokens_used = tokens[:n_chunks * args.seq_len]

    rand_top1 = 1.0 / V
    rand_top5 = 5.0 / V

    results = {
        "model": args.model,
        "layers": layers, "ks": ks, "d": d, "vocab": V,
        "max_tokens": args.max_tokens,
        "grid": {},  # grid[layer][k] -> {top1, top5}
    }

    print(f"\n{'=' * 60}")
    print(f"=== training grid ===")
    print(f"{'=' * 60}")

    for l in layers:
        results["grid"][str(l)] = {}
        for k in ks:
            t0 = time.time()
            X, Y = build_pairs(states[l], tokens_used, args.seq_len, k)
            n = X.shape[0]
            n_val = n // 5
            perm = torch.randperm(n)
            X_val, Y_val = X[perm[:n_val]], Y[perm[:n_val]]
            X_tr, Y_tr = X[perm[n_val:]], Y[perm[n_val:]]

            head = ProbeHead(d, model.model.norm, model.lm_head)
            res = train_probe(head, X_tr, Y_tr, X_val, Y_val, device,
                              epochs=args.epochs)
            dur = time.time() - t0
            results["grid"][str(l)][str(k)] = {
                "top1": res["top1"], "top5": res["top5"],
                "n_train": int(X_tr.shape[0]),
                "duration_s": dur,
            }
            layer_label = f"L{l}"
            if l == final_idx:
                layer_label += " (final+norm)"
            print(f"  {layer_label:>18s}  k={k}:  top1={res['top1']:.4f}  "
                  f"top5={res['top5']:.4f}  ({dur:.0f}s)")

            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary matrix
    print(f"\n{'=' * 60}\n=== accuracy matrix (top-1) ===\n{'=' * 60}")
    header = f"  {'layer':>8s}  " + "  ".join(f"k={k:<5d}" for k in ks)
    print(header)
    for l in layers:
        label = f"L{l}"
        if l == final_idx: label += "+norm"
        row = f"  {label:>8s}  " + "  ".join(
            f"{results['grid'][str(l)][str(k)]['top1']:>6.3f} "
            for k in ks)
        print(row)

    print(f"\n=== accuracy matrix (top-5) ===")
    print(header)
    for l in layers:
        label = f"L{l}"
        if l == final_idx: label += "+norm"
        row = f"  {label:>8s}  " + "  ".join(
            f"{results['grid'][str(l)][str(k)]['top5']:>6.3f} "
            for k in ks)
        print(row)

    # Direct hypothesis test: L14 vs L28 at k>=2
    print(f"\n=== hypothesis test: does L14 beat L28 (final) for k≥2? ===")
    if 14 in layers and 28 in layers:
        for k in ks:
            if k < 2: continue
            t14 = results["grid"]["14"][str(k)]["top1"]
            t28 = results["grid"]["28"][str(k)]["top1"]
            winner = "L14 (throat)" if t14 > t28 else "L28 (final)"
            delta = t14 - t28
            print(f"  k={k}:  L14={t14:.3f}  vs  L28={t28:.3f}  "
                  f"→ {winner} wins by {abs(delta):.3f}")

    # Rough spec-decode estimate: sum of acceptance probabilities
    # Use top-1 as acceptance probability proxy for this rough estimate
    print(f"\n=== rough spec-decode estimate ===")
    print("  (sum of top-1 accuracies at each k = expected tokens per round)")
    for l in layers:
        acc_sum = sum(results["grid"][str(l)][str(k)]["top1"] for k in ks)
        print(f"  using only L{l} heads for k∈{ks}:  expected {acc_sum:.2f} tokens/round")

    # Cross-layer: best head per k
    best_per_k = {}
    for k in ks:
        best_l = max(layers, key=lambda l: results["grid"][str(l)][str(k)]["top1"])
        best_per_k[k] = (best_l, results["grid"][str(best_l)][str(k)]["top1"])
    print(f"\n  best layer per k:")
    for k, (l, acc) in best_per_k.items():
        print(f"    k={k}:  L{l} (top1={acc:.3f})")
    ensemble_sum = sum(acc for _, acc in best_per_k.values())
    print(f"  ensemble (best per k): expected {ensemble_sum:.2f} tokens/round")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
