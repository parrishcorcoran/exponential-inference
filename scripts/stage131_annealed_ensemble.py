"""
Stage 131 — Annealed Medusa ensemble across layers.

Stage 130 showed L28 (final+norm) wins single-head accuracy at every k,
but the exit gate L21 is close (-3%). Question: does adding heads at
lower layers as an ENSEMBLE add independent information at each k,
enabling cross-layer consensus checks?

Procedure:
  1. Train probes at all (L, k) in an expanded grid
  2. For each k, greedily anneal layer additions:
     - Start with best single layer (top-1)
     - Try adding each remaining layer's logits to the ensemble
     - Keep if ensemble top-1 improves by > threshold
     - Stop when no addition helps
  3. Report the best layer set per k
  4. Estimate cross-layer consensus benefit

Extended grid: L5, L10, L14, L18, L21, L25, L28.
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
def collect_all_states(model, tokens, layer_indices, device, seq_len=256, final_idx=None):
    per_layer = {l: [] for l in layer_indices}
    n = len(tokens) // seq_len
    for i in range(n):
        window = tokens[i*seq_len:(i+1)*seq_len]
        if len(window) < 2: continue
        ids = torch.tensor([window], dtype=torch.long, device=device)
        out = model(ids, use_cache=False, output_hidden_states=True)
        for l in layer_indices:
            h = out.hidden_states[l][0]
            if final_idx is not None and l == final_idx:
                h = model.model.norm(h)
            per_layer[l].append(h.float().cpu())
    return {l: torch.cat(v, dim=0) for l, v in per_layer.items()}


def build_pairs_index(n_chunks, seq_len, k, total_tokens):
    """Return (state_indices, token_indices) for valid (t, t+k) pairs,
       where state_indices points into states tensor and token_indices
       into the original token stream."""
    sel_states = []
    sel_targets = []
    for c in range(n_chunks):
        for pos in range(seq_len - k):
            global_t = c * seq_len + pos
            target_t = global_t + k
            if target_t >= total_tokens: break
            sel_states.append(c * seq_len + pos)
            sel_targets.append(target_t)
    return np.array(sel_states), np.array(sel_targets)


def train_probe(head, X_tr, Y_tr, device, epochs=5, batch=128, lr=5e-4):
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    n = X_tr.shape[0]
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start+batch]
            xb = X_tr[idx].to(device)
            yb = Y_tr[idx].to(device)
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def get_logits(head, X, device, batch=256):
    """Collect logits on a dataset."""
    head.eval()
    all_logits = []
    for start in range(0, X.shape[0], batch):
        xb = X[start:start+batch].to(device)
        logits = head(xb)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def eval_top_k(logits, Y, top_k):
    top = logits.topk(top_k, dim=-1).indices
    correct = (top == Y.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return correct


def eval_top1(logits, Y):
    return (logits.argmax(dim=-1) == Y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage131_annealed_ensemble.json")
    p.add_argument("--device", default=None)
    p.add_argument("--layers", default="5,10,14,18,21,25,28")
    p.add_argument("--ks", default="1,2,3,5")
    p.add_argument("--max-tokens", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--improvement-thresh", type=float, default=0.003,
                   help="min top-1 improvement to keep an added layer")
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

    print(f"loading WikiText-2 tokens (max {args.max_tokens})...")
    tokens = load_tokens(tok, args.max_tokens, "train")

    final_idx = L if L in layers else None
    print(f"collecting states at {layers}...")
    t0 = time.time()
    states = collect_all_states(model, tokens, layers, device,
                                  seq_len=args.seq_len, final_idx=final_idx)
    print(f"  collected in {time.time()-t0:.0f}s")
    n_chunks = states[layers[0]].shape[0] // args.seq_len

    results = {"model": args.model, "layers": layers, "ks": ks,
               "max_tokens": args.max_tokens, "per_k": {}}

    # For each k, train heads at each layer, then do greedy ensemble
    for k in ks:
        print(f"\n{'=' * 60}\n=== k = {k} ===\n{'=' * 60}")
        sel_states, sel_targets = build_pairs_index(n_chunks, args.seq_len,
                                                      k, len(tokens))
        n_total = len(sel_states)
        n_val = n_total // 5
        perm = np.random.RandomState(0).permutation(n_total)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Y_tr = torch.tensor([tokens[sel_targets[i]] for i in tr_idx], dtype=torch.long)
        Y_val = torch.tensor([tokens[sel_targets[i]] for i in val_idx], dtype=torch.long)

        # Train a probe at each layer, collect val logits
        per_layer_logits = {}  # l -> logits [n_val, V]
        per_layer_top1 = {}
        for l in layers:
            X_full = states[l][sel_states]
            X_tr = X_full[tr_idx]
            X_val = X_full[val_idx]
            head = ProbeHead(d, model.model.norm, model.lm_head)
            t0 = time.time()
            train_probe(head, X_tr, Y_tr, device, epochs=args.epochs)
            val_logits = get_logits(head, X_val, device)
            dur = time.time() - t0
            per_layer_logits[l] = val_logits
            per_layer_top1[l] = eval_top1(val_logits, Y_val)
            print(f"  L{l}  trained  top1={per_layer_top1[l]:.4f}  ({dur:.0f}s)")

        # Greedy ensemble: start with best single layer, add layers that help
        sorted_layers = sorted(layers, key=lambda l: -per_layer_top1[l])
        greedy_set = [sorted_layers[0]]
        ensemble_logits = per_layer_logits[greedy_set[0]].clone()
        best_top1 = per_layer_top1[greedy_set[0]]
        greedy_history = [{"set": list(greedy_set), "top1": best_top1,
                            "top5": eval_top_k(ensemble_logits, Y_val, 5)}]
        print(f"  start:   set={greedy_set}  top1={best_top1:.4f}")

        for candidate in sorted_layers[1:]:
            # Try adding: average logits
            new_logits = (ensemble_logits * len(greedy_set)
                           + per_layer_logits[candidate]) / (len(greedy_set) + 1)
            new_top1 = eval_top1(new_logits, Y_val)
            delta = new_top1 - best_top1
            action = "+"
            if delta > args.improvement_thresh:
                greedy_set.append(candidate)
                ensemble_logits = new_logits
                best_top1 = new_top1
                greedy_history.append({"set": list(greedy_set),
                                        "top1": new_top1,
                                        "top5": eval_top_k(new_logits, Y_val, 5)})
                print(f"    + L{candidate}:  ensemble top1={new_top1:.4f}  "
                      f"Δ={delta:+.4f}  KEPT")
            else:
                print(f"    - L{candidate}:  would be top1={new_top1:.4f}  "
                      f"Δ={delta:+.4f}  skipped")

        # Try the "all-layers" ensemble as a sanity check
        all_logits = torch.stack([per_layer_logits[l] for l in layers]).mean(0)
        all_top1 = eval_top1(all_logits, Y_val)
        all_top5 = eval_top_k(all_logits, Y_val, 5)
        print(f"  sanity:  ALL layers ensemble top1={all_top1:.4f}  top5={all_top5:.4f}")

        # Consensus analysis: how often do top layers agree on top-1?
        top_l = sorted_layers[0]  # best single
        second_l = sorted_layers[1]
        pred_top = per_layer_logits[top_l].argmax(dim=-1)
        pred_second = per_layer_logits[second_l].argmax(dim=-1)
        agree = (pred_top == pred_second)
        agree_rate = agree.float().mean().item()
        if agree.sum() > 0:
            accuracy_when_agree = (pred_top[agree] == Y_val[agree]).float().mean().item()
        else:
            accuracy_when_agree = 0.0
        print(f"  consensus: L{top_l} & L{second_l} agree on {agree_rate*100:.1f}% of positions; "
              f"when they agree, correct {accuracy_when_agree*100:.1f}% of the time")

        results["per_k"][str(k)] = {
            "per_layer_top1": per_layer_top1,
            "greedy_set": greedy_set,
            "greedy_top1": best_top1,
            "greedy_top5": greedy_history[-1]["top5"],
            "greedy_history": greedy_history,
            "all_layers_top1": all_top1,
            "all_layers_top5": all_top5,
            "consensus": {
                "layers": [int(top_l), int(second_l)],
                "agree_rate": agree_rate,
                "accuracy_when_agree": accuracy_when_agree,
            },
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    print(f"  {'k':>3s}  {'best single':>12s}  {'greedy set':>22s}  {'greedy top1':>12s}  {'all-layer top1':>15s}")
    for k in ks:
        r = results["per_k"][str(k)]
        best_l = max(layers, key=lambda l: r["per_layer_top1"][l])
        print(f"  {k:>3d}  L{best_l:>10s} ({r['per_layer_top1'][best_l]:.3f})  "
              f"  {str(r['greedy_set']):>22s}  {r['greedy_top1']:>12.4f}  "
              f"{r['all_layers_top1']:>15.4f}")

    # Tokens per round estimate
    print(f"\n  === tokens per round estimate ===")
    greedy_sum = sum(results["per_k"][str(k)]["greedy_top1"] for k in ks)
    all_sum = sum(results["per_k"][str(k)]["all_layers_top1"] for k in ks)
    single_sum = sum(max(results["per_k"][str(k)]["per_layer_top1"].values()) for k in ks)
    print(f"  best single layer per k: expected {single_sum:.3f} tokens/round")
    print(f"  greedy ensemble per k:   expected {greedy_sum:.3f} tokens/round")
    print(f"  all-layers ensemble per k: expected {all_sum:.3f} tokens/round")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
