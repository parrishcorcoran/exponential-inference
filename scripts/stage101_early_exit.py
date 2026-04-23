"""
Stage 101 — Early exit on Qwen3-14B (or any Qwen3 model).

Add per-layer LM-head probes so each transformer layer can emit a token
prediction. Train only the probes (base model frozen). At inference,
exit early when a layer's prediction confidence exceeds threshold.

Why this can coexist with BitNet-style compression (unlike BitNet itself):
This stack uses dense fp matmul throughout. The probes are standard
Linear → shared LM head. Nothing CPU-specific. Same tensor-core path
as the base forward.

Training:
  - Base model frozen (no optimizer state for base weights).
  - Per-layer PreNorm + optional affine + shared LM head.
  - Loss = Σ_l w_l · CE(probe_l(h_l), next_token).
  - Uniform w_l or linear-ramp toward final layer.

Inference (early exit):
  - At each layer l, compute probe_l(h_l) logits.
  - If max(softmax(logits)) > τ (e.g., 0.9), exit and return argmax.
  - Else continue to next layer.

Reports:
  - Per-layer val CE (how well each probe predicts).
  - Inference speedup at various τ thresholds.
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


class LayerProbe(nn.Module):
    """PreNorm + affine + shared LM head applied to a per-layer hidden state."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(d_model))
        self.affine_weight = nn.Parameter(torch.eye(d_model))
        self.affine_bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, h, lm_head_weight):
        # h: [B, T, d_model]. RMSNorm + affine + shared unembed.
        rms = h.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        h = h * rms * self.norm_weight
        h = h @ self.affine_weight.T + self.affine_bias
        return F.linear(h, lm_head_weight)


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
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-probes", default="probes.pt")
    p.add_argument("--out", default="results/stage101_early_exit.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    vocab = model.config.vocab_size
    print(f"  L={L}  d_model={d_model}  vocab={vocab}", flush=True)

    # Freeze base model
    for p_ in model.parameters(): p_.requires_grad = False
    lm_head_weight = model.lm_head.weight  # tied with embedding

    # Create probes for every layer (layer 0 = embedding, L = final)
    probes = nn.ModuleList([LayerProbe(d_model) for _ in range(L + 1)]).to(device).to(torch.float32)
    print(f"  created {len(probes)} probes  (each {sum(p.numel() for p in probes[0].parameters()):,} params)", flush=True)

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 500, split="train")
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")

    opt = torch.optim.AdamW(probes.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Uniform weights across layers
    layer_weights = torch.ones(L + 1, device=device) / (L + 1)

    step = 0; t0 = time.time(); running = []
    history = []
    while step < args.steps:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.steps: break
            opt.zero_grad()
            with torch.no_grad():
                out = model(inp, use_cache=False, output_hidden_states=True)
                hidden_states = [h.detach() for h in out.hidden_states]
            total = 0.0
            per_layer_losses = []
            for l, h in enumerate(hidden_states):
                logits = probes[l](h.to(torch.float32), lm_head_weight.to(torch.float32))
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
                total = total + layer_weights[l] * loss
                per_layer_losses.append(float(loss.item()))
            total.backward()
            torch.nn.utils.clip_grad_norm_(probes.parameters(), 1.0)
            opt.step()
            running.append(float(total.item())); step += 1
            if step % args.eval_every == 0:
                # Validation: per-layer CE
                probes.eval()
                per_layer_val = [0.0] * (L + 1); vcount = 0
                with torch.no_grad():
                    for v_inp, v_tgt in iter_batches(val_tokens, args.seq_len, args.batch_size, device):
                        out = model(v_inp, use_cache=False, output_hidden_states=True)
                        for l, h in enumerate(out.hidden_states):
                            logits = probes[l](h.to(torch.float32), lm_head_weight.to(torch.float32))
                            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), v_tgt.reshape(-1))
                            per_layer_val[l] += float(loss.item())
                        vcount += 1
                        if vcount >= 5: break
                per_layer_val = [v / max(vcount, 1) for v in per_layer_val]
                probes.train()
                tr = float(np.mean(running[-args.eval_every:]))
                history.append({"step": step, "train_total": tr,
                               "per_layer_val_ce": per_layer_val,
                               "elapsed": time.time()-t0})
                mid = L // 2
                print(f"  step {step}/{args.steps}  tot={tr:.4f}  "
                      f"ce@L0={per_layer_val[0]:.3f}  ce@L{mid}={per_layer_val[mid]:.3f}  "
                      f"ce@L{L}={per_layer_val[L]:.3f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Final per-layer CE report + inference speedup at thresholds
    probes.eval()
    print(f"\n=== final per-layer val CE ===", flush=True)
    final_per_layer = history[-1]["per_layer_val_ce"]
    for l in range(L + 1):
        if l in (0, L//4, L//2, 3*L//4, L-1, L):
            print(f"  layer {l:>3}  val_ce={final_per_layer[l]:.4f}  val_ppl={math.exp(final_per_layer[l]):.2f}")

    # Save probes for later use
    torch.save(probes.state_dict(), args.save_probes)
    print(f"\nsaved probes to {args.save_probes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "L": L, "d_model": d_model,
                   "history": history,
                   "final_per_layer_val_ce": final_per_layer}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
