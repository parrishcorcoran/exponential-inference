"""
Stage 108 — 0.6B marginal cost WITH aware fine-tune.

Stage 107 showed the post-hoc cost curve. Now test: does fine-tune rescue
the cliffs?

For each compression config, apply it, then fine-tune N steps, measure
val_ppl. Compare to post-hoc (stage 107) to see where fine-tune helps.

Configs tested (the ones where post-hoc showed cliff or moderate cost):
  - Weight Q4 (post-hoc +31.4)
  - Weight Q3 (post-hoc +38K)
  - Weight Q2 (post-hoc +52M)
  - Embed Q3 (post-hoc +12.5)
  - Embed Q2 (post-hoc +6688)
  - d_ffn 2048 (post-hoc +113)
  - d_ffn 1536 (post-hoc +235)
  - d_ffn 1024 (post-hoc +1097)

Plus the already-cheap ones to confirm fine-tune doesn't hurt:
  - Weight Q8, Q6
  - Embed Q8, Q6, Q4

For weight quantization, use a learnable per-tensor α (BitNet-style)
with straight-through estimator. Model keeps fp32 master weights, but
forward uses quantized.

For embed quantization, just pure post-hoc + fine-tune (embed stays
fp, but other weights adapt).

For d_ffn shrink, truncate then fine-tune the remaining weights.
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
def eval_ppl(model, tokens, seq_len, device, max_batches=15):
    model.eval()
    total, count = 0.0, 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item(); count += 1
        if count >= max_batches: break
    model.train()
    return total / max(count, 1)


# -------- Quantization-aware modules --------

class IntSTE(torch.autograd.Function):
    """Symmetric int quantization with straight-through estimator."""
    @staticmethod
    def forward(ctx, W, bits, per_channel_dim=None):
        levels = 2 ** bits
        if per_channel_dim is not None:
            max_abs = W.abs().amax(dim=1 - per_channel_dim, keepdim=True)
        else:
            max_abs = W.abs().max()
        scale = (max_abs / (levels / 2 - 1)).clamp(min=1e-8)
        q = torch.round(W / scale).clamp(-(levels/2 - 1), levels/2 - 1)
        return q * scale
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None


class QATLinear(nn.Module):
    """Linear with fp32 master weight, forward uses per-channel int quantization
       via STE. alpha is learnable scalar for additional amplitude control."""
    def __init__(self, in_features, out_features, bits, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.W_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.alpha = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, bits):
        m = cls(linear.in_features, linear.out_features, bits, bias=(linear.bias is not None))
        with torch.no_grad():
            m.W_fp.data = linear.weight.data.clone().float()
            if linear.bias is not None:
                m.bias.data = linear.bias.data.clone().float()
        return m

    def forward(self, x):
        W_q = IntSTE.apply(self.W_fp, self.bits, 0)
        W_eff = W_q * self.alpha
        return F.linear(x.float(), W_eff, self.bias).to(x.dtype)


def convert_body_to_qat(model, bits):
    n = 0
    for layer in model.model.layers:
        for parent, name in [(layer.self_attn, "q_proj"), (layer.self_attn, "k_proj"),
                             (layer.self_attn, "v_proj"), (layer.self_attn, "o_proj"),
                             (layer.mlp, "gate_proj"), (layer.mlp, "up_proj"),
                             (layer.mlp, "down_proj")]:
            old = getattr(parent, name)
            new = QATLinear.from_linear(old, bits)
            setattr(parent, name, new)
            n += 1
    return n


def apply_embed_quantization_ste(model, bits):
    """Post-hoc quantize embedding (tied with lm_head)."""
    W = model.get_input_embeddings().weight.data
    levels = 2 ** bits
    max_abs = W.abs().amax(dim=1, keepdim=True)
    scale = (max_abs / (levels / 2 - 1)).clamp(min=1e-8)
    q = torch.round(W / scale).clamp(-(levels/2 - 1), levels/2 - 1)
    W_q = q * scale
    model.get_input_embeddings().weight.data.copy_(W_q.to(W.dtype))


def shrink_dffn(model, keep):
    for layer in model.model.layers:
        g = layer.mlp.gate_proj
        u = layer.mlp.up_proj
        d = layer.mlp.down_proj
        k = min(keep, g.weight.shape[0])
        g.weight.data = g.weight.data[:k].contiguous()
        u.weight.data = u.weight.data[:k].contiguous()
        d.weight.data = d.weight.data[:, :k].contiguous()
        g.out_features = k
        u.out_features = k
        d.in_features = k


def load_fresh(model_id, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)
    return m


def run_aware_config(args, device, tokenizer, train_tokens, val_tokens, teacher_ppl, config):
    """Apply compression config, fine-tune, measure val_ppl over training."""
    label = config["label"]
    print(f"\n=== {label} ===", flush=True)
    t_start = time.time()

    model = load_fresh(args.model, device)
    teacher_init_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=5)

    axis = config["axis"]; value = config["value"]

    if axis == "weight_bits":
        convert_body_to_qat(model, value)
    elif axis == "embed_bits":
        apply_embed_quantization_ste(model, value)
    elif axis == "d_ffn":
        if value < 3072:
            shrink_dffn(model, value)
    elif axis == "none":
        pass

    model = model.to(device)

    pre_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=5)
    pre_ppl = math.exp(pre_ce)
    print(f"  post-compress val_ppl={pre_ppl:.3f}  (teacher={teacher_ppl:.3f})", flush=True)

    # Fine-tune
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    step = 0
    trajectory = [{"step": 0, "val_ppl": pre_ppl, "event": "post-compress"}]
    while step < args.steps:
        for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
            if step >= args.steps: break
            opt.zero_grad()
            logits = model(inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0:
                val_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=5)
                val_ppl = math.exp(val_ce)
                trajectory.append({"step": step, "val_ppl": val_ppl, "train_ce": float(loss.item())})
                print(f"  step {step}/{args.steps}  val_ppl={val_ppl:.3f}  "
                      f"Δ={val_ppl-teacher_ppl:+.3f}", flush=True)

    final_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=5)
    final_ppl = math.exp(final_ce)
    elapsed = time.time() - t_start

    result = {
        "label": label,
        "axis": axis,
        "value": value,
        "post_compress_ppl": pre_ppl,
        "final_tune_ppl": final_ppl,
        "delta_from_teacher": final_ppl - teacher_ppl,
        "recovery_from_compress": pre_ppl - final_ppl,
        "elapsed_sec": elapsed,
        "trajectory": trajectory,
    }
    print(f"  FINAL: val_ppl={final_ppl:.3f}  Δ={final_ppl-teacher_ppl:+.3f}  "
          f"({elapsed:.0f}s)", flush=True)

    del model; gc.collect()
    if device == "mps": torch.mps.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default="results/stage108_06b_aware.json")
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

    print("teacher baseline...", flush=True)
    model = load_fresh(args.model, device)
    val_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 20, split="validation")
    teacher_ce = eval_ppl(model, val_tokens, args.seq_len, device, max_batches=15)
    teacher_ppl = math.exp(teacher_ce)
    print(f"  teacher val_ppl={teacher_ppl:.3f}", flush=True)
    del model; gc.collect()
    if device == "mps": torch.mps.empty_cache()

    train_tokens = load_tokens(tokenizer, max_tokens=args.seq_len * 200, split="train")

    # Configs to test: focus on where post-hoc had cliff, and a few cheap ones for comparison
    configs = [
        # post-hoc cheap — sanity check fine-tune doesn't hurt
        {"label": "weight_Q8", "axis": "weight_bits", "value": 8},
        {"label": "weight_Q6", "axis": "weight_bits", "value": 6},
        # post-hoc cliff — does fine-tune rescue?
        {"label": "weight_Q4", "axis": "weight_bits", "value": 4},
        {"label": "weight_Q3", "axis": "weight_bits", "value": 3},
        # extreme
        {"label": "weight_Q2_ternary", "axis": "weight_bits", "value": 2},
        # embed cliff
        {"label": "embed_Q3", "axis": "embed_bits", "value": 3},
        {"label": "embed_Q2", "axis": "embed_bits", "value": 2},
        # d_ffn shrink — does fine-tune recover the lost capacity?
        {"label": "dffn_2048", "axis": "d_ffn", "value": 2048},
        {"label": "dffn_1536", "axis": "d_ffn", "value": 1536},
        {"label": "dffn_1024", "axis": "d_ffn", "value": 1024},
    ]

    results = []
    for cfg in configs:
        try:
            r = run_aware_config(args, device, tokenizer, train_tokens, val_tokens, teacher_ppl, cfg)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {cfg['label']}: {e}", flush=True)
            results.append({"label": cfg['label'], "error": str(e)})

    print(f"\n{'='*60}\n=== 0.6B AWARE marginal cost summary ===\n{'='*60}")
    print(f"teacher val_ppl: {teacher_ppl:.3f}")
    print(f"{'config':>20}  {'post-hoc':>10}  {'post-tune':>10}  {'recovery':>10}  verdict")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:>20}  ERROR: {r['error']}")
            continue
        post_hoc = r.get("post_compress_ppl", float('nan'))
        final = r.get("final_tune_ppl", float('nan'))
        recovery = post_hoc - final
        verdict = "RESCUED" if (post_hoc > 50 and final < 50) else \
                  "improved" if recovery > 0 else "no-change"
        print(f"  {r['label']:>20}  {post_hoc:>10.2f}  {final:>10.2f}  "
              f"{recovery:>10.2f}  {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "teacher_val_ppl": teacher_ppl,
                   "args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
