"""
Stage 83 — Iterative Holographic Transformer (IHT) prototype at Qwen3-0.6B scale.

Tests the holographic-physics thesis: can an architecture that explicitly
implements holographic retrieval (HRR-style binding + iterative rotation)
match a standard transformer of matched dimensions on language modeling,
with FEWER parameters per layer?

Design (IHT):
  Stage 1 (one pass): build causal hologram state S_t via cumulative
    outer products:  S_t = S_{t-1} + k_t ⊗ v_t   where k_t, v_t are from
    linear projections of embed(token_t).

  Stage 2 (per-target position, L iterations): given S_{t-1}, starting
    from h = embed(token_t), iterate:
        h = R_l · h                    # learned rotation (d × d)
        q = W_Q_l · h                  # per-layer query
        c = S_{t-1} · q                # context retrieval (one matmul)
        h = h + α · c                  # residual integration

  Readout: logits = lm_head · final_h

No softmax attention, no per-layer MLP. Each layer is cheap; architecture
lets us stack many with low compute.

Baseline: standard transformer at matched d_model=1024, L=28 (matches
Qwen3-0.6B architecture dimensions), MHA with n_heads=16 head_dim=64,
SwiGLU MLP d_ffn=3072. ~580M params.

IHT at same dims: ~215M params (no per-layer MLP, just rotation + retrieval).

Comparison: same training budget (steps, seq, batch). Does IHT match
baseline ppl with fewer params?

Training: wikitext-2, from scratch. Suggested: 5000-10000 steps at
seq=128 batch=4 on GPU (Strix-appropriate).
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ Shared components ------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope(seq_len, head_dim, base=10000.0, device="cpu"):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)[None, None, :, :]
    sin_full = torch.cat([sin, sin], dim=-1)[None, None, :, :]
    return (q * cos_full) + (rotate_half(q) * sin_full), (k * cos_full) + (rotate_half(k) * sin_full)


# ------------------ Baseline: standard transformer ------------------

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class StandardLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = StandardAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ffn)
    def forward(self, h, cos, sin):
        h = h + self.attn(self.attn_norm(h), cos, sin)
        h = h + self.mlp(self.mlp_norm(h))
        return h


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layers=28, n_heads=16, d_ffn=3072, max_seq=512):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([StandardLayer(d_model, n_heads, d_ffn) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        cos, sin = precompute_rope(max_seq, d_model // n_heads)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids):
        T = input_ids.shape[1]
        cos = self.rope_cos[:T]; sin = self.rope_sin[:T]
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.final_norm(h)
        return self.lm_head(h)


# ------------------ IHT: Iterative Holographic Transformer ------------------

class HolographicContextEncoder(nn.Module):
    """Stage 1: builds causal hologram state S_t via cumulative outer products.
    S_t shape: [B, T, d, d]. For each position, accumulates k_t ⊗ v_t."""
    def __init__(self, d_model):
        super().__init__()
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: [B, T, d_model]. Returns S: [B, T, d, d] (cumulative causal state)."""
        x_n = self.norm(x)
        k = self.W_K(x_n)  # [B, T, d]
        v = self.W_V(x_n)  # [B, T, d]
        # Outer products k_t ⊗ v_t: [B, T, d, d]
        # Memory-friendly: compute in-place with cumulative sum
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, T, d, d]
        S = torch.cumsum(outer, dim=1)  # causal accumulation
        return S


class HolographicLayer(nn.Module):
    """One iteration — rotation + holographic retrieval + SwiGLU cleanup.

    Stage 86-92 (2026-04-22) finding: linear retrieval alone cannot compose
    across depth — per-layer angular error compounds to ~1% next-token match
    after 28 layers. The bilinear SwiGLU gate (silu(W_gate h) ⊙ (W_up h)) is
    necessary as a drift-cleanup operator between holographic retrievals —
    it pulls the residual stream back onto the learned manifold at each step.

    Two residual paths per layer: holographic attention + SwiGLU MLP.
    """
    def __init__(self, d_model, d_ffn, use_mlp=True):
        super().__init__()
        # Attention path: normalize, rotate, retrieve, residual-add
        self.attn_norm = RMSNorm(d_model)
        # Rotation matrix, initialized near identity (will drift during training)
        self.R = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
        # Query projection for context read
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        # Scale factor for context contribution
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # MLP cleanup path: required per stage-86-92 findings
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp_norm = RMSNorm(d_model)
            self.mlp = SwiGLU(d_model, d_ffn)

    def forward(self, h, S_prev):
        """h: [B, T, d], S_prev: [B, T, d, d]. S_prev[b, t] is context at position t."""
        # --- attention path ---
        h_n = self.attn_norm(h)
        # Rotation
        h_rot = h_n @ self.R.T  # [B, T, d]
        # Query
        q = self.W_Q(h_rot)  # [B, T, d]
        # Holographic retrieval
        c = (S_prev @ q.unsqueeze(-1)).squeeze(-1)  # [B, T, d]
        h = h + h_rot + self.alpha * c  # residual
        # --- mlp cleanup path ---
        if self.use_mlp:
            h = h + self.mlp(self.mlp_norm(h))
        return h


class IterativeHolographic(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layers=28, max_seq=512):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.context_encoder = HolographicContextEncoder(d_model)
        self.layers = nn.ModuleList([HolographicLayer(d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        h_init = self.embed(input_ids)  # [B, T, d]
        # Stage 1: build causal hologram state
        S = self.context_encoder(h_init)  # [B, T, d, d]
        # Causal shift: S_prev[b, t] should contain context up to t-1
        # Prepend zeros at position 0
        zeros = torch.zeros(B, 1, self.d_model, self.d_model, device=S.device, dtype=S.dtype)
        S_prev = torch.cat([zeros, S[:, :-1]], dim=1)  # [B, T, d, d]
        # Stage 2: iterate layers
        h = h_init
        for layer in self.layers:
            h = layer(h, S_prev)
        h = self.final_norm(h)
        return self.lm_head(h)


# ------------------ Data + training ------------------

def load_tokens(tokenizer, split, max_tokens):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def make_batches(tokens, seq_len, batch_size, rng):
    n = len(tokens) - seq_len - 1
    while True:
        idx = rng.choice(n, size=batch_size, replace=False)
        x = torch.stack([torch.tensor(tokens[i:i + seq_len], dtype=torch.long) for i in idx])
        y = torch.stack([torch.tensor(tokens[i + 1:i + seq_len + 1], dtype=torch.long) for i in idx])
        yield x, y


@torch.no_grad()
def eval_ppl(model, tokens, device, seq_len=128, max_tokens=4000):
    model.eval()
    total_loss = 0.0; total = 0
    for i in range(0, min(len(tokens) - seq_len - 1, max_tokens), seq_len):
        x = torch.tensor([tokens[i:i + seq_len]], dtype=torch.long, device=device)
        y = torch.tensor([tokens[i + 1:i + seq_len + 1]], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits[0].float(), y[0], reduction="sum")
        total_loss += float(loss.item()); total += y.numel()
    return math.exp(total_loss / max(total, 1))


def train_one(model, model_name, train_tokens, eval_tokens, device,
              steps, seq_len, batch_size, lr, eval_every):
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== training {model_name} ({n_params:,} params) ===", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = 500
    def lr_at(step):
        if step < warmup: return lr * step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    rng = np.random.default_rng(42)
    batch_iter = make_batches(train_tokens, seq_len, batch_size, rng)
    history = []
    t_start = time.time()
    running_loss = 0.0; running_n = 0

    model.train()
    for step in range(1, steps + 1):
        cur_lr = lr_at(step)
        for g in opt.param_groups: g["lr"] = cur_lr

        x, y = next(batch_iter)
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += float(loss.item()); running_n += 1
        if step % 50 == 0 or step == steps:
            avg = running_loss / running_n
            print(f"  step {step}/{steps}  avg_loss={avg:.4f}  lr={cur_lr:.2e}  "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)
            running_loss = 0.0; running_n = 0
        if step % eval_every == 0 or step == steps:
            ppl = eval_ppl(model, eval_tokens, device, seq_len)
            print(f"  [eval step {step}] val ppl = {ppl:.4f}", flush=True)
            history.append({"step": step, "val_ppl": ppl, "train_loss": avg})
            model.train()
    return {"model_name": model_name, "n_params": n_params, "history": history,
            "final_val_ppl": history[-1]["val_ppl"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--d-ffn", type=int, default=3072)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--models", nargs="+", default=["baseline", "iht"],
                   help="which models to train")
    p.add_argument("--out", default="results/stage83_iht_prototype.json")
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"=== stage 83: IHT prototype at Qwen3-0.6B scale ===", flush=True)
    print(f"device={device}  d_model={args.d_model}  L={args.n_layers}  steps={args.steps}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = tokenizer.vocab_size

    print("loading wikitext-2...", flush=True)
    train_tokens = load_tokens(tokenizer, "train", max_tokens=2_000_000)
    eval_tokens = load_tokens(tokenizer, "validation", max_tokens=50_000)
    print(f"  train: {len(train_tokens)}  val: {len(eval_tokens)}", flush=True)

    results = {}

    if "baseline" in args.models:
        m = StandardTransformer(vocab, d_model=args.d_model, n_layers=args.n_layers,
                                 n_heads=args.n_heads, d_ffn=args.d_ffn,
                                 max_seq=args.seq_len + 8)
        results["baseline"] = train_one(m, "standard transformer", train_tokens, eval_tokens, device,
                                         args.steps, args.seq_len, args.batch_size, args.lr, args.eval_every)
        del m; gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        elif device == "mps": torch.mps.empty_cache()

    if "iht" in args.models:
        m = IterativeHolographic(vocab, d_model=args.d_model, n_layers=args.n_layers,
                                  max_seq=args.seq_len + 8)
        results["iht"] = train_one(m, "IHT", train_tokens, eval_tokens, device,
                                    args.steps, args.seq_len, args.batch_size, args.lr, args.eval_every)
        del m; gc.collect()

    # Compare
    print(f"\n{'='*60}\n=== FINAL COMPARISON ===\n{'='*60}", flush=True)
    if "baseline" in results:
        b = results["baseline"]
        print(f"baseline  val ppl: {b['final_val_ppl']:.4f}  ({b['n_params']:,} params)")
    if "iht" in results:
        i = results["iht"]
        print(f"IHT       val ppl: {i['final_val_ppl']:.4f}  ({i['n_params']:,} params)")
    if "baseline" in results and "iht" in results:
        ratio = results["iht"]["final_val_ppl"] / results["baseline"]["final_val_ppl"]
        param_ratio = results["iht"]["n_params"] / results["baseline"]["n_params"]
        print(f"\nIHT / baseline ppl ratio: {ratio:.4f}")
        print(f"IHT / baseline param ratio: {param_ratio:.4f}")
        if ratio < 1.05:
            print(f"→ IHT MATCHES baseline at {param_ratio*100:.0f}% of params — thesis supported")
        else:
            print(f"→ IHT worse by {(ratio-1)*100:.1f}% at {param_ratio*100:.0f}% of params")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
