"""
Stage 60 — Architecture bake-off: standard transformer vs two-channel holographic.

Purpose: before committing Strix time to the holographic architecture,
prove it's not a worse architecture at small scale. Both models trained
on identical data, same optimizer, same steps. Compare held-out perplexity.

Outcomes:
  - holographic beats standard: architecture is working, proceed to scale
  - holographic ≈ standard: architecture is benign, the win (if any)
    will come from manifold-target training later
  - holographic loses: architectural choice is hurting, revert to standard
    and put the novelty into the training objective instead

Both architectures are matched to ~20M params and trained from scratch
(not distillation) on wikitext-2. We're measuring architectural quality,
not distillation quality.

Usage:
  python scripts/stage60_bakeoff.py --arch standard --steps 3000
  python scripts/stage60_bakeoff.py --arch holographic --steps 3000
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_wikitext(tokenizer, max_train, max_heldout):
    from datasets import load_dataset
    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    heldout_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    def collect(ds, n):
        toks = []
        for row in ds:
            t = row["text"].strip()
            if not t or t.startswith("="): continue
            ids = tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids[0]
            toks.append(ids)
            if sum(len(x) for x in toks) >= n: break
        return torch.cat(toks)[:n]
    return collect(train_ds, max_train), collect(heldout_ds, max_heldout)


# ===== Standard Transformer baseline =====

class StandardBlock(nn.Module):
    def __init__(self, d, n_heads, head_dim, bulk):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d)
        self.mlp_norm = nn.LayerNorm(d)
        q_dim = n_heads * head_dim
        self.q_proj = nn.Linear(d, q_dim, bias=False)
        self.k_proj = nn.Linear(d, head_dim, bias=False)    # 1 KV head
        self.v_proj = nn.Linear(d, head_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, d, bias=False)
        self.gate_proj = nn.Linear(d, bulk, bias=False)
        self.up_proj = nn.Linear(d, bulk, bias=False)
        self.down_proj = nn.Linear(bulk, d, bias=False)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, h):
        B, T, D = h.shape
        residual = h
        h_n = self.attn_norm(h)
        q = self.q_proj(h_n).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_n).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_n).view(B, T, 1, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.matmul(F.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        h = residual + self.o_proj(attn)
        residual = h
        h_n = self.mlp_norm(h)
        h = residual + self.down_proj(F.silu(self.gate_proj(h_n)) * self.up_proj(h_n))
        return h


class StandardTransformer(nn.Module):
    def __init__(self, vocab, hidden=256, n_heads=4, head_dim=64, bulk=1024, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([
            StandardBlock(hidden, n_heads, head_dim, bulk) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight   # tie

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.final_norm(h))


# ===== Holographic two-channel student =====

class HoloBlock(nn.Module):
    """Two-channel layer: carry + flip. π-phase operator on flip.
    Small mix_angle between channels."""
    def __init__(self, d_c, d_f, n_heads, head_dim, bulk_c, bulk_f):
        super().__init__()
        self.carry_attn_norm = nn.LayerNorm(d_c)
        self.carry_mlp_norm = nn.LayerNorm(d_c)
        self.flip_attn_norm = nn.LayerNorm(d_f)
        self.flip_mlp_norm = nn.LayerNorm(d_f)
        q_c = n_heads * head_dim
        q_f = n_heads * head_dim
        self.cq = nn.Linear(d_c, q_c, bias=False)
        self.ck = nn.Linear(d_c, head_dim, bias=False)
        self.cv = nn.Linear(d_c, head_dim, bias=False)
        self.co = nn.Linear(q_c, d_c, bias=False)
        self.fq = nn.Linear(d_f, q_f, bias=False)
        self.fk = nn.Linear(d_f, head_dim, bias=False)
        self.fv = nn.Linear(d_f, head_dim, bias=False)
        self.fo = nn.Linear(q_f, d_f, bias=False)
        self.cg = nn.Linear(d_c, bulk_c, bias=False)
        self.cu = nn.Linear(d_c, bulk_c, bias=False)
        self.cd = nn.Linear(bulk_c, d_c, bias=False)
        self.fg = nn.Linear(d_f, bulk_f, bias=False)
        self.fu = nn.Linear(d_f, bulk_f, bias=False)
        self.fd = nn.Linear(bulk_f, d_f, bias=False)
        # Small mix between channels per layer (walking basis, stage 59)
        self.mix_angle = nn.Parameter(torch.tensor(0.1))
        # Cross-channel mixing needs same dim — require d_c == d_f
        assert d_c == d_f, "carry and flip dims must match for mixing"
        self.n_heads = n_heads
        self.head_dim = head_dim

    def _attn(self, h, q_proj, k_proj, v_proj, o_proj):
        B, T, D = h.shape
        q = q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k_proj(h).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v_proj(h).view(B, T, 1, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.matmul(F.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return o_proj(attn)

    def forward(self, h_c, h_f):
        # Per-channel attention
        h_c = h_c + self._attn(self.carry_attn_norm(h_c), self.cq, self.ck, self.cv, self.co)
        h_f = h_f + self._attn(self.flip_attn_norm(h_f), self.fq, self.fk, self.fv, self.fo)
        # Per-channel MLP (bulk preserved per channel)
        hcn = self.carry_mlp_norm(h_c)
        h_c = h_c + self.cd(F.silu(self.cg(hcn)) * self.cu(hcn))
        hfn = self.flip_mlp_norm(h_f)
        h_f = h_f + self.fd(F.silu(self.fg(hfn)) * self.fu(hfn))
        # π-phase operator on flip channel (the key architectural move)
        h_f = -h_f
        # Small mixing (walking basis)
        c, s = torch.cos(self.mix_angle), torch.sin(self.mix_angle)
        new_c = c * h_c + s * h_f
        new_f = -s * h_c + c * h_f
        return new_c, new_f


class HolographicStudent(nn.Module):
    """Two-channel holographic transformer.
    Carry: stable "preserved information" channel.
    Flip: sign-inverting "phase-π" channel.
    Coherent readout: carry - flip (phase-aligned sum)."""
    def __init__(self, vocab, carry_dim=128, flip_dim=128, n_heads=2, head_dim=64,
                 bulk=512, n_layers=6):
        super().__init__()
        self.embed_c = nn.Embedding(vocab, carry_dim)
        self.embed_f = nn.Embedding(vocab, flip_dim)
        self.blocks = nn.ModuleList([
            HoloBlock(carry_dim, flip_dim, n_heads, head_dim, bulk, bulk)
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(carry_dim)
        self.lm_head = nn.Linear(carry_dim, vocab, bias=False)
        self.lm_head.weight = self.embed_c.weight   # tie to carry embedding

    def forward(self, input_ids):
        h_c = self.embed_c(input_ids)
        h_f = self.embed_f(input_ids)
        for block in self.blocks:
            h_c, h_f = block(h_c, h_f)
        # Coherent readout: phase-0 carry + phase-π flip → carry - flip
        coherent = h_c - h_f
        return self.lm_head(self.final_norm(coherent))


# ===== Training + eval =====

def make_batches(ids, seq_len):
    pairs = []
    for i in range(seq_len, len(ids) - 1, seq_len):
        ctx = ids[i - seq_len:i]
        tgt = ids[i - seq_len + 1:i + 1]
        if len(ctx) == seq_len and len(tgt) == seq_len:
            pairs.append((ctx, tgt))
    return pairs


def evaluate(model, pairs, device, vocab, n_eval=200):
    model.eval()
    nll = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for ctx, tgt in pairs[:n_eval]:
            logits = model(ctx.unsqueeze(0).to(device))[0].float()
            tgt = tgt.to(device)
            logp = F.log_softmax(logits, dim=-1)
            nll -= float(logp.gather(1, tgt.unsqueeze(1)).sum().item())
            correct += int((logits.argmax(dim=-1) == tgt).sum().item())
            total += tgt.shape[0]
    return {"ppl": math.exp(nll / total), "acc": correct / total, "n": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["standard", "holographic"], required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B",
                   help="tokenizer to use (doesn't need to match training model)")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-tokens", type=int, default=600_000)
    p.add_argument("--heldout-tokens", type=int, default=30_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-pairs", type=int, default=150)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    if args.out is None:
        args.out = f"results/stage60_bakeoff_{args.arch}.json"
    random.seed(0); torch.manual_seed(0)
    print(f"device={device}  arch={args.arch}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab = tokenizer.vocab_size
    print(f"  tokenizer={args.tokenizer}  vocab={vocab}")

    print(f"\n=== tokenizing ===")
    train_ids, heldout_ids = load_wikitext(tokenizer, args.train_tokens, args.heldout_tokens)
    train_pairs = make_batches(train_ids, args.seq_len)
    heldout_pairs = make_batches(heldout_ids, args.seq_len)
    print(f"  train: {len(train_pairs)} chunks  heldout: {len(heldout_pairs)} chunks")

    print(f"\n=== building {args.arch} model ===")
    if args.arch == "standard":
        # Target ~20M trainable: hidden 256, bulk 1024, 6 layers
        model = StandardTransformer(vocab, hidden=256, n_heads=4, head_dim=64,
                                     bulk=1024, n_layers=6).to(device)
    else:
        # Target ~20M trainable: carry=flip=128, bulk 512 per channel, 6 layers
        # (fewer params per channel to keep total in the same ballpark)
        model = HolographicStudent(vocab, carry_dim=128, flip_dim=128, n_heads=2,
                                    head_dim=64, bulk=512, n_layers=6).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable/1e6:.2f}M  total: {total/1e6:.2f}M")

    print(f"\n=== initial eval (before training) ===")
    baseline = evaluate(model, heldout_pairs, device, vocab, n_eval=args.eval_pairs)
    print(f"  initial ppl: {baseline['ppl']:.2f}  acc: {baseline['acc']:.3f}")

    print(f"\n=== training {args.steps} steps ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    loss_hist = []
    ckpts = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        idxs = random.sample(range(len(train_pairs)),
                             min(args.batch_size, len(train_pairs)))
        ctxs = torch.stack([train_pairs[i][0] for i in idxs]).to(device)
        tgts = torch.stack([train_pairs[i][1] for i in idxs]).to(device)

        logits = model(ctxs)
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgts.reshape(-1))
        if not torch.isfinite(loss):
            optimizer.zero_grad(); continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_hist.append(float(loss.item()))
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            print(f"  step {step:>5d}  loss={loss.item():.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0:
            ck = evaluate(model, heldout_pairs, device, vocab, n_eval=args.eval_pairs)
            ck["step"] = step + 1
            ck["train_loss"] = float(loss.item())
            ckpts.append(ck)
            print(f"  [eval step {step+1}] ppl={ck['ppl']:.2f}  acc={ck['acc']:.3f}",
                  flush=True)
            model.train()

    total_time = time.perf_counter() - t0

    print(f"\n=== final trajectory ===")
    print(f"  {'step':>6}  {'ppl':>10}  {'acc':>7}")
    print(f"  {0:>6}  {baseline['ppl']:>10.2f}  {baseline['acc']:>7.3f}  (initial)")
    for ck in ckpts:
        print(f"  {ck['step']:>6}  {ck['ppl']:>10.2f}  {ck['acc']:>7.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "arch": args.arch, "tokenizer": args.tokenizer, "vocab": vocab,
            "trainable_params": trainable, "total_params": total,
            "config": vars(args), "baseline": baseline,
            "checkpoints": ckpts, "total_wall_seconds": total_time,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
