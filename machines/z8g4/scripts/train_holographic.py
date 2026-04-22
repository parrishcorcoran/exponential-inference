"""
Train the Holographic Multi-View Transformer on wikitext-2.

Standard next-token prediction (cross-entropy). No teacher needed.
The holographic architecture IS the hypothesis — if it learns language
with fewer sequential steps than a standard transformer, the
multi-view approach works.

Also trains a standard transformer baseline with matched parameter
count for comparison.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from holographic_model import build_holographic_model, HolographicTransformer


class StandardTransformer(nn.Module):
    """Minimal standard transformer for baseline comparison."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, head_dim, d_int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(StandardBlock(d_model, n_heads, head_dim, d_int))
        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(vocab_size, d_model, bias=False)
        self.lm_head.weight = self.embed.weight  # tie

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, input_ids, labels=None, **kw):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = F.linear(h, self.embed.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        tied = self.embed.weight.numel()
        return total - tied


class StandardBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_int, bias=False)
        self.up = nn.Linear(d_model, d_int, bias=False)
        self.down = nn.Linear(d_int, d_model, bias=False)

    def forward(self, h):
        B, T, _ = h.shape
        residual = h
        h = self.norm1(h)
        q = self.q(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.full((T, T), float('-inf'), device=h.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(h.dtype)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        h = residual + self.o(out)
        residual = h
        h = self.norm2(h)
        h = residual + self.down(F.silu(self.gate(h)) * self.up(h))
        return h


class WikitextDataset(Dataset):
    """Load wikitext and chunk into fixed-length sequences."""
    def __init__(self, tokenizer, seq_len=256, split="train", max_tokens=None, variant="wikitext-2-raw-v1"):
        from datasets import load_dataset
        ds = load_dataset("wikitext", variant, split=split)
        text = "\n".join(ds["text"])
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        if max_tokens:
            tokens = tokens[:max_tokens]
        # Chunk
        n_chunks = len(tokens) // (seq_len + 1)
        tokens = tokens[:n_chunks * (seq_len + 1)]
        self.chunks = tokens.view(n_chunks, seq_len + 1)
        print(f"  {split}: {len(tokens):,} tokens, {n_chunks} chunks of {seq_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]  # input, target


def train_epoch(model, loader, optimizer, scheduler, device, max_steps=None):
    model.train()
    total_loss = 0
    n_steps = 0
    t0 = time.time()

    for batch_idx, (input_ids, targets) in enumerate(loader):
        if max_steps and batch_idx >= max_steps:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        out = model(input_ids=input_ids, labels=targets)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_steps += 1

        if n_steps % 50 == 0:
            avg = total_loss / n_steps
            dt = time.time() - t0
            print(f"    step {n_steps:5d} | loss {avg:.4f} | {dt/n_steps*1000:.0f}ms/step", flush=True)

    return total_loss / max(n_steps, 1), n_steps


@torch.inference_mode()
def eval_model(model, loader, device, max_steps=100):
    model.eval()
    total_loss = 0
    n = 0
    for batch_idx, (input_ids, targets) in enumerate(loader):
        if batch_idx >= max_steps:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        out = model(input_ids=input_ids, labels=targets)
        total_loss += out["loss"].item()
        n += 1
    return total_loss / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-views", type=int, default=8)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--d-int", type=int, default=1536)
    p.add_argument("--n-kv-heads", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps-per-epoch", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=500000)
    p.add_argument("--baseline-layers", type=int, default=8,
                   help="Number of layers for the standard transformer baseline")
    p.add_argument("--dataset", default="wikitext-2-raw-v1",
                   help="wikitext-2-raw-v1 or wikitext-103-raw-v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="machines/z8g4/results/holographic_train.json")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    # Data
    print("\nLoading wikitext-2...", flush=True)
    train_ds = WikitextDataset(tokenizer, args.seq_len, "train", args.max_tokens, args.dataset)
    val_ds = WikitextDataset(tokenizer, args.seq_len, "validation", 50000, args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    results = {}

    # ================================================================
    # Train Holographic Model
    # ================================================================
    print("\n" + "="*60)
    print("HOLOGRAPHIC MODEL")
    print("="*60, flush=True)

    holo_model = build_holographic_model(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        n_views=args.n_views,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        d_int=args.d_int,
        n_kv_heads=args.n_kv_heads,
    ).to(device)

    holo_params = sum(p.numel() for p in holo_model.parameters())
    optimizer = torch.optim.AdamW(holo_model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = args.epochs * min(len(train_loader), args.max_steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    holo_history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        train_loss, steps = train_epoch(holo_model, train_loader, optimizer, scheduler,
                                        device, args.max_steps_per_epoch)
        val_loss = eval_model(holo_model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20))
        holo_history.append({"epoch": epoch+1, "train_loss": train_loss,
                            "val_loss": val_loss, "val_ppl": val_ppl})
        print(f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)

    holo_time = time.time() - t0

    # Generate sample
    holo_model.eval()
    sample_ids = tokenizer("The discovery that", return_tensors="pt")["input_ids"].to(device)
    gen = holo_model.generate(sample_ids, max_new_tokens=50)
    holo_sample = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"\n  Sample: {holo_sample[:200]}")

    results["holographic"] = {
        "params": holo_params,
        "history": holo_history,
        "final_val_loss": holo_history[-1]["val_loss"],
        "final_val_ppl": holo_history[-1]["val_ppl"],
        "train_time_s": holo_time,
        "sample": holo_sample[:300],
        "config": {
            "d_model": args.d_model, "n_blocks": args.n_blocks, "n_views": args.n_views,
            "n_heads": args.n_heads, "d_int": args.d_int,
        },
    }

    del holo_model, optimizer, scheduler

    # ================================================================
    # Train Standard Baseline (matched params)
    # ================================================================
    print("\n" + "="*60)
    print("STANDARD TRANSFORMER BASELINE")
    print("="*60, flush=True)

    std_model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.baseline_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        d_int=args.d_int,
    ).to(device)

    std_params = std_model.count_params()
    print(f"Standard Transformer: {std_params/1e6:.1f}M params, {args.baseline_layers} layers")

    optimizer = torch.optim.AdamW(std_model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    std_history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        train_loss, steps = train_epoch(std_model, train_loader, optimizer, scheduler,
                                        device, args.max_steps_per_epoch)
        val_loss = eval_model(std_model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20))
        std_history.append({"epoch": epoch+1, "train_loss": train_loss,
                           "val_loss": val_loss, "val_ppl": val_ppl})
        print(f"  train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.1f}", flush=True)

    std_time = time.time() - t0

    results["standard"] = {
        "params": std_params,
        "history": std_history,
        "final_val_loss": std_history[-1]["val_loss"],
        "final_val_ppl": std_history[-1]["val_ppl"],
        "train_time_s": std_time,
        "config": {"d_model": args.d_model, "n_layers": args.baseline_layers,
                   "n_heads": args.n_heads, "d_int": args.d_int},
    }

    # ================================================================
    # Comparison
    # ================================================================
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    h = results["holographic"]
    s = results["standard"]
    print(f"{'':20s} | {'Holographic':>12s} | {'Standard':>12s}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'Parameters':20s} | {h['params']/1e6:>10.1f}M | {s['params']/1e6:>10.1f}M")
    print(f"{'Val Loss':20s} | {h['final_val_loss']:>12.4f} | {s['final_val_loss']:>12.4f}")
    print(f"{'Val PPL':20s} | {h['final_val_ppl']:>12.1f} | {s['final_val_ppl']:>12.1f}")
    print(f"{'Train time':20s} | {h['train_time_s']/60:>10.1f}m | {s['train_time_s']/60:>10.1f}m")
    winner = "HOLOGRAPHIC" if h['final_val_ppl'] < s['final_val_ppl'] else "STANDARD"
    print(f"\n  Winner: {winner}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
