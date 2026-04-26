"""
KV-Medusa on CPU: predict future K/V cache entries from hidden states.

Adapted from Strix's GPU pipeline. Uses base model (no compression).
Trains 10 heads one at a time, ~100 steps each.

Each KV-Medusa head predicts K and V at position t+offset given
hidden state at position t. Target dimension = n_kv_heads * head_dim.

Architecture per head:
  k_pred: Linear(d_model, d_model//2) -> SiLU -> Linear(d_model//2, d_kv)
  v_pred: same

Trained on OpenWebText, evaluated on held-out samples.
Measures cosine similarity and acceptance rate.
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVMedusaHead(nn.Module):
    """Predict future K and V cache entries from hidden state."""
    def __init__(self, d_model, n_kv_heads, head_dim):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        d_kv = n_kv_heads * head_dim
        self.k_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_kv, bias=False),
        )
        self.v_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_kv, bias=False),
        )

    def forward(self, h):
        """h: [batch, seq, d_model] -> K,V: [batch, seq, n_kv_heads, head_dim]"""
        b, s, _ = h.shape
        k = self.k_pred(h).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.v_pred(h).view(b, s, self.n_kv_heads, self.head_dim)
        return k, v


def load_data(tokenizer, seq_len=256, max_tokens=500_000):
    """Load OpenWebText for training."""
    from datasets import load_dataset
    print("  Loading OpenWebText...", flush=True)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts = []
    count = 0
    for ex in ds:
        texts.append(ex["text"])
        count += len(ex["text"]) // 4
        if count >= max_tokens * 1.2:
            break
    all_text = "\n\n".join(texts)
    tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]
    # Split train/val
    val_end = 50_000
    val_tokens = tokens[:val_end]
    train_tokens = tokens[val_end:val_end + max_tokens]

    def chunk(toks):
        n = len(toks) // (seq_len + 1)
        return toks[:n * (seq_len + 1)].view(n, seq_len + 1)

    train = chunk(train_tokens)
    val = chunk(val_tokens)
    print(f"  Train: {len(train)} chunks, Val: {len(val)} chunks")
    return train, val


def main():
    torch.set_num_threads(32)

    model_name = "Qwen/Qwen3-32B"
    seq_len = 256
    max_offsets = 10
    steps_per_head = 100
    eval_every = 25
    lr = 5e-4
    n_val = 10

    print("=" * 60)
    print(f"KV-MEDUSA ON CPU: {model_name}")
    print(f"  {max_offsets} heads (t+1 through t+{max_offsets})")
    print(f"  {steps_per_head} steps per head")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading data...", flush=True)
    train_chunks, val_chunks = load_data(tokenizer, seq_len)

    print(f"\nLoading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    d_model = model.config.hidden_size
    n_kv_heads = model.config.num_key_value_heads
    n_heads = model.config.num_attention_heads
    # head_dim from config if available, else derive from KV cache probe
    head_dim = getattr(model.config, 'head_dim', None)
    if head_dim is None:
        # Probe actual KV cache shape with a dummy forward
        with torch.no_grad():
            dummy = torch.tensor([[1, 2, 3]])
            dummy_out = model(input_ids=dummy, use_cache=True)
            pkv = dummy_out.past_key_values
            if hasattr(pkv, 'key_cache'):
                head_dim = pkv.key_cache[0].shape[-1]
            elif hasattr(pkv, 'layers'):
                head_dim = pkv.layers[0].keys.shape[-1]
            else:
                head_dim = pkv[0][0].shape[-1]
        del dummy_out
        gc.collect()
    d_kv = n_kv_heads * head_dim
    L = model.config.num_hidden_layers
    target_layer = L // 2

    print(f"  d_model={d_model}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, d_kv={d_kv}")
    print(f"  Target layer: {target_layer} (middle of {L})")
    print(flush=True)

    save_dir = "z8_pipeline_32b/kv_medusa_results"
    os.makedirs(save_dir, exist_ok=True)

    all_results = []
    t_start = time.time()

    for offset in range(1, max_offsets + 1):
        print(f"\n{'='*60}")
        print(f"KV-MEDUSA HEAD {offset} -- predict K/V at t+{offset}")
        print(f"{'='*60}", flush=True)

        head = KVMedusaHead(d_model, n_kv_heads, head_dim).float()
        head_params = sum(p.numel() for p in head.parameters())
        print(f"  Head params: {head_params/1e6:.1f}M", flush=True)

        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
        head.train()

        # Shuffle training order
        indices = list(range(len(train_chunks)))
        random.shuffle(indices)
        train_iter = iter(indices)

        history = []
        for step in range(1, steps_per_head + 1):
            # Get next batch
            try:
                idx = next(train_iter)
            except StopIteration:
                random.shuffle(indices)
                train_iter = iter(indices)
                idx = next(train_iter)

            batch = train_chunks[idx:idx+1]
            inp = batch[:, :seq_len]

            # Forward through base model to get hidden states and KV cache
            with torch.no_grad():
                out = model(input_ids=inp, use_cache=True, output_hidden_states=True)
                # Hidden states from last layer
                h = out.hidden_states[-1][:, :-offset].detach().float()

                # Actual K,V at target layer
                pkv = out.past_key_values
                # Access KV cache - handle different cache formats
                if hasattr(pkv, 'layers'):
                    layer_cache = pkv.layers[target_layer]
                    actual_k = layer_cache.keys    # [1, n_kv_heads, seq, head_dim]
                    actual_v = layer_cache.values
                elif hasattr(pkv, 'key_cache'):
                    actual_k = pkv.key_cache[target_layer]
                    actual_v = pkv.value_cache[target_layer]
                else:
                    # Tuple format
                    actual_k = pkv[target_layer][0]
                    actual_v = pkv[target_layer][1]

                # Target: KV at positions [offset:]
                target_k = actual_k[:, :, offset:].permute(0, 2, 1, 3).detach().float()
                target_v = actual_v[:, :, offset:].permute(0, 2, 1, 3).detach().float()

            # Trim to match
            min_len = min(h.shape[1], target_k.shape[1])
            h = h[:, :min_len]
            target_k = target_k[:, :min_len]
            target_v = target_v[:, :min_len]

            # Predict
            pred_k, pred_v = head(h)

            # Loss
            loss_k = F.mse_loss(pred_k, target_k)
            loss_v = F.mse_loss(pred_v, target_v)
            loss = loss_k + loss_v

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            if step % eval_every == 0 or step == steps_per_head:
                with torch.no_grad():
                    cos_k = F.cosine_similarity(
                        pred_k.reshape(-1, head_dim),
                        target_k.reshape(-1, head_dim), dim=-1).mean().item()
                    cos_v = F.cosine_similarity(
                        pred_v.reshape(-1, head_dim),
                        target_v.reshape(-1, head_dim), dim=-1).mean().item()
                    rel_k = ((pred_k - target_k).norm() / target_k.norm()).item()
                    rel_v = ((pred_v - target_v).norm() / target_v.norm()).item()

                elapsed = time.time() - t_start
                print(f"  step {step:>4}: loss={loss.item():.4f} "
                      f"cos_k={cos_k:.3f} cos_v={cos_v:.3f} "
                      f"rel_k={rel_k:.3f} rel_v={rel_v:.3f} "
                      f"[{elapsed:.0f}s]", flush=True)
                history.append({
                    "step": step, "loss": round(loss.item(), 5),
                    "cos_k": round(cos_k, 4), "cos_v": round(cos_v, 4),
                    "rel_k": round(rel_k, 4), "rel_v": round(rel_v, 4),
                })

        # Validation eval
        head.eval()
        val_cos_k, val_cos_v, val_rel_k, val_rel_v = [], [], [], []
        for vi in range(min(n_val, len(val_chunks))):
            vbatch = val_chunks[vi:vi+1, :seq_len]
            with torch.no_grad():
                out = model(input_ids=vbatch, use_cache=True, output_hidden_states=True)
                h = out.hidden_states[-1][:, :-offset].float()

                pkv = out.past_key_values
                if hasattr(pkv, 'layers'):
                    lc = pkv.layers[target_layer]
                    ak, av = lc.keys, lc.values
                elif hasattr(pkv, 'key_cache'):
                    ak = pkv.key_cache[target_layer]
                    av = pkv.value_cache[target_layer]
                else:
                    ak, av = pkv[target_layer]

                tk = ak[:, :, offset:].permute(0, 2, 1, 3).float()
                tv = av[:, :, offset:].permute(0, 2, 1, 3).float()

                ml = min(h.shape[1], tk.shape[1])
                pk, pv = head(h[:, :ml])

                val_cos_k.append(F.cosine_similarity(
                    pk.reshape(-1, head_dim), tk[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
                val_cos_v.append(F.cosine_similarity(
                    pv.reshape(-1, head_dim), tv[:, :ml].reshape(-1, head_dim), dim=-1).mean().item())
                val_rel_k.append(((pk - tk[:, :ml]).norm() / tk[:, :ml].norm()).item())
                val_rel_v.append(((pv - tv[:, :ml]).norm() / tv[:, :ml].norm()).item())

        fck = sum(val_cos_k) / len(val_cos_k)
        fcv = sum(val_cos_v) / len(val_cos_v)
        frk = sum(val_rel_k) / len(val_rel_k)
        frv = sum(val_rel_v) / len(val_rel_v)

        # Acceptance rate: what fraction of predicted K vectors have cos > 0.7?
        accept_k = sum(1 for c in val_cos_k if c > 0.7) / len(val_cos_k)

        quality = "GOOD" if fck > 0.8 else "OK" if fck > 0.5 else "WEAK"
        print(f"\n  HEAD {offset} FINAL: cos_k={fck:.3f} cos_v={fcv:.3f} "
              f"rel_k={frk:.3f} rel_v={frv:.3f} accept_k={accept_k:.1%} [{quality}]",
              flush=True)

        all_results.append({
            "offset": offset,
            "final_cos_k": round(fck, 4),
            "final_cos_v": round(fcv, 4),
            "final_rel_k": round(frk, 4),
            "final_rel_v": round(frv, 4),
            "accept_rate_k_07": round(accept_k, 4),
            "head_params_M": round(head_params / 1e6, 1),
            "history": history,
        })

        # Save incrementally
        with open(f"{save_dir}/kv_medusa_results.json", "w") as f:
            json.dump({"results": all_results, "model": model_name,
                       "target_layer": target_layer}, f, indent=2)

        del head, opt
        gc.collect()

    # Summary
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"KV-MEDUSA SUMMARY -- {model_name}")
    print(f"{'='*60}")
    print(f"  Target layer: {target_layer}")
    print(f"  d_kv: {d_kv}")
    print(f"  Steps per head: {steps_per_head}")
    print(f"  Total time: {elapsed_h:.2f}h")
    print()
    print(f"  {'Offset':>6} | {'cos_k':>6} {'cos_v':>6} | {'rel_k':>6} {'rel_v':>6} | {'Quality':>7}")
    print(f"  {'-'*6}-+-{'-'*13}-+-{'-'*13}-+-{'-'*7}")
    for r in all_results:
        quality = "GOOD" if r["final_cos_k"] > 0.8 else "OK" if r["final_cos_k"] > 0.5 else "WEAK"
        print(f"  t+{r['offset']:>3} | {r['final_cos_k']:.3f}  {r['final_cos_v']:.3f} | "
              f"{r['final_rel_k']:.3f}  {r['final_rel_v']:.3f} | {quality:>7}")

    good = sum(1 for r in all_results if r["final_cos_k"] > 0.7)
    ok = sum(1 for r in all_results if r["final_cos_k"] > 0.5)
    print(f"\n  Heads with cos_k > 0.7: {good}/{max_offsets}")
    print(f"  Heads with cos_k > 0.5: {ok}/{max_offsets}")
    print(f"  Potential draft tokens per step: {good}")

    # Estimate speedup
    if good > 0:
        # Each verified step = 1 model forward pass
        # With N good heads, we get N+1 tokens per step (1 real + N drafts)
        tokens_per_step = good + 1
        print(f"\n  Projected: {tokens_per_step} tokens per decode step")
        print(f"  Estimated speedup: {tokens_per_step:.1f}x over standard decoding")


if __name__ == "__main__":
    main()
