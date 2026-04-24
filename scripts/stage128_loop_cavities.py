"""
Stage 128 — Test the cavities-as-loops hypothesis on 0.6B.

Finding 15 identified two cavity clusters in Qwen3-0.6B:
  - L8-L10 triple cavity (three rank-1 layers in a row)
  - L23-L24 double cavity

Hypothesis: cavities are redundant refinement that could be done by
iterating the preceding wall layer's computation. If true, replacing
cavity sequences with loops of the preceding wall should recover most
of the quality at fewer effective compute steps.

Each test measures PPL on WikiText-2 val.

Tests:
  baseline: unmodified model
  drop_L8_10: skip L8, L9, L10 entirely (identity passthrough)
  drop_L23_24: skip L23, L24 entirely
  loop_L7_into_8_10: replace L8, L9, L10 with L7's module (shared weights,
                     3 extra applications of L7 to the residual)
  loop_L7_once: replace L8 with L7; drop L9, L10
  loop_L7_twice: replace L8, L9 with L7; drop L10
  drop_all_cavities: skip L4, L6, L8-L10, L15, L23, L24 (all cavities)
  loop_L22_into_23_24: replace L23, L24 with L22 module
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityDecoderLayer(nn.Module):
    """Drop-in replacement for a Qwen decoder layer that just passes
       hidden_states through unchanged. Copies attention_type attribute
       because Qwen3's model.forward reads it per-layer."""
    def __init__(self, orig_layer):
        super().__init__()
        self.attention_type = getattr(orig_layer, "attention_type", "full_attention")

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def set_layer(model, idx, new_layer):
    """Install new_layer at position idx of model.model.layers."""
    model.model.layers[idx] = new_layer


def restore_original(model, originals):
    for idx, layer in originals.items():
        model.model.layers[idx] = layer


def load_tokens(tokenizer, max_tokens, split="validation"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def iter_batches(tokens, seq_len):
    n = (len(tokens) - 1) // seq_len
    for i in range(n):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        yield window


@torch.no_grad()
def eval_ppl(model, tokens, seq_len, device, n_batches=15):
    model.eval()
    total = 0.0
    n = 0
    for window in iter_batches(tokens, seq_len):
        inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([window[1:]], dtype=torch.long, device=device)
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1))
        total += loss.item()
        n += 1
        if n >= n_batches:
            break
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage128_loop_cavities.json")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()

    L = model.config.num_hidden_layers
    print(f"L={L}")

    print("loading WikiText-2 val tokens...")
    val_tokens = load_tokens(tok, 5000, "validation")

    # Keep originals so we can restore after each test
    originals = {i: model.model.layers[i] for i in range(L)}

    def reset():
        for i in range(L):
            model.model.layers[i] = originals[i]

    def measure(label):
        loss = eval_ppl(model, val_tokens, 256, device)
        ppl = float(np.exp(loss))
        print(f"  {label:40s}  loss={loss:.4f}  PPL={ppl:.2f}")
        return {"loss": loss, "ppl": ppl}

    print("\n=== baseline ===")
    reset()
    results = {"baseline": measure("baseline (unmodified)")}

    # --- Tests on L8-L10 triple cavity ---
    print("\n=== L8-L10 triple cavity ===")

    reset()
    for i in [8, 9, 10]:
        set_layer(model, i, IdentityDecoderLayer(originals[i]))
    results["drop_L8_10"] = measure("drop L8, L9, L10 (3 layers)")

    reset()
    set_layer(model, 8, originals[7])  # share L7's weights at position 8
    for i in [9, 10]:
        set_layer(model, i, IdentityDecoderLayer(originals[i]))
    results["loop_L7_once"] = measure("replace L8 with L7, drop L9+L10")

    reset()
    set_layer(model, 8, originals[7])
    set_layer(model, 9, originals[7])
    set_layer(model, 10, IdentityDecoderLayer(originals[10]))
    results["loop_L7_twice"] = measure("L8=L9=L7, drop L10")

    reset()
    for i in [8, 9, 10]:
        set_layer(model, i, originals[7])
    results["loop_L7_3x"] = measure("replace L8, L9, L10 all with L7")

    # --- Tests on L23-L24 double cavity ---
    print("\n=== L23-L24 double cavity ===")

    reset()
    for i in [23, 24]:
        set_layer(model, i, IdentityDecoderLayer(originals[i]))
    results["drop_L23_24"] = measure("drop L23, L24")

    reset()
    set_layer(model, 23, originals[22])
    set_layer(model, 24, IdentityDecoderLayer(originals[24]))
    results["loop_L22_once"] = measure("replace L23 with L22, drop L24")

    reset()
    set_layer(model, 23, originals[22])
    set_layer(model, 24, originals[22])
    results["loop_L22_twice"] = measure("L23=L24=L22")

    # --- Combined: drop all cavities ---
    print("\n=== all cavities ===")
    all_cavities = [4, 6, 8, 9, 10, 15, 23, 24]

    reset()
    for i in all_cavities:
        set_layer(model, i, IdentityDecoderLayer(originals[i]))
    results["drop_all_8_cavities"] = measure(f"drop all 8 cavities {all_cavities}")

    # --- Single-layer sanity checks: drop the hardest wall (L21) ---
    print("\n=== single-layer ablation sanity checks ===")
    reset()
    set_layer(model, 21, IdentityDecoderLayer(originals[21]))
    results["drop_L21_only"] = measure("drop L21 (hardest wall — should break)")

    reset()
    set_layer(model, 5, IdentityDecoderLayer(originals[5]))
    results["drop_L5_only"] = measure("drop L5 (entry wall — should break)")

    reset()
    set_layer(model, 8, IdentityDecoderLayer(originals[8]))
    results["drop_L8_only"] = measure("drop L8 (pure cavity — should barely hurt)")

    # Summary
    print(f"\n{'=' * 60}\n=== summary ===\n{'=' * 60}")
    base_ppl = results["baseline"]["ppl"]
    base_loss = results["baseline"]["loss"]
    for label, r in results.items():
        if label == "baseline": continue
        delta = r["loss"] - base_loss
        rel_ppl = r["ppl"] / base_ppl
        marker = " ✓ " if delta < 0.05 else " ~ " if delta < 0.3 else " ! " if delta < 1.0 else "XXX"
        print(f"  {label:28s}  PPL={r['ppl']:>8.2f}  "
              f"{rel_ppl:>5.2f}× baseline  Δloss={delta:+.3f}  {marker}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
