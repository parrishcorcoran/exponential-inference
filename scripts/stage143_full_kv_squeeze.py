"""
Stage 143 — Full physical KV squeeze: all four KV levers, per-layer schedules.

Levers (all annealed simultaneously with finetune):
  Axis A4. K projection rank (per-layer schedule)
  Axis A5. V projection rank (uniform — V is content-rich everywhere)
  Axis A6. K cache bits (uniform Q-anneal)
  Axis A7. V cache bits (uniform Q-anneal)

Per-layer schedule from finding 15 + stage 138 topography:
  K rank floors:
    Mouth (L0-L5): 16
    Cavities (L6-L10, L12, L15, L23, L24): 16
    Walls (L8, L11, L13): 64
    Throat (L14): 32
    Exit gates (L19-L21): 96
    Mouth 2 (L22, L25-L27): 32-64

  V rank floor: 128 uniform (V is uniform high per stage 138)

  Bits floor: Q4 uniform on K and V

Thermostat: each iteration picks the (axis, layer) with most headroom,
takes a step, finetunes, accepts/reverts. Continues until all axes at
their floors or all rejected.

Excludes H2O eviction — awaiting our certainty-based replacement (D3).
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


class FactoredLinear(nn.Module):
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out


def factorize_linear(linear, rank, device, dtype):
    W = linear.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    bias = linear.bias.data.to(dtype).to(device) if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def refactorize(fac_linear, rank, device, dtype):
    with torch.no_grad():
        W_eff = (fac_linear.A.data.float().cpu() @ fac_linear.B.data.float().cpu())
    U, S, Vt = torch.linalg.svd(W_eff, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A_new = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B_new = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    fac_linear.A = nn.Parameter(A_new)
    fac_linear.B = nn.Parameter(B_new)


def quantize_inplace(fac_linear, bits, device, dtype):
    """Quantize A and B to target bits, write back as float (simulates lossy storage)."""
    if bits >= 16: return
    qmax = 2 ** (bits - 1) - 1 if bits > 1 else 1
    for name in ["A", "B"]:
        p = getattr(fac_linear, name)
        x = p.data.float()
        if bits == 1:
            scale = x.abs().mean(dim=-1, keepdim=True).clamp(min=1e-10)
            q = torch.sign(x) * scale
        else:
            scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-10) / qmax
            q = (torch.round(x / scale).clamp(-qmax, qmax)) * scale
        p.data.copy_(q.to(dtype).to(device))


# Floors set to absolute minimum (rank 1, bits 1) — let the thermostat
# discover the true per-layer floor by REJECTION, not pre-specification.
# Each (axis, layer) freezes when it can't recover with finetune.
K_RANK_FLOOR = {l: 1 for l in range(28)}
V_RANK_FLOOR = {l: 1 for l in range(28)}
K_BITS_FLOOR = 1
V_BITS_FLOOR = 1


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
    n = (len(tokens) - 1) // seq_len
    idx = list(range(n))
    if shuffle:
        import random
        random.shuffle(idx)
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
def eval_loss(model, tokens, seq_len, device, n_batches=10):
    model.eval()
    total = 0.0; n = 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device, shuffle=False):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        total += loss.item()
        n += 1
        if n >= n_batches: break
    return total / max(1, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage143_full_kv_squeeze.json")
    p.add_argument("--device", default=None)
    p.add_argument("--ft-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-tokens", type=int, default=80000)
    p.add_argument("--val-tokens", type=int, default=4000)
    p.add_argument("--tolerance-loss", type=float, default=0.5)
    p.add_argument("--max-iterations", type=int, default=80)
    p.add_argument("--rank-step-factor", type=float, default=0.85,
                   help="multiply rank by this each accepted step")
    p.add_argument("--bit-step", type=int, default=2,
                   help="bits decreased per step")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    print(f"L={L}")

    print("loading WikiText-2...")
    train_tokens = load_tokens(tokenizer, args.train_tokens, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")

    loss_base = eval_loss(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # Initial: factorize all k_proj, v_proj at full rank
    factored = {}
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in ["k_proj", "v_proj"]:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            factored[(l, name)] = fac

    loss_init = eval_loss(model, val_tokens, args.seq_len, device)
    print(f"  factorized full-rank sanity: loss={loss_init:.4f} PPL={np.exp(loss_init):.2f}")
    if loss_init - loss_base > 0.5:
        print("  SANITY CHECK FAILED")
        return

    # Per-layer state
    cur_K_rank = {l: factored[(l, "k_proj")].A.shape[1] for l in range(L)}
    cur_V_rank = {l: factored[(l, "v_proj")].A.shape[1] for l in range(L)}
    cur_K_bits = {l: 16 for l in range(L)}
    cur_V_bits = {l: 16 for l in range(L)}

    # Rejection memory: don't keep retrying same (axis, layer) at same target
    frozen = set()  # set of (axis, layer)

    # Freeze all params except factored A/B and norm
    for p_ in model.parameters(): p_.requires_grad = False
    for m in factored.values():
        m.A.requires_grad = True; m.B.requires_grad = True
    for p_ in model.model.norm.parameters(): p_.requires_grad = True

    def trainable_params():
        ps = []
        for m in factored.values():
            ps += [m.A, m.B]
        for p_ in model.model.norm.parameters():
            ps.append(p_)
        return ps

    def finetune(n_steps):
        opt = torch.optim.AdamW(trainable_params(), lr=args.lr, weight_decay=0.01)
        model.train()
        step = 0
        last_loss = None
        while step < n_steps:
            for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
                if step >= n_steps: break
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params(), 1.0)
                opt.step()
                last_loss = loss.item()
                step += 1
        return last_loss

    def headroom(axis, layer):
        """How much room is there to reduce on (axis, layer)?
           Returns ratio current/floor — higher = more room."""
        if (axis, layer) in frozen: return 0.0
        if axis == "K_rank":
            return cur_K_rank[layer] / max(K_RANK_FLOOR[layer], 1)
        elif axis == "V_rank":
            return cur_V_rank[layer] / max(V_RANK_FLOOR[layer], 1)
        elif axis == "K_bits":
            return cur_K_bits[layer] / K_BITS_FLOOR
        elif axis == "V_bits":
            return cur_V_bits[layer] / V_BITS_FLOOR
        return 0.0

    history = []
    accepted_steps = 0
    rejected_steps = 0

    print("\n=== thermostat squeeze ===")
    print(f"  K rank floors: {K_RANK_FLOOR}")
    print(f"  V rank floor: {V_RANK_FLOOR[0]} (uniform)")
    print(f"  K bits floor: {K_BITS_FLOOR}, V bits floor: {V_BITS_FLOOR}")

    for it in range(args.max_iterations):
        # Find (axis, layer) with most headroom, not frozen
        best_axis_layer = None
        best_room = 1.05  # only step if room > 1.05× over floor
        for axis in ["K_rank", "V_rank", "K_bits", "V_bits"]:
            for l in range(L):
                r = headroom(axis, l)
                if r > best_room:
                    best_room = r
                    best_axis_layer = (axis, l)

        if best_axis_layer is None:
            print(f"\n  iter {it}: all axes/layers at floor or frozen. Halting.")
            break

        axis, target_l = best_axis_layer

        # Snapshot state for revert
        k_state = {kk: v.data.clone() for kk, v in
                   factored[(target_l, "k_proj")].state_dict().items()}
        v_state = {kk: v.data.clone() for kk, v in
                   factored[(target_l, "v_proj")].state_dict().items()}
        old_k_rank = cur_K_rank[target_l]
        old_v_rank = cur_V_rank[target_l]
        old_k_bits = cur_K_bits[target_l]
        old_v_bits = cur_V_bits[target_l]

        # Apply step
        if axis == "K_rank":
            new_v = max(K_RANK_FLOOR[target_l],
                        int(round(cur_K_rank[target_l] * args.rank_step_factor)))
            refactorize(factored[(target_l, "k_proj")], new_v, device, dtype)
            if cur_K_bits[target_l] < 16:
                quantize_inplace(factored[(target_l, "k_proj")],
                                 cur_K_bits[target_l], device, dtype)
            factored[(target_l, "k_proj")].A.requires_grad = True
            factored[(target_l, "k_proj")].B.requires_grad = True
            step_label = f"K_rank L{target_l}: {old_k_rank} → {new_v}"
            cur_K_rank[target_l] = new_v
        elif axis == "V_rank":
            new_v = max(V_RANK_FLOOR[target_l],
                        int(round(cur_V_rank[target_l] * args.rank_step_factor)))
            refactorize(factored[(target_l, "v_proj")], new_v, device, dtype)
            if cur_V_bits[target_l] < 16:
                quantize_inplace(factored[(target_l, "v_proj")],
                                 cur_V_bits[target_l], device, dtype)
            factored[(target_l, "v_proj")].A.requires_grad = True
            factored[(target_l, "v_proj")].B.requires_grad = True
            step_label = f"V_rank L{target_l}: {old_v_rank} → {new_v}"
            cur_V_rank[target_l] = new_v
        elif axis == "K_bits":
            new_v = max(K_BITS_FLOOR, cur_K_bits[target_l] - args.bit_step)
            quantize_inplace(factored[(target_l, "k_proj")], new_v, device, dtype)
            factored[(target_l, "k_proj")].A.requires_grad = True
            factored[(target_l, "k_proj")].B.requires_grad = True
            step_label = f"K_bits L{target_l}: {old_k_bits} → {new_v}"
            cur_K_bits[target_l] = new_v
        elif axis == "V_bits":
            new_v = max(V_BITS_FLOOR, cur_V_bits[target_l] - args.bit_step)
            quantize_inplace(factored[(target_l, "v_proj")], new_v, device, dtype)
            factored[(target_l, "v_proj")].A.requires_grad = True
            factored[(target_l, "v_proj")].B.requires_grad = True
            step_label = f"V_bits L{target_l}: {old_v_bits} → {new_v}"
            cur_V_bits[target_l] = new_v

        # Finetune
        ft_loss = finetune(args.ft_steps)
        cur_loss = eval_loss(model, val_tokens, args.seq_len, device)
        delta = cur_loss - loss_base

        if delta < args.tolerance_loss:
            accepted_steps += 1
            print(f"  iter {it} [{accepted_steps:>3d}A] {step_label}  "
                  f"loss={cur_loss:.4f}  Δ={delta:+.3f}  ✓")
            history.append({"iter": it, "axis": axis, "layer": target_l,
                            "step": step_label, "loss": cur_loss, "delta": delta,
                            "accepted": True})
        else:
            rejected_steps += 1
            print(f"  iter {it} [{rejected_steps:>3d}R] {step_label}  "
                  f"loss={cur_loss:.4f}  Δ={delta:+.3f}  ✗ rejecting")
            # Revert
            fac_k = factored[(target_l, "k_proj")]
            fac_v = factored[(target_l, "v_proj")]
            fac_k.A = nn.Parameter(k_state["A"]); fac_k.B = nn.Parameter(k_state["B"])
            fac_v.A = nn.Parameter(v_state["A"]); fac_v.B = nn.Parameter(v_state["B"])
            cur_K_rank[target_l] = old_k_rank
            cur_V_rank[target_l] = old_v_rank
            cur_K_bits[target_l] = old_k_bits
            cur_V_bits[target_l] = old_v_bits
            frozen.add((axis, target_l))
            history.append({"iter": it, "axis": axis, "layer": target_l,
                            "step": step_label, "loss": cur_loss, "delta": delta,
                            "accepted": False})

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "baseline_loss": loss_base, "baseline_ppl": ppl_base,
                "tolerance": args.tolerance_loss,
                "history": history,
                "current_K_rank": cur_K_rank,
                "current_V_rank": cur_V_rank,
                "current_K_bits": cur_K_bits,
                "current_V_bits": cur_V_bits,
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "frozen": [list(x) for x in frozen],
            }, f, indent=2)

    # Final summary
    final_loss = eval_loss(model, val_tokens, args.seq_len, device)
    print(f"\n{'='*60}\n=== final state ===\n{'='*60}")
    print(f"  baseline: PPL={ppl_base:.2f}")
    print(f"  final:    PPL={np.exp(final_loss):.2f}  Δ={final_loss - loss_base:+.3f}")
    print(f"  accepted/rejected steps: {accepted_steps}/{rejected_steps}")
    avg_K = sum(cur_K_rank.values()) / L
    avg_V = sum(cur_V_rank.values()) / L
    avg_K_bits = sum(cur_K_bits.values()) / L
    avg_V_bits = sum(cur_V_bits.values()) / L
    print(f"  avg K rank: {avg_K:.1f}  V rank: {avg_V:.1f}")
    print(f"  avg K bits: {avg_K_bits:.1f}  V bits: {avg_V_bits:.1f}")

    # Compute total cache compression
    orig_cache_per_token = 2 * 1024 * 16  # K + V at d_kv=1024 in 16-bit
    new_cache_per_token = sum(
        cur_K_rank[l] * cur_K_bits[l] + cur_V_rank[l] * cur_V_bits[l]
        for l in range(L)) / L
    compression = orig_cache_per_token / max(new_cache_per_token, 1)
    print(f"  cache compression: {compression:.1f}× (per token, per layer avg)")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
