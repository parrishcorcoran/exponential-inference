"""
Stage 137 — Multi-axis squeeze on 0.6B with shape-aware schedule.

Combines findings 13-19 into a single trained-aware compression run:

  Axes annealed simultaneously (thermostat policy):
    1. K rank (per-layer, schedule from stage 138 topography)
    2. V rank (uniform moderate, since stage 138 showed V is uniform)
    3. K bits (uniform Q-anneal from 16 → 4 → 3)
    4. V bits (uniform, lags K since V is more rank-rich)

  Per-layer rank schedule (from stage 138 EVR-95 K profile):
    - Mouth/cavity layers (L0-L7, L10, L12, L15, L27): aggressive (target rank 16)
    - Wall layers (L5, L8, L11, L13): moderate (target rank 64)
    - Throat (L14): low (target rank 32)
    - Exit gates (L19-L21): protect (target rank 128)
    - Mouth 2 (L22-L26): moderate (target rank 64)

  Schedule: thermostat — try a step on any axis, accept if PPL doesn't
  climb past tolerance. Mix axes in round-robin so all advance.

  Fine-tune ~80 steps between each successful step.

Estimate target: ~30-50× cache compression at quality on 0.6B.
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


def quantize_factored(fac_linear, bits, device, dtype):
    """Quantize the factored A and B in place. Symmetric per-row scale."""
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
            q = torch.round(x / scale).clamp(-qmax, qmax) * scale
        setattr(fac_linear, name, nn.Parameter(q.to(dtype).to(device)))


# Per-layer rank targets, derived from stage 138 topography
# Indexed by layer 0..27. K targets and V targets.
K_RANK_FLOOR = {
    # Cavity/mouth: aggressive
    0: 16, 1: 16, 2: 16, 3: 16, 4: 16, 5: 32, 6: 64, 7: 32,
    # Throat walls and corridor (with finding 15 walls)
    8: 64, 9: 32, 10: 16, 11: 64, 12: 16, 13: 64, 14: 32,
    # Exit transition
    15: 16, 16: 64, 17: 48, 18: 64, 19: 96, 20: 96, 21: 128,
    # Buffer + mouth 2
    22: 16, 23: 16, 24: 16, 25: 64, 26: 64, 27: 32,
}
V_RANK_FLOOR = {l: 192 for l in range(28)}  # V is uniformly higher rank


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
    p.add_argument("--out", default="results/stage137_multi_axis.json")
    p.add_argument("--device", default=None)
    p.add_argument("--ft-steps", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-tokens", type=int, default=80000)
    p.add_argument("--val-tokens", type=int, default=4000)
    p.add_argument("--tolerance-loss", type=float, default=0.5,
                   help="max Δloss vs baseline before back-off")
    p.add_argument("--max-iterations", type=int, default=20,
                   help="thermostat rounds")
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
    d = model.config.hidden_size
    print(f"L={L}  d={d}")

    print("loading WikiText-2...")
    train_tokens = load_tokens(tokenizer, args.train_tokens, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")

    loss_base = eval_loss(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # Initial state: factor all k_proj, v_proj at full rank (= identity)
    factored = {}
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in ["k_proj", "v_proj"]:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            factored[(l, name)] = fac

    # Sanity: full-rank factorization should match baseline
    loss_init = eval_loss(model, val_tokens, args.seq_len, device)
    print(f"  factorized at full rank (sanity): loss={loss_init:.4f}  PPL={np.exp(loss_init):.2f}")
    if loss_init - loss_base > 0.5:
        print(f"  WARNING: factorization sanity check failed.")
        return

    # Per-layer current ranks (start at full)
    cur_K_rank = {l: factored[(l, "k_proj")].A.shape[1] for l in range(L)}
    cur_V_rank = {l: factored[(l, "v_proj")].A.shape[1] for l in range(L)}
    cur_K_bits = 16
    cur_V_bits = 16

    # Freeze all params except factored A/B and final norm
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

    history = []
    accepted_steps = 0

    print("\n=== thermostat squeeze ===")
    for it in range(args.max_iterations):
        # Round-robin try a step on each axis
        # Axis 1: K rank (per-layer towards floor)
        # Axis 2: V rank (uniform towards floor)
        # Axis 3: K bits (towards floor)
        # Axis 4: V bits (towards floor)

        progressed = False

        # AXIS 1: K rank step — multiplicative reduction on ALL layers at once
        # toward their per-layer floors
        any_k_room = any(cur_K_rank[l] > K_RANK_FLOOR[l] for l in range(L))
        if any_k_room:
            old_states_k = {}
            new_K_rank = {}
            for l in range(L):
                if cur_K_rank[l] > K_RANK_FLOOR[l]:
                    new_r = max(K_RANK_FLOOR[l], int(round(cur_K_rank[l] * 0.85)))
                    new_K_rank[l] = new_r
                    old_states_k[l] = {k: v.data.clone() for k, v in
                                       factored[(l, "k_proj")].state_dict().items()}
                    refactorize(factored[(l, "k_proj")], new_r, device, dtype)
                    factored[(l, "k_proj")].A.requires_grad = True
                    factored[(l, "k_proj")].B.requires_grad = True
                else:
                    new_K_rank[l] = cur_K_rank[l]

            print(f"\n  iter {it} A1: K rank uniform×0.85 (toward per-layer floors)")
            ft_loss = finetune(args.ft_steps)
            cur_loss = eval_loss(model, val_tokens, args.seq_len, device)
            delta = cur_loss - loss_base
            print(f"    K avg rank: {sum(new_K_rank.values())/L:.0f}  "
                  f"loss={cur_loss:.4f}  Δ baseline={delta:+.3f}")

            if delta < args.tolerance_loss:
                for l in range(L): cur_K_rank[l] = new_K_rank[l]
                accepted_steps += 1
                progressed = True
                history.append({"iter": it, "axis": "K_rank_all",
                                "avg_rank": sum(new_K_rank.values())/L,
                                "loss": cur_loss, "delta": delta, "accepted": True})
            else:
                print(f"    REJECTED. Reverting K ranks.")
                for l, st in old_states_k.items():
                    fac = factored[(l, "k_proj")]
                    fac.A = nn.Parameter(st["A"])
                    fac.B = nn.Parameter(st["B"])
                    if "bias" in st and st["bias"] is not None:
                        fac.bias = nn.Parameter(st["bias"])
                history.append({"iter": it, "axis": "K_rank_all",
                                "loss": cur_loss, "delta": delta, "accepted": False})

        # AXIS 2: V rank step (uniform)
        target_v = max(cur_V_rank.values())
        if target_v > V_RANK_FLOOR[0]:
            new_rank = max(V_RANK_FLOOR[0], int(round(target_v * 0.85)))
            old_states = {l: {k: v.data.clone() for k, v in
                              factored[(l, "v_proj")].state_dict().items()}
                          for l in range(L)}
            for l in range(L):
                if cur_V_rank[l] > new_rank:
                    refactorize(factored[(l, "v_proj")], new_rank, device, dtype)
                    factored[(l, "v_proj")].A.requires_grad = True
                    factored[(l, "v_proj")].B.requires_grad = True

            print(f"\n  iter {it} A2: V rank uniform {target_v} → {new_rank}")
            ft_loss = finetune(args.ft_steps)
            cur_loss = eval_loss(model, val_tokens, args.seq_len, device)
            delta = cur_loss - loss_base
            print(f"    loss={cur_loss:.4f}  Δ baseline={delta:+.3f}")

            if delta < args.tolerance_loss:
                for l in range(L):
                    cur_V_rank[l] = min(cur_V_rank[l], new_rank)
                accepted_steps += 1
                progressed = True
                history.append({"iter": it, "axis": "V_rank", "from": target_v,
                                "to": new_rank, "loss": cur_loss, "delta": delta,
                                "accepted": True})
            else:
                print(f"    REJECTED. Reverting V rank.")
                for l in range(L):
                    fac = factored[(l, "v_proj")]
                    fac.A = nn.Parameter(old_states[l]["A"])
                    fac.B = nn.Parameter(old_states[l]["B"])
                    if "bias" in old_states[l] and old_states[l]["bias"] is not None:
                        fac.bias = nn.Parameter(old_states[l]["bias"])
                history.append({"iter": it, "axis": "V_rank", "from": target_v,
                                "to": new_rank, "loss": cur_loss, "delta": delta,
                                "accepted": False})

        # Save incremental
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "baseline_loss": loss_base, "baseline_ppl": ppl_base,
                "tolerance": args.tolerance_loss,
                "history": history,
                "current_K_ranks": cur_K_rank,
                "current_V_ranks": cur_V_rank,
                "accepted_steps": accepted_steps,
            }, f, indent=2)

        if not progressed:
            print(f"\n  iter {it}: no axis could progress. Halting.")
            break

    # Final eval
    final_loss = eval_loss(model, val_tokens, args.seq_len, device)
    print(f"\n{'='*60}\n=== final state ===\n{'='*60}")
    print(f"  baseline: PPL={ppl_base:.2f}")
    print(f"  final:    PPL={np.exp(final_loss):.2f}  Δ={final_loss - loss_base:+.3f}")
    print(f"  accepted steps: {accepted_steps}")
    avg_K = sum(cur_K_rank.values()) / L
    avg_V = sum(cur_V_rank.values()) / L
    print(f"  avg K rank: {avg_K:.1f} (down from {min(model.config.hidden_size, 1024)})")
    print(f"  avg V rank: {avg_V:.1f}")

    # Final per-layer breakdown
    print(f"\n  per-layer K ranks: {dict(sorted(cur_K_rank.items()))}")
    print(f"  per-layer V ranks: {dict(sorted(cur_V_rank.items()))}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
