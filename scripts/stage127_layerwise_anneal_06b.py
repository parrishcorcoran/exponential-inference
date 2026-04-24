"""
Stage 127 — Layer-by-layer sequential rank anneal with ASVD whitening.

Process layers 2..L-2 sequentially. For each layer:
  1. Seed rank from the layer's weight singular-value spectrum at 95% EVR
  2. Anneal rank geometrically (× 0.85) down until PPL damage > threshold
  3. Back off, freeze that layer, move to next

Uses ASVD (activation-aware) whitening so SVD truncation minimizes
output error under the calibration activation distribution, not
reconstruction error on the weight matrix itself. Activation stats
collected once at the start from the unmodified model.

This is better-designed than stage 126:
  - Per-layer floors, not zone averages
  - Each layer's anneal sees predecessors' compression (sequential like GPTQ)
  - Initial rank seeded from weight spectrum, not full rank
  - ASVD whitening = minimizes loss-equivalent error, lower floor than naive SVD
  - Edges skipped natively (layers 0-1 and L-1 untouched)

Outputs: per-layer rank floor curve + cumulative PPL.
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
    """W ≈ A B with optional input-side unscaling (for ASVD).

       forward: out = ((x @ unscale.diag()) @ B.T) @ A.T
       If unscale is 1, reduces to vanilla factored linear.
       unscale absorbs the D^-1 from ASVD whitening.
    """
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)  # [out, k]
        self.B = nn.Parameter(B)  # [k, in]  -- already has unscaling folded in
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        # Explicit matmul: F.linear has precision issues on MPS for non-square.
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None:
            out = out + self.bias
        return out


def asvd_factorize(W, s_in, rank, device, dtype):
    """Activation-aware SVD:
       W̃ = W * s_in (scale W's cols by activation RMS per input dim)
       SVD(W̃) → Ũ Σ Ṽ^T
       A = Ũ_k diag(sqrt(Σ_k))   [out, k]
       B = diag(sqrt(Σ_k)) Ṽ_k^T diag(1/s_in)   [k, in]  (unscaling folded)
       Forward: y = x @ B^T @ A^T = x @ diag(1/s_in) @ Ṽ^T diag(Σ) Ũ^T = x @ W^T.
    """
    W_cpu = W.float().cpu()
    s_cpu = s_in.float().cpu().clamp(min=1e-6)
    Wt = W_cpu * s_cpu  # scales each column by s[i]
    U, S, Vt = torch.linalg.svd(Wt, full_matrices=False)
    k = min(rank, len(S))
    sqrtS = S[:k].sqrt()
    A = (U[:, :k] * sqrtS).to(dtype)
    B_pre = (sqrtS.unsqueeze(1) * Vt[:k])  # [k, in], not yet unscaled
    B = (B_pre / s_cpu).to(dtype)  # absorb D^-1
    return A.to(device), B.to(device)


def effective_rank_at_evr(W, s_in, target_evr=0.95):
    """Return the minimum rank needed to capture target_evr of the
       activation-weighted spectrum."""
    W_cpu = W.float().cpu()
    s_cpu = s_in.float().cpu().clamp(min=1e-6)
    Wt = W_cpu * s_cpu
    _, S, _ = torch.linalg.svd(Wt, full_matrices=False)
    S_sq = S ** 2
    evr = S_sq.cumsum(0) / S_sq.sum()
    idx = (evr >= target_evr).nonzero(as_tuple=True)[0]
    return int(idx[0]) + 1 if len(idx) else len(S)


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


def iter_batches(tokens, seq_len, batch_size, device, shuffle=False):
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
def eval_ppl(model, tokens, seq_len, device, n_batches=10):
    model.eval()
    total = 0.0
    n = 0
    for inp, tgt in iter_batches(tokens, seq_len, 1, device):
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            tgt.reshape(-1))
        total += loss.item()
        n += 1
        if n >= n_batches:
            break
    return total / max(1, n)


@torch.no_grad()
def collect_input_rms(model, tokenizer, passages, device,
                       layer_indices, proj_names, max_length=256):
    """For each (layer_idx, proj_name), compute per-input-dim RMS activation
       by hooking the Linear's input."""
    sum_sq = {(l, n): None for l in layer_indices for n in proj_names}
    count = {(l, n): 0 for l in layer_indices for n in proj_names}

    handles = []
    def make_hook(key):
        def hook(module, inputs):
            x = inputs[0]  # [..., in]
            x2 = x.float().pow(2).reshape(-1, x.shape[-1]).sum(dim=0).cpu()
            n = x.shape[:-1].numel()
            if sum_sq[key] is None:
                sum_sq[key] = x2
            else:
                sum_sq[key] = sum_sq[key] + x2
            count[key] += int(n)
        return hook

    for l in layer_indices:
        for n in proj_names:
            proj = getattr(model.model.layers[l].self_attn, n)
            handles.append(proj.register_forward_pre_hook(make_hook((l, n))))

    model.eval()
    for sent in passages:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(device)
        _ = model(ids, use_cache=False)

    for h in handles:
        h.remove()

    rms = {}
    for key, sq in sum_sq.items():
        rms[key] = (sq / max(1, count[key])).sqrt()
    return rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="results/stage127_layerwise_anneal.json")
    p.add_argument("--device", default=None)
    p.add_argument("--skip-edges", type=int, default=2,
                   help="Skip first N and last N layers")
    p.add_argument("--seed-evr", type=float, default=0.95,
                   help="Initial rank captures this fraction of spectrum")
    p.add_argument("--anneal-factor", type=float, default=0.85,
                   help="Rank multiplier per anneal step")
    p.add_argument("--loss-tolerance", type=float, default=0.05,
                   help="Max per-layer Δ loss before back-off")
    p.add_argument("--val-tokens", type=int, default=4000)
    p.add_argument("--n-calib-sents", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--min-rank", type=int, default=1)
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
    d = model.config.hidden_size
    print(f"L={L}  d={d}")

    # Layer range
    target_layers = list(range(args.skip_edges, L - args.skip_edges))
    print(f"target layers (skip {args.skip_edges} on each edge): "
          f"{target_layers[0]}..{target_layers[-1]} ({len(target_layers)} layers)")

    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Val tokens (for PPL)
    print("loading WikiText-2 val tokens for PPL...")
    val_tokens = load_tokens(tok, args.val_tokens, "validation")

    # Calibration sentences for ASVD stats (fixed throughout)
    print(f"loading {args.n_calib_sents} calibration sentences from WikiText-2 train...")
    train_tokens = load_tokens(tok, 30000, "train")
    # decode back a reasonable number of sentences: chop into seq_len windows
    calib_sents = []
    for start in range(0, len(train_tokens) - args.seq_len, args.seq_len):
        window = train_tokens[start:start + args.seq_len]
        text = tok.decode(window, skip_special_tokens=True)
        calib_sents.append(text)
        if len(calib_sents) >= args.n_calib_sents:
            break
    print(f"  using {len(calib_sents)} calibration sentences")

    # Collect activation RMS per (layer, proj) on UNMODIFIED model
    print("\ncollecting activation RMS for ASVD whitening...")
    t0 = time.time()
    rms = collect_input_rms(model, tok, calib_sents, device, target_layers, proj_names)
    print(f"  collected in {time.time()-t0:.0f}s")

    # Baseline PPL
    loss_base = eval_ppl(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"baseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # Results structure
    results = {
        "model": args.model,
        "baseline_loss": loss_base, "baseline_ppl": ppl_base,
        "target_layers": target_layers,
        "seed_evr": args.seed_evr,
        "anneal_factor": args.anneal_factor,
        "loss_tolerance": args.loss_tolerance,
        "per_layer": {},
        "running_loss_after_each_layer": [],
    }

    orig_linears = {}  # (l, name) -> nn.Linear (for restoration if needed)
    running_loss = loss_base

    for li, l in enumerate(target_layers):
        attn = model.model.layers[l].self_attn
        print(f"\n{'='*60}")
        print(f"=== layer {l} ({li+1}/{len(target_layers)}) ===")
        print(f"{'='*60}")

        # Seed ranks for each projection from activation-weighted spectrum
        seeds = {}
        max_ranks = {}
        for name in proj_names:
            proj = getattr(attn, name)
            orig_linears[(l, name)] = proj
            s_in = rms[(l, name)]
            max_r = min(proj.weight.shape)
            max_ranks[name] = max_r
            seeds[name] = min(effective_rank_at_evr(proj.weight.data, s_in,
                                                     args.seed_evr), max_r)
        print(f"  seed ranks @ EVR={args.seed_evr}: "
              + "  ".join(f"{n}={seeds[n]}/{max_ranks[n]}" for n in proj_names))

        # Anneal: treat all 4 projections uniformly via a multiplier on their seeds
        # mult starts at 1.0, decreases by anneal_factor each step
        mult = 1.0
        last_ok_mult = 1.0
        last_ok_ranks = dict(seeds)
        anneal_history = []

        # First: plant the seed rank
        for name in proj_names:
            A, B = asvd_factorize(orig_linears[(l, name)].weight.data,
                                   rms[(l, name)], seeds[name], device, dtype)
            fac = FactoredLinear(A, B, bias=None)
            setattr(attn, name, fac)

        # Evaluate at seed
        loss_seed = eval_ppl(model, val_tokens, args.seq_len, device)
        delta_seed = loss_seed - running_loss
        print(f"  seed: loss={loss_seed:.4f}  Δ vs pre-layer={delta_seed:+.4f}")
        anneal_history.append({"mult": 1.0, "ranks": dict(seeds),
                                "loss": loss_seed, "delta": delta_seed})

        if delta_seed > args.loss_tolerance:
            # Even the seed exceeds tolerance. Back off to higher rank.
            print(f"  seed exceeds tolerance; bumping rank up")
            # Try higher rank = 1/anneal_factor * seed
            bump_mult = 1.0 / args.anneal_factor
            bumped_ranks = {n: min(int(round(seeds[n] * bump_mult)), max_ranks[n])
                            for n in proj_names}
            for name in proj_names:
                A, B = asvd_factorize(orig_linears[(l, name)].weight.data,
                                       rms[(l, name)], bumped_ranks[name], device, dtype)
                setattr(attn, name, FactoredLinear(A, B, None))
            loss_bump = eval_ppl(model, val_tokens, args.seq_len, device)
            if loss_bump - running_loss <= args.loss_tolerance:
                last_ok_mult = bump_mult
                last_ok_ranks = bumped_ranks
                running_loss = loss_bump
                print(f"    bump ok: loss={loss_bump:.4f}  Δ={loss_bump - running_loss:+.4f}")
            else:
                # Give up, restore original linear
                for name in proj_names:
                    setattr(attn, name, orig_linears[(l, name)])
                loss_restore = eval_ppl(model, val_tokens, args.seq_len, device)
                print(f"    even bumped not ok. Restoring original. loss={loss_restore:.4f}")
                running_loss = loss_restore
                results["per_layer"][str(l)] = {
                    "status": "restored_to_original",
                    "seeds": seeds, "history": anneal_history,
                }
                results["running_loss_after_each_layer"].append(running_loss)
                continue
        else:
            last_ok_mult = 1.0
            last_ok_ranks = dict(seeds)
            running_loss = loss_seed

        # Geometric anneal down
        while True:
            mult *= args.anneal_factor
            trial_ranks = {n: max(args.min_rank,
                                   int(round(seeds[n] * mult)))
                            for n in proj_names}
            # Check if any rank hit floor
            if all(trial_ranks[n] <= args.min_rank for n in proj_names):
                print(f"  hit min_rank={args.min_rank}, stopping")
                break
            for name in proj_names:
                A, B = asvd_factorize(orig_linears[(l, name)].weight.data,
                                       rms[(l, name)], trial_ranks[name], device, dtype)
                setattr(attn, name, FactoredLinear(A, B, None))
            loss_trial = eval_ppl(model, val_tokens, args.seq_len, device)
            delta = loss_trial - running_loss
            anneal_history.append({"mult": mult, "ranks": dict(trial_ranks),
                                    "loss": loss_trial, "delta": delta})
            print(f"  mult={mult:.3f}  ranks={trial_ranks}  loss={loss_trial:.4f}  Δ={delta:+.4f}")
            if delta > args.loss_tolerance:
                # Back off to last ok
                for name in proj_names:
                    A, B = asvd_factorize(orig_linears[(l, name)].weight.data,
                                           rms[(l, name)], last_ok_ranks[name],
                                           device, dtype)
                    setattr(attn, name, FactoredLinear(A, B, None))
                print(f"  backed off to ranks={last_ok_ranks}")
                running_loss = eval_ppl(model, val_tokens, args.seq_len, device)
                print(f"  frozen: loss={running_loss:.4f}  "
                      f"Δ vs baseline={running_loss - loss_base:+.4f}")
                break
            last_ok_mult = mult
            last_ok_ranks = dict(trial_ranks)
            running_loss = loss_trial

        # Record
        n_fac_params = sum(last_ok_ranks[n] * (orig_linears[(l, n)].weight.shape[0]
                                                + orig_linears[(l, n)].weight.shape[1])
                           for n in proj_names)
        n_orig_params = sum(orig_linears[(l, n)].weight.numel() for n in proj_names)
        results["per_layer"][str(l)] = {
            "status": "annealed",
            "seeds": seeds,
            "frozen_ranks": last_ok_ranks,
            "history": anneal_history,
            "running_loss_after": running_loss,
            "delta_vs_baseline": running_loss - loss_base,
            "n_factored_params": n_fac_params,
            "n_original_params": n_orig_params,
            "compression_ratio": n_orig_params / max(1, n_fac_params),
        }
        results["running_loss_after_each_layer"].append(running_loss)

        # Save incrementally
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}")
    print(f"  baseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")
    print(f"  final:    loss={running_loss:.4f}  PPL={np.exp(running_loss):.2f}  "
          f"Δ={running_loss - loss_base:+.4f}")
    print("\n  per-layer floors (frozen ranks, q/k/v/o):")
    for l_str, r in results["per_layer"].items():
        fr = r.get("frozen_ranks", {})
        if fr:
            print(f"    L{int(l_str):>2}: q={fr['q_proj']:>4}  k={fr['k_proj']:>4}  "
                  f"v={fr['v_proj']:>4}  o={fr['o_proj']:>4}  "
                  f"(compr {r['compression_ratio']:.1f}×)")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
