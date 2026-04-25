"""
Z8 Layer-wise Compression Pipeline for Qwen3-32B.

Self-contained script. Runs on CPU with 700GB RAM (Z8 hardware).
Uses layer-wise cached calibration (GPTQ/AWQ-style) instead of
full-model finetuning. ~50-400× faster per step.

Phases:
  1. Shape measurement (per-layer wormhole)
  2. Teacher capture (one forward pass, save activations)
  3a. Gentle weight anneal (q_proj, o_proj — layer-wise)
  3b. Aggressive KV anneal (k_proj, v_proj — layer-wise)
  4. Optional cleanup full-model FT
  5. Save compressed model + config

Expected wall time on Z8: ~12-30 hours for 32B.
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


# ============================================================
# Factored linear (matmul-based to be portable)
# ============================================================

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
    fac = FactoredLinear(A, B, bias)
    del W, U, S, Vt, sqrt_S
    gc.collect()
    return fac


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
    del W_eff, U, S, Vt, sqrt_S
    gc.collect()


# ============================================================
# Helpers
# ============================================================

def participation_ratio(X):
    if X.shape[0] == 0: return 0.0
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    return float((s2.sum().pow(2) / s2.pow(2).sum().clamp(min=1e-20)).item())


def free_memory():
    gc.collect()


def load_tokens(tokenizer, max_tokens, split="train"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= max_tokens: break
    return toks[:max_tokens]


def build_batches(tokens, seq_len, batch_size):
    n = (len(tokens) - 1) // seq_len
    batches = []
    cur = []
    for i in range(n):
        start = i * seq_len
        window = tokens[start:start + seq_len + 1]
        if len(window) < 2: continue
        cur.append(window)
        if len(cur) == batch_size:
            t = torch.tensor(cur, dtype=torch.long)
            batches.append((t[:, :-1], t[:, 1:]))
            cur = []
    return batches


@torch.no_grad()
def eval_loss(model, val_batches, device, n_batches=8):
    model.eval()
    total = 0.0; n = 0
    for inp, tgt in val_batches[:n_batches]:
        inp_d = inp.to(device)
        tgt_d = tgt.to(device)
        logits = model(inp_d, use_cache=False).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt_d.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(1, n)


# ============================================================
# Layer-wise calibration — the speedup engine
# ============================================================

def capture_layer_io(model, batches, layer_idx, proj_name, device):
    """Capture (input, output) at one specific Linear via hooks.

    Returns X [N_total_tokens, in_dim], Y [N_total_tokens, out_dim].
    """
    target = getattr(model.model.layers[layer_idx].self_attn, proj_name)
    X_list, Y_list = [], []

    def hook(module, inputs, output):
        # Flatten batch and seq dims
        X_list.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu())
        Y_list.append(output.detach().reshape(-1, output.shape[-1]).cpu())

    h = target.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for inp, _ in batches:
            _ = model(inp.to(device), use_cache=False)
    h.remove()
    return torch.cat(X_list, dim=0), torch.cat(Y_list, dim=0)


def train_factored_on_cache(fac_linear, X, Y, device, dtype,
                              n_steps=80, batch_size=512, lr=1e-3):
    """Train factored linear A, B against cached (X, Y) pairs via MSE."""
    fac_linear.A.requires_grad = True
    fac_linear.B.requires_grad = True
    opt = torch.optim.AdamW([fac_linear.A, fac_linear.B], lr=lr, weight_decay=0.0)
    n = X.shape[0]
    last_loss = None
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,))
        xb = X[idx].to(device).to(dtype)
        yb = Y[idx].to(device).to(dtype)
        pred = fac_linear(xb)
        loss = F.mse_loss(pred.float(), yb.float())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([fac_linear.A, fac_linear.B], 1.0)
        opt.step()
        last_loss = loss.item()
    del opt
    free_memory()
    return last_loss


# ============================================================
# Main pipeline
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--out", default="z8_pipeline_32b/results/pipeline_results.json")
    p.add_argument("--save-dir", default="z8_pipeline_32b/checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--calib-batches", type=int, default=64,
                   help="N batches captured for layer-wise calibration")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--per-layer-steps", type=int, default=80,
                   help="Gradient steps per layer per anneal stage")
    p.add_argument("--per-layer-lr", type=float, default=1e-3)
    # Weight anneal (gentle on q, o)
    p.add_argument("--weight-step-factor", type=float, default=0.95)
    p.add_argument("--weight-target-ratio", type=float, default=0.80)
    p.add_argument("--weight-max-stages", type=int, default=8)
    # KV anneal (aggressive on k, v)
    p.add_argument("--kv-step-factor", type=float, default=0.85)
    p.add_argument("--kv-target-rank", type=int, default=64)
    p.add_argument("--kv-max-stages", type=int, default=15)
    p.add_argument("--recapture-every", type=int, default=3,
                   help="Re-run teacher every N stages to refresh captures "
                        "(mitigates compounding errors)")
    p.add_argument("--val-tokens", type=int, default=4000)
    p.add_argument("--quality-tolerance", type=float, default=0.5)
    args = p.parse_args()

    device = args.device
    dtype = torch.float32  # CPU-only; bf16 not widely supported pre-Sapphire-Rapids
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\n{'='*60}\n=== Loading {args.model} ===\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  L={L}  d={d}  loaded in {time.time()-t0:.0f}s")

    print("\nloading WikiText-103 tokens...")
    train_tokens_n = args.calib_batches * args.batch_size * (args.seq_len + 1) + 1000
    train_tokens = load_tokens(tokenizer, train_tokens_n, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")
    train_batches = build_batches(train_tokens, args.seq_len, args.batch_size)[:args.calib_batches]
    val_batches = build_batches(val_tokens, args.seq_len, 1)
    print(f"  train batches: {len(train_batches)}  val batches: {len(val_batches)}")

    results = {"model": args.model, "L": L, "d": d}

    # ====================================================================
    # PHASE 1: Shape measurement
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 1: Shape measurement ===\n{'='*60}")
    t0 = time.time()
    loss_base = eval_loss(model, val_batches, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    print("\nmeasuring per-layer wormhole shape on 1 sequence...")
    ids = train_batches[0][0][:1].to(device)
    with torch.no_grad():
        out_meas = model(ids, use_cache=True, output_hidden_states=True)
    res_pr = [participation_ratio(h[0].float().cpu()) for h in out_meas.hidden_states]
    res_norm = [float(h[0].float().pow(2).sum().sqrt().item()) for h in out_meas.hidden_states]
    throat_pr = min(res_pr)
    throat_layer = res_pr.index(throat_pr)
    mag_pump = max(res_norm) / max(min(res_norm), 1e-6)
    print(f"  → throat PR={throat_pr:.2f} at L{throat_layer}  pump={mag_pump:.0f}×")
    print(f"  → measured in {time.time()-t0:.0f}s")
    del out_meas
    free_memory()

    results["wormhole_shape"] = {
        "residual_pr": res_pr, "throat_pr": throat_pr,
        "throat_layer": throat_layer, "magnitude_pump": mag_pump,
    }
    results["baseline_loss"] = loss_base
    results["baseline_ppl"] = ppl_base

    # ====================================================================
    # PHASE 2: Factorize all attention projections at full rank
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 2: Factorize all attention projections ===\n{'='*60}")
    weight_factored = {}  # q_proj, o_proj
    kv_factored = {}      # k_proj, v_proj
    print("\n  factorizing all attention projections at full rank...")
    t0 = time.time()
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in ["q_proj", "o_proj"]:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            weight_factored[(l, name)] = fac
        for name in ["k_proj", "v_proj"]:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            kv_factored[(l, name)] = fac
    print(f"  done in {time.time()-t0:.0f}s")

    loss_init = eval_loss(model, val_batches, device)
    print(f"  factorize sanity: loss={loss_init:.4f}  Δ={loss_init - loss_base:+.4f}")
    if loss_init - loss_base > 0.5:
        print("  SANITY CHECK FAILED")
        return

    cur_ranks = {(l, name): weight_factored[(l, name)].A.shape[1]
                  for (l, name) in weight_factored}
    cur_ranks.update({(l, name): kv_factored[(l, name)].A.shape[1]
                      for (l, name) in kv_factored})

    # ====================================================================
    # PHASE 3a: Gentle weight anneal — layer-wise on q_proj, o_proj
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 3a: Gentle weight anneal (layer-wise) ===\n{'='*60}")
    print(f"  step factor: {args.weight_step_factor}  target: {args.weight_target_ratio*100:.0f}% retained")

    initial_ranks_w = {k: cur_ranks[k] for k in weight_factored}
    weight_stages = []

    # Capture baseline I/O for q_proj, o_proj at every layer (one teacher pass)
    print(f"\n  capturing teacher I/O for q_proj, o_proj (one full forward)...")
    t0 = time.time()
    captures_w = {}
    for l in range(L):
        for name in ["q_proj", "o_proj"]:
            X, Y = capture_layer_io(model, train_batches[:8], l, name, device)
            captures_w[(l, name)] = (X, Y)
        if (l + 1) % 5 == 0:
            print(f"    captured up to L{l}")
    print(f"  capture done in {time.time()-t0:.0f}s")

    for stage in range(args.weight_max_stages):
        any_changed = False
        for (l, name), fac in weight_factored.items():
            target = max(int(initial_ranks_w[(l, name)] * args.weight_target_ratio), 1)
            new_r = max(target, int(round(cur_ranks[(l, name)] * args.weight_step_factor)))
            if new_r < cur_ranks[(l, name)]:
                refactorize(fac, new_r, device, dtype)
                cur_ranks[(l, name)] = new_r
                any_changed = True
        if not any_changed: break

        print(f"\n  weight stage {stage+1}: layer-wise training {len(weight_factored)} projections")
        t0 = time.time()
        ft_losses = []
        for (l, name), fac in weight_factored.items():
            X, Y = captures_w[(l, name)]
            l_loss = train_factored_on_cache(
                fac, X, Y, device, dtype,
                n_steps=args.per_layer_steps, lr=args.per_layer_lr)
            ft_losses.append(l_loss)
        loss_post = eval_loss(model, val_batches, device)
        delta = loss_post - loss_base
        avg_w = sum(cur_ranks[k] for k in weight_factored) / len(weight_factored)
        dur = time.time() - t0
        print(f"    avg rank={avg_w:.0f}  val loss={loss_post:.4f}  Δ={delta:+.3f}  "
              f"({dur:.0f}s, mean ft_mse={np.mean(ft_losses):.6f})")
        weight_stages.append({"stage": stage+1, "avg_rank": avg_w,
                              "val_loss": loss_post, "delta": delta,
                              "mean_ft_mse": float(np.mean(ft_losses)),
                              "duration_s": dur})

        # Save incremental
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            results_partial = dict(results)
            results_partial["weight_stages"] = weight_stages
            json.dump(results_partial, f, indent=2)

        if delta > args.quality_tolerance:
            print(f"    halting weight anneal (Δ {delta:+.3f} > tolerance)")
            break

    initial_avg_w = sum(initial_ranks_w.values()) / len(initial_ranks_w)
    final_avg_w = sum(cur_ranks[k] for k in weight_factored) / len(weight_factored)
    weight_compression = initial_avg_w / max(final_avg_w, 1)
    print(f"\n  weight anneal: avg rank {initial_avg_w:.0f} → {final_avg_w:.0f}  "
          f"({weight_compression:.2f}× compression)")

    del captures_w
    free_memory()

    # ====================================================================
    # PHASE 3b: Aggressive KV anneal — layer-wise on k_proj, v_proj
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 3b: Aggressive KV anneal (layer-wise) ===\n{'='*60}")
    print(f"  step factor: {args.kv_step_factor}  target rank: {args.kv_target_rank}")

    initial_ranks_kv = {k: cur_ranks[k] for k in kv_factored}
    kv_stages = []

    captures_kv = None  # will (re-)build periodically

    for stage in range(args.kv_max_stages):
        any_changed = False
        for k in kv_factored:
            new_r = max(args.kv_target_rank, int(round(cur_ranks[k] * args.kv_step_factor)))
            if new_r < cur_ranks[k]:
                refactorize(kv_factored[k], new_r, device, dtype)
                cur_ranks[k] = new_r
                any_changed = True
        if not any_changed: break

        # Re-capture every N stages (to track drift from compression of earlier layers)
        if stage == 0 or stage % args.recapture_every == 0:
            print(f"\n  KV stage {stage+1}: re-capturing teacher I/O ({len(kv_factored)} projections)...")
            t_cap = time.time()
            captures_kv = {}
            for l in range(L):
                for name in ["k_proj", "v_proj"]:
                    X, Y = capture_layer_io(model, train_batches[:8], l, name, device)
                    captures_kv[(l, name)] = (X, Y)
            print(f"    re-capture in {time.time()-t_cap:.0f}s")
        else:
            print(f"\n  KV stage {stage+1}: using captures from stage {(stage // args.recapture_every) * args.recapture_every}")

        t0 = time.time()
        ft_losses = []
        for (l, name), fac in kv_factored.items():
            X, Y = captures_kv[(l, name)]
            l_loss = train_factored_on_cache(
                fac, X, Y, device, dtype,
                n_steps=args.per_layer_steps, lr=args.per_layer_lr)
            ft_losses.append(l_loss)
        loss_post = eval_loss(model, val_batches, device)
        delta = loss_post - loss_base
        avg_kv = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)
        dur = time.time() - t0
        print(f"    avg KV rank={avg_kv:.0f}  val loss={loss_post:.4f}  Δ={delta:+.3f}  "
              f"({dur:.0f}s, mean ft_mse={np.mean(ft_losses):.6f})")
        kv_stages.append({"stage": stage+1, "avg_kv_rank": avg_kv,
                          "val_loss": loss_post, "delta": delta,
                          "mean_ft_mse": float(np.mean(ft_losses)),
                          "duration_s": dur})

        # Save incremental
        with open(args.out, "w") as f:
            results_partial = dict(results)
            results_partial["weight_stages"] = weight_stages
            results_partial["kv_stages"] = kv_stages
            json.dump(results_partial, f, indent=2)

        if delta > args.quality_tolerance:
            print(f"    halting KV anneal (Δ {delta:+.3f} > tolerance)")
            break

    initial_avg_kv = sum(initial_ranks_kv.values()) / len(initial_ranks_kv)
    final_avg_kv = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)
    kv_compression = initial_avg_kv / max(final_avg_kv, 1)
    print(f"\n  KV anneal: avg rank {initial_avg_kv:.0f} → {final_avg_kv:.0f}  "
          f"({kv_compression:.2f}× compression)")

    # ====================================================================
    # PHASE 4: (Optional) Cleanup full-model FT — skipped by default on Z8
    # Z8 CPU is too slow for full FT. Layer-wise quality is the deliverable.
    # ====================================================================

    final_loss = eval_loss(model, val_batches, device)
    final_ppl = float(np.exp(final_loss))

    results.update({
        "weight_stages": weight_stages,
        "kv_stages": kv_stages,
        "weight_compression_ratio": weight_compression,
        "kv_compression_ratio": kv_compression,
        "final_loss": final_loss, "final_ppl": final_ppl,
        "final_delta": final_loss - loss_base,
    })

    # ====================================================================
    # SAVE
    # ====================================================================
    print(f"\n{'='*60}\n=== Saving compressed model ===\n{'='*60}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "base_model": args.model, "L": L, "d": d,
        "ranks": {f"{l}.{name}": cur_ranks[(l, name)] for (l, name) in cur_ranks},
        "throat_layer": throat_layer,
        "weight_compression_ratio": weight_compression,
        "kv_compression_ratio": kv_compression,
        "baseline_ppl": ppl_base, "final_ppl": final_ppl,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model_state.pt")

    loader_code = '''"""Load Z8-compressed Qwen-32B model."""
import json, torch
import torch.nn as nn
from pathlib import Path


class FactoredLinear(nn.Module):
    def __init__(self, A, B, bias=None):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias = nn.Parameter(bias) if bias is not None else None
    def forward(self, x):
        out = (x @ self.B.T) @ self.A.T
        if self.bias is not None: out = out + self.bias
        return out


def load_compressed_model(checkpoint_dir, device="cpu", dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cd = Path(checkpoint_dir)
    config = json.load(open(cd / "config.json"))
    base = config["base_model"]

    print(f"loading base {base}...")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device)

    for l in range(config["L"]):
        attn = model.model.layers[l].self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"{l}.{name}"
            if key not in config["ranks"]: continue
            r = config["ranks"][key]
            proj = getattr(attn, name)
            out_dim, in_dim = proj.weight.shape
            A = torch.zeros(out_dim, r)
            B = torch.zeros(r, in_dim)
            bias = torch.zeros(proj.bias.shape) if proj.bias is not None else None
            setattr(attn, name, FactoredLinear(A, B, bias))

    state = torch.load(cd / "model_state.pt", map_location=device)
    model.load_state_dict(state)
    return model, tokenizer
'''
    with open(save_dir / "load.py", "w") as f:
        f.write(loader_code)
    print(f"  saved {save_dir}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}")
    print(f"  baseline PPL: {ppl_base:.2f}")
    print(f"  final PPL:    {final_ppl:.2f}  Δ={final_loss - loss_base:+.3f}")
    print(f"  weight compression: {initial_avg_w:.0f} → {final_avg_w:.0f}  ({weight_compression:.2f}×)")
    print(f"  KV compression:     {initial_avg_kv:.0f} → {final_avg_kv:.0f}  ({kv_compression:.2f}×)")
    print(f"  throat layer: L{throat_layer}  (PR={throat_pr:.2f})")
    print(f"\n  ready for downstream Medusa experiments")
    print(f"  load with: from z8_pipeline_32b.checkpoints.load import load_compressed_model")


if __name__ == "__main__":
    main()
