"""
Pipeline smoke test on Qwen3-0.6B.

Phase 1 → Phase 3a (gentle weight) → Phase 3b (aggressive KV) → Save.
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
# Factored linear (uses explicit matmul to avoid MPS F.linear bug)
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
        import random; random.shuffle(idx)
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
def eval_loss(model, tokens, seq_len, device, n_batches=8):
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


def free_mps_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================
# Pipeline
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="pipeline_smoke_06b/results/smoke_results.json")
    p.add_argument("--save-dir", default="pipeline_smoke_06b/checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--ft-steps", type=int, default=30,
                   help="FT steps per anneal stage")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train-tokens", type=int, default=20000)
    p.add_argument("--val-tokens", type=int, default=2000)
    # Gentle weight rank reduction (5% per step, target ~80% retained = 1.25× compression)
    p.add_argument("--weight-step-factor", type=float, default=0.95,
                   help="Gentle 5% reduction per step on weights")
    p.add_argument("--weight-target-ratio", type=float, default=0.85,
                   help="Stop weight anneal at this fraction of full rank")
    p.add_argument("--weight-tolerance", type=float, default=0.3)
    # Aggressive KV reduction (15% per step, target rank 64-128 from 1024 = 8-16×)
    p.add_argument("--kv-step-factor", type=float, default=0.75,
                   help="Aggressive 25% reduction per step on KV")
    p.add_argument("--kv-target-rank", type=int, default=64,
                   help="Push KV rank toward this floor")
    p.add_argument("--kv-tolerance", type=float, default=0.5)
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
    print(f"\n{'='*60}\n=== Loading {args.model} ===\n{'='*60}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  L={L}  d={d}  loaded in {time.time()-t0:.0f}s")

    print("\nloading WikiText-2...")
    train_tokens = load_tokens(tokenizer, args.train_tokens, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")

    results = {"model": args.model, "L": L, "d": d}

    # ====================================================================
    # PHASE 1: Shape measurement
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 1: Shape measurement ===\n{'='*60}")

    loss_base = eval_loss(model, val_tokens, args.seq_len, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    print("\nmeasuring per-layer wormhole shape...")
    t0 = time.time()
    ids = torch.tensor([train_tokens[:args.seq_len]], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
    res_pr = [participation_ratio(h[0].float().cpu()) for h in out.hidden_states]
    res_norm = [float(h[0].float().pow(2).sum().sqrt().item()) for h in out.hidden_states]
    kv = out.past_key_values
    if hasattr(kv, "layers") and kv.layers:
        pairs = [(c.keys, c.values) for c in kv.layers]
    elif hasattr(kv, "to_legacy_cache"):
        pairs = kv.to_legacy_cache()
    else:
        pairs = list(kv)
    K_pr = []; V_pr = []
    for K, V in pairs:
        Kf = K[0].transpose(0, 1).reshape(K.shape[2], -1).cpu().float()
        Vf = V[0].transpose(0, 1).reshape(V.shape[2], -1).cpu().float()
        K_pr.append(participation_ratio(Kf))
        V_pr.append(participation_ratio(Vf))
        del Kf, Vf

    throat_pr = min(res_pr)
    mouth_pr = max(res_pr)
    throat_layer = res_pr.index(throat_pr)
    mag_pump = max(res_norm) / max(min(res_norm), 1e-6)

    print(f"  measured in {time.time()-t0:.0f}s")
    print(f"  → throat PR={throat_pr:.2f} at L{throat_layer}  mouth PR={mouth_pr:.2f}  pump={mag_pump:.0f}×")
    print(f"  → K cache PR mean: {np.mean(K_pr):.2f}  V cache PR mean: {np.mean(V_pr):.2f}")

    results["wormhole_shape"] = {
        "residual_pr": res_pr, "residual_norm": res_norm,
        "K_pr": K_pr, "V_pr": V_pr,
        "throat_pr": throat_pr, "mouth_pr": mouth_pr,
        "throat_layer": throat_layer, "magnitude_pump": mag_pump,
    }

    del out; free_mps_memory()

    # ====================================================================
    # PHASE 3a: Gentle weight rank anneal
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 3a: Gentle weight rank anneal (~5% reductions) ===\n{'='*60}")
    print(f"  step factor: {args.weight_step_factor}, target {args.weight_target_ratio*100:.0f}% rank retention")

    # Factorize all q_proj, k_proj, v_proj, o_proj at full rank
    weight_factored = {}
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    print("\n  factorizing all attention projections at full rank...")
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in proj_names:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            weight_factored[(l, name)] = fac
    free_mps_memory()

    loss_init = eval_loss(model, val_tokens, args.seq_len, device)
    print(f"  factorize sanity: loss={loss_init:.4f}  Δ={loss_init - loss_base:+.4f}")
    if loss_init - loss_base > 0.5:
        print("  SANITY CHECK FAILED")
        return

    # Freeze most params, train only A/B and norm
    for p_ in model.parameters(): p_.requires_grad = False
    for m in weight_factored.values():
        m.A.requires_grad = True; m.B.requires_grad = True
    for p_ in model.model.norm.parameters(): p_.requires_grad = True

    def trainable_params():
        ps = []
        for m in weight_factored.values():
            ps += [m.A, m.B]
        for p_ in model.model.norm.parameters(): ps.append(p_)
        return ps

    def finetune(n_steps):
        opt = torch.optim.AdamW(trainable_params(), lr=args.lr, weight_decay=0.01)
        model.train()
        step = 0
        while step < n_steps:
            for inp, tgt in iter_batches(train_tokens, args.seq_len, args.batch_size, device):
                if step >= n_steps: break
                logits = model(inp, use_cache=False).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params(), 1.0)
                opt.step()
                step += 1
        del opt
        free_mps_memory()

    # Anneal weights gently — multiplicative reduction until target ratio reached
    cur_ranks = {(l, name): weight_factored[(l, name)].A.shape[1] for (l, name) in weight_factored}
    initial_ranks = dict(cur_ranks)
    weight_stages = []
    weight_stage_idx = 0
    max_weight_stages = 8  # ~5% per step × 8 = ~33% reduction floor
    while True:
        # Check if any layer can still reduce
        can_progress = False
        for (l, name) in weight_factored:
            target = max(int(initial_ranks[(l, name)] * args.weight_target_ratio), 1)
            if cur_ranks[(l, name)] > target:
                can_progress = True
                break
        if not can_progress or weight_stage_idx >= max_weight_stages:
            break

        weight_stage_idx += 1
        print(f"\n  weight stage {weight_stage_idx}: ×{args.weight_step_factor}")
        t0 = time.time()
        # Step all weights down
        for (l, name), fac in weight_factored.items():
            target = max(int(initial_ranks[(l, name)] * args.weight_target_ratio), 1)
            new_r = max(target, int(round(cur_ranks[(l, name)] * args.weight_step_factor)))
            if new_r < cur_ranks[(l, name)]:
                refactorize(fac, new_r, device, dtype)
                fac.A.requires_grad = True
                fac.B.requires_grad = True
                cur_ranks[(l, name)] = new_r
        free_mps_memory()

        finetune(args.ft_steps)
        loss_post = eval_loss(model, val_tokens, args.seq_len, device)
        delta = loss_post - loss_base
        avg_rank = sum(cur_ranks.values()) / len(cur_ranks)
        dur = time.time() - t0
        print(f"    avg rank={avg_rank:.0f}  loss={loss_post:.4f}  Δ={delta:+.3f}  ({dur:.0f}s)")
        weight_stages.append({
            "stage": weight_stage_idx,
            "avg_rank": avg_rank,
            "loss": loss_post, "delta": delta, "duration_s": dur,
        })

        if delta > args.weight_tolerance:
            print(f"    weight Δ {delta:+.3f} > {args.weight_tolerance}, halting weight anneal")
            break

    initial_avg_rank = sum(initial_ranks.values()) / len(initial_ranks)
    final_avg_rank = sum(cur_ranks.values()) / len(cur_ranks)
    weight_compression = initial_avg_rank / max(final_avg_rank, 1)
    results["weight_rank"] = {
        "initial_avg_rank": initial_avg_rank,
        "final_avg_rank": final_avg_rank,
        "compression_ratio": weight_compression,
        "stages": weight_stages,
    }
    print(f"\n  weight anneal done: avg rank {initial_avg_rank:.0f} → {final_avg_rank:.0f}  "
          f"({weight_compression:.2f}× param reduction on attention)")

    # ====================================================================
    # PHASE 3b: Aggressive KV cache compression
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 3b: Aggressive KV cache compression ===\n{'='*60}")
    print(f"  step factor: {args.kv_step_factor}, target rank ~{args.kv_target_rank}")

    # K and V are already factored from phase 3a — keep going
    kv_factored = {(l, name): weight_factored[(l, name)] for l in range(L) for name in ["k_proj", "v_proj"]}
    kv_initial_ranks = {k: cur_ranks[k] for k in kv_factored}
    kv_stages = []
    kv_stage_idx = 0
    max_kv_stages = 12

    while True:
        can_progress = False
        for k in kv_factored:
            if cur_ranks[k] > args.kv_target_rank:
                can_progress = True
                break
        if not can_progress or kv_stage_idx >= max_kv_stages:
            break

        kv_stage_idx += 1
        print(f"\n  KV stage {kv_stage_idx}: ×{args.kv_step_factor}")
        t0 = time.time()
        for (l, name), fac in kv_factored.items():
            new_r = max(args.kv_target_rank, int(round(cur_ranks[(l, name)] * args.kv_step_factor)))
            if new_r < cur_ranks[(l, name)]:
                refactorize(fac, new_r, device, dtype)
                fac.A.requires_grad = True
                fac.B.requires_grad = True
                cur_ranks[(l, name)] = new_r
        free_mps_memory()

        finetune(args.ft_steps)
        loss_post = eval_loss(model, val_tokens, args.seq_len, device)
        delta = loss_post - loss_base
        avg_kv_rank = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)
        dur = time.time() - t0
        print(f"    avg KV rank={avg_kv_rank:.0f}  loss={loss_post:.4f}  Δ={delta:+.3f}  ({dur:.0f}s)")
        kv_stages.append({
            "stage": kv_stage_idx,
            "avg_kv_rank": avg_kv_rank,
            "loss": loss_post, "delta": delta, "duration_s": dur,
        })

        if delta > args.kv_tolerance:
            print(f"    KV Δ {delta:+.3f} > {args.kv_tolerance}, halting KV anneal")
            break

    initial_avg_kv = sum(kv_initial_ranks.values()) / len(kv_initial_ranks)
    final_avg_kv = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)
    kv_compression = initial_avg_kv / max(final_avg_kv, 1)
    results["kv_compression"] = {
        "initial_avg_kv_rank": initial_avg_kv,
        "final_avg_kv_rank": final_avg_kv,
        "compression_ratio": kv_compression,
        "stages": kv_stages,
    }
    print(f"\n  KV anneal done: avg rank {initial_avg_kv:.0f} → {final_avg_kv:.0f}  "
          f"({kv_compression:.2f}× cache compression)")

    # Final eval
    final_loss = eval_loss(model, val_tokens, args.seq_len, device)
    final_ppl = float(np.exp(final_loss))
    results["baseline_loss"] = loss_base
    results["baseline_ppl"] = ppl_base
    results["final_loss"] = final_loss
    results["final_ppl"] = final_ppl
    results["final_delta"] = final_loss - loss_base

    # ====================================================================
    # SAVE
    # ====================================================================
    print(f"\n{'='*60}\n=== Saving compressed model ===\n{'='*60}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "base_model": args.model,
        "L": L, "d": d,
        "ranks": {f"{l}.{name}": cur_ranks[(l, name)]
                   for (l, name) in cur_ranks},
        "throat_layer": throat_layer,
        "wormhole_residual_pr": res_pr,
        "weight_compression_ratio": weight_compression,
        "kv_compression_ratio": kv_compression,
        "baseline_ppl": ppl_base,
        "final_ppl": final_ppl,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model_state.pt")

    loader_code = '''"""Load compressed Qwen model from pipeline smoke test."""
import json
import torch
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


def load_compressed_model(checkpoint_dir):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cd = Path(checkpoint_dir)
    config = json.load(open(cd / "config.json"))
    base = config["base_model"]

    print(f"loading base {base}...")
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager")

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

    state = torch.load(cd / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    return model, tokenizer
'''
    with open(save_dir / "load.py", "w") as f:
        f.write(loader_code)
    print(f"  saved {save_dir}")

    # Final results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}")
    print(f"  baseline PPL: {ppl_base:.2f}")
    print(f"  final PPL: {final_ppl:.2f}  Δ={final_loss - loss_base:+.3f}")
    print(f"  weight compression: {weight_compression:.2f}× (gentle)")
    print(f"  KV cache compression: {kv_compression:.2f}× (aggressive)")
    print(f"  throat layer: L{throat_layer} (PR={throat_pr:.2f})")
    print(f"\n  ready for downstream Medusa experiments")
    print(f"  load with: from pipeline_smoke_06b.checkpoints.load import load_compressed_model")
    print(f"\n  results: {out_path}")


if __name__ == "__main__":
    main()
