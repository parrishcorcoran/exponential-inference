"""
Smoke test v2 — cached teacher distillation.

Same pipeline as v1 but with one key change:
  - Teacher runs ONCE per batch (cached) before any compression
  - During FT, student trains against cached teacher logits (KL) +
    cached teacher final hidden state (MSE regularizer)
  - No teacher forward during FT loop → much cheaper per step

Expected: same wall time, but ~3-5× more effective FT steps, which
should let KV anneal go much deeper than v1's halt-at-stage-1.

Cache cost: per batch (seq=256, d=1024, vocab=151936):
  - Final hidden state: 1MB (fp32)
  - Top-k=20 logits: ~16KB (indices + values)
  - Total per batch: ~1MB, ~50MB for 50 batches
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


def participation_ratio(X):
    if X.shape[0] == 0: return 0.0
    s = torch.linalg.svdvals(X.float())
    s2 = s.pow(2)
    return float((s2.sum().pow(2) / s2.pow(2).sum().clamp(min=1e-20)).item())


def free_mps_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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


def build_batches(tokens, seq_len, batch_size):
    """Pre-build all batches, return list of (input_ids, target_ids)."""
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
def cache_teacher_outputs(model, batches, device, top_k=50):
    """Run teacher forward on all batches once. Cache final hidden state
       and top-k logits per position.

       Returns list of dicts per batch with:
         - input_ids: [batch, seq]
         - target_ids: [batch, seq]
         - h_final: [batch, seq, d]  (post-norm hidden state, what LM head sees)
         - top_k_indices: [batch, seq, k]
         - top_k_logits: [batch, seq, k]
    """
    cache = []
    final_norm = model.model.norm
    for bi, (inp, tgt) in enumerate(batches):
        inp_d = inp.to(device)
        out = model(inp_d, use_cache=False, output_hidden_states=True)
        # Final hidden state BEFORE norm — we re-norm during training so
        # adapter stays consistent. Or use post-norm. Use post-norm since
        # that's what the LM head consumes.
        h_pre_norm = out.hidden_states[-1][0]  # [seq, d]
        h_final = final_norm(h_pre_norm).float().cpu()
        logits = out.logits[0].float().cpu()  # [seq, V]
        # Top-k for memory efficiency
        topk = logits.topk(top_k, dim=-1)
        cache.append({
            "input_ids": inp.cpu(),
            "target_ids": tgt.cpu(),
            "h_final": h_final.unsqueeze(0),  # [1, seq, d] to match batch dim
            "top_k_indices": topk.indices.unsqueeze(0),
            "top_k_logits": topk.values.unsqueeze(0),
        })
        if (bi + 1) % 20 == 0:
            print(f"    cached {bi+1}/{len(batches)} batches")
    return cache


def cached_distill_loss(student_logits, student_h_final, batch_cache, device,
                         alpha_mse=0.5, alpha_kl=1.0, alpha_ce=0.5):
    """Combined loss: KL on top-k logits + MSE on final hidden + CE on ground truth."""
    # Cached teacher data (move to device)
    target_ids = batch_cache["target_ids"].to(device)
    h_teacher = batch_cache["h_final"].to(device)
    topk_idx = batch_cache["top_k_indices"].to(device)
    topk_logits_teacher = batch_cache["top_k_logits"].to(device)

    # CE loss on ground truth (standard)
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(),
        target_ids.reshape(-1))

    # MSE on final hidden states
    mse_loss = F.mse_loss(student_h_final.float(), h_teacher.float())

    # KL on top-k: gather student's logits at the same indices
    student_topk = student_logits.gather(-1, topk_idx)
    teacher_logp = F.log_softmax(topk_logits_teacher.float(), dim=-1)
    student_logp = F.log_softmax(student_topk.float(), dim=-1)
    teacher_p = teacher_logp.exp()
    kl_loss = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean()

    total = alpha_ce * ce_loss + alpha_mse * mse_loss + alpha_kl * kl_loss
    return total, ce_loss.item(), mse_loss.item(), kl_loss.item()


@torch.no_grad()
def eval_loss_cached(model, val_batches, device, n_batches=8):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="pipeline_smoke_06b/results/smoke_v2_cached.json")
    p.add_argument("--save-dir", default="pipeline_smoke_06b/checkpoints_v2")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--cache-batches", type=int, default=40,
                   help="N teacher batches to cache")
    p.add_argument("--ft-steps", type=int, default=120,
                   help="FT steps per anneal stage (much higher than v1, since teacher is cached)")
    p.add_argument("--val-tokens", type=int, default=2000)
    p.add_argument("--weight-step-factor", type=float, default=0.95)
    p.add_argument("--weight-target-ratio", type=float, default=0.80)
    p.add_argument("--weight-tolerance", type=float, default=0.3)
    p.add_argument("--kv-step-factor", type=float, default=0.85)
    p.add_argument("--kv-target-rank", type=int, default=128)
    p.add_argument("--kv-tolerance", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=50)
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

    # Load tokens, build batches
    print("\nloading WikiText-2...")
    train_tokens_n = args.cache_batches * args.batch_size * (args.seq_len + 1) + 1000
    train_tokens = load_tokens(tokenizer, train_tokens_n, "train")
    val_tokens = load_tokens(tokenizer, args.val_tokens, "validation")
    train_batches = build_batches(train_tokens, args.seq_len, args.batch_size)[:args.cache_batches]
    val_batches = build_batches(val_tokens, args.seq_len, 1)
    print(f"  train batches: {len(train_batches)}  val batches: {len(val_batches)}")

    # ====================================================================
    # PHASE 0: TEACHER CACHE — run teacher once, cache outputs
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 0: Teacher cache ===\n{'='*60}")
    t0 = time.time()
    teacher_cache = cache_teacher_outputs(model, train_batches, device, top_k=args.top_k)
    print(f"  cached {len(teacher_cache)} batches in {time.time()-t0:.0f}s")
    free_mps_memory()

    # Baseline PPL using val batches
    loss_base = eval_loss_cached(model, val_batches, device)
    ppl_base = float(np.exp(loss_base))
    print(f"\nbaseline: loss={loss_base:.4f}  PPL={ppl_base:.2f}")

    # ====================================================================
    # PHASE 1: Shape measurement
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 1: Shape measurement ===\n{'='*60}")
    t0 = time.time()
    ids = train_batches[0][0][:1].to(device)
    with torch.no_grad():
        out_meas = model(ids, use_cache=True, output_hidden_states=True)
    res_pr = [participation_ratio(h[0].float().cpu()) for h in out_meas.hidden_states]
    res_norm = [float(h[0].float().pow(2).sum().sqrt().item()) for h in out_meas.hidden_states]
    throat_pr = min(res_pr)
    throat_layer = res_pr.index(throat_pr)
    mag_pump = max(res_norm) / max(min(res_norm), 1e-6)
    print(f"  measured in {time.time()-t0:.0f}s")
    print(f"  → throat PR={throat_pr:.2f} at L{throat_layer}  pump={mag_pump:.0f}×")
    del out_meas
    free_mps_memory()

    results = {
        "model": args.model, "L": L, "d": d,
        "wormhole_shape": {
            "residual_pr": res_pr, "throat_pr": throat_pr,
            "throat_layer": throat_layer, "magnitude_pump": mag_pump,
        },
        "baseline_loss": loss_base, "baseline_ppl": ppl_base,
    }

    # ====================================================================
    # PHASE 2: Factorize attention projections, train with cached teacher
    # ====================================================================
    print(f"\n{'='*60}\n=== PHASE 2: Factorize + cached-distill anneal ===\n{'='*60}")

    # Factorize ONLY q_proj, o_proj for weight axis. Keep k_proj, v_proj
    # at full rank — those are the KV target for aggressive Phase 3.
    weight_factored = {}
    weight_proj_names = ["q_proj", "o_proj"]
    print("\n  factorizing q_proj, o_proj at full rank...")
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in weight_proj_names:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            weight_factored[(l, name)] = fac

    # Factorize k_proj, v_proj at full rank too — needed for refactorize() later
    kv_factored = {}
    for l in range(L):
        attn = model.model.layers[l].self_attn
        for name in ["k_proj", "v_proj"]:
            proj = getattr(attn, name)
            max_r = min(proj.weight.shape)
            fac = factorize_linear(proj, max_r, device, dtype)
            setattr(attn, name, fac)
            kv_factored[(l, name)] = fac
    free_mps_memory()

    loss_init = eval_loss_cached(model, val_batches, device)
    print(f"  factorize sanity: loss={loss_init:.4f}  Δ={loss_init - loss_base:+.4f}")
    if loss_init - loss_base > 0.5:
        print("  SANITY CHECK FAILED")
        return

    # Freeze most params, train only A/B and final norm
    for p_ in model.parameters(): p_.requires_grad = False
    for fdict in [weight_factored, kv_factored]:
        for m in fdict.values():
            m.A.requires_grad = True; m.B.requires_grad = True
    for p_ in model.model.norm.parameters(): p_.requires_grad = True

    def trainable_params():
        ps = []
        for fdict in [weight_factored, kv_factored]:
            for m in fdict.values():
                ps += [m.A, m.B]
        for p_ in model.model.norm.parameters():
            ps.append(p_)
        return ps

    def cached_finetune(n_steps):
        """Fine-tune using cached teacher outputs."""
        opt = torch.optim.AdamW(trainable_params(), lr=args.lr, weight_decay=0.01)
        model.train()
        step = 0
        n_cache = len(teacher_cache)
        running_loss = 0.0
        while step < n_steps:
            # Cycle through cache
            batch_cache = teacher_cache[step % n_cache]
            inp = batch_cache["input_ids"].to(device)
            # Forward — need final hidden state, so output_hidden_states
            out = model(inp, use_cache=False, output_hidden_states=True)
            student_logits = out.logits  # [batch, seq, V]
            student_h = model.model.norm(out.hidden_states[-1])
            loss, ce, mse, kl = cached_distill_loss(
                student_logits, student_h, batch_cache, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params(), 1.0)
            opt.step()
            running_loss = loss.item()
            step += 1
        del opt
        free_mps_memory()
        return running_loss

    # ====================================================================
    # PHASE 2a: Gentle weight anneal (q_proj, o_proj)
    # ====================================================================
    print(f"\n  --- PHASE 2a: gentle weight (q_proj, o_proj) anneal ---")
    cur_ranks = {(l, name): weight_factored[(l, name)].A.shape[1] for (l, name) in weight_factored}
    cur_ranks.update({(l, name): kv_factored[(l, name)].A.shape[1] for (l, name) in kv_factored})
    initial_ranks_w = {k: cur_ranks[k] for k in weight_factored}
    weight_stages = []
    for stage in range(8):
        # Step weight ranks
        any_changed = False
        for (l, name) in weight_factored:
            target = max(int(initial_ranks_w[(l, name)] * args.weight_target_ratio), 1)
            new_r = max(target, int(round(cur_ranks[(l, name)] * args.weight_step_factor)))
            if new_r < cur_ranks[(l, name)]:
                refactorize(weight_factored[(l, name)], new_r, device, dtype)
                weight_factored[(l, name)].A.requires_grad = True
                weight_factored[(l, name)].B.requires_grad = True
                cur_ranks[(l, name)] = new_r
                any_changed = True
        if not any_changed: break

        ft_loss = cached_finetune(args.ft_steps)
        loss_post = eval_loss_cached(model, val_batches, device)
        delta = loss_post - loss_base
        avg_w = sum(cur_ranks[k] for k in weight_factored) / len(weight_factored)
        print(f"    stage {stage+1}: avg_w_rank={avg_w:.0f}  ft_loss={ft_loss:.4f}  "
              f"val_loss={loss_post:.4f}  Δ={delta:+.3f}")
        weight_stages.append({"stage": stage+1, "avg_rank": avg_w,
                              "ft_loss": ft_loss, "val_loss": loss_post, "delta": delta})
        if delta > args.weight_tolerance:
            print(f"    halting weight anneal")
            break

    # ====================================================================
    # PHASE 2b: Aggressive KV anneal (k_proj, v_proj)
    # ====================================================================
    print(f"\n  --- PHASE 2b: aggressive KV anneal ---")
    initial_ranks_kv = {k: cur_ranks[k] for k in kv_factored}
    kv_stages = []
    for stage in range(15):
        any_changed = False
        for k in kv_factored:
            new_r = max(args.kv_target_rank,
                        int(round(cur_ranks[k] * args.kv_step_factor)))
            if new_r < cur_ranks[k]:
                refactorize(kv_factored[k], new_r, device, dtype)
                kv_factored[k].A.requires_grad = True
                kv_factored[k].B.requires_grad = True
                cur_ranks[k] = new_r
                any_changed = True
        if not any_changed: break

        ft_loss = cached_finetune(args.ft_steps)
        loss_post = eval_loss_cached(model, val_batches, device)
        delta = loss_post - loss_base
        avg_kv = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)
        print(f"    stage {stage+1}: avg_kv_rank={avg_kv:.0f}  ft_loss={ft_loss:.4f}  "
              f"val_loss={loss_post:.4f}  Δ={delta:+.3f}")
        kv_stages.append({"stage": stage+1, "avg_kv_rank": avg_kv,
                          "ft_loss": ft_loss, "val_loss": loss_post, "delta": delta})
        if delta > args.kv_tolerance:
            print(f"    halting KV anneal at stage {stage+1}")
            break

    # Final summary
    initial_avg_w = sum(initial_ranks_w.values()) / len(initial_ranks_w)
    final_avg_w = sum(cur_ranks[k] for k in weight_factored) / len(weight_factored)
    initial_avg_kv = sum(initial_ranks_kv.values()) / len(initial_ranks_kv)
    final_avg_kv = sum(cur_ranks[k] for k in kv_factored) / len(kv_factored)

    final_loss = eval_loss_cached(model, val_batches, device)
    final_ppl = float(np.exp(final_loss))

    results.update({
        "weight_stages": weight_stages,
        "kv_stages": kv_stages,
        "weight_compression_ratio": initial_avg_w / max(final_avg_w, 1),
        "kv_compression_ratio": initial_avg_kv / max(final_avg_kv, 1),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "final_delta": final_loss - loss_base,
    })

    # Save
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model": args.model, "L": L, "d": d,
        "ranks": {f"{l}.{name}": cur_ranks[(l, name)] for (l, name) in cur_ranks},
        "throat_layer": throat_layer,
        "baseline_ppl": ppl_base, "final_ppl": final_ppl,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    torch.save(model.state_dict(), save_dir / "model_state.pt")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\n=== SUMMARY (v2 cached distill) ===\n{'='*60}")
    print(f"  baseline PPL: {ppl_base:.2f}")
    print(f"  final PPL:    {final_ppl:.2f}  Δ={final_loss - loss_base:+.3f}")
    print(f"  weight compression: {initial_avg_w:.0f} → {final_avg_w:.0f}  "
          f"({initial_avg_w/max(final_avg_w,1):.2f}×)")
    print(f"  KV compression:     {initial_avg_kv:.0f} → {final_avg_kv:.0f}  "
          f"({initial_avg_kv/max(final_avg_kv,1):.2f}×)")
    print(f"  saved to {save_dir}")


if __name__ == "__main__":
    main()
