"""
Manifold Resonator: long training run.

Project Q,K to 10D manifold, compute attention there, retrieve full V.
Teacher-student MSE matching, then PID blend.

Phase 1: 5000 steps MSE matching (codebook only)
Phase 2: PID blend standard → manifold attention + LM fine-tune
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


class ManifoldResonator(nn.Module):
    def __init__(self, head_dim, manifold_dim=10):
        super().__init__()
        self.q_proj = nn.Linear(head_dim, manifold_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, manifold_dim, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        self.manifold_dim = manifold_dim

    def forward(self, Q, K, V):
        Q_m = self.q_proj(Q)
        K_m = self.k_proj(K)
        scores = Q_m @ K_m.transpose(-2, -1) / math.sqrt(self.manifold_dim)
        S = scores.shape[-1]
        mask = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return attn @ V


def load_data(seq_len=128, max_val=50_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    return tokens, tokens[:max_val]


@torch.inference_mode()
def eval_ppl(model, val_tokens, seq_len=128, n=15):
    model.eval()
    total = 0; c = 0
    n_chunks = len(val_tokens) // (seq_len + 1)
    chunks = val_tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)
    for i in range(min(n, n_chunks)):
        inp = chunks[i:i+1, :seq_len]
        tgt = chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        c += 1
    return math.exp(total / max(c, 1))


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--manifold-dim", type=int, default=10)
    ap.add_argument("--phase1-steps", type=int, default=5000)
    ap.add_argument("--phase2-rounds", type=int, default=100)
    ap.add_argument("--phase2-steps", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=128)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"MANIFOLD RESONATOR: {cli.model}")
    print(f"  Project Q,K to {cli.manifold_dim}D, attention there")
    print(f"  Phase 1: {cli.phase1_steps} steps MSE matching")
    print(f"  Phase 2: PID blend + LM fine-tune")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', d_model // n_heads)
    kv_groups = n_heads // n_kv

    print(f"  L={L}, d={d_model}, heads={n_heads}/{n_kv}, head_dim={hd}")

    all_tokens, val_tokens = load_data(seq_len=cli.seq_len)
    teacher_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # Resonators
    resonators = nn.ModuleList([
        ManifoldResonator(hd, manifold_dim=cli.manifold_dim) for _ in range(L)
    ])
    print(f"  Resonator params: {sum(p.numel() for p in resonators.parameters()):,}")

    # Pre-compute rotary + mask
    with torch.no_grad():
        dummy_inp = all_tokens[:cli.seq_len].unsqueeze(0)
        dummy_out = model(input_ids=dummy_inp, use_cache=False, output_hidden_states=True)
        position_ids = torch.arange(cli.seq_len).unsqueeze(0)
        rotary = model.model.rotary_emb(dummy_out.hidden_states[0], position_ids)
        causal = torch.triu(torch.ones(cli.seq_len, cli.seq_len), diagonal=1).bool()
        attn_mask = torch.where(causal.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min, 0.0)

    # ====================
    # PHASE 1: MSE matching
    # ====================
    print(f"\n{'='*60}")
    print(f"PHASE 1: MSE matching ({cli.phase1_steps} steps)")
    print(f"{'='*60}", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(resonators.parameters(), lr=5e-5, weight_decay=0.01)
    resonators.train()

    n_available = len(all_tokens) // (cli.seq_len + 1) - 1
    t_start = time.time()
    best_mse = float('inf')

    for step in range(1, cli.phase1_steps + 1):
        idx = random.randint(0, n_available - 1)
        start = idx * cli.seq_len
        inp = all_tokens[start:start + cli.seq_len].unsqueeze(0)

        with torch.no_grad():
            out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
            hidden_states = out.hidden_states

        total_mse = 0.0
        for li in range(L):
            h = hidden_states[li]
            B, S, D = h.shape
            attn_mod = model.model.layers[li].self_attn

            with torch.no_grad():
                teacher_out = attn_mod(h, position_embeddings=rotary, attention_mask=attn_mask)[0]

            q = attn_mod.q_proj(h).view(B, S, n_heads, hd).transpose(1, 2)
            k = attn_mod.k_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
            v = attn_mod.v_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
            k = k.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)
            v = v.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)

            res_out = resonators[li](q, k, v)
            res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * hd)
            res_out = attn_mod.o_proj(res_out)

            total_mse = total_mse + F.mse_loss(res_out, teacher_out)

        avg_mse = total_mse / L
        optimizer.zero_grad(set_to_none=True)
        avg_mse.backward()
        torch.nn.utils.clip_grad_norm_(resonators.parameters(), 1.0)
        optimizer.step()

        if avg_mse.item() < best_mse:
            best_mse = avg_mse.item()

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"  step {step:>5}: avg_mse={avg_mse.item():.4f}  best={best_mse:.4f}  "
                  f"[{elapsed:.0f}s]", flush=True)

    del optimizer

    # Per-layer check
    print(f"\nPhase 1 per-layer results:", flush=True)
    resonators.eval()
    with torch.no_grad():
        inp = all_tokens[:cli.seq_len].unsqueeze(0)
        out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
        hidden_states = out.hidden_states

        layer_results = []
        for li in range(L):
            h = hidden_states[li]
            B, S, D = h.shape
            attn_mod = model.model.layers[li].self_attn
            teacher_out = attn_mod(h, position_embeddings=rotary, attention_mask=attn_mask)[0]

            q = attn_mod.q_proj(h).view(B, S, n_heads, hd).transpose(1, 2)
            k = attn_mod.k_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
            v = attn_mod.v_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
            k = k.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)
            v = v.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)

            res_out = resonators[li](q, k, v)
            res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * hd)
            res_out = attn_mod.o_proj(res_out)

            cos = F.cosine_similarity(teacher_out.reshape(-1), res_out.reshape(-1), dim=0).item()
            mse = F.mse_loss(res_out, teacher_out).item()
            layer_results.append({"layer": li, "mse": round(mse, 4), "cos": round(cos, 4)})
            print(f"  L{li:>2}: MSE={mse:>10.4f}  cos={cos:.4f}", flush=True)

    # ====================
    # PHASE 2: PID blend
    # ====================
    print(f"\n{'='*60}")
    print(f"PHASE 2: PID blend + LM fine-tune")
    print(f"{'='*60}", flush=True)

    fade = 1.0
    hooks = []

    for li in range(L):
        attn_mod = model.model.layers[li].self_attn
        orig_fwd = attn_mod.forward
        res = resonators[li]

        def make_hook(orig, resonator, layer_i):
            def hooked(hidden_states, *args, **kwargs):
                std_out = orig(hidden_states, *args, **kwargs)
                std_attn = std_out[0] if isinstance(std_out, tuple) else std_out

                if fade > 0.999:
                    return std_out

                B, S, D = hidden_states.shape
                am = model.model.layers[layer_i].self_attn
                q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                k = am.k_proj(hidden_states).view(B, S, n_kv, hd).transpose(1, 2)
                v = am.v_proj(hidden_states).view(B, S, n_kv, hd).transpose(1, 2)
                k = k.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)
                v = v.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)

                res_out = resonator(q, k, v)
                res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * hd)
                res_out = am.o_proj(res_out)

                with torch.no_grad():
                    s_s = std_attn.std().clamp(min=1e-6)
                    r_s = res_out.std().clamp(min=1e-6)
                res_norm = (res_out - res_out.mean()) / r_s * s_s + std_attn.mean()

                blended = fade * std_attn + (1.0 - fade) * res_norm
                if isinstance(std_out, tuple):
                    return (blended,) + std_out[1:]
                return blended
            return hooked

        attn_mod.forward = make_hook(orig_fwd, res, li)
        hooks.append((attn_mod, orig_fwd))

    verify = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    setpoint = teacher_ppl * 1.05
    print(f"  Verify fade=1.0: PPL={verify:.2f}")
    print(f"  Setpoint: {setpoint:.2f}", flush=True)

    for p in model.parameters():
        p.requires_grad_(True)

    results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "manifold_dim": cli.manifold_dim,
        "phase1_layers": layer_results, "phase2": [],
    }
    consecutive_over = 0

    print(f"\n  {'Round':>5} | {'Fade':>5} | {'PPL':>8} | {'Ratio':>6} | {'Res%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}", flush=True)

    for rnd in range(1, cli.phase2_rounds + 1):
        ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
        ratio = ppl / teacher_ppl
        res_pct = (1.0 - fade) * 100
        status = "OK" if ratio <= 1.05 else "OVER"

        if status == "OK":
            consecutive_over = 0
        else:
            consecutive_over += 1

        print(f"  {rnd:5d} | {fade:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | {res_pct:4.1f}% | {status}", flush=True)

        results["phase2"].append({
            "round": rnd, "fade": round(fade, 4), "ppl": round(ppl, 2),
            "ratio": round(ratio, 4), "res_pct": round(res_pct, 1),
        })
        with open(Path(save_dir) / "manifold_resonator.json", "w") as f:
            json.dump(results, f, indent=2)

        if fade <= 0.01:
            print(f"\n  PURE MANIFOLD ATTENTION!")
            break
        if consecutive_over >= 10:
            print(f"\n  WALL at fade={fade:.3f}")
            break

        if ratio <= 1.05:
            step = min(0.02, 0.01 * (setpoint - ppl) / setpoint)
            fade = max(0.0, fade - step)

        model.train(); resonators.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params.extend(resonators.parameters())
        opt = torch.optim.AdamW(all_params, lr=2e-5, weight_decay=0.01)
        for s in range(cli.phase2_steps):
            idx = random.randint(0, n_available - 1)
            start = idx * cli.seq_len
            inp = all_tokens[start:start + cli.seq_len].unsqueeze(0)
            tgt = all_tokens[start + 1:start + cli.seq_len + 1].unsqueeze(0)
            logits = model(input_ids=inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
        del opt; gc.collect()
        model.eval(); resonators.eval()

    final_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n  Teacher: {teacher_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Fade: {fade:.3f}")
    print(f"  Time: {elapsed_h:.2f}h")

    save_path = Path(save_dir) / "manifold_resonator_06b"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    torch.save(resonators.state_dict(), save_path / "resonators.pt")
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
