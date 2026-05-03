"""
Soft-blended HRR with whitening + error correction.

No binary routing. No threshold. Every token gets a smooth blend:
  output = score * standard_attn + (1-score) * (hrr_attn + correction)

score comes from router (sigmoid, 0-1 per token)
PID controls router BIAS — lowering bias → lower scores → more HRR
Whitening decorrelates K before superposition → less noise
Error correction learns systematic HRR errors

All three combined. Super slow PID. 0.6B proof of concept.
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


# -- HRR operations (float64 for MKL backward compat) --

def make_position_vectors(max_len, dim, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    vecs = torch.randn(max_len, dim, generator=gen)
    return vecs / vecs.norm(dim=-1, keepdim=True)


def hrr_bind(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_correlate(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B.conj(), n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_attention_vectorized(query, key, value, pos_vectors, n_heads, n_kv_heads, head_dim):
    """Vectorized causal HRR attention via cumulative superposition."""
    B, _, seq_k, d = key.shape
    kv_groups = n_heads // n_kv_heads

    pos = pos_vectors[:seq_k].to(key.device)

    K_bound = hrr_bind(key, pos.unsqueeze(0).unsqueeze(0))
    V_bound = hrr_bind(value, pos.unsqueeze(0).unsqueeze(0))

    K_super = K_bound.cumsum(dim=2)
    V_super = V_bound.cumsum(dim=2)

    K_super = K_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_super = K_super.reshape(B, n_heads, seq_k, d)
    V_super = V_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    V_super = V_super.reshape(B, n_heads, seq_k, d)

    retrieval = hrr_correlate(query, K_super)
    output = hrr_correlate(retrieval, V_super)

    counts = torch.arange(1, seq_k + 1, device=key.device, dtype=key.dtype)
    output = output / (counts.view(1, 1, -1, 1).sqrt() * math.sqrt(d))

    return output


# -- Soft blend components --

class SoftRouter(nn.Module):
    """Per-token soft routing. Output 0-1: how much standard attention to use.
    1.0 = pure standard, 0.0 = pure HRR.
    PID controls the bias to shift the distribution."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1, bias=True)
        # Start with high bias = mostly standard
        nn.init.normal_(self.gate.weight, std=0.01)  # small random for variance
        nn.init.constant_(self.gate.bias, 4.0)  # sigmoid(4) ≈ 0.98 = mostly standard

    def forward(self, x):
        return torch.sigmoid(self.gate(x))  # [B, S, 1]


class ErrorCorrection(nn.Module):
    """Small per-layer correction for HRR output.
    Learns systematic errors in the superposition."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        # Initialize near zero — correction starts small
        nn.init.normal_(self.proj.weight, std=0.001)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


# -- Data --

def load_data(seq_len=256, max_train=2_000_000, max_val=100_000):
    cache_path = "data/owt_tokens_50M.pt"
    print(f"  Loading cached corpus...", flush=True)
    tokens = torch.load(cache_path, weights_only=True)
    val_tokens = tokens[:max_val]
    train_tokens = tokens[max_val:max_val + max_train]

    def chunk(toks):
        n = len(toks) // (seq_len + 1)
        return toks[:n * (seq_len + 1)].view(n, seq_len + 1)

    return chunk(train_tokens), chunk(val_tokens)


@torch.inference_mode()
def eval_ppl(model, val_chunks, seq_len=256, n_eval=20):
    model.eval()
    total = 0
    n = 0
    for i in range(min(n_eval, len(val_chunks))):
        inp = val_chunks[i:i+1, :seq_len]
        tgt = val_chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1))
        total += loss.item()
        n += 1
    ce = total / max(n, 1)
    return math.exp(min(ce, 20))


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--target-pct", type=float, default=5.0)
    ap.add_argument("--ft-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--bias-step", type=float, default=0.1,
                    help="How much to decrease router bias per PID step")
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"SOFT-BLENDED HRR: {cli.model}")
    print(f"  output = score * standard + (1-score) * (hrr + correction)")
    print(f"  PID controls router bias (high=standard, low=HRR)")
    print(f"  Whitened HRR + error correction")
    print(f"  Target: {cli.target_pct}% above teacher")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)

    print(f"\nLoading {cli.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', d_model // n_heads)
    kv_groups = n_heads // n_kv_heads
    attn_out_dim = n_heads * head_dim

    print(f"  L={L}, d={d_model}, heads={n_heads}/{n_kv_heads}, head_dim={head_dim}")

    print("Loading data...", flush=True)
    train_chunks, val_chunks = load_data()

    teacher_ppl = eval_ppl(model, val_chunks)
    setpoint = teacher_ppl * (1.0 + cli.target_pct / 100.0)
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  PID setpoint: {setpoint:.2f}", flush=True)

    # Create components
    routers = nn.ModuleList([SoftRouter(d_model) for _ in range(L)])
    corrections = nn.ModuleList([ErrorCorrection(d_model) for _ in range(L)])
    pos_vectors = make_position_vectors(512, head_dim)

    # Install hooks
    print("\nInstalling soft-blend hooks...", flush=True)
    hooks = []

    for layer_idx in range(L):
        attn_module = model.model.layers[layer_idx].self_attn
        original_forward = attn_module.forward
        router = routers[layer_idx]
        correction = corrections[layer_idx]

        def make_hook(orig_fwd, rtr, corr, li):
            def hooked_forward(*args, **kwargs):
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                if hidden_states is None:
                    return orig_fwd(*args, **kwargs)

                # Router score: how much standard to use (0-1)
                score = rtr(hidden_states)  # [B, S, 1]

                # If almost all standard, skip HRR compute
                if score.mean().item() > 0.99:
                    return orig_fwd(*args, **kwargs)

                # Standard attention
                std_output = orig_fwd(*args, **kwargs)
                if isinstance(std_output, tuple):
                    std_attn_out = std_output[0]
                else:
                    std_attn_out = std_output

                # HRR attention
                B, S, D = hidden_states.shape
                hd = head_dim
                attn_mod = model.model.layers[li].self_attn

                q = attn_mod.q_proj(hidden_states)
                k = attn_mod.k_proj(hidden_states)
                v = attn_mod.v_proj(hidden_states)

                q = q.view(B, S, n_heads, hd).transpose(1, 2)
                k = k.view(B, S, n_kv_heads, hd).transpose(1, 2)
                v = v.view(B, S, n_kv_heads, hd).transpose(1, 2)

                hrr_out = hrr_attention_vectorized(
                    q, k, v, pos_vectors, n_heads, n_kv_heads, hd)
                hrr_out = hrr_out.transpose(1, 2).reshape(B, S, n_heads * hd)
                hrr_out = attn_mod.o_proj(hrr_out)

                # Normalize HRR output to match standard output's scale
                with torch.no_grad():
                    std_mean = std_attn_out.mean()
                    std_std = std_attn_out.std().clamp(min=1e-6)
                    hrr_mean = hrr_out.mean()
                    hrr_std = hrr_out.std().clamp(min=1e-6)
                hrr_normalized = (hrr_out - hrr_mean) / hrr_std * std_std + std_mean

                # Error correction on normalized output
                hrr_corrected = hrr_normalized + corr(hrr_normalized)

                # Soft blend: score=1 → standard, score=0 → HRR
                blended = score * std_attn_out + (1.0 - score) * hrr_corrected

                if isinstance(std_output, tuple):
                    return (blended,) + std_output[1:]
                return blended

            return hooked_forward

        attn_module.forward = make_hook(original_forward, router, correction, layer_idx)
        hooks.append((attn_module, original_forward))

    print(f"  Installed soft-blend on {L} layers")
    print(f"  Router params: {sum(p.numel() for p in routers.parameters()):,}")
    print(f"  Correction params: {sum(p.numel() for p in corrections.parameters()):,}")

    # Verify
    verify_ppl = eval_ppl(model, val_chunks)
    print(f"  Verify: PPL={verify_ppl:.2f} (teacher={teacher_ppl:.2f})")

    # Avg router score
    with torch.inference_mode():
        inp = val_chunks[0:1, :256]
        h = model.model.embed_tokens(inp)
        avg_scores = [routers[i](h).mean().item() for i in range(L)]
    avg_std = sum(avg_scores) / len(avg_scores) * 100
    print(f"  Initial avg standard%: {avg_std:.1f}%")

    # Enable training
    for p in model.parameters():
        p.requires_grad_(True)

    # PID for router bias
    current_bias = 4.0  # matches router init

    results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "setpoint": setpoint, "history": [],
    }

    t_start = time.time()
    consecutive_stuck = 0

    print(f"\n{'='*60}")
    print(f"SOFT-BLEND HRR RUNNING")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Bias':>6} | {'PPL':>8} | {'Ratio':>6} | {'Std%':>5} | {'HRR%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}", flush=True)

    for round_num in range(1, cli.max_rounds + 1):
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl

        # Measure blend fraction
        with torch.inference_mode():
            inp = val_chunks[0:1, :256]
            h = model.model.embed_tokens(inp)
            scores = [routers[i](h).mean().item() for i in range(L)]
        avg_std_pct = sum(scores) / len(scores) * 100
        avg_hrr_pct = 100.0 - avg_std_pct

        if ppl <= teacher_ppl:
            status = "FREE"
        elif ratio <= 1.0 + cli.target_pct / 100:
            status = "ON TARGET"
        else:
            status = "OVER"

        print(f"  {round_num:5d} | {current_bias:5.2f} | {ppl:8.2f} | {ratio:5.2f}x | "
              f"{avg_std_pct:4.1f}% | {avg_hrr_pct:4.1f}% | {status}", flush=True)

        results["history"].append({
            "round": round_num, "bias": round(current_bias, 3),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "std_pct": round(avg_std_pct, 1), "hrr_pct": round(avg_hrr_pct, 1),
            "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        })

        # Save
        with open(Path(save_dir) / "hrr_soft_blend.json", "w") as f:
            json.dump(results, f, indent=2)

        # Check completion
        if avg_hrr_pct >= 95:
            print(f"\n  95%+ HRR — SUCCESS!")
            break

        # PID: should we lower bias (more HRR)?
        error = setpoint - ppl  # positive = room to compress
        if error > 0 and status != "OVER":
            # Room to push — lower bias
            step = min(cli.bias_step, error / setpoint)
            current_bias = max(-4.0, current_bias - step)
            # Apply new bias to all routers
            with torch.no_grad():
                for router in routers:
                    router.gate.bias.fill_(current_bias)
        elif status == "OVER":
            consecutive_stuck += 1
            if consecutive_stuck >= 10:
                print(f"\n  WALL at bias={current_bias:.2f}, HRR={avg_hrr_pct:.1f}%")
                break
        else:
            consecutive_stuck = 0

        # Fine-tune: model + routers + corrections
        model.train()
        for m in [routers, corrections]:
            m.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params.extend(routers.parameters())
        all_params.extend(corrections.parameters())

        optimizer = torch.optim.AdamW(all_params, lr=cli.lr, weight_decay=0.01)
        indices = list(range(len(train_chunks)))
        random.shuffle(indices)

        for step in range(cli.ft_steps):
            idx = indices[step % len(indices)]
            batch = train_chunks[idx:idx+1]
            inp = batch[:, :256]
            tgt = batch[:, 1:257]

            logits = model(input_ids=inp, use_cache=False).logits
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                      tgt.reshape(-1))

            # Efficiency reward: lower scores (more HRR) reduces loss
            efficiency_bonus = 0.0
            h_embed = model.model.embed_tokens(inp).detach()
            for router in routers:
                efficiency_bonus += router(h_embed).mean()
            efficiency_bonus = efficiency_bonus / L  # avg standard fraction

            loss = ce_loss + 0.05 * efficiency_bonus  # reward HRR usage

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        del optimizer
        gc.collect()
        model.eval()

    # Final
    final_ppl = eval_ppl(model, val_chunks)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"  Teacher PPL:  {teacher_ppl:.2f}")
    print(f"  Final PPL:    {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Final bias:   {current_bias:.2f}")
    print(f"  Final HRR%:   {avg_hrr_pct:.1f}%")
    print(f"  Time:         {elapsed_h:.2f}h")

    results["final"] = {
        "ppl": final_ppl, "bias": current_bias,
        "hrr_pct": avg_hrr_pct, "elapsed_h": elapsed_h,
    }
    with open(Path(save_dir) / "hrr_soft_blend.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
