"""
Fade-out standard attention, HRR takes over.

Like nGPT magnitude annealing but for the attention mechanism:
  output = fade * standard_attn + hrr_corrected

fade starts at 1.0, PID shrinks it toward 0.0.
HRR stays full strength. Model adapts because standard is dying.
When fade=0, delete standard attention. Pure HRR.

No routing. No threshold. No normalization matching.
Just turn down the volume on standard and train the model to hear HRR.
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


# -- HRR --

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


def hrr_attention(query, key, value, pos_vectors, n_heads, n_kv_heads, head_dim):
    """Vectorized causal HRR via cumulative superposition."""
    B, _, seq_k, d = key.shape
    kv_groups = n_heads // n_kv_heads
    pos = pos_vectors[:seq_k].to(key.device)

    K_bound = hrr_bind(key, pos.unsqueeze(0).unsqueeze(0))
    V_bound = hrr_bind(value, pos.unsqueeze(0).unsqueeze(0))
    K_super = K_bound.cumsum(dim=2)
    V_super = V_bound.cumsum(dim=2)

    K_super = K_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d).reshape(B, n_heads, seq_k, d)
    V_super = V_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d).reshape(B, n_heads, seq_k, d)

    retrieval = hrr_correlate(query, K_super)
    output = hrr_correlate(retrieval, V_super)

    counts = torch.arange(1, seq_k + 1, device=key.device, dtype=key.dtype)
    return output / (counts.view(1, 1, -1, 1).sqrt() * math.sqrt(d))


class ErrorCorrection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


# -- Data --

def load_data(seq_len=256, max_train=2_000_000, max_val=100_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    val_t = tokens[:max_val]
    train_t = tokens[max_val:max_val + max_train]
    def chunk(t):
        n = len(t) // (seq_len + 1)
        return t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return chunk(train_t), chunk(val_t)


@torch.inference_mode()
def eval_ppl(model, chunks, seq_len=256, n=20):
    model.eval()
    total = 0; c = 0
    for i in range(min(n, len(chunks))):
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
    ap.add_argument("--target-pct", type=float, default=5.0)
    ap.add_argument("--ft-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-rounds", type=int, default=300)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"FADE-OUT STANDARD ATTENTION: {cli.model}")
    print(f"  output = fade * standard + hrr_corrected")
    print(f"  fade: 1.0 -> 0.0 (PID controlled)")
    print(f"  Target: {cli.target_pct}% above teacher")
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
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, 'head_dim', d_model // n_heads)

    print(f"  L={L}, d={d_model}, heads={n_heads}/{n_kv_heads}, head_dim={head_dim}")

    print("Loading data...", flush=True)
    train_chunks, val_chunks = load_data()

    teacher_ppl = eval_ppl(model, val_chunks)
    setpoint = teacher_ppl * (1.0 + cli.target_pct / 100.0)
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  Setpoint: {setpoint:.2f}", flush=True)

    # Components
    corrections = nn.ModuleList([ErrorCorrection(d_model) for _ in range(L)])
    pos_vectors = make_position_vectors(512, head_dim)
    fade = 1.0  # start: 100% standard

    # Install hooks
    print("Installing fade hooks...", flush=True)
    hooks = []

    for li in range(L):
        attn_mod = model.model.layers[li].self_attn
        orig_fwd = attn_mod.forward
        corr = corrections[li]

        def make_hook(orig, cr, layer_i):
            def hooked(hidden_states, *args, **kwargs):
                # Standard attention
                std_out = orig(hidden_states, *args, **kwargs)
                if isinstance(std_out, tuple):
                    std_attn = std_out[0]
                else:
                    std_attn = std_out

                # If fade is 1.0, pure standard — skip HRR entirely
                if fade > 0.999:
                    return std_out

                # HRR attention
                B, S, D = hidden_states.shape
                hd = head_dim
                am = model.model.layers[layer_i].self_attn

                q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                k = am.k_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)
                v = am.v_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)

                hrr_out = hrr_attention(q, k, v, pos_vectors, n_heads, n_kv_heads, hd)
                hrr_out = hrr_out.transpose(1, 2).reshape(B, S, n_heads * hd)
                hrr_out = am.o_proj(hrr_out)

                # Normalize HRR to match standard scale
                with torch.no_grad():
                    s_std = std_attn.std().clamp(min=1e-6)
                    h_std = hrr_out.std().clamp(min=1e-6)
                    s_mean = std_attn.mean()
                    h_mean = hrr_out.mean()
                hrr_norm = (hrr_out - h_mean) / h_std * s_std + s_mean

                # Error correction
                hrr_corrected = hrr_norm + cr(hrr_norm)

                # Fade: as fade→0, standard disappears, HRR takes over
                output = fade * std_attn + (1.0 - fade) * hrr_corrected

                if isinstance(std_out, tuple):
                    return (output,) + std_out[1:]
                return output

            return hooked

        attn_mod.forward = make_hook(orig_fwd, corr, li)
        hooks.append((attn_mod, orig_fwd))

    print(f"  Hooks installed on {L} layers")

    # Verify
    verify = eval_ppl(model, val_chunks)
    print(f"  Verify fade=1.0: PPL={verify:.2f} (teacher={teacher_ppl:.2f})", flush=True)

    # Enable training
    for p in model.parameters():
        p.requires_grad_(True)

    results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "setpoint": setpoint, "history": [],
    }

    t_start = time.time()
    consecutive_stuck = 0

    print(f"\n{'='*60}")
    print(f"FADING STANDARD ATTENTION")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Fade':>6} | {'PPL':>8} | {'Ratio':>6} | {'HRR%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}", flush=True)

    for rnd in range(1, cli.max_rounds + 1):
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl
        hrr_pct = (1.0 - fade) * 100

        if ppl <= teacher_ppl:
            status = "FREE"
        elif ratio <= 1.0 + cli.target_pct / 100:
            status = "ON TARGET"
        else:
            status = "OVER"

        print(f"  {rnd:5d} | {fade:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | "
              f"{hrr_pct:4.1f}% | {status}", flush=True)

        results["history"].append({
            "round": rnd, "fade": round(fade, 4),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "hrr_pct": round(hrr_pct, 1), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        })

        with open(Path(save_dir) / "hrr_fade_standard.json", "w") as f:
            json.dump(results, f, indent=2)

        if fade <= 0.01:
            print(f"\n  FADE COMPLETE — PURE HRR ATTENTION!")
            break

        # PID: should we lower fade?
        error = setpoint - ppl
        if error > 0:
            # Room to compress — lower fade
            step = min(0.01, 0.005 * error / setpoint)  # max 1% per round
            fade = max(0.0, fade - step)
            consecutive_stuck = 0
        else:
            consecutive_stuck += 1
            if consecutive_stuck >= 10:
                print(f"\n  WALL at fade={fade:.3f}, HRR={hrr_pct:.1f}%")
                break

        # Fine-tune
        model.train()
        corrections.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
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
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        del optimizer
        gc.collect()
        model.eval()
        corrections.eval()

    # Final
    final_ppl = eval_ppl(model, val_chunks)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"  Teacher:  {teacher_ppl:.2f}")
    print(f"  Final:    {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Fade:     {fade:.3f}")
    print(f"  HRR%:     {(1-fade)*100:.1f}%")
    print(f"  Time:     {elapsed_h:.2f}h")

    results["final"] = {"ppl": final_ppl, "fade": fade, "elapsed_h": elapsed_h}
    with open(Path(save_dir) / "hrr_fade_standard.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
