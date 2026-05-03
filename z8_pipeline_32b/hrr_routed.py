"""
Routed HRR Attention: per-token routing between standard and HRR attention.

Like MoE but for the attention mechanism:
  - Tiny router per layer decides per-token: standard or HRR?
  - Start 100% standard, train router to shift tokens to HRR
  - PID controls the routing threshold to maintain quality
  - Goal: 80%+ tokens on HRR = massive speedup

Router: single linear [hidden_dim → 1] + sigmoid
  score > threshold → standard attention (expensive, accurate)
  score < threshold → HRR attention (cheap, approximate)

Training:
  1. Start with threshold=1.0 (all standard)
  2. PID lowers threshold gradually
  3. Router learns which tokens NEED standard attention
  4. Fine-tune all params + router at each step

The HRR uses vectorized sliding window (no per-position loop).
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


# -- HRR operations (vectorized) --

def make_position_vectors(max_len, dim, seed=42):
    gen = torch.Generator()
    gen.manual_seed(seed)
    vecs = torch.randn(max_len, dim, generator=gen)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    return vecs


def hrr_bind_batch(a, b):
    """Batch circular convolution via FFT. Uses float64 for MKL backward compatibility."""
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_correlate_batch(a, b):
    """Batch circular correlation via FFT. Uses float64 for MKL backward compatibility."""
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B.conj(), n=a.shape[-1], dim=-1).to(a.dtype)


def hrr_attention_vectorized(query, key, value, pos_vectors, n_heads, n_kv_heads, head_dim):
    """Vectorized HRR attention using causal cumulative superposition.

    Instead of per-position loop, build cumulative superposition:
    K_super[i] = sum(bind(K[j], pos[j]) for j in 0..i)

    This is a cumulative sum of bound K vectors = O(n) total.
    """
    B, _, seq_k, d = key.shape
    kv_groups = n_heads // n_kv_heads

    pos = pos_vectors[:seq_k].to(key.device)  # [seq_k, d]

    # Bind all K and V with position vectors
    K_bound = hrr_bind_batch(key, pos.unsqueeze(0).unsqueeze(0))  # [B, n_kv_heads, seq_k, d]
    V_bound = hrr_bind_batch(value, pos.unsqueeze(0).unsqueeze(0))

    # Cumulative sum = causal superposition at each position
    K_super = K_bound.cumsum(dim=2)  # [B, n_kv_heads, seq_k, d]
    V_super = V_bound.cumsum(dim=2)

    # Expand for GQA
    K_super = K_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    K_super = K_super.reshape(B, n_heads, seq_k, d)
    V_super = V_super.unsqueeze(2).expand(B, n_kv_heads, kv_groups, seq_k, d)
    V_super = V_super.reshape(B, n_heads, seq_k, d)

    # Correlate each Q with its causal K_super
    retrieval = hrr_correlate_batch(query, K_super)

    # Retrieve V via correlation
    output = hrr_correlate_batch(retrieval, V_super)

    # Normalize by number of superposed items at each position
    counts = torch.arange(1, seq_k + 1, device=key.device, dtype=key.dtype)
    output = output / (counts.view(1, 1, -1, 1).sqrt() * math.sqrt(d))

    return output


# -- Router --

class AttentionRouter(nn.Module):
    """Per-token router: decides standard vs HRR attention.
    Outputs score in [0,1]. High = needs standard, low = HRR is fine."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1, bias=True)
        # Initialize to output ~0.5 (neutral — router learns which tokens need standard)
        nn.init.constant_(self.gate.weight, 0.0)
        nn.init.constant_(self.gate.bias, 0.0)  # sigmoid(0) = 0.5

    def forward(self, hidden_states):
        """hidden_states: [B, seq, hidden_dim] → scores: [B, seq, 1]"""
        return torch.sigmoid(self.gate(hidden_states))


# -- Data + eval --

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


def train_steps(model, routers, train_chunks, n_steps, lr, seq_len=256, threshold=0.5):
    model.train()
    for r in routers:
        r.train()
    all_params = [p for p in model.parameters() if p.requires_grad]
    for r in routers:
        all_params.extend(r.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)
    indices = list(range(len(train_chunks)))
    random.shuffle(indices)
    idx_iter = iter(indices)

    for step in range(n_steps):
        try:
            idx = next(idx_iter)
        except StopIteration:
            random.shuffle(indices)
            idx_iter = iter(indices)
            idx = next(idx_iter)

        batch = train_chunks[idx:idx+1]
        inp = batch[:, :seq_len]
        tgt = batch[:, 1:seq_len+1]

        logits = model(input_ids=inp, use_cache=False).logits
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                  tgt.reshape(-1))

        # Compute cost: reward using HRR (cheap path)
        # Measure what fraction of tokens used standard (expensive)
        compute_cost = 0.0
        for rtr in routers:
            scores = rtr(model.model.embed_tokens(inp).detach())
            # Fraction using standard = fraction above threshold
            std_frac = (scores > threshold).float().mean()
            compute_cost = compute_cost + std_frac
        compute_cost = compute_cost / len(routers)

        # Total loss: language quality + efficiency incentive
        # alpha=0.1 means "saving 10% compute is worth 0.1 nats of CE"
        loss = ce_loss + 0.1 * compute_cost

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

    model.eval()
    for r in routers:
        r.eval()
    del optimizer
    gc.collect()


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    torch.set_num_threads(32)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    cli_args = ap.parse_args()
    model_name = cli_args.model
    target_pct = 5.0
    ft_steps = 200
    lr = 2e-5
    max_rounds = 200
    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"ROUTED HRR ATTENTION: {model_name}")
    print(f"  Per-token routing: standard vs HRR")
    print(f"  Like MoE but for attention mechanism")
    print(f"  Target: {target_pct}% above teacher")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
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

    # Teacher baseline
    teacher_ppl = eval_ppl(model, val_chunks)
    setpoint = teacher_ppl * (1.0 + target_pct / 100.0)
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  PID setpoint: {setpoint:.2f}", flush=True)

    # Create routers (one per layer)
    routers = [AttentionRouter(d_model) for _ in range(L)]
    pos_vectors = make_position_vectors(512, head_dim)

    # Install hooks
    print("\nInstalling routed attention hooks...", flush=True)
    hooks = []
    threshold = 0.50  # at router's initial output — ~50% tokens route to HRR

    for layer_idx in range(L):
        attn_module = model.model.layers[layer_idx].self_attn
        original_forward = attn_module.forward
        router = routers[layer_idx]

        def make_hook(orig_fwd, rtr, li):
            def hooked_forward(*args, **kwargs):
                # Get hidden states
                hidden_states = args[0] if args else kwargs.get('hidden_states')
                if hidden_states is None:
                    return orig_fwd(*args, **kwargs)

                # Router decision
                scores = rtr(hidden_states.detach())  # [B, S, 1]
                # Tokens with score BELOW threshold → HRR
                # Tokens with score ABOVE threshold → standard
                hrr_mask = (scores < threshold).float()  # [B, S, 1]
                std_mask = 1.0 - hrr_mask

                hrr_frac = hrr_mask.mean().item()

                # If all standard, skip HRR entirely
                if hrr_frac < 0.01:
                    return orig_fwd(*args, **kwargs)

                # Run standard attention
                std_output = orig_fwd(*args, **kwargs)
                if isinstance(std_output, tuple):
                    std_attn_out = std_output[0]
                else:
                    std_attn_out = std_output

                # Compute HRR attention
                B, S, D = hidden_states.shape
                attn_mod = model.model.layers[li].self_attn

                q = attn_mod.q_proj(hidden_states)
                k = attn_mod.k_proj(hidden_states)
                v = attn_mod.v_proj(hidden_states)

                q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, S, n_kv_heads, head_dim).transpose(1, 2)
                v = v.view(B, S, n_kv_heads, head_dim).transpose(1, 2)

                hrr_out = hrr_attention_vectorized(
                    q, k, v, pos_vectors, n_heads, n_kv_heads, head_dim)
                hrr_out = hrr_out.transpose(1, 2).reshape(B, S, n_heads * head_dim)
                hrr_out = attn_mod.o_proj(hrr_out)

                # Route: blend per-token
                # std_mask: [B, S, 1], expand to [B, S, D]
                blended = std_mask * std_attn_out + hrr_mask * hrr_out

                if isinstance(std_output, tuple):
                    return (blended,) + std_output[1:]
                return blended

            return hooked_forward

        attn_module.forward = make_hook(original_forward, router, layer_idx)
        hooks.append((attn_module, original_forward))

    print(f"  Installed routed attention on {L} layers ({L} routers)")

    # Verify baseline
    verify_ppl = eval_ppl(model, val_chunks)
    print(f"  Verify threshold=1.0: PPL={verify_ppl:.2f} (should match {teacher_ppl:.2f})")

    # Enable training
    for p in model.parameters():
        p.requires_grad_(True)

    # PID for threshold
    class ThresholdPID:
        def __init__(self, setpoint):
            self.setpoint = setpoint
            self.integral = 0.0
            self.prev_error = 0.0

        def update(self, ppl):
            error = self.setpoint - ppl  # positive = room to compress
            self.integral = max(-5, min(5, self.integral + error))
            deriv = error - self.prev_error
            self.prev_error = error
            # Output: how much to LOWER threshold (positive = lower it)
            output = 0.05 * error + 0.005 * self.integral + 0.01 * deriv
            output = output / max(self.setpoint, 1.0)
            return max(0.0, min(0.05, output))  # max 5% threshold step

    pid = ThresholdPID(setpoint)

    results = {
        "model": model_name, "teacher_ppl": teacher_ppl,
        "setpoint": setpoint, "history": [],
    }

    t_start = time.time()
    consecutive_stuck = 0

    print(f"\n{'='*60}")
    print(f"ROUTED HRR RUNNING")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Thresh':>6} | {'PPL':>8} | {'Ratio':>6} | {'HRR%':>5} | {'PID':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}", flush=True)

    for round_num in range(1, max_rounds + 1):
        # Measure HRR fraction
        with torch.inference_mode():
            inp = val_chunks[0:1, :256]
            hidden = model.model.embed_tokens(inp)
            hrr_fracs = []
            for li, router in enumerate(routers):
                scores = router(hidden)
                hrr_frac = (scores < threshold).float().mean().item()
                hrr_fracs.append(hrr_frac)
            avg_hrr = sum(hrr_fracs) / len(hrr_fracs) * 100

        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl
        pid_out = pid.update(ppl)

        if ppl <= teacher_ppl:
            status = "FREE"
        elif ratio <= 1.0 + target_pct / 100:
            status = "ON TARGET"
        else:
            status = "OVER"

        print(f"  {round_num:5d} | {threshold:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | "
              f"{avg_hrr:4.1f}% | {pid_out:5.3f} | {status}", flush=True)

        results["history"].append({
            "round": round_num, "threshold": round(threshold, 4),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "hrr_pct": round(avg_hrr, 1),
            "pid_output": round(pid_out, 4), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        })

        # Save
        with open(Path(save_dir) / "hrr_routed.json", "w") as f:
            json.dump(results, f, indent=2)

        # Check completion
        if avg_hrr >= 95:
            print(f"\n  95%+ TOKENS ON HRR — SUCCESS!")
            break

        if pid_out <= 0.001 and ppl > setpoint:
            consecutive_stuck += 1
            if consecutive_stuck >= 5:
                print(f"\n  WALL at threshold={threshold:.3f}, HRR={avg_hrr:.1f}%")
                break
        else:
            consecutive_stuck = 0

        # Raise threshold (pushes more tokens to HRR)
        if pid_out > 0.001:
            threshold = min(1.0, threshold + pid_out)

        # Fine-tune model + routers (with compute efficiency reward)
        train_steps(model, routers, train_chunks, ft_steps, lr, threshold=threshold)

    # Final
    final_ppl = eval_ppl(model, val_chunks)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"  Teacher PPL:    {teacher_ppl:.2f}")
    print(f"  Final PPL:      {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Final threshold:{threshold:.3f}")
    print(f"  HRR tokens:     {avg_hrr:.1f}%")
    print(f"  Time:           {elapsed_h:.2f}h")

    results["final"] = {
        "ppl": final_ppl, "threshold": threshold,
        "hrr_pct": avg_hrr, "elapsed_h": elapsed_h,
    }
    with open(Path(save_dir) / "hrr_routed.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
