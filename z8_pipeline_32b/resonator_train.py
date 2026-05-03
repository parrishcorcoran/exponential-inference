"""
Train a model to use resonator attention with a learned codebook.

Don't try to match standard attention. Let the model learn
its own holographic retrieval patterns from scratch.

Architecture:
  - Take Qwen3-0.6B
  - Replace attention with resonator retrieval
  - The model learns:
    1. How to bind K,V into superposition
    2. A codebook of retrieval patterns
    3. How to resonate the correct answer out

PID-controlled anneal from standard → resonator attention.
The model adapts its representations to be resonator-friendly.

Start with tiny amount of resonator, slowly increase.
The weights learn the codebook as they adapt.
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


def bind(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).to(a.dtype)


def correlate(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B.conj(), n=a.shape[-1], dim=-1).to(a.dtype)


class LearnedResonatorAttention(nn.Module):
    """Resonator attention with learned codebook.

    Instead of using V vectors as codebook (which doesn't work),
    learn a FIXED codebook of resonance patterns. The model learns
    to encode information into patterns that resonate cleanly.

    codebook: [n_codes, head_dim] — learned resonance patterns
    Each code is a "channel" the model can store/retrieve through.

    Store: bind each code with a learned projection of the value
    Retrieve: probe with query, resonate against codebook
    """
    def __init__(self, n_heads, n_kv_heads, head_dim, n_codes=32, max_seq=512):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_codes = n_codes
        self.kv_groups = n_heads // n_kv_heads

        # Learned codebook: resonance patterns
        self.codebook = nn.Parameter(torch.randn(n_codes, head_dim) * 0.02)

        # Learned binding keys (position-like, but learned)
        self.bind_keys = nn.Parameter(torch.randn(max_seq, head_dim) * 0.02)

        # Project V into codebook space
        self.v_to_codes = nn.Linear(head_dim, n_codes, bias=False)
        nn.init.normal_(self.v_to_codes.weight, std=0.02)

        # Project resonator output back to head_dim
        self.codes_to_out = nn.Linear(n_codes, head_dim, bias=False)
        nn.init.normal_(self.codes_to_out.weight, std=0.02)

    def forward(self, query, key, value):
        """
        query: [B, n_heads, seq, head_dim]
        key:   [B, n_kv_heads, seq, head_dim]
        value: [B, n_kv_heads, seq, head_dim]
        """
        B, _, seq_q, d = query.shape
        seq_k = key.shape[2]

        # Expand KV for GQA
        K = key.unsqueeze(2).expand(B, self.n_kv_heads, self.kv_groups, seq_k, d)
        K = K.reshape(B, self.n_heads, seq_k, d)
        V = value.unsqueeze(2).expand(B, self.n_kv_heads, self.kv_groups, seq_k, d)
        V = V.reshape(B, self.n_heads, seq_k, d)

        # Step 1: Encode values into codebook coefficients
        # For each position j, compute how much of each code to store
        v_codes = self.v_to_codes(V)  # [B, n_heads, seq_k, n_codes]

        # Step 2: Build superposed memory using learned bind keys
        # M = sum_j bind(bind_key[j], K[j]) * v_codes[j]
        # Simplified: M = sum_j v_codes[j] * bind(bind_key[j], K[j])
        bk = self.bind_keys[:seq_k]  # [seq_k, d]

        # Bind keys with positions
        K_bound = bind(K, bk.unsqueeze(0).unsqueeze(0))  # [B, heads, seq_k, d]

        # Weight by value code coefficients and superpose
        # For each code c: M_c = sum_j v_codes[j,c] * K_bound[j]
        # M_c: [B, heads, d]
        # v_codes: [B, heads, seq_k, n_codes]
        # K_bound: [B, heads, seq_k, d]
        M = torch.einsum('bhsc,bhsd->bhcd', v_codes, K_bound)  # [B, heads, n_codes, d]

        # Step 3: Probe each code channel with query
        # For each query position i, correlate with each code's memory
        # probe_c[i] = correlate(Q[i], M_c)
        # Then resonate: which codes activate?

        output = torch.zeros_like(query)  # [B, heads, seq_q, d]

        for i in range(seq_q):
            q_i = query[:, :, i:i+1, :]  # [B, heads, 1, d]

            # Causal: only use memory up to position i
            # Rebuild M causally (expensive but correct)
            v_codes_causal = v_codes[:, :, :i+1, :]  # [B, heads, i+1, n_codes]
            K_bound_causal = K_bound[:, :, :i+1, :]  # [B, heads, i+1, d]
            M_causal = torch.einsum('bhsc,bhsd->bhcd',
                                     v_codes_causal, K_bound_causal)  # [B, heads, n_codes, d]

            # Probe each code
            code_scores = torch.zeros(B, self.n_heads, self.n_codes, device=query.device)
            for c in range(self.n_codes):
                m_c = M_causal[:, :, c, :]  # [B, heads, d]
                probe = correlate(q_i.squeeze(2), m_c)  # [B, heads, d]
                # Score: how much does the probe match the codebook pattern?
                code_scores[:, :, c] = (probe * self.codebook[c]).sum(-1)

            # Softmax over codes (which channels resonate?)
            code_weights = F.softmax(code_scores, dim=-1)  # [B, heads, n_codes]

            # Output: weighted sum of codebook patterns, projected to head_dim
            out_i = self.codes_to_out(code_weights)  # [B, heads, d]
            output[:, :, i, :] = out_i

        return output


def load_data(seq_len=128, max_train=1_000_000, max_val=50_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    val_t = tokens[:max_val]
    train_t = tokens[max_val:max_val + max_train]
    def chunk(t):
        n = len(t) // (seq_len + 1)
        return t[:n * (seq_len + 1)].view(n, seq_len + 1)
    return chunk(train_t), chunk(val_t)


@torch.inference_mode()
def eval_ppl(model, chunks, seq_len=128, n=15):
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
    ap.add_argument("--n-codes", type=int, default=32)
    ap.add_argument("--ft-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-rounds", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=128)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"RESONATOR TRAINING: {cli.model}")
    print(f"  Learned codebook with {cli.n_codes} resonance patterns")
    print(f"  PID anneal from standard → resonator attention")
    print(f"  Model learns its own holographic retrieval")
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
    train_chunks, val_chunks = load_data(seq_len=cli.seq_len)

    teacher_ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    setpoint = teacher_ppl * 1.05
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  Setpoint: {setpoint:.2f}", flush=True)

    # Create resonator modules (one per layer)
    resonators = nn.ModuleList([
        LearnedResonatorAttention(n_heads, n_kv_heads, head_dim, n_codes=cli.n_codes)
        for _ in range(L)
    ])
    res_params = sum(p.numel() for p in resonators.parameters())
    print(f"  Resonator params: {res_params:,} ({res_params/1e6:.1f}M)")

    # Install hooks: soft blend standard + resonator
    fade = 1.0  # start 100% standard
    hooks = []

    for li in range(L):
        attn_mod = model.model.layers[li].self_attn
        orig_fwd = attn_mod.forward
        res = resonators[li]

        def make_hook(orig, resonator, layer_i):
            def hooked(hidden_states, *args, **kwargs):
                # Standard attention
                std_out = orig(hidden_states, *args, **kwargs)
                if isinstance(std_out, tuple):
                    std_attn = std_out[0]
                else:
                    std_attn = std_out

                if fade > 0.999:
                    return std_out

                # Resonator attention
                B, S, D = hidden_states.shape
                hd = head_dim
                am = model.model.layers[layer_i].self_attn

                q = am.q_proj(hidden_states).view(B, S, n_heads, hd).transpose(1, 2)
                k = am.k_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)
                v = am.v_proj(hidden_states).view(B, S, n_kv_heads, hd).transpose(1, 2)

                res_out = resonator(q, k, v)
                res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * hd)
                res_out = am.o_proj(res_out)

                # Normalize resonator output to match standard scale
                with torch.no_grad():
                    s_std = std_attn.std().clamp(min=1e-6)
                    r_std = res_out.std().clamp(min=1e-6)
                    s_mean = std_attn.mean()
                    r_mean = res_out.mean()
                res_norm = (res_out - r_mean) / r_std * s_std + s_mean

                # Blend
                blended = fade * std_attn + (1.0 - fade) * res_norm

                if isinstance(std_out, tuple):
                    return (blended,) + std_out[1:]
                return blended

            return hooked

        attn_mod.forward = make_hook(orig_fwd, res, li)
        hooks.append((attn_mod, orig_fwd))

    print(f"  Hooks installed on {L} layers")

    # Verify
    verify = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    print(f"  Verify fade=1.0: PPL={verify:.2f}", flush=True)

    # Training
    for p in model.parameters():
        p.requires_grad_(True)

    results = {"model": cli.model, "teacher_ppl": teacher_ppl, "history": []}
    t_start = time.time()
    consecutive_over = 0

    print(f"\n{'='*60}")
    print(f"TRAINING: anneal standard → resonator")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Fade':>5} | {'PPL':>8} | {'Ratio':>6} | {'Res%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}", flush=True)

    for rnd in range(1, cli.max_rounds + 1):
        ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
        ratio = ppl / teacher_ppl
        res_pct = (1.0 - fade) * 100

        if ratio <= 1.05:
            status = "OK"
            consecutive_over = 0
        else:
            status = "OVER"
            consecutive_over += 1

        print(f"  {rnd:5d} | {fade:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | {res_pct:4.1f}% | {status}", flush=True)

        results["history"].append({
            "round": rnd, "fade": round(fade, 4), "ppl": round(ppl, 2),
            "ratio": round(ratio, 4), "res_pct": round(res_pct, 1),
            "status": status, "elapsed_s": round(time.time() - t_start, 1),
        })

        with open(Path(save_dir) / "resonator_train.json", "w") as f:
            json.dump(results, f, indent=2)

        if fade <= 0.01:
            print(f"\n  PURE RESONATOR ATTENTION ACHIEVED!")
            break

        if consecutive_over >= 10:
            print(f"\n  WALL at fade={fade:.3f}")
            break

        # PID: lower fade if quality holds
        if ratio <= 1.05:
            step = min(0.01, 0.005 * (setpoint - ppl) / setpoint)
            fade = max(0.0, fade - step)

        # Train model + resonators
        model.train()
        resonators.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params.extend(resonators.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=cli.lr, weight_decay=0.01)

        indices = list(range(len(train_chunks)))
        random.shuffle(indices)
        for step in range(cli.ft_steps):
            idx = indices[step % len(indices)]
            batch = train_chunks[idx:idx+1]
            inp = batch[:, :cli.seq_len]
            tgt = batch[:, 1:cli.seq_len+1]
            logits = model(input_ids=inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

        del optimizer
        gc.collect()
        model.eval()
        resonators.eval()

    final_ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Teacher: {teacher_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Fade: {fade:.3f} ({(1-fade)*100:.1f}% resonator)")
    print(f"  Time: {elapsed_h:.2f}h")

    # Save model + resonators
    save_path = Path(save_dir) / f"resonator_06b_fade{fade:.2f}"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    torch.save(resonators.state_dict(), save_path / "resonators.pt")
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
