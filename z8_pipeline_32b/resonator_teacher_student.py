"""
Teacher-Student Resonator Training.

Teacher: standard attention (frozen, provides target outputs)
Student: resonator with learned codebook (trained to match teacher)

Per-layer MSE: student_output should match teacher_output exactly.
Like LOLCATS but with resonator instead of linear attention.

Phase 1: Train resonator codebook only (model frozen)
         Loss = MSE(resonator_output, standard_output) per layer
Phase 2: Fine-tune entire model + resonator on language modeling
         PID blend from standard → resonator
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


class LearnedResonator(nn.Module):
    """Resonator with learned codebook.

    Vectorized: no per-position loop.
    Uses cumulative superposition for causal memory.
    """
    def __init__(self, n_heads, n_kv_heads, head_dim, n_codes=32, max_seq=256):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_codes = n_codes
        self.kv_groups = n_heads // n_kv_heads

        # Learned codebook patterns
        self.codebook = nn.Parameter(torch.randn(n_codes, head_dim) * 0.02)

        # Learned position-like binding keys
        self.bind_keys = nn.Parameter(torch.randn(max_seq, head_dim) * 0.02)

        # Encode V into code coefficients
        self.v_encoder = nn.Linear(head_dim, n_codes, bias=False)
        nn.init.normal_(self.v_encoder.weight, std=0.02)

        # Decode code activations back to head_dim
        self.decoder = nn.Linear(n_codes, head_dim, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, query, key, value):
        B, _, seq, d = query.shape
        kv_groups = self.kv_groups

        # Expand KV for GQA
        K = key.unsqueeze(2).expand(B, self.n_kv_heads, kv_groups, seq, d)
        K = K.reshape(B, self.n_heads, seq, d)
        V = value.unsqueeze(2).expand(B, self.n_kv_heads, kv_groups, seq, d)
        V = V.reshape(B, self.n_heads, seq, d)

        # Encode V into codebook coefficients
        v_codes = self.v_encoder(V)  # [B, heads, seq, n_codes]

        # Bind K with learned position keys
        bk = self.bind_keys[:seq]
        K_bound = bind(K, bk.unsqueeze(0).unsqueeze(0))  # [B, heads, seq, d]

        # Build per-code memory: M_c = cumsum(v_codes[j,c] * K_bound[j])
        # weighted_K[c] = v_codes[:,:,:,c:c+1] * K_bound
        # M_c = cumsum(weighted_K[c])
        # Shape: [B, heads, n_codes, seq, d]
        weighted_K = v_codes.unsqueeze(-1) * K_bound.unsqueeze(3)  # [B, h, seq, n_codes, d]
        weighted_K = weighted_K.permute(0, 1, 3, 2, 4)  # [B, h, n_codes, seq, d]
        M = weighted_K.cumsum(dim=3)  # [B, h, n_codes, seq, d] — causal memory

        # Probe: correlate Q with each code's memory at each position
        # Q: [B, h, seq, d] → [B, h, 1, seq, d]
        # M: [B, h, n_codes, seq, d]
        Q_exp = query.unsqueeze(2)  # [B, h, 1, seq, d]

        # Correlate Q with M for each code
        probes = correlate(Q_exp.expand_as(M), M)  # [B, h, n_codes, seq, d]

        # Score each code: dot product of probe with codebook pattern
        # probes: [B, h, n_codes, seq, d]
        # codebook: [n_codes, d]
        code_scores = (probes * self.codebook.view(1, 1, self.n_codes, 1, d)).sum(-1)
        # [B, h, n_codes, seq]

        code_scores = code_scores.permute(0, 1, 3, 2)  # [B, h, seq, n_codes]

        # Softmax over codes
        code_weights = F.softmax(code_scores, dim=-1)  # [B, h, seq, n_codes]

        # Decode
        output = self.decoder(code_weights)  # [B, h, seq, d]

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


ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-codes", type=int, default=32)
    ap.add_argument("--phase1-steps", type=int, default=500,
                    help="Steps for Phase 1 (MSE matching)")
    ap.add_argument("--phase2-steps", type=int, default=200,
                    help="Steps per round for Phase 2 (LM fine-tune)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-rounds", type=int, default=100)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"TEACHER-STUDENT RESONATOR: {cli.model}")
    print(f"  Phase 1: MSE match standard attention output (codebook only)")
    print(f"  Phase 2: PID blend + LM fine-tune (all params)")
    print(f"  {cli.n_codes} codebook patterns per layer")
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

    train_chunks, val_chunks = load_data(seq_len=cli.seq_len)

    teacher_ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    print(f"  Teacher PPL: {teacher_ppl:.2f}", flush=True)

    # Create resonators
    resonators = nn.ModuleList([
        LearnedResonator(n_heads, n_kv_heads, head_dim, n_codes=cli.n_codes)
        for _ in range(L)
    ])
    print(f"  Resonator params: {sum(p.numel() for p in resonators.parameters()):,}")

    # ==========================================
    # PHASE 1: Teacher-Student MSE matching
    # ==========================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Train resonator codebook via MSE matching")
    print(f"  {cli.phase1_steps} steps, codebook only, model frozen")
    print(f"{'='*60}", flush=True)

    # Freeze model, only train resonators
    for p in model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(resonators.parameters(), lr=cli.lr, weight_decay=0.01)
    resonators.train()

    indices = list(range(len(train_chunks)))
    random.shuffle(indices)

    for step in range(1, cli.phase1_steps + 1):
        idx = indices[step % len(indices)]
        batch = train_chunks[idx:idx+1]
        inp = batch[:, :cli.seq_len]

        # Get hidden states at each layer from teacher
        with torch.no_grad():
            out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
            hidden_states = out.hidden_states  # L+1 hidden states

        # Get position embeddings and attention mask from the model
        B_size, S_size = inp.shape
        with torch.no_grad():
            # Create causal mask
            causal_mask = torch.triu(torch.ones(S_size, S_size, device=inp.device), diagonal=1)
            causal_mask = causal_mask.bool()
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
            attn_mask = torch.where(attn_mask, torch.finfo(torch.float32).min, 0.0)

            # Get position embeddings from model's rotary embedding
            position_ids = torch.arange(S_size, device=inp.device).unsqueeze(0)
            rotary = model.model.rotary_emb(hidden_states[0], position_ids)

        # For each layer, compute teacher attention output and student resonator output
        total_mse = 0.0
        for li in range(L):
            h = hidden_states[li]  # [B, S, D] — input to this layer's attention
            B_l, S_l, D_l = h.shape

            attn_mod = model.model.layers[li].self_attn

            # Teacher: standard attention output (with proper args)
            with torch.no_grad():
                teacher_out = attn_mod(h, position_embeddings=rotary, attention_mask=attn_mask)[0]

            # Student: resonator output
            q = attn_mod.q_proj(h).view(B_l, S_l, n_heads, head_dim).transpose(1, 2)
            k = attn_mod.k_proj(h).view(B_l, S_l, n_kv_heads, head_dim).transpose(1, 2)
            v = attn_mod.v_proj(h).view(B_l, S_l, n_kv_heads, head_dim).transpose(1, 2)

            res_out = resonators[li](q, k, v)  # [B, heads, seq, d]
            res_out = res_out.transpose(1, 2).reshape(B_l, S_l, n_heads * head_dim)
            res_out = attn_mod.o_proj(res_out)  # [B, S, D]

            mse = F.mse_loss(res_out, teacher_out)
            total_mse = total_mse + mse

        avg_mse = total_mse / L
        optimizer.zero_grad(set_to_none=True)
        avg_mse.backward()
        torch.nn.utils.clip_grad_norm_(resonators.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"  step {step:>4}: avg_mse={avg_mse.item():.6f}", flush=True)

    del optimizer
    gc.collect()

    # Check Phase 1 result: how close is resonator to teacher?
    print(f"\nPhase 1 done. Checking per-layer MSE...", flush=True)
    resonators.eval()
    with torch.no_grad():
        inp = val_chunks[0:1, :cli.seq_len]
        out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
        hidden_states = out.hidden_states

        S_v = cli.seq_len
        position_ids_v = torch.arange(S_v, device=inp.device).unsqueeze(0)
        rotary_v = model.model.rotary_emb(hidden_states[0], position_ids_v)
        causal_v = torch.triu(torch.ones(S_v, S_v, device=inp.device), diagonal=1).bool()
        attn_mask_v = torch.where(causal_v.unsqueeze(0).unsqueeze(0), torch.finfo(torch.float32).min, 0.0)

        for li in [0, L//4, L//2, 3*L//4, L-1]:
            h = hidden_states[li]
            B_v, S_v2, D_v = h.shape
            attn_mod = model.model.layers[li].self_attn
            teacher_out = attn_mod(h, position_embeddings=rotary_v, attention_mask=attn_mask_v)[0]

            q = attn_mod.q_proj(h).view(B_v, S_v2, n_heads, head_dim).transpose(1, 2)
            k = attn_mod.k_proj(h).view(B_v, S_v2, n_kv_heads, head_dim).transpose(1, 2)
            v = attn_mod.v_proj(h).view(B_v, S_v2, n_kv_heads, head_dim).transpose(1, 2)
            res_out = resonators[li](q, k, v)
            res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * head_dim)
            res_out = attn_mod.o_proj(res_out)

            cos = F.cosine_similarity(teacher_out.reshape(-1), res_out.reshape(-1), dim=0).item()
            mse = F.mse_loss(res_out, teacher_out).item()
            print(f"  L{li:>2}: MSE={mse:.6f} cos={cos:.4f}")

    # ==========================================
    # PHASE 2: PID blend + LM fine-tune
    # ==========================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: PID blend standard → resonator + LM loss")
    print(f"{'='*60}", flush=True)

    # Install blend hooks
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
                q = am.q_proj(hidden_states).view(B, S, n_heads, head_dim).transpose(1, 2)
                k = am.k_proj(hidden_states).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
                v = am.v_proj(hidden_states).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
                res_out = resonator(q, k, v)
                res_out = res_out.transpose(1, 2).reshape(B, S, n_heads * head_dim)
                res_out = am.o_proj(res_out)

                # Normalize
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

    verify = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    print(f"  Verify fade=1.0: PPL={verify:.2f}")

    # Unfreeze model
    for p in model.parameters():
        p.requires_grad_(True)

    setpoint = teacher_ppl * 1.05
    results = {"teacher_ppl": teacher_ppl, "phase2": []}
    t_start = time.time()
    consecutive_over = 0

    print(f"\n  {'Round':>5} | {'Fade':>5} | {'PPL':>8} | {'Ratio':>6} | {'Res%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}", flush=True)

    for rnd in range(1, cli.max_rounds + 1):
        ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
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
        with open(Path(save_dir) / "resonator_teacher_student.json", "w") as f:
            json.dump(results, f, indent=2)

        if fade <= 0.01:
            print(f"\n  PURE RESONATOR ACHIEVED!")
            break
        if consecutive_over >= 10:
            print(f"\n  WALL at fade={fade:.3f}")
            break

        # PID
        if ratio <= 1.05:
            step = min(0.02, 0.01 * (setpoint - ppl) / setpoint)
            fade = max(0.0, fade - step)

        # Train
        model.train(); resonators.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params.extend(resonators.parameters())
        opt = torch.optim.AdamW(all_params, lr=cli.lr * 0.2, weight_decay=0.01)
        random.shuffle(indices)
        for step in range(cli.phase2_steps):
            idx = indices[step % len(indices)]
            batch = train_chunks[idx:idx+1]
            inp = batch[:, :cli.seq_len]
            tgt = batch[:, 1:cli.seq_len+1]
            logits = model(input_ids=inp, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
        del opt; gc.collect()
        model.eval(); resonators.eval()

    # Save
    final_ppl = eval_ppl(model, val_chunks, seq_len=cli.seq_len)
    elapsed_h = (time.time() - t_start) / 3600
    print(f"\n  Teacher: {teacher_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Fade: {fade:.3f} ({(1-fade)*100:.1f}% resonator)")
    print(f"  Time: {elapsed_h:.2f}h")

    save_path = Path(save_dir) / f"resonator_ts_06b"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    torch.save(resonators.state_dict(), save_path / "resonators.pt")
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
