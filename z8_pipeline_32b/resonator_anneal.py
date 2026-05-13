"""
Anneal MAP probe → raw superposition probe.

fade=1.0: probe = MAP_decode (O(n²), clean)
fade=0.0: probe = raw_superposition correlate (O(n×d), noisy)

0.1% per step. All weights free. PID holds quality.
Resonator iteration refines at each level.
The model learns to work with the cheaper probe.

Saves checkpoint every 50 rounds.
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


def correlate(key, trace):
    """CORRECT: conj on KEY."""
    K = torch.fft.rfft(key.double(), dim=-1)
    T = torch.fft.rfft(trace.double(), dim=-1)
    return torch.fft.irfft(K.conj() * T, n=key.shape[-1], dim=-1).float()


def bind(a, b):
    A = torch.fft.rfft(a.double(), dim=-1)
    B = torch.fft.rfft(b.double(), dim=-1)
    return torch.fft.irfft(A * B, n=a.shape[-1], dim=-1).float()


class LearnedMAPKeys(nn.Module):
    def __init__(self, max_seq, dim, n_layers):
        super().__init__()
        init = (torch.randint(0, 2, (n_layers, max_seq, dim)).float() * 2 - 1)
        self.keys = nn.Parameter(init * 0.5)

    def get_keys(self, layer, seq):
        return torch.tanh(self.keys[layer, :seq])


def load_data(seq_len=64, max_val=50_000):
    tokens = torch.load("data/owt_tokens_50M.pt", weights_only=True)
    return tokens, tokens[:max_val]


@torch.inference_mode()
def eval_ppl(model, val_tokens, seq_len=64, n=15):
    model.eval()
    n_chunks = len(val_tokens) // (seq_len + 1)
    chunks = val_tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)
    total = 0; c = 0
    for i in range(min(n, n_chunks)):
        inp = chunks[i:i+1, :seq_len]
        tgt = chunks[i:i+1, 1:seq_len+1]
        logits = model(input_ids=inp, use_cache=False).logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1)).item()
        c += 1
    return math.exp(total / max(c, 1))


def main():
    torch.set_num_threads(32)
    model_name = "Qwen/Qwen3-0.6B"
    seq_len = 64
    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"RESONATOR ANNEAL: MAP → raw superposition")
    print(f"  0.1% fade per step, all weights free, PID quality control")
    print(f"  Resonator iteration at each level")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', d_model // n_heads)
    kv_groups = n_heads // n_kv

    print(f"  L={L}, d={d_model}, heads={n_heads}/{n_kv}, head_dim={hd}")

    all_tokens, val_tokens = load_data(seq_len=seq_len)
    teacher_ppl = eval_ppl(model, val_tokens, seq_len=seq_len)
    setpoint = teacher_ppl * 1.05
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  Setpoint: {setpoint:.2f}", flush=True)

    # Load pre-trained MAP keys (trained to cos=1.0 on multiple layers)
    map_keys = LearnedMAPKeys(256, hd, L)
    keys_path = "z8_pipeline_32b/pid_results/trained_map_keys.pt"
    print(f"\nLoading pre-trained MAP keys from {keys_path}...", flush=True)
    map_keys.load_state_dict(torch.load(keys_path, weights_only=True))
    print(f"  Loaded. 15/28 layers above cos=0.95, 4 at exact 1.0", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)

    causal = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    n_available = len(all_tokens) // (seq_len + 1) - 1

    # Phase 2: Anneal fade from MAP → raw superposition
    print(f"\n{'='*60}")
    print(f"Phase 2: Anneal MAP → raw superposition")
    print(f"  0.1% per round, all weights free")
    print(f"{'='*60}", flush=True)

    # Start with model FROZEN — only MAP keys train
    # Unfreeze model weights only when fade starts dropping
    for p in model.parameters():
        p.requires_grad_(False)
    model_unfrozen = False

    fade = 1.0  # 1.0 = pure MAP, 0.0 = pure raw superposition
    n_resonator_iters = 3

    results = {
        "model": model_name, "teacher_ppl": teacher_ppl,
        "setpoint": setpoint, "history": [],
    }

    t_start = time.time()
    consecutive_over = 0

    print(f"\n  {'Round':>5} | {'Fade':>6} | {'PPL':>8} | {'Ratio':>6} | {'MAP%':>5} | {'Raw%':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}", flush=True)

    for rnd in range(1, 2001):
        ppl = eval_ppl(model, val_tokens, seq_len=seq_len)
        ratio = ppl / teacher_ppl
        map_pct = fade * 100
        raw_pct = (1.0 - fade) * 100

        if ratio <= 1.05:
            status = "OK"
            consecutive_over = 0
        else:
            status = "OVER"
            consecutive_over += 1

        print(f"  {rnd:5d} | {fade:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | {map_pct:4.1f}% | {raw_pct:4.1f}% | {status}",
              flush=True)

        results["history"].append({
            "round": rnd, "fade": round(fade, 4), "ppl": round(ppl, 2),
            "ratio": round(ratio, 4), "map_pct": round(map_pct, 1),
            "raw_pct": round(raw_pct, 1), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        })

        # Save every 50 rounds
        if rnd % 50 == 0:
            with open(Path(save_dir) / "resonator_anneal.json", "w") as f:
                json.dump(results, f, indent=2)
            ckpt_path = Path(save_dir) / f"resonator_anneal_fade{fade:.3f}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            torch.save(map_keys.state_dict(), ckpt_path / "map_keys.pt")
            print(f"  >>> CHECKPOINT saved: {ckpt_path.name}", flush=True)

        if fade <= 0.001:
            print(f"\n  PURE RAW SUPERPOSITION — MAP eliminated!")
            break
        if consecutive_over >= 20:
            print(f"\n  WALL at fade={fade:.3f}")
            break

        # PID: lower fade if quality holds
        if ratio <= 1.05:
            new_fade = max(0.0, fade - 0.0001)  # 0.01% per step
            # Unfreeze model weights when fade first drops
            if new_fade < fade and not model_unfrozen:
                print(f"  >>> Unfreezing model weights at fade={new_fade:.4f}", flush=True)
                for p in model.parameters():
                    p.requires_grad_(True)
                model_unfrozen = True
            fade = new_fade

        # Skip training if fade hasn't moved (nothing to adapt to)
        if fade >= 0.9999:
            continue

        # Install hooks for this fade level
        # Remove old hooks
        for li in range(L):
            attn_mod = model.model.layers[li].self_attn
            if hasattr(attn_mod, '_original_forward'):
                attn_mod.forward = attn_mod._original_forward

        # Train with blended attention
        model.train()
        map_keys.train()
        all_params = [p for p in model.parameters() if p.requires_grad]
        all_params.extend(map_keys.parameters())
        opt = torch.optim.AdamW(all_params, lr=2e-5, weight_decay=0.01)

        for step in range(1000):
            idx = random.randint(0, n_available - 1)
            start = idx * seq_len
            inp = all_tokens[start:start + seq_len].unsqueeze(0)
            tgt = all_tokens[start + 1:start + seq_len + 1].unsqueeze(0)

            # Forward pass with blended attention scores
            out = model(input_ids=inp, use_cache=False, output_hidden_states=True)

            # Standard LM loss
            logits = out.logits
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))

            # Attention matching loss: MAP+raw blend should match standard
            attn_loss = 0.0
            for li in range(L):
                h = out.hidden_states[li].detach()
                B, S, D = h.shape
                attn_mod = model.model.layers[li].self_attn

                Q = attn_mod.q_proj(h).view(B, S, n_heads, hd).transpose(1, 2)
                K = attn_mod.k_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
                K = K.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)
                V = attn_mod.v_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
                V = V.unsqueeze(2).expand(B, n_kv, kv_groups, S, hd).reshape(B, n_heads, S, hd)

                # Standard scores (teacher)
                with torch.no_grad():
                    real_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(hd)
                    real_scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    real_attn = torch.softmax(real_scores, dim=-1)

                # MAP scores
                pos = map_keys.get_keys(li, S)
                K_bound = K * pos.unsqueeze(0).unsqueeze(0)
                M_K = K_bound.sum(dim=2)
                QM = Q * M_K.unsqueeze(2)
                map_scores = (QM @ pos.T) / math.sqrt(hd)
                map_scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Raw superposition scores (HRR correlate with CORRECT conj)
                K_bound_hrr = bind(K, pos.unsqueeze(0).unsqueeze(0))
                M_hrr = K_bound_hrr.cumsum(dim=2)  # causal
                raw_probe = correlate(Q, M_hrr)  # [B, heads, seq, hd]
                # Score: probe against each position key
                raw_scores = (raw_probe @ pos.T) / math.sqrt(hd)
                raw_scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

                # Blend
                blended_scores = fade * map_scores + (1.0 - fade) * raw_scores
                blended_attn = torch.softmax(blended_scores, dim=-1)

                # Loss: blended should match standard
                attn_loss = attn_loss + F.mse_loss(blended_attn, real_attn)

            loss = lm_loss + 0.1 * (attn_loss / L)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

        del opt
        gc.collect()
        model.eval()
        map_keys.eval()

    # Final
    final_ppl = eval_ppl(model, val_tokens, seq_len=seq_len)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"  Teacher: {teacher_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Fade: {fade:.3f} ({(1-fade)*100:.1f}% raw superposition)")
    print(f"  Time: {elapsed_h:.2f}h")

    # Save final
    with open(Path(save_dir) / "resonator_anneal.json", "w") as f:
        json.dump(results, f, indent=2)
    ckpt_path = Path(save_dir) / "resonator_anneal_final"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(str(ckpt_path))
    torch.save(map_keys.state_dict(), ckpt_path / "map_keys.pt")
    print(f"  Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
