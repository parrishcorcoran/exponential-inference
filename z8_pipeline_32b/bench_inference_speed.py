"""
Wall clock inference speed: full model vs factored at various ranks.
Does the rank reduction actually make forward passes faster?
"""

import gc
import time
import torch
import torch.nn as nn

ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]
ALL_PROJS = ATTN_PROJS + MLP_PROJS


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


def factorize(linear, rank):
    W = linear.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).contiguous()
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).contiguous()
    bias = linear.bias.data if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def get_proj(model, layer_idx, proj_name):
    layer = model.model.layers[layer_idx]
    if proj_name in ATTN_PROJS:
        return getattr(layer.self_attn, proj_name)
    else:
        return getattr(layer.mlp, proj_name)


def set_proj(model, layer_idx, proj_name, module):
    layer = model.model.layers[layer_idx]
    if proj_name in ATTN_PROJS:
        setattr(layer.self_attn, proj_name, module)
    else:
        setattr(layer.mlp, proj_name, module)


@torch.inference_mode()
def time_forward(model, tokenizer, n_runs=5, seq_len=256):
    """Time forward passes, return avg ms."""
    text = "The theory of general relativity predicts that massive objects warp spacetime " * 20
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"]

    # Warmup
    for _ in range(2):
        model(input_ids=ids, use_cache=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=False)
        times.append(time.perf_counter() - t0)

    return sum(times) / len(times) * 1000  # ms


@torch.inference_mode()
def time_generation(model, tokenizer, n_tokens=32):
    """Time autoregressive generation, return ms/token."""
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"]

    # Warmup
    model.generate(ids, max_new_tokens=4, do_sample=False)

    t0 = time.perf_counter()
    model.generate(ids, max_new_tokens=n_tokens, do_sample=False)
    elapsed = (time.perf_counter() - t0) * 1000  # ms
    return elapsed / n_tokens  # ms per token


def main():
    torch.set_num_threads(32)

    print("=" * 70)
    print("INFERENCE SPEED BENCHMARK: Full vs Factored (Qwen3-14B)")
    print("=" * 70, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ranks_to_test = [1024, 512, 256, 128]

    # -- Full model baseline --
    print(f"\nLoading full model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"  L={L}, d={d}", flush=True)

    # Count params
    full_params = sum(p.numel() for p in model.parameters())

    print(f"\nTiming full model forward (seq_len=256)...", flush=True)
    full_fwd_ms = time_forward(model, tokenizer)
    print(f"  Full forward: {full_fwd_ms:.0f}ms", flush=True)

    print(f"Timing full model generation (32 tokens)...", flush=True)
    full_gen_ms = time_generation(model, tokenizer)
    print(f"  Full generation: {full_gen_ms:.0f}ms/token", flush=True)

    # -- Factored models --
    print(f"\n{'='*70}")
    print(f"{'Rank':>6} | {'Params':>8} | {'Fwd ms':>8} {'Speedup':>8} | {'Gen ms/tok':>10} {'Speedup':>8} | {'Compress':>8}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*8}-+-{'-'*8}")
    print(f"{'full':>6} | {full_params/1e9:7.1f}B | {full_fwd_ms:7.0f}ms {'1.00x':>8} | {full_gen_ms:9.0f}ms {'1.00x':>8} | {'1.00x':>8}")

    for rank in ranks_to_test:
        del model
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="eager").eval()

        for l in range(L):
            for pname in ALL_PROJS:
                proj = get_proj(model, l, pname)
                max_r = min(proj.weight.shape)
                r = min(rank, max_r)
                fac = factorize(proj, r)
                set_proj(model, l, pname, fac)

        gc.collect()

        fac_params = sum(p.numel() for p in model.parameters())
        compress = full_params / fac_params

        fwd_ms = time_forward(model, tokenizer)
        fwd_speedup = full_fwd_ms / fwd_ms

        gen_ms = time_generation(model, tokenizer)
        gen_speedup = full_gen_ms / gen_ms

        print(f"{rank:6d} | {fac_params/1e9:7.1f}B | {fwd_ms:7.0f}ms {fwd_speedup:7.2f}x | "
              f"{gen_ms:9.0f}ms {gen_speedup:7.2f}x | {compress:7.2f}x",
              flush=True)

    print(f"\n  seq_len=256, 14B model, fp32, 32 threads")
    print(f"  Forward = single forward pass (prefill)")
    print(f"  Generation = autoregressive decode, ms per token")


if __name__ == "__main__":
    main()
