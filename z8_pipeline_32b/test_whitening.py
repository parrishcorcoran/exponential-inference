"""
Quick A/B test: raw SVD vs whitened SVD on Qwen3-14B.

Whitened SVD (from SVD-LLM):
  1. Run calibration data through model, collect input activations per projection
  2. Compute covariance C = X^T X
  3. Cholesky: L = cholesky(C)
  4. Whiten: W' = W @ L
  5. SVD on W' instead of W
  6. Undo whitening in factored form: A @ (B @ L^{-1})

This aligns the SVD truncation with what the model actually sees,
not just the raw weight spectrum.
"""

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def factorize_raw(linear, rank):
    """Standard truncated SVD."""
    W = linear.weight.data.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).contiguous()
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).contiguous()
    bias = linear.bias.data if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def factorize_whitened(linear, cov, rank):
    """Activation-whitened SVD.

    cov: X^T X covariance of input activations [in, in]

    Steps:
      1. Cholesky: L such that C = L @ L^T
      2. W' = W @ L  (whiten)
      3. SVD(W') = U S Vt
      4. A = U[:, :k] * sqrt(S[:k])
      5. B_whitened = sqrt(S[:k]) * Vt[:k]
      6. B = B_whitened @ L^{-1}  (undo whitening)
    """
    W = linear.weight.data.float()  # [out, in]

    # Regularize covariance for numerical stability
    cov = cov.float()
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)

    # Cholesky
    try:
        L = torch.linalg.cholesky(cov)  # C = L @ L^T
    except torch.linalg.LinAlgError:
        print("    Cholesky failed, falling back to raw SVD")
        return factorize_raw(linear, rank)

    # Whiten
    W_prime = W @ L  # [out, in]

    # SVD on whitened weights
    U, S, Vt = torch.linalg.svd(W_prime, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()

    A = (U[:, :k] * sqrt_S).contiguous()  # [out, k]
    B_whitened = (sqrt_S.unsqueeze(1) * Vt[:k]).contiguous()  # [k, in]

    # Undo whitening: B = B_whitened @ L^{-1}
    L_inv = torch.linalg.inv(L)
    B = (B_whitened @ L_inv).contiguous()  # [k, in]

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
def eval_ppl(model, val_chunks, seq_len=256, n_eval=50):
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
    return ce, math.exp(min(ce, 20))


@torch.inference_mode()
def collect_activations(model, calibration_chunks, seq_len=256, n_calib=32):
    """Run calibration data and collect input activations per projection.
    Returns dict: (layer, proj_name) -> covariance matrix X^T X.
    """
    model.eval()
    L = model.config.num_hidden_layers

    # We need to hook into each projection to capture inputs
    covariances = {}
    counts = {}
    hooks = []

    def make_hook(layer_idx, proj_name):
        key = (layer_idx, proj_name)
        def hook_fn(module, input, output):
            x = input[0].float()  # [batch, seq, hidden]
            x = x.reshape(-1, x.shape[-1])  # [batch*seq, hidden]
            if key not in covariances:
                covariances[key] = torch.zeros(x.shape[1], x.shape[1])
                counts[key] = 0
            covariances[key] += x.T @ x  # [hidden, hidden]
            counts[key] += x.shape[0]
        return hook_fn

    # Register hooks
    for l in range(L):
        for pname in ALL_PROJS:
            proj = get_proj(model, l, pname)
            h = proj.register_forward_hook(make_hook(l, pname))
            hooks.append(h)

    # Run calibration
    for i in range(min(n_calib, len(calibration_chunks))):
        inp = calibration_chunks[i:i+1, :seq_len]
        model(input_ids=inp, use_cache=False)
        if (i + 1) % 8 == 0:
            print(f"    Calibration {i+1}/{min(n_calib, len(calibration_chunks))}", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize
    for key in covariances:
        covariances[key] /= max(counts[key], 1)

    return covariances


def load_data(tokenizer, seq_len=256, max_tokens=500_000):
    from datasets import load_dataset
    print("  Loading OpenWebText sample...", flush=True)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts = []
    count = 0
    for ex in ds:
        texts.append(ex["text"])
        count += len(ex["text"]) // 4
        if count >= max_tokens * 1.2:
            break
    all_text = "\n\n".join(texts)
    tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]
    n = len(tokens) // (seq_len + 1)
    chunks = tokens[:n * (seq_len + 1)].view(n, seq_len + 1)
    print(f"  {len(chunks)} chunks ({len(chunks) * seq_len / 1e6:.1f}M tokens)")
    return chunks


def main():
    torch.set_num_threads(32)

    print("=" * 70)
    print("A/B TEST: Raw SVD vs Whitened SVD on Qwen3-14B")
    print("=" * 70, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()
    L = model.config.num_hidden_layers
    print(f"  L={L}, loaded", flush=True)

    # Data
    chunks = load_data(tokenizer)

    # Teacher baseline
    print("\nTeacher baseline...", flush=True)
    teacher_ce, teacher_ppl = eval_ppl(model, chunks)
    print(f"  Teacher: CE={teacher_ce:.4f} PPL={teacher_ppl:.2f}", flush=True)

    # Collect activations for whitening
    print("\nCollecting activations for whitening (32 calibration samples)...", flush=True)
    t0 = time.time()
    covariances = collect_activations(model, chunks, n_calib=32)
    print(f"  Done in {time.time()-t0:.0f}s, {len(covariances)} covariance matrices", flush=True)

    # Test multiple ranks
    ranks_to_test = [128, 256, 512, 1024]

    print(f"\n{'='*70}")
    print(f"RANK SWEEP: Raw vs Whitened SVD")
    print(f"{'='*70}")
    print(f"  {'Rank':>6} | {'Raw PPL':>10} {'Raw ratio':>10} | {'White PPL':>10} {'White ratio':>10} | {'Improvement':>12}")
    print(f"  {'-'*6}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}-+-{'-'*12}")

    for rank in ranks_to_test:
        # -- Raw SVD --
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
                fac = factorize_raw(proj, r)
                set_proj(model, l, pname, fac)

        gc.collect()
        raw_ce, raw_ppl = eval_ppl(model, chunks)
        raw_ratio = raw_ppl / teacher_ppl

        # -- Whitened SVD --
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
                key = (l, pname)
                cov = covariances.get(key)
                if cov is not None:
                    fac = factorize_whitened(proj, cov, r)
                else:
                    fac = factorize_raw(proj, r)
                set_proj(model, l, pname, fac)

        gc.collect()
        white_ce, white_ppl = eval_ppl(model, chunks)
        white_ratio = white_ppl / teacher_ppl

        improvement = raw_ppl / max(white_ppl, 0.01)
        print(f"  {rank:6d} | {raw_ppl:10.2f} {raw_ratio:9.2f}x | "
              f"{white_ppl:10.2f} {white_ratio:9.2f}x | {improvement:10.1f}x better",
              flush=True)

    print(f"\n  Teacher PPL: {teacher_ppl:.2f}")
    print(f"\n  If whitening helps significantly, we integrate it into the 32B thermostat.")
    print(f"  The improvement should be largest at low ranks (128-512) where")
    print(f"  alignment with activation distribution matters most.")


if __name__ == "__main__":
    main()
