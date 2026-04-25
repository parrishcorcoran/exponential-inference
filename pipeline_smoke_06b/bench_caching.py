"""
Benchmark: full-model FT vs layer-wise cached calibration.

Measures actual wall-clock for the same compression operation:
  Compress one layer's k_proj at rank 256, train its A, B for 30 steps.

Method A: full-model FT — student forward through 28 layers + CE loss + backward
Method B: layer-wise cached — capture (x, y) once, then per-step run only
          the compressed layer + MSE loss against captured y

Reports actual speedup on YOUR hardware.
"""
import gc
import time
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
        if self.bias is not None: out = out + self.bias
        return out


def factorize_linear(linear, rank, device, dtype):
    W = linear.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, len(S))
    sqrt_S = S[:k].sqrt()
    A = (U[:, :k] * sqrt_S).to(dtype).to(device)
    B = (sqrt_S.unsqueeze(1) * Vt[:k]).to(dtype).to(device)
    bias = linear.bias.data.to(dtype).to(device) if linear.bias is not None else None
    return FactoredLinear(A, B, bias)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    print(f"device={device}  dtype={dtype}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    print("loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="eager").to(device).eval()
    L = model.config.num_hidden_layers
    print(f"  loaded in {time.time()-t0:.0f}s, L={L}")

    print("loading tokens...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    toks = []
    for item in ds:
        t = item.get("text", "")
        if not t.strip(): continue
        toks.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(toks) >= 8000: break
    toks = toks[:8000]

    seq_len = 256
    n_batches = 8
    batch_size = 1
    batches = []
    for i in range(n_batches):
        start = i * seq_len
        if start + seq_len + 1 > len(toks): break
        window = toks[start:start + seq_len + 1]
        inp = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([window[1:]], dtype=torch.long, device=device)
        batches.append((inp, tgt))

    target_layer = 14  # throat
    target_rank = 256
    n_steps = 30

    # ============================================================
    # METHOD A: FULL MODEL FT
    # ============================================================
    print(f"\n{'='*60}")
    print(f"METHOD A: full-model FT ({n_steps} steps, layer {target_layer} k_proj rank {target_rank})")
    print(f"{'='*60}")

    # Factorize one layer's k_proj
    attn = model.model.layers[target_layer].self_attn
    orig_kproj = attn.k_proj
    fac_a = factorize_linear(orig_kproj, target_rank, device, dtype)
    attn.k_proj = fac_a

    # Freeze all except this one factored layer's A, B
    for p in model.parameters(): p.requires_grad = False
    fac_a.A.requires_grad = True
    fac_a.B.requires_grad = True

    opt = torch.optim.AdamW([fac_a.A, fac_a.B], lr=5e-5)
    model.train()

    # Warmup: 1 step (load CUDA kernels etc)
    inp, tgt = batches[0]
    logits = model(inp, use_cache=False).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

    # Time the n_steps
    if device == "mps": torch.mps.synchronize()
    t0 = time.time()
    for step in range(n_steps):
        inp, tgt = batches[step % len(batches)]
        logits = model(inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "mps": torch.mps.synchronize()
    full_ft_time = time.time() - t0
    print(f"  Method A (full FT): {full_ft_time:.2f}s  ({full_ft_time/n_steps*1000:.0f}ms/step)")

    # Restore original
    attn.k_proj = orig_kproj
    del fac_a, opt
    gc.collect()
    if device == "mps": torch.mps.empty_cache()

    # ============================================================
    # METHOD B: LAYER-WISE CACHED CALIBRATION
    # ============================================================
    print(f"\n{'='*60}")
    print(f"METHOD B: layer-wise cached calibration ({n_steps} steps)")
    print(f"{'='*60}")

    # Capture (x, y) at k_proj on calibration batches (one teacher forward)
    captured_x = []
    captured_y = []

    def capture_hook(module, inputs, output):
        captured_x.append(inputs[0].detach().clone())
        captured_y.append(output.detach().clone())

    h = orig_kproj.register_forward_hook(capture_hook)
    if device == "mps": torch.mps.synchronize()
    t_capture_start = time.time()
    with torch.no_grad():
        for inp, _ in batches:
            _ = model(inp, use_cache=False)
    if device == "mps": torch.mps.synchronize()
    capture_time = time.time() - t_capture_start
    h.remove()
    print(f"  capture phase: {capture_time:.2f}s  ({len(captured_x)} batches)")

    # Stack
    X = torch.cat(captured_x, dim=1).reshape(-1, captured_x[0].shape[-1])  # [N, in]
    Y = torch.cat(captured_y, dim=1).reshape(-1, captured_y[0].shape[-1])  # [N, out]
    print(f"  cached pairs: X={tuple(X.shape)}  Y={tuple(Y.shape)}")

    # Standalone factored linear, trained against (X, Y) only
    fac_b = factorize_linear(orig_kproj, target_rank, device, dtype)
    opt = torch.optim.AdamW([fac_b.A, fac_b.B], lr=5e-5)

    # Warmup
    Xb = X[:batch_size * seq_len]
    Yb = Y[:batch_size * seq_len]
    pred = fac_b(Xb)
    loss = F.mse_loss(pred, Yb)
    opt.zero_grad(); loss.backward(); opt.step()

    # Time the n_steps
    if device == "mps": torch.mps.synchronize()
    t0 = time.time()
    n_total = X.shape[0]
    for step in range(n_steps):
        # Sample a batch from cached pairs
        idx_start = (step * batch_size * seq_len) % (n_total - batch_size * seq_len)
        idx_end = idx_start + batch_size * seq_len
        Xb = X[idx_start:idx_end]
        Yb = Y[idx_start:idx_end]
        pred = fac_b(Xb)
        loss = F.mse_loss(pred, Yb)
        opt.zero_grad(); loss.backward(); opt.step()
    if device == "mps": torch.mps.synchronize()
    cached_train_time = time.time() - t0
    print(f"  Method B (cached train): {cached_train_time:.2f}s  ({cached_train_time/n_steps*1000:.0f}ms/step)")

    # Total Method B = capture + train (since for fairness, capture happens once per anneal stage in practice)
    method_b_total = capture_time + cached_train_time
    print(f"  Method B total (capture + train): {method_b_total:.2f}s")

    # ============================================================
    # COMPARISON
    # ============================================================
    print(f"\n{'='*60}\n=== COMPARISON ===\n{'='*60}")
    print(f"  Method A (full FT) {n_steps} steps:        {full_ft_time:.2f}s  ({full_ft_time/n_steps*1000:.0f}ms/step)")
    print(f"  Method B (cached train) {n_steps} steps:   {cached_train_time:.2f}s  ({cached_train_time/n_steps*1000:.0f}ms/step)")
    print(f"  Method B + capture (one-time):             {method_b_total:.2f}s")
    print(f"")
    print(f"  Per-step speedup: {full_ft_time/cached_train_time:.1f}× faster (training only)")
    print(f"  Total speedup (with capture):    {full_ft_time/method_b_total:.1f}× faster")
    print(f"")
    print(f"  Note: capture cost amortizes over MANY anneal stages.")
    print(f"  If we do 10 rank stages × {n_steps} steps each:")
    a10 = full_ft_time * 10
    b10 = capture_time + cached_train_time * 10
    print(f"    Method A (10 stages full FT):    {a10:.0f}s")
    print(f"    Method B (1 capture + 10 stages cached):  {b10:.0f}s")
    print(f"    Effective speedup at 10 stages: {a10/b10:.1f}×")


if __name__ == "__main__":
    main()
