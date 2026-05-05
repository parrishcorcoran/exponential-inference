"""Anneal α from A1's per-row values toward a configurable target.

Reusable for multiple "baby steps" along A1 → strict nGPT → ... → A2:

  TARGET_AGGREGATION  TARGET_VALUE  Result
  ──────────────────  ────────────  ─────────────────────────
  per_tensor          mean          STRICT nGPT (per-tensor s = mean of A1 α)
  per_tensor          1.0           A2 (all layers s=1, magnitude removed)
  per_head            mean          per-head s for attention (Loshchilov-faithful)
  per_row             0.5_to_mean   intermediate (CV halved)

Mechanism:
  α_eff[i] = (1-τ) · α_init[i] + τ · target[i]     where target depends on config
  τ ramps 0 → 1 over training
  W̃ projected to unit-norm after each optimizer step (nGPT constraint)
  Loss: CE + KL(student||frozen-base-teacher) + hidden state MSE
  Corpus: diverse round-robin (OWT + wikitext + C4)
  Measurement gates: val_ce + W̃ norms + coherency every EVAL_EVERY_TOKENS

Saves new artifact at OUTPUT_DIR with full training summary + the new α tensors.
"""
import os
import sys
import math
import time
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngpt_lossless_convert import NGPTLinear, TARGETS  # noqa: E402
from ngpt_load import load_ngpt_model  # noqa: E402


# ─── config ──────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "model_package/Qwen3-0.6B-nGPT-perfect"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "model_package/Qwen3-0.6B-nGPT-strict"))

# What to anneal α toward
TARGET_AGGREGATION = os.environ.get("TARGET_AGGREGATION", "per_tensor")  # per_tensor | per_head | per_row
TARGET_VALUE = os.environ.get("TARGET_VALUE", "mean")                    # mean | 1.0 | float
# Per-head aggregation needs to know which Linears are attention QKV (use 16 heads for Qwen3-0.6B)
N_HEADS = int(os.environ.get("N_HEADS", "16"))

TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "20000000"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LR = float(os.environ.get("LR", "5e-6"))
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "1.0"))
HIDDEN_MSE_WEIGHT = float(os.environ.get("HIDDEN_MSE_WEIGHT", "0.1"))
EVAL_EVERY_TOKENS = int(os.environ.get("EVAL_EVERY_TOKENS", "2500000"))
N_VAL_TOKENS = int(os.environ.get("N_VAL_TOKENS", "131072"))
SEED = int(os.environ.get("SEED", "42"))

# Hard kill gates (kill the run if any threshold crossed during training)
KILL_VAL_CE_DELTA_NATS = float(os.environ.get("KILL_VAL_CE_DELTA_NATS", "0.05"))   # 0.05 nats above base = kill
KILL_W_TILDE_NORM_DEV = float(os.environ.get("KILL_W_TILDE_NORM_DEV", "0.05"))     # |row_norm - 1| > 0.05 = kill

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


# ─── target computation ──────────────────────────────────────────────────────
def compute_target_alpha(model):
    """For each NGPTLinear, compute the target α tensor that current α should anneal toward.

    Returns dict: {module_name: target_tensor (same shape as current α)}
    """
    targets = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, NGPTLinear):
            continue
        a = mod.alpha.data.float()  # shape [out_features]
        out_f = a.shape[0]

        if TARGET_AGGREGATION == "per_tensor":
            agg_value = a.mean()
            target = torch.full_like(a, agg_value.item())
        elif TARGET_AGGREGATION == "per_head":
            # Only meaningful for attention Linears; everything else falls back to per_tensor
            is_attn = any(t in name for t in ("q_proj", "k_proj", "v_proj"))
            if is_attn and out_f % N_HEADS == 0:
                head_dim = out_f // N_HEADS
                # Mean per head
                head_means = a.view(N_HEADS, head_dim).mean(dim=-1, keepdim=True)
                target = head_means.expand(N_HEADS, head_dim).reshape(out_f)
            else:
                target = torch.full_like(a, a.mean().item())
        elif TARGET_AGGREGATION == "per_row":
            # No aggregation — just anneal each row toward TARGET_VALUE (e.g., 1.0)
            target = a.clone()
        else:
            raise ValueError(f"unknown TARGET_AGGREGATION: {TARGET_AGGREGATION}")

        # Override target value if requested (e.g., 1.0 for A2)
        if TARGET_VALUE == "mean":
            pass  # target already set above
        elif TARGET_VALUE == "1.0":
            target = torch.ones_like(target)
        else:
            try:
                v = float(TARGET_VALUE)
                target = torch.full_like(target, v)
            except ValueError:
                raise ValueError(f"unknown TARGET_VALUE: {TARGET_VALUE}")

        targets[name] = target.to(a.device).to(a.dtype)
    return targets


def apply_alpha_anneal(model, alpha_init, alpha_target, tau):
    """Set each NGPTLinear's α to (1-τ)·α_init + τ·α_target."""
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, NGPTLinear):
                a_eff = (1 - tau) * alpha_init[name] + tau * alpha_target[name]
                mod.alpha.data.copy_(a_eff.to(mod.alpha.dtype))


def project_w_tilde(model):
    n = 0
    for mod in model.modules():
        if isinstance(mod, NGPTLinear):
            W = mod.weight.data
            rn = W.float().norm(dim=-1, keepdim=True).clamp(min=1e-12)
            mod.weight.data.copy_((W.float() / rn).to(W.dtype))
            n += 1
    return n


# ─── corpus ──────────────────────────────────────────────────────────────────
def load_diverse_corpus():
    paths = [
        ("OWT", Path("data/owt_tokens_200M.pt")),
        ("wikitext", Path("data/wikitext_tokens_100M.pt")),
        ("C4", Path("data/c4_tokens_200M.pt")),
    ]
    out = []
    for name, p in paths:
        if p.exists():
            print(f"  loading {name}: {p}")
            out.append((name, torch.load(p, weights_only=False).long()))
    if not out:
        raise SystemExit("no corpus caches found")
    return out


def make_train_iterator(sources, seq_len, batch_size, device):
    cursors = {n: 0 for n, _ in sources}
    src_idx = 0
    while True:
        name, toks = sources[src_idx % len(sources)]
        src_idx += 1
        n_per_batch = seq_len * batch_size
        if cursors[name] + n_per_batch > toks.numel():
            cursors[name] = 0
        chunk = toks[cursors[name]:cursors[name] + n_per_batch]
        cursors[name] += n_per_batch
        yield chunk.view(batch_size, seq_len).to(device)


# ─── hidden hooks ────────────────────────────────────────────────────────────
class HiddenCapture:
    def __init__(self, model, n_layers):
        self.cache = [None] * n_layers
        self.handles = []
        layers_root = model.model.layers if hasattr(model, "model") else model.layers
        for i, layer in enumerate(layers_root):
            def make_hook(idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self.cache[idx] = h
                return hook
            self.handles.append(layer.register_forward_hook(make_hook(i)))

    def clear(self):
        self.cache = [None] * len(self.cache)

    def remove(self):
        for h in self.handles:
            h.remove()


# ─── eval ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_val_ce(model, val_tokens, seq_len=2048, batch=4):
    model.eval()
    n = (val_tokens.numel() // seq_len) * seq_len
    tokens = val_tokens[:n].view(-1, seq_len).to(DEVICE)
    total_loss = 0.0
    total_tok = 0
    for i in range(0, tokens.size(0), batch):
        batch_ids = tokens[i:i+batch]
        logits = model(batch_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tok += shift_labels.numel()
    model.train()
    return total_loss / total_tok


@torch.no_grad()
def measure_w_tilde_norms(model):
    norms = []
    for mod in model.modules():
        if isinstance(mod, NGPTLinear):
            n = mod.weight.data.float().norm(dim=-1)
            norms.append(n)
    norms = torch.cat(norms)
    return (norms.mean().item(), norms.min().item(), norms.max().item(),
            (norms - 1.0).abs().max().item())


@torch.no_grad()
def coherency_check(model, tokenizer, max_new_tokens=20):
    prompts = [
        "The capital of France is",
        "Once upon a time,",
        "In quantum mechanics,",
        "The president of the United States",
    ]
    model.eval()
    out = []
    for p in prompts:
        ids = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
        gen = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id or 0)
        out.append((p, tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True)))
    model.train()
    return out


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"=== α anneal — {TARGET_AGGREGATION} / target={TARGET_VALUE} ===")
    print(f"  input:  {INPUT_DIR}")
    print(f"  output: {OUTPUT_DIR}")
    print(f"  target_tokens={TARGET_TOKENS:,}  LR={LR}  batch={BATCH_SIZE}")
    print(f"  KL={KL_WEIGHT}  HS_MSE={HIDDEN_MSE_WEIGHT}")
    print(f"  kill gates: val_ce_delta>{KILL_VAL_CE_DELTA_NATS}  W̃_dev>{KILL_W_TILDE_NORM_DEV}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

    print(f"\nloading frozen base teacher: {CHECKPOINT}")
    teacher = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"\nloading student (input artifact): {INPUT_DIR}")
    student = load_ngpt_model(INPUT_DIR, CHECKPOINT, DEVICE, DTYPE)
    student.train()

    # Capture initial α and compute target
    print("\ncomputing α targets...")
    alpha_init = {}
    for name, mod in student.named_modules():
        if isinstance(mod, NGPTLinear):
            alpha_init[name] = mod.alpha.data.clone().detach()
    alpha_target = compute_target_alpha(student)
    sample = next(iter(alpha_init))
    print(f"  sample {sample}")
    print(f"    init   range=[{alpha_init[sample].float().min():.4f}, {alpha_init[sample].float().max():.4f}]"
          f"  mean={alpha_init[sample].float().mean():.4f}")
    print(f"    target range=[{alpha_target[sample].float().min():.4f}, {alpha_target[sample].float().max():.4f}]"
          f"  mean={alpha_target[sample].float().mean():.4f}")

    # Hidden state capture for KL + MSE
    n_layers = len(student.model.layers if hasattr(student, "model") else student.layers)
    student_hidden = HiddenCapture(student, n_layers)
    teacher_hidden = HiddenCapture(teacher, n_layers)

    # Optimizer
    w_params, alpha_params, other_params = [], [], []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".alpha"):
            alpha_params.append(p)
        elif any(t in name for t in TARGETS) and name.endswith(".weight"):
            w_params.append(p)
        else:
            other_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": w_params, "lr": LR},
        {"params": alpha_params, "lr": LR * 5.0},
        {"params": other_params, "lr": LR},
    ])

    # Corpus + val
    print("\nloading diverse corpus...")
    sources = load_diverse_corpus()
    train_iter = make_train_iterator(sources, SEQ_LEN, BATCH_SIZE, DEVICE)
    owt_full = next((t for n, t in sources if n == "OWT"), sources[0][1])
    val_tokens = owt_full[-N_VAL_TOKENS:].long()

    print("\nbaseline measurements...")
    base_val_ce = compute_val_ce(teacher, val_tokens)
    initial_val_ce = compute_val_ce(student, val_tokens)
    print(f"  base val_ce:    {base_val_ce:.6f}")
    print(f"  student initial: {initial_val_ce:.6f}  (delta vs base: {initial_val_ce-base_val_ce:+.6f})")

    tokens_per_step = SEQ_LEN * BATCH_SIZE
    total_steps = TARGET_TOKENS // tokens_per_step
    eval_every_steps = EVAL_EVERY_TOKENS // tokens_per_step
    print(f"\ntotal steps: {total_steps:,}  eval every: {eval_every_steps:,}")

    history = []
    print("\n" + "="*70)
    print("training")
    print("="*70)
    t_start = time.time()
    killed = False
    for step in range(1, total_steps + 1):
        # τ ramps 0 → 1 across training
        tau = step / total_steps
        apply_alpha_anneal(student, alpha_init, alpha_target, tau)

        batch_ids = next(train_iter)

        with torch.no_grad():
            teacher_hidden.clear()
            t_logits = teacher(batch_ids).logits

        student_hidden.clear()
        s_logits = student(batch_ids).logits

        shift_logits = s_logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )
        T = 2.0
        s_logp = F.log_softmax(s_logits[:, :-1, :].float() / T, dim=-1)
        t_p = F.softmax(t_logits[:, :-1, :].float() / T, dim=-1)
        kl_loss = F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)

        hs_loss = 0.0
        if HIDDEN_MSE_WEIGHT > 0:
            for s_h, t_h in zip(student_hidden.cache, teacher_hidden.cache):
                if s_h is None or t_h is None:
                    continue
                hs_loss = hs_loss + F.mse_loss(s_h.float(), t_h.float())
            hs_loss = hs_loss / max(1, len(student_hidden.cache))

        loss = ce_loss + KL_WEIGHT * kl_loss + HIDDEN_MSE_WEIGHT * hs_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        project_w_tilde(student)

        if step % 50 == 0:
            elapsed = time.time() - t_start
            tps = step * tokens_per_step / elapsed
            print(f"  step {step:>5}/{total_steps}  τ={tau:.4f}  ce={ce_loss.item():.4f}  "
                  f"kl={kl_loss.item():.4f}  hs={float(hs_loss):.4f}  tok/s={tps:.0f}", flush=True)

        if step % eval_every_steps == 0 or step == total_steps:
            v = compute_val_ce(student, val_tokens)
            n_mean, n_min, n_max, n_dev = measure_w_tilde_norms(student)
            delta = v - base_val_ce
            print(f"\n  ── eval @ step {step} (τ={tau:.4f}, {step*tokens_per_step/1_000_000:.1f}M tokens) ──")
            print(f"     val_ce = {v:.6f}  (base={base_val_ce:.6f}, delta={delta:+.6f})")
            print(f"     W̃ norms: mean={n_mean:.6f}  range=[{n_min:.6f}, {n_max:.6f}]  max_dev={n_dev:.6f}")
            history.append({"step": step, "tau": tau, "tokens": step * tokens_per_step,
                            "val_ce": v, "delta_vs_base": delta,
                            "w_tilde_norm_mean": n_mean, "w_tilde_norm_max_dev": n_dev})

            # Hard kill gates
            if delta > KILL_VAL_CE_DELTA_NATS:
                print(f"     ★ KILL: val_ce delta {delta:.6f} > threshold {KILL_VAL_CE_DELTA_NATS}")
                killed = True
                break
            if n_dev > KILL_W_TILDE_NORM_DEV:
                print(f"     ★ KILL: W̃ row norm dev {n_dev:.6f} > threshold {KILL_W_TILDE_NORM_DEV}")
                killed = True
                break

            # Coherency check
            coh = coherency_check(student, tokenizer, max_new_tokens=15)
            for p, c in coh:
                print(f"     {p!r} → {c!r}")
            print()

    # Finalize
    print("\n" + "="*70)
    print("done" if not killed else "KILLED — kill gates triggered")
    print("="*70)
    final_val_ce = compute_val_ce(student, val_tokens)
    n_mean, n_min, n_max, n_dev = measure_w_tilde_norms(student)
    print(f"  base val_ce:   {base_val_ce:.6f}  ppl={math.exp(base_val_ce):.4f}")
    print(f"  final val_ce:  {final_val_ce:.6f}  ppl={math.exp(final_val_ce):.4f}  "
          f"delta={final_val_ce-base_val_ce:+.6f}")
    print(f"  W̃ norms: mean={n_mean:.6f}  range=[{n_min:.6f}, {n_max:.6f}]  max_dev={n_dev:.6f}")

    # Save
    if not killed:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), OUTPUT_DIR / "ngpt_state_dict.pt")
        alphas = {name + ".alpha": p.detach().cpu()
                  for name, p in student.named_parameters() if name.endswith(".alpha")}
        torch.save(alphas, OUTPUT_DIR / "alphas.pt")
        tokenizer.save_pretrained(OUTPUT_DIR)
        with open(OUTPUT_DIR / "training_summary.json", "w") as f:
            json.dump({
                "input_dir": str(INPUT_DIR),
                "target_aggregation": TARGET_AGGREGATION,
                "target_value": TARGET_VALUE,
                "tokens_trained": step * tokens_per_step,
                "lr": LR,
                "kl_weight": KL_WEIGHT,
                "hidden_mse_weight": HIDDEN_MSE_WEIGHT,
                "base_val_ce": base_val_ce,
                "initial_val_ce": initial_val_ce,
                "final_val_ce": final_val_ce,
                "delta_vs_base": final_val_ce - base_val_ce,
                "w_tilde_norm_mean": n_mean,
                "w_tilde_norm_max_dev": n_dev,
                "history": history,
                "killed": killed,
            }, f, indent=2)
        print(f"\n  saved: {OUTPUT_DIR}/")
    else:
        print(f"\n  NOT saved due to kill gate trigger")

    student_hidden.remove()
    teacher_hidden.remove()


if __name__ == "__main__":
    main()
