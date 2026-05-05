"""A1: polish A0 to perfect nGPT.

Loads A0 (W̃ + α split). Fine-tunes both W̃ and α under the nGPT constraint:
  - After every optimizer step, project W̃ rows to unit-norm
  - α drifts freely as a learnable per-row magnitude
  - This is the actual nGPT training dynamic

Goal: close the +0.0014 nats bf16 split gap so val_ce returns to base or
better, with W̃ rows held exactly unit-norm by the constraint.

Loss: CE + α_kl·KL(student || frozen_base_teacher) + α_hs·MSE(hidden_student, hidden_teacher)
Corpus: diverse round-robin (OWT + wikitext + C4 caches)

Stops early when val_ce ≤ base for 2 consecutive evals (stability) or token budget reached.

Usage:
    NGPT_DIR=model_package/Qwen3-0.6B-nGPT-form \\
    OUTPUT_DIR=model_package/Qwen3-0.6B-nGPT-perfect \\
    TARGET_TOKENS=100000000 \\
    python scripts/recipe_a1_perfect_ngpt.py
"""
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngpt_lossless_convert import NGPTLinear, replace_with_ngpt, TARGETS  # noqa: E402
from ngpt_load import load_ngpt_model  # noqa: E402


# ─── config ──────────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
NGPT_DIR = Path(os.environ.get("NGPT_DIR", "model_package/Qwen3-0.6B-nGPT-form"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "model_package/Qwen3-0.6B-nGPT-perfect"))

TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "100000000"))   # 100M default
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LR = float(os.environ.get("LR", "1e-5"))
LR_ALPHA_MULT = float(os.environ.get("LR_ALPHA_MULT", "5.0"))   # α gets higher LR (nGPT convention)
KL_WEIGHT = float(os.environ.get("KL_WEIGHT", "1.0"))
HIDDEN_MSE_WEIGHT = float(os.environ.get("HIDDEN_MSE_WEIGHT", "0.1"))
EVAL_EVERY_TOKENS = int(os.environ.get("EVAL_EVERY_TOKENS", "5000000"))   # eval every 5M tokens
N_VAL_TOKENS = int(os.environ.get("N_VAL_TOKENS", "131072"))
EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "2"))   # stop after N evals at/below base

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


# ─── data loading ────────────────────────────────────────────────────────────
def load_diverse_corpus():
    """Round-robin through OWT + wikitext + C4 caches."""
    paths = [
        ("OWT", Path("data/owt_tokens_200M.pt")),
        ("wikitext", Path("data/wikitext_tokens_100M.pt")),
        ("C4", Path("data/c4_tokens_200M.pt")),
    ]
    sources = []
    for name, p in paths:
        if p.exists():
            print(f"  loading {name}: {p}")
            t = torch.load(p, weights_only=False).long()
            sources.append((name, t))
        else:
            print(f"  skipping {name}: not at {p}")
    if not sources:
        raise SystemExit("no corpus caches found in data/")
    return sources


def make_train_iterator(sources, seq_len, batch_size, device):
    """Round-robin from each source. Random offset per batch."""
    cursors = {name: 0 for name, _ in sources}
    src_idx = 0
    while True:
        name, toks = sources[src_idx % len(sources)]
        src_idx += 1
        n_per_batch = seq_len * batch_size
        if cursors[name] + n_per_batch > toks.numel():
            cursors[name] = 0  # wrap
        chunk = toks[cursors[name]:cursors[name] + n_per_batch]
        cursors[name] += n_per_batch
        yield chunk.view(batch_size, seq_len).to(device)


# ─── unit-norm projection ────────────────────────────────────────────────────
@torch.no_grad()
def project_w_tilde(model):
    """After each optimizer step, project W̃ rows to unit-norm.

    This is the constraint that makes it nGPT during training.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, NGPTLinear):
            W = mod.weight.data
            rn = W.float().norm(dim=-1, keepdim=True).clamp(min=1e-12)
            mod.weight.data.copy_((W.float() / rn).to(W.dtype))
            n += 1
    return n


# ─── hidden state hooks ──────────────────────────────────────────────────────
class HiddenCapture:
    """Capture per-layer hidden states from forward hooks."""
    def __init__(self, model, n_layers):
        self.cache = [None] * n_layers
        self.handles = []
        # Hook on each transformer layer's output
        # Qwen3 layers are at model.model.layers[i]
        layers_root = model.model.layers if hasattr(model, "model") else model.layers
        for i, layer in enumerate(layers_root):
            def make_hook(idx):
                def hook(mod, inp, out):
                    # out is a tuple — first is hidden state
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
def coherency_check(model, tokenizer, max_new_tokens=15):
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


@torch.no_grad()
def measure_w_tilde_norms(model):
    """Verify W̃ rows are unit-norm. Returns (mean, min, max) across all NGPTLinears."""
    norms = []
    for mod in model.modules():
        if isinstance(mod, NGPTLinear):
            n = mod.weight.data.float().norm(dim=-1)
            norms.append(n)
    norms = torch.cat(norms)
    return norms.mean().item(), norms.min().item(), norms.max().item()


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print(f"device={DEVICE}  dtype={DTYPE}")
    print(f"target_tokens={TARGET_TOKENS:,}  seq={SEQ_LEN}  batch={BATCH_SIZE}")
    print(f"LR={LR}  LR_ALPHA_MULT={LR_ALPHA_MULT}  KL={KL_WEIGHT}  HS_MSE={HIDDEN_MSE_WEIGHT}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

    print(f"\nloading frozen base teacher: {CHECKPOINT}")
    teacher = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"\nloading A0 student: {NGPT_DIR}")
    student = load_ngpt_model(NGPT_DIR, CHECKPOINT, DEVICE, DTYPE)
    student.train()

    # Verify A0 starting state
    m, lo, hi = measure_w_tilde_norms(student)
    print(f"  A0 W̃ row norms: mean={m:.6f}  min={lo:.6f}  max={hi:.6f}")

    # Snap W̃ to exact unit-norm before training (ensure starting on the constraint manifold)
    project_w_tilde(student)
    m, lo, hi = measure_w_tilde_norms(student)
    print(f"  after projection: mean={m:.6f}  min={lo:.6f}  max={hi:.6f}")

    # Setup hidden state capture for both models
    n_layers = len(student.model.layers if hasattr(student, "model") else student.layers)
    student_hidden = HiddenCapture(student, n_layers)
    teacher_hidden = HiddenCapture(teacher, n_layers)

    # Optimizer: separate LRs for W̃ and α (nGPT convention: α gets higher LR)
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
        {"params": w_params, "lr": LR, "name": "W_tilde"},
        {"params": alpha_params, "lr": LR * LR_ALPHA_MULT, "name": "alpha"},
        {"params": other_params, "lr": LR, "name": "other"},
    ])
    print(f"  optimizer: W̃ params={sum(p.numel() for p in w_params):,}  "
          f"α params={sum(p.numel() for p in alpha_params):,}  "
          f"other={sum(p.numel() for p in other_params):,}")

    # Load corpus
    print("\nloading diverse corpus...")
    sources = load_diverse_corpus()
    train_iter = make_train_iterator(sources, SEQ_LEN, BATCH_SIZE, DEVICE)

    # Validation tokens (held out from end of OWT)
    owt_full = next((t for n, t in sources if n == "OWT"), sources[0][1])
    val_tokens = owt_full[-N_VAL_TOKENS:].long()

    # Baseline measurements
    print("\nbaseline measurements...")
    base_val_ce = compute_val_ce(teacher, val_tokens)
    student_val_ce = compute_val_ce(student, val_tokens)
    print(f"  base   val_ce = {base_val_ce:.6f}  ppl = {math.exp(base_val_ce):.4f}")
    print(f"  A0     val_ce = {student_val_ce:.6f}  ppl = {math.exp(student_val_ce):.4f}")
    print(f"  gap    = {student_val_ce - base_val_ce:+.6f} nats (this is what we close)")

    # ─── train loop ──────────────────────────────────────────────────────────
    tokens_per_step = SEQ_LEN * BATCH_SIZE
    total_steps = TARGET_TOKENS // tokens_per_step
    eval_every_steps = EVAL_EVERY_TOKENS // tokens_per_step
    print(f"\ntotal steps: {total_steps:,}   eval every: {eval_every_steps:,} steps "
          f"({EVAL_EVERY_TOKENS//1_000_000}M tokens)")

    best_val_ce = float("inf")
    consecutive_at_or_below_base = 0
    history = []

    print("\n" + "="*70)
    print("training")
    print("="*70)
    t_start = time.time()
    for step in range(1, total_steps + 1):
        batch_ids = next(train_iter)

        # Teacher forward
        with torch.no_grad():
            teacher_hidden.clear()
            t_out = teacher(batch_ids)
            t_logits = t_out.logits

        # Student forward
        student_hidden.clear()
        s_out = student(batch_ids)
        s_logits = s_out.logits

        # Losses
        shift_logits = s_logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )

        # KL on logits (forward KL: student matches teacher)
        T = 2.0
        s_logp = F.log_softmax(s_logits[:, :-1, :].float() / T, dim=-1)
        t_p = F.softmax(t_logits[:, :-1, :].float() / T, dim=-1)
        kl_loss = F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)

        # Hidden state MSE
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

        # The constraint: project W̃ rows to unit-norm
        project_w_tilde(student)

        if step % 50 == 0:
            elapsed = time.time() - t_start
            tps = step * tokens_per_step / elapsed
            print(f"  step {step:>5}/{total_steps}  ce={ce_loss.item():.4f}  "
                  f"kl={kl_loss.item():.4f}  hs={float(hs_loss):.4f}  "
                  f"loss={loss.item():.4f}  tok/s={tps:.0f}", flush=True)

        # Eval
        if step % eval_every_steps == 0 or step == total_steps:
            v = compute_val_ce(student, val_tokens)
            m_norm, lo_norm, hi_norm = measure_w_tilde_norms(student)
            print(f"\n  ── eval @ step {step} ({step*tokens_per_step/1_000_000:.0f}M tokens) ──")
            print(f"     val_ce = {v:.6f}  (base={base_val_ce:.6f}, gap={v-base_val_ce:+.6f})")
            print(f"     W̃ norms: mean={m_norm:.6f}  range=[{lo_norm:.6f}, {hi_norm:.6f}]")
            history.append({"step": step, "tokens": step * tokens_per_step,
                           "val_ce": v, "gap_nats": v - base_val_ce,
                           "w_tilde_norm_mean": m_norm})

            # Save best
            if v < best_val_ce:
                best_val_ce = v
                if v <= base_val_ce:
                    consecutive_at_or_below_base += 1
                    print(f"     ★ at/below base ({consecutive_at_or_below_base}/{EARLY_STOP_PATIENCE})")
                    if consecutive_at_or_below_base >= EARLY_STOP_PATIENCE:
                        print(f"\n     EARLY STOP: stable at val_ce ≤ base for {EARLY_STOP_PATIENCE} evals")
                        break
                else:
                    consecutive_at_or_below_base = 0
            print()

    # ─── finalize ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("training done — final eval + save")
    print("="*70)
    final_val_ce = compute_val_ce(student, val_tokens)
    final_coh = coherency_check(student, tokenizer)
    base_coh = coherency_check(teacher, tokenizer)
    m_norm, lo_norm, hi_norm = measure_w_tilde_norms(student)

    print(f"\n  base val_ce:      {base_val_ce:.6f}  ppl={math.exp(base_val_ce):.4f}")
    print(f"  A0 val_ce:        {student_val_ce:.6f}  (gap before training: {student_val_ce-base_val_ce:+.6f})")
    print(f"  perfect val_ce:   {final_val_ce:.6f}  (gap now: {final_val_ce-base_val_ce:+.6f})")
    print(f"  W̃ row norms: mean={m_norm:.6f}  range=[{lo_norm:.6f}, {hi_norm:.6f}]")

    print(f"\n  coherency:")
    n_match = 0
    for (p, b), (_, n) in zip(base_coh, final_coh):
        match = "✓" if b == n else "✗"
        if b == n:
            n_match += 1
        print(f"    {match} {p!r}")
        print(f"        base:    {b!r}")
        print(f"        perfect: {n!r}")
    print(f"  match: {n_match}/4")

    # Save artifact
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sd = student.state_dict()
    torch.save(sd, OUTPUT_DIR / "ngpt_state_dict.pt")
    alphas = {name + ".alpha": p.detach().cpu()
              for name, p in student.named_parameters() if name.endswith(".alpha")}
    torch.save(alphas, OUTPUT_DIR / "alphas.pt")
    import json
    meta = {
        "base_checkpoint": CHECKPOINT,
        "init_from": str(NGPT_DIR),
        "tokens_trained": step * tokens_per_step,
        "lr": LR,
        "lr_alpha_mult": LR_ALPHA_MULT,
        "kl_weight": KL_WEIGHT,
        "hidden_mse_weight": HIDDEN_MSE_WEIGHT,
        "val_ce_base": base_val_ce,
        "val_ce_a0": student_val_ce,
        "val_ce_perfect": final_val_ce,
        "gap_a0_nats": student_val_ce - base_val_ce,
        "gap_perfect_nats": final_val_ce - base_val_ce,
        "ppl_base": math.exp(base_val_ce),
        "ppl_perfect": math.exp(final_val_ce),
        "w_tilde_norm_mean": m_norm,
        "w_tilde_norm_min": lo_norm,
        "w_tilde_norm_max": hi_norm,
        "coherency_match": [b == n for (_, b), (_, n) in zip(base_coh, final_coh)],
        "history": history,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump(meta, f, indent=2)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n  saved: {OUTPUT_DIR}/")
    print(f"    ngpt_state_dict.pt")
    print(f"    alphas.pt")
    print(f"    training_summary.json")
    print(f"    tokenizer files")

    student_hidden.remove()
    teacher_hidden.remove()


if __name__ == "__main__":
    main()
