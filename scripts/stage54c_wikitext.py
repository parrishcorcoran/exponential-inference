"""
Stage 54c — Theory #6 trained on wikitext-2 (richer data).

Fixes stage 54's training objective. Cosine-to-single-target didn't
generalize (model learned to match the training target but couldn't
distinguish it from other tokens). Replaced with cross-entropy over
the full projected-embedding vocab: the true token must have higher
dot product with the prediction than every other token, IN THE
PROJECTED MANIFOLD SPACE.

This is "standard LM training with lm_head = projected-embedding-matrix".
At low rank k (e.g. 10), it tests whether the manifold-dim representation
alone is sufficient for vocab prediction — the strong form of the
manifold-as-target hypothesis.

If student held-out accuracy approaches or exceeds teacher's at any
rank, Theory #6's pipeline is validated. Real ceiling-break test
(multi-teacher ensemble, larger scale) is Strix's job.
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def load_wikitext(tokenizer, max_train_tokens, max_heldout_tokens):
    """Load wikitext-2-raw-v1 and tokenize into train + held-out token streams."""
    from datasets import load_dataset
    print("  loading wikitext-2-raw-v1 from HuggingFace...")
    train_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    heldout_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")

    def tokenize_stream(ds, max_tokens):
        all_ids = []
        for row in ds:
            text = row["text"].strip()
            if not text or text.startswith("="): continue
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            all_ids.append(ids)
            if sum(len(x) for x in all_ids) >= max_tokens:
                break
        return torch.cat(all_ids)[:max_tokens]

    train_ids = tokenize_stream(train_ds, max_train_tokens)
    heldout_ids = tokenize_stream(heldout_ds, max_heldout_tokens)
    return train_ids, heldout_ids


# Kept for reference / compatibility:
_LEGACY_HELDOUT = [
    "Game theory analyzes strategic interactions between rational decision makers.",
    "Bayesian inference updates a prior probability distribution using observed data.",
    "The immune system recognizes pathogens through pattern recognition receptors.",
    "The Riemann zeta function encodes deep information about the distribution of primes.",
]


class TinyGeodesic(nn.Module):
    """1 attention (1 KV head) + 1 SwiGLU MLP + project-to-embedding.
    Predicts a vector in embedding space given context tokens."""

    def __init__(self, embed, hidden_dim, intermediate=None, head_dim=128, n_heads=8):
        super().__init__()
        self.embed = embed  # frozen
        for p in self.embed.parameters(): p.requires_grad = False
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        inter = intermediate or hidden_dim * 2

        # 1 KV head attention
        q_dim = n_heads * head_dim
        kv_dim = 1 * head_dim
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_dim, bias=False)

        # MLP
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, inter, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_dim, bias=False)

        # Final projection
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids):
        # Cast embedding to fp32 (avoid mixed dtype on MPS)
        h = self.embed(input_ids).float()      # [B, T, H]
        B, T, H = h.shape
        hd = self.head_dim

        # Attention (manual, to avoid MPS SDPA backward bugs)
        residual = h
        h_n = self.attn_norm(h)
        q = self.q_proj(h_n).view(B, T, self.n_heads, hd).transpose(1, 2)  # [B, H, T, hd]
        k = self.k_proj(h_n).view(B, T, 1, hd).transpose(1, 2)              # [B, 1, T, hd]
        v = self.v_proj(h_n).view(B, T, 1, hd).transpose(1, 2)              # [B, 1, T, hd]

        # Manual attention: scores = Q @ K.T / sqrt(d); causal mask; softmax; @ V
        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale                # [B, H, T, T]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn = torch.matmul(attn_weights, v)                                 # [B, H, T, hd]

        attn = attn.transpose(1, 2).contiguous().view(B, T, self.n_heads * hd)
        h = residual + self.o_proj(attn)

        # MLP
        residual = h
        h_n = self.mlp_norm(h)
        h = residual + self.down_proj(F.silu(self.gate_proj(h_n)) * self.up_proj(h_n))
        return self.out_norm(h)


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def pca_basis(X, k):
    """X: [N, d] -> (P [d, k], mean [d])."""
    mu = X.mean(dim=0)
    Xc = X - mu
    cov = Xc.T @ Xc / max(Xc.shape[0] - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    k = min(k, eigvecs.shape[1])
    P = eigvecs[:, -k:].flip(dims=[1])
    return P, mu


def tokenize_chunks_from_ids(token_ids, context_len, device, pad_id=0, max_pairs=None):
    """Takes a 1D token-id tensor. Returns [(ctx, next_id), ...] pairs with
    sliding-window context of fixed length context_len."""
    pairs = []
    n = token_ids.shape[0]
    for i in range(context_len, n):
        ctx = token_ids[i - context_len:i]
        next_id = int(token_ids[i].item())
        pairs.append((ctx.to(device), next_id))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def train_student(student, pairs, E_proj_all, embed_basis_mean, P, device, steps, batch_size, lr):
    """Train student with CONTRASTIVE cross-entropy: softmax over projected embeddings.
    E_proj_all: [V, k] — projected embedding of every vocab token."""
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    loss_history = []
    correct_history = []

    # Instead of F.normalize (MPS buggy), pre-divide by stable norms
    eps = 1e-6
    E_norms = E_proj_all.norm(dim=-1, keepdim=True).clamp_min(eps)  # [V, 1]
    E_proj_norm = (E_proj_all / E_norms).detach()                   # [V, k]
    TEMP = 10.0
    student.train()

    for step in range(steps):
        idxs = random.sample(range(len(pairs)), min(batch_size, len(pairs)))
        ctxs = torch.stack([pairs[i][0] for i in idxs]).to(device)
        target_ids = torch.tensor([pairs[i][1] for i in idxs], device=device)

        # Student forward
        h_out = student(ctxs)                                     # [B, T, H]
        pred_last = h_out[:, -1, :].float()                       # [B, H]
        pred_proj = (pred_last - embed_basis_mean) @ P            # [B, k]

        # Manual normalize with eps clamp for MPS stability
        pred_norms = pred_proj.norm(dim=-1, keepdim=True).clamp_min(eps)
        pred_norm = pred_proj / pred_norms                        # [B, k]
        logits = TEMP * (pred_norm @ E_proj_norm.T)               # [B, V]
        loss = F.cross_entropy(logits, target_ids)

        # NaN guard — abort the step if anything went bad
        if not torch.isfinite(loss):
            print(f"    step {step}: non-finite loss, skipping step", flush=True)
            optimizer.zero_grad()
            continue

        train_correct = (logits.argmax(dim=-1) == target_ids).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        # Check for NaN grads before stepping
        has_nan_grad = False
        for p_ in params:
            if p_.grad is not None and not torch.isfinite(p_.grad).all():
                has_nan_grad = True
                break
        if has_nan_grad:
            print(f"    step {step}: NaN grad detected, zeroing grads", flush=True)
            optimizer.zero_grad()
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        loss_history.append(float(loss.item()))
        correct_history.append(train_correct)
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"    step {step:>4d}/{steps}  loss={loss.item():.4f}  "
                  f"train_acc={train_correct:.3f}", flush=True)

    return loss_history


def evaluate(student, teacher, pairs, E_full, P, embed_basis_mean, tokenizer, device):
    """Compare student's next-token accuracy to teacher's on held-out pairs."""
    student.eval()
    teacher.eval()
    n_correct_student = 0
    n_correct_teacher = 0
    n_match_student_teacher = 0
    n_total = len(pairs)

    with torch.inference_mode():
        for ctx, true_id in pairs:
            ctx_b = ctx.unsqueeze(0)

            # Student: predict projected-embedding, argmax over all embeddings
            h_out = student(ctx_b)
            pred_last = h_out[0, -1].float()
            pred_proj = (pred_last - embed_basis_mean) @ P        # [k]
            # All token embeddings projected
            E_proj = (E_full - embed_basis_mean) @ P               # [V, k]
            E_proj_norm = F.normalize(E_proj, dim=-1)
            pred_norm = F.normalize(pred_proj.unsqueeze(0), dim=-1)
            sims = (pred_norm @ E_proj_norm.T)[0]                   # [V]
            student_pred = int(sims.argmax().item())

            # Teacher: standard forward, argmax logits
            t_out = teacher(input_ids=ctx_b, use_cache=False)
            teacher_pred = int(t_out.logits[0, -1].argmax().item())

            if student_pred == true_id: n_correct_student += 1
            if teacher_pred == true_id: n_correct_teacher += 1
            if student_pred == teacher_pred: n_match_student_teacher += 1

    return {
        "student_acc": n_correct_student / n_total,
        "teacher_acc": n_correct_teacher / n_total,
        "student_teacher_agreement": n_match_student_teacher / n_total,
        "n_total": n_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--ranks", default="10,64,512")
    p.add_argument("--context-len", type=int, default=24)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-tokens", type=int, default=80000,
                   help="wikitext tokens to use for training")
    p.add_argument("--heldout-tokens", type=int, default=8000,
                   help="wikitext tokens to use for held-out eval")
    p.add_argument("--max-heldout-pairs", type=int, default=500,
                   help="cap evaluation pairs (teacher forward each)")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage54c_wikitext.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    random.seed(0); torch.manual_seed(0)

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    H = model.config.hidden_size
    E_full = model.model.embed_tokens.weight.detach().float().to(device)
    print(f"  H={H}  V={E_full.shape[0]}")

    print(f"\n=== loading wikitext-2-raw-v1 ===")
    train_ids, heldout_ids = load_wikitext(
        tokenizer, args.train_tokens, args.heldout_tokens)
    print(f"  train stream: {len(train_ids)} tokens")
    print(f"  heldout stream: {len(heldout_ids)} tokens")

    train_pairs = tokenize_chunks_from_ids(train_ids, args.context_len, device,
                                            pad_id=tokenizer.pad_token_id or 0)
    heldout_pairs = tokenize_chunks_from_ids(
        heldout_ids, args.context_len, device,
        pad_id=tokenizer.pad_token_id or 0, max_pairs=args.max_heldout_pairs)
    print(f"  train: {len(train_pairs)} pairs   heldout: {len(heldout_pairs)} pairs")

    # Teacher baseline on held-out
    print(f"\n=== teacher baseline next-token accuracy on held-out ===")
    n_correct = 0
    with torch.inference_mode():
        for ctx, true_id in heldout_pairs:
            out = model(input_ids=ctx.unsqueeze(0), use_cache=False)
            if int(out.logits[0, -1].argmax().item()) == true_id:
                n_correct += 1
    teacher_holdout_acc = n_correct / len(heldout_pairs)
    print(f"  teacher held-out top-1 accuracy: {teacher_holdout_acc:.3f}  "
          f"({n_correct}/{len(heldout_pairs)})")

    ranks = [int(x) for x in args.ranks.split(",")]
    all_results = []
    for k in ranks:
        print(f"\n{'='*60}")
        print(f"=== rank k = {k} ===")
        print(f"{'='*60}")

        # PCA basis from embedding matrix
        P_cpu, mu_cpu = pca_basis(E_full.cpu(), k)
        P = P_cpu.to(device)
        mu = mu_cpu.to(device)

        # Build student
        student = TinyGeodesic(model.model.embed_tokens,
                                hidden_dim=H, intermediate=H * 2,
                                head_dim=128, n_heads=8).to(device)
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"  trainable params: {trainable/1e6:.2f}M")

        # Precompute projected embedding of every vocab token
        E_proj_all = (E_full - mu) @ P      # [V, k]

        # Train
        t0 = time.perf_counter()
        print(f"  training {args.steps} steps, batch={args.batch_size}, lr={args.lr}")
        loss_hist = train_student(
            student, train_pairs, E_proj_all, mu, P, device,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr)
        train_dt = time.perf_counter() - t0
        print(f"  training done in {train_dt:.1f}s")

        # Eval
        print(f"  evaluating on held-out...")
        eval_stats = evaluate(student, model, heldout_pairs,
                              E_full, P, mu, tokenizer, device)
        print(f"    student held-out acc: {eval_stats['student_acc']:.3f}")
        print(f"    teacher held-out acc: {eval_stats['teacher_acc']:.3f}  "
              f"(should equal {teacher_holdout_acc:.3f})")
        print(f"    student↔teacher agreement: {eval_stats['student_teacher_agreement']:.3f}")

        all_results.append({
            "rank": k,
            "trainable_params": trainable,
            "final_loss": loss_hist[-1] if loss_hist else None,
            "train_wall_seconds": train_dt,
            **eval_stats,
        })

    print(f"\n=== summary ===")
    print(f"  teacher held-out acc:           {teacher_holdout_acc:.3f}")
    print(f"  {'rank':>5}  {'params':>8}  {'student_acc':>12}  {'vs teacher':>12}")
    for r in all_results:
        delta = r["student_acc"] - teacher_holdout_acc
        sign = "+" if delta >= 0 else ""
        print(f"  {r['rank']:>5}  {r['trainable_params']/1e6:>6.1f}M  "
              f"{r['student_acc']:>12.3f}  {sign}{delta:.3f}")

    # Verdict
    best = max(r["student_acc"] for r in all_results)
    print(f"\n=== verdict ===")
    if best > teacher_holdout_acc:
        print(f"  STUDENT EXCEEDS TEACHER (by {best - teacher_holdout_acc:.3f}) — teacher ceiling scratched (pipeline test scale)")
    elif best > 0.8 * teacher_holdout_acc:
        print(f"  Student within 80% of teacher — pipeline works, scaling may get to parity")
    elif best > 0.3:
        print(f"  Student at {best:.3f} — nontrivial but well below teacher")
    else:
        print(f"  Student near random — training didn't learn at this scale")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "H": H,
            "train_tokens": args.train_tokens, "heldout_tokens": args.heldout_tokens,
            "train_pairs": len(train_pairs), "heldout_pairs": len(heldout_pairs),
            "teacher_heldout_acc": teacher_holdout_acc,
            "context_len": args.context_len, "steps": args.steps,
            "results": all_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
