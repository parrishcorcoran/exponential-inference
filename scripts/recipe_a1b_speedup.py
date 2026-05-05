"""A1b: prove nGPT trains faster than vanilla on continued pretraining.

Two paired runs (controlled by MODE env var):

  MODE=ngpt    — load A1 perfect-nGPT, train W̃ + α with unit-norm projection
  MODE=vanilla — load fresh base Qwen3-0.6B, train all params, no projection

Both runs use IDENTICAL:
  - data: diverse corpus round-robin
  - loss: CE only (no teacher — pure continued pretraining)
  - LR, batch, seq_len, optimizer, seeds
  - eval cadence + val tokens

Compare val_ce trajectories. If A1 (nGPT) reaches lower val_ce faster, the
training-speedup claim of Loshchilov's nGPT paper holds for our post-hoc
conversion.

Output: results/a1b_{mode}_trajectory.json with per-eval val_ce + step.
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
from ngpt_lossless_convert import NGPTLinear, replace_with_ngpt, TARGETS  # noqa: E402
from ngpt_load import load_ngpt_model  # noqa: E402


MODE = os.environ.get("MODE", "ngpt").lower()
assert MODE in ("ngpt", "vanilla"), f"MODE must be ngpt or vanilla, got {MODE}"

CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
NGPT_DIR = Path(os.environ.get("NGPT_DIR", "model_package/Qwen3-0.6B-nGPT-perfect"))

TARGET_TOKENS = int(os.environ.get("TARGET_TOKENS", "50000000"))   # 50M default
SEQ_LEN = int(os.environ.get("SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
LR = float(os.environ.get("LR", "5e-6"))
EVAL_EVERY_TOKENS = int(os.environ.get("EVAL_EVERY_TOKENS", "2500000"))
N_VAL_TOKENS = int(os.environ.get("N_VAL_TOKENS", "131072"))
SEED = int(os.environ.get("SEED", "42"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_diverse_corpus():
    paths = [
        ("OWT", Path("data/owt_tokens_200M.pt")),
        ("wikitext", Path("data/wikitext_tokens_100M.pt")),
        ("C4", Path("data/c4_tokens_200M.pt")),
    ]
    sources = []
    for name, p in paths:
        if p.exists():
            t = torch.load(p, weights_only=False).long()
            sources.append((name, t))
    if not sources:
        raise SystemExit("no corpus caches found in data/")
    return sources


def make_train_iterator(sources, seq_len, batch_size, device, seed):
    """Identical iteration order across runs given same seed."""
    rng = random.Random(seed)
    cursors = {name: 0 for name, _ in sources}
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


@torch.no_grad()
def project_w_tilde(model):
    n = 0
    for mod in model.modules():
        if isinstance(mod, NGPTLinear):
            W = mod.weight.data
            rn = W.float().norm(dim=-1, keepdim=True).clamp(min=1e-12)
            mod.weight.data.copy_((W.float() / rn).to(W.dtype))
            n += 1
    return n


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


def main():
    set_all_seeds(SEED)
    print(f"=== A1b speedup proof — MODE={MODE.upper()} ===")
    print(f"  device={DEVICE}  dtype={DTYPE}  seed={SEED}")
    print(f"  target_tokens={TARGET_TOKENS:,}  seq={SEQ_LEN}  batch={BATCH_SIZE}  LR={LR}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

    if MODE == "ngpt":
        print(f"\nloading A1 perfect-nGPT student: {NGPT_DIR}")
        student = load_ngpt_model(NGPT_DIR, CHECKPOINT, DEVICE, DTYPE)
    else:
        print(f"\nloading vanilla base Qwen3 student: {CHECKPOINT}")
        student = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE)
    student.train()

    print("\nloading diverse corpus...")
    sources = load_diverse_corpus()
    train_iter = make_train_iterator(sources, SEQ_LEN, BATCH_SIZE, DEVICE, SEED)

    owt_full = next((t for n, t in sources if n == "OWT"), sources[0][1])
    val_tokens = owt_full[-N_VAL_TOKENS:].long()

    print("\nbaseline measurement...")
    initial_val_ce = compute_val_ce(student, val_tokens)
    print(f"  initial val_ce = {initial_val_ce:.6f}  ppl = {math.exp(initial_val_ce):.4f}")

    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR)

    tokens_per_step = SEQ_LEN * BATCH_SIZE
    total_steps = TARGET_TOKENS // tokens_per_step
    eval_every_steps = EVAL_EVERY_TOKENS // tokens_per_step
    print(f"\ntotal steps: {total_steps:,}  eval every: {eval_every_steps:,} steps "
          f"({EVAL_EVERY_TOKENS//1_000_000}M tokens)")

    history = [{"step": 0, "tokens": 0, "val_ce": initial_val_ce}]
    print("\n" + "="*70)
    print(f"training [{MODE.upper()}]")
    print("="*70)
    t_start = time.time()
    for step in range(1, total_steps + 1):
        batch_ids = next(train_iter)
        s_logits = student(batch_ids).logits
        shift_logits = s_logits[:, :-1, :].contiguous()
        shift_labels = batch_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        if MODE == "ngpt":
            project_w_tilde(student)

        if step % 50 == 0:
            elapsed = time.time() - t_start
            tps = step * tokens_per_step / elapsed
            print(f"  step {step:>5}/{total_steps}  ce={loss.item():.4f}  tok/s={tps:.0f}",
                  flush=True)

        if step % eval_every_steps == 0 or step == total_steps:
            v = compute_val_ce(student, val_tokens)
            print(f"\n  ── eval @ step {step} ({step*tokens_per_step/1_000_000:.1f}M tokens) ──")
            print(f"     val_ce = {v:.6f}  (initial {initial_val_ce:.6f}, delta {v-initial_val_ce:+.6f})")
            history.append({
                "step": step,
                "tokens": step * tokens_per_step,
                "val_ce": v,
                "delta_from_initial": v - initial_val_ce,
            })
            print()

    # Save trajectory
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"a1b_{MODE}_trajectory.json"
    summary = {
        "mode": MODE,
        "checkpoint": CHECKPOINT,
        "init_from": str(NGPT_DIR) if MODE == "ngpt" else CHECKPOINT,
        "seed": SEED,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "target_tokens": TARGET_TOKENS,
        "tokens_per_step": tokens_per_step,
        "tokens_trained": step * tokens_per_step,
        "initial_val_ce": initial_val_ce,
        "final_val_ce": history[-1]["val_ce"],
        "history": history,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved trajectory: {out_path}")
    print(f"\n=== {MODE.upper()} done ===")


if __name__ == "__main__":
    main()
