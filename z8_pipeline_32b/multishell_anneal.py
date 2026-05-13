"""
Multi-Shell nGPT Anneal — CORRECT implementation.

Like Strix's pipeline_unit_norm_anneal.py but with per-component shell radii
instead of forcing unit 1.0.

KEY: master weights stay untouched. Tau projection applied at FORWARD TIME only.
Training updates master weights via gradients. Norms compensate.

For each projection in each layer:
  shell_radius = mean row norm (measured once at start)
  W_eff = W / ||W_row|| * ((1-tau)*||W_row|| + tau*shell_radius)

At tau=0: W_eff = W (original)
At tau=1: every row has norm = shell_radius (on its shell)
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


PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
              'gate_proj', 'up_proj', 'down_proj']
ATTN_PROJS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Global tau — closures read from this
_TAU = 0.0
def set_tau(t): global _TAU; _TAU = t
def get_tau(): return _TAU


def project_rows_to_shell(W, tau, target_radius):
    """Project rows toward target_radius, controlled by tau.
    tau=0: W unchanged. tau=1: all rows have norm = target_radius."""
    if tau <= 0.0:
        return W
    row_norms = W.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target_norms = (1.0 - tau) * row_norms + tau * target_radius
    return W / row_norms * target_norms


def patch_linear(module, target_radius):
    """Replace forward to project W toward shell at forward time.
    Master weights stay untouched."""
    weight = module.weight
    bias = module.bias

    def projected_forward(x):
        W = project_rows_to_shell(weight, get_tau(), target_radius)
        return F.linear(x, W, bias)

    module.forward = projected_forward


def measure_shells(model, L):
    """Measure mean row norm per component — the natural shell radius."""
    shells = {}
    for li in range(L):
        shells[li] = {}
        for name in PROJ_NAMES:
            if name in ATTN_PROJS:
                w = getattr(model.model.layers[li].self_attn, name).weight.data
            else:
                w = getattr(model.model.layers[li].mlp, name).weight.data
            norms = w.norm(dim=1)
            shells[li][name] = norms.mean().item()
    return shells


def measure_cv(model, L):
    """Average coefficient of variation of row norms."""
    cvs = []
    for li in range(L):
        for name in PROJ_NAMES:
            if name in ATTN_PROJS:
                w = getattr(model.model.layers[li].self_attn, name).weight.data
            else:
                w = getattr(model.model.layers[li].mlp, name).weight.data
            norms = w.norm(dim=1)
            cvs.append((norms.std() / norms.mean()).item())
    return sum(cvs) / len(cvs)


@torch.inference_mode()
def eval_ppl(model, val_tokens, seq_len=256, n=20):
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

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--target-pct", type=float, default=5.0)
    ap.add_argument("--tau-step", type=float, default=0.1, help="Tau increment per drop")
    ap.add_argument("--steps-per-drop", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-holds", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=256)
    cli = ap.parse_args()

    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"MULTI-SHELL ANNEAL (correct): {cli.model}")
    print(f"  Master weights preserved. Tau at forward time only.")
    print(f"  Per-component shell radii (not unit 1.0)")
    print(f"  Tau step: {cli.tau_step}, {cli.steps_per_drop} steps per drop")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cli.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cli.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    L = model.config.num_hidden_layers
    print(f"  L={L}, d={model.config.hidden_size}")

    # Load diverse corpus
    corpus_path = "data/diverse_corpus_100M.pt"
    print(f"\n  Loading {corpus_path}...", flush=True)
    tokens = torch.load(corpus_path, weights_only=True)
    val_tokens = tokens[:100_000]
    train_tokens = tokens[100_000:]
    print(f"  {len(tokens)/1e6:.1f}M tokens")

    # Baseline
    teacher_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    threshold = 0.5  # max CE increase allowed
    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  Threshold: +{threshold} CE", flush=True)

    # Measure natural shells
    shells = measure_shells(model, L)
    init_cv = measure_cv(model, L)
    print(f"  Initial CV: {init_cv:.4f}")

    # Patch all target linears
    print(f"\n  Patching {L * len(PROJ_NAMES)} linears...", flush=True)
    for li in range(L):
        for name in PROJ_NAMES:
            if name in ATTN_PROJS:
                module = getattr(model.model.layers[li].self_attn, name)
            else:
                module = getattr(model.model.layers[li].mlp, name)
            patch_linear(module, shells[li][name])

    # Verify tau=0 matches baseline
    set_tau(0.0)
    verify = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    print(f"  Verify tau=0: PPL={verify:.2f} (should match {teacher_ppl:.2f})", flush=True)

    # Get baseline CE for threshold
    base_ce = math.log(teacher_ppl)

    # Strix recipe: all weights EXCEPT embeddings and lm_head
    for name, p in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} (weights + norms, no embed/lm_head)")

    # Tau schedule: 0.1 steps from 0 to 1
    tau_schedule = [round(cli.tau_step * (i + 1), 2) for i in range(int(1.0 / cli.tau_step))]
    print(f"  Tau schedule: {tau_schedule}", flush=True)

    n_available = len(train_tokens) // (cli.seq_len + 1) - 1
    results = {
        "model": cli.model, "teacher_ppl": teacher_ppl,
        "shells": {str(k): v for k, v in shells.items()},
        "drops": [], "history": [],
    }

    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"ANNEALING")
    print(f"{'='*60}", flush=True)

    for tau in tau_schedule:
        set_tau(tau)
        holds = 0

        while holds <= cli.max_holds:
            # Train
            model.train()
            optimizer = torch.optim.AdamW(trainable, lr=cli.lr, weight_decay=0.01)

            for step in range(1, cli.steps_per_drop + 1):
                idx = random.randint(0, n_available - 1)
                start = idx * cli.seq_len
                inp = train_tokens[start:start + cli.seq_len].unsqueeze(0)
                tgt = train_tokens[start + 1:start + cli.seq_len + 1].unsqueeze(0)

                logits = model(input_ids=inp, use_cache=False).logits
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                if step % 250 == 0:
                    val_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
                    val_ce = math.log(val_ppl)
                    delta = val_ce - base_ce
                    cv = measure_cv(model, L)
                    results["history"].append({
                        "step": step, "tau": tau, "val_ce": round(val_ce, 4),
                        "delta": round(delta, 4), "cv": round(cv, 4),
                    })
                    print(f"  tau={tau:.2f} step {step:>4}/{cli.steps_per_drop}: "
                          f"PPL={val_ppl:.2f} delta={delta:+.4f} CV={cv:.4f}", flush=True)

            del optimizer; gc.collect()
            model.eval()

            # Check quality
            val_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
            val_ce = math.log(val_ppl)
            delta = val_ce - base_ce
            cv = measure_cv(model, L)

            if delta <= threshold:
                outcome = "pass"
                print(f"  >>> TAU={tau:.2f} PASSED: delta={delta:+.4f} CV={cv:.4f}", flush=True)
                results["drops"].append({
                    "tau": tau, "holds": holds, "final_ppl": round(val_ppl, 2),
                    "delta": round(delta, 4), "cv": round(cv, 4), "outcome": "pass",
                })
                break
            else:
                holds += 1
                if holds > cli.max_holds:
                    print(f"  >>> TAU={tau:.2f} FAILED after {cli.max_holds} holds", flush=True)
                    results["drops"].append({
                        "tau": tau, "holds": holds, "final_ppl": round(val_ppl, 2),
                        "delta": round(delta, 4), "outcome": "fail",
                    })
                else:
                    print(f"  >>> TAU={tau:.2f} HOLD {holds}/{cli.max_holds}: "
                          f"delta={delta:+.4f}, retraining...", flush=True)

        # Save checkpoint at EVERY tau level
        with open(Path(save_dir) / "multishell_anneal.json", "w") as f:
            json.dump(results, f, indent=2)
        ckpt = Path(save_dir) / f"multishell_tau{tau:.2f}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        print(f"  >>> CHECKPOINT: {ckpt.name}", flush=True)

        if results["drops"][-1]["outcome"] == "fail":
            print(f"\n  STOPPED at tau={tau:.2f}")
            break

    # Final
    final_ppl = eval_ppl(model, val_tokens, seq_len=cli.seq_len)
    final_cv = measure_cv(model, L)
    elapsed_h = (time.time() - t_start) / 3600
    highest_tau = max(d["tau"] for d in results["drops"] if d["outcome"] == "pass") if any(d["outcome"] == "pass" for d in results["drops"]) else 0

    print(f"\n{'='*60}")
    print(f"  Teacher: {teacher_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f}")
    print(f"  Highest tau passed: {highest_tau}")
    print(f"  CV: {init_cv:.4f} → {final_cv:.4f}")
    print(f"  Time: {elapsed_h:.2f}h")

    ckpt = Path(save_dir) / "multishell_final"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    with open(Path(save_dir) / "multishell_anneal.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {ckpt}")


if __name__ == "__main__":
    main()
