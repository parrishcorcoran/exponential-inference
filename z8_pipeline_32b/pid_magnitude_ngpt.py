"""
Two-phase PID magnitude anneal to nGPT form.

Phase 1: Freeze weights, shrink magnitude, train only norms.
  - Each step: multiply all weights by (1 - step_size)
  - PID controls step_size based on quality vs 5% target
  - Find the floor where norms-only can't recover

Phase 2: Unfreeze everything, continue shrinking.
  - Train all parameters (weights + norms)
  - PID continues controlling compression rate
  - Push toward magnitude = 0 (unit norm rows = nGPT form)

Each step is 0.01 magnitude (1% of remaining).
Fine-tune 200 steps between each compression step.
"""

import argparse
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


# ── Data ─────────────────────────────────────────────────────────────

def load_data(tokenizer, seq_len=256, max_train=2_000_000, max_val=100_000):
    cache_path = "data/owt_tokens_50M.pt"
    if os.path.exists(cache_path):
        print(f"  Loading cached corpus from {cache_path}...", flush=True)
        tokens = torch.load(cache_path)
        print(f"  {len(tokens)/1e6:.1f}M tokens from cache")
    else:
        from datasets import load_dataset
        print("  Loading OpenWebText (streaming)...", flush=True)
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        texts = []
        count = 0
        for ex in ds:
            texts.append(ex["text"])
            count += len(ex["text"]) // 4
            if count >= (max_train + max_val) * 1.2:
                break
        all_text = "\n\n".join(texts)
        tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]

    val_tokens = tokens[:max_val]
    train_tokens = tokens[max_val:max_val + max_train]

    def chunk(toks):
        n = len(toks) // (seq_len + 1)
        return toks[:n * (seq_len + 1)].view(n, seq_len + 1)

    return chunk(train_tokens), chunk(val_tokens)


@torch.inference_mode()
def eval_ppl(model, val_chunks, seq_len=256, n_eval=30):
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
    return math.exp(min(ce, 20))


def train_steps(model, train_chunks, trainable, n_steps, lr, seq_len=256):
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    indices = list(range(len(train_chunks)))
    random.shuffle(indices)
    idx_iter = iter(indices)

    for step in range(n_steps):
        try:
            idx = next(idx_iter)
        except StopIteration:
            random.shuffle(indices)
            idx_iter = iter(indices)
            idx = next(idx_iter)

        batch = train_chunks[idx:idx+1]
        inp = batch[:, :seq_len]
        tgt = batch[:, 1:seq_len+1]

        logits = model(input_ids=inp, use_cache=False).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

    model.eval()
    del optimizer
    gc.collect()


# ── Magnitude operations ─────────────────────────────────────────────

def get_weight_params(model):
    """Get all weight parameters (not norms, not embeddings)."""
    params = []
    for name, p in model.named_parameters():
        if "norm" not in name.lower() and "embed" not in name.lower() and "lm_head" not in name.lower():
            params.append((name, p))
    return params


def get_norm_params(model):
    """Get all norm parameters."""
    params = []
    for name, p in model.named_parameters():
        if "norm" in name.lower():
            params.append(p)
    return params


def get_all_trainable(model):
    """Get all parameters for phase 2."""
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    return params


def avg_magnitude(model):
    """Average absolute weight magnitude (excluding norms/embeds)."""
    total = 0
    count = 0
    for name, p in get_weight_params(model):
        total += p.data.float().abs().mean().item()
        count += 1
    return total / max(count, 1)


def avg_row_norm(model):
    """Average row L2 norm across all linear projections."""
    total = 0
    count = 0
    for name, p in get_weight_params(model):
        if p.dim() == 2:
            norms = p.data.float().norm(dim=1)
            total += norms.mean().item()
            count += 1
    return total / max(count, 1)


def shrink_magnitude(model, factor):
    """Multiply all weights by factor (e.g., 0.99 for 1% shrink)."""
    with torch.no_grad():
        for name, p in get_weight_params(model):
            p.mul_(factor)


def row_norm_cv(model):
    """Coefficient of variation of row norms — measures sphericality."""
    all_norms = []
    for name, p in get_weight_params(model):
        if p.dim() == 2:
            norms = p.data.float().norm(dim=1).tolist()
            all_norms.extend(norms)
    import numpy as np
    arr = np.array(all_norms)
    return float(arr.std() / arr.mean()), float(arr.mean())


# ── PID Controller ───────────────────────────────────────────────────

class PIDController:
    def __init__(self, setpoint, kp=0.5, ki=0.02, kd=0.15):
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_ppl):
        error = self.setpoint - current_ppl
        self.integral = max(-10, min(10, self.integral + error))
        derivative = error - self.prev_error
        self.prev_error = error
        output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        output = output / max(self.setpoint, 1.0)
        return max(0.0, min(0.02, output))  # 0 to 2% max step


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--target-pct", type=float, default=5.0)
    p.add_argument("--ft-steps-p1", type=int, default=200, help="FT steps per round, phase 1 (norms only)")
    p.add_argument("--ft-steps-p2", type=int, default=300, help="FT steps per round, phase 2 (all params)")
    p.add_argument("--lr-p1", type=float, default=5e-5)
    p.add_argument("--lr-p2", type=float, default=2e-5)
    p.add_argument("--max-rounds", type=int, default=300)
    p.add_argument("--save-dir", default="z8_pipeline_32b/pid_results")
    args = p.parse_args()

    torch.set_num_threads(32)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print(f"TWO-PHASE PID MAGNITUDE ANNEAL TO nGPT")
    print(f"  Model: {args.model}")
    print(f"  Target: {args.target_pct}% above teacher")
    print(f"  Phase 1: freeze weights, train norms")
    print(f"  Phase 2: unfreeze all, push to unit norm")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\nLoading {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    print("Loading data...", flush=True)
    train_chunks, val_chunks = load_data(tokenizer)

    teacher_ppl = eval_ppl(model, val_chunks)
    setpoint = teacher_ppl * (1.0 + args.target_pct / 100.0)
    init_mag = avg_magnitude(model)
    init_cv, init_mean_norm = row_norm_cv(model)

    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  PID setpoint: {setpoint:.2f}")
    print(f"  Initial magnitude: {init_mag:.6f}")
    print(f"  Initial row norm CV: {init_cv:.4f} (mean={init_mean_norm:.3f})")
    print(flush=True)

    pid = PIDController(setpoint)

    results = {
        "model": args.model, "teacher_ppl": teacher_ppl,
        "init_magnitude": init_mag, "init_cv": init_cv,
        "init_mean_norm": init_mean_norm,
        "phase1": [], "phase2": [],
    }

    t_start = time.time()
    current_mag_ratio = 1.0
    phase = 1
    phase_switched = False
    consecutive_stuck = 0

    print(f"{'='*60}")
    print(f"PHASE 1: Freeze weights, train norms only")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Phase':>5} | {'Mag':>6} | {'PPL':>8} | {'Ratio':>6} | {'CV':>6} | {'MeanN':>6} | {'PID':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}", flush=True)

    for round_num in range(1, args.max_rounds + 1):
        # Eval
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl
        cv, mean_norm = row_norm_cv(model)
        pid_out = pid.update(ppl)

        # Status
        if ppl <= teacher_ppl:
            status = "FREE"
        elif ratio <= 1.0 + args.target_pct / 100:
            status = "ON TARGET"
        else:
            status = "OVER"

        print(f"  {round_num:5d} | P{phase:>4} | {current_mag_ratio:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | "
              f"{cv:5.3f} | {mean_norm:5.3f} | {pid_out:5.3f} | {status}", flush=True)

        entry = {
            "round": round_num, "phase": phase,
            "mag_ratio": round(current_mag_ratio, 4),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "cv": round(cv, 4), "mean_norm": round(mean_norm, 4),
            "pid_output": round(pid_out, 4), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        }

        if phase == 1:
            results["phase1"].append(entry)
        else:
            results["phase2"].append(entry)

        # Save incrementally
        rpath = Path(args.save_dir) / "pid_magnitude_ngpt.json"
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2)

        # Phase 1→2 transition: when magnitude drops below 0.2
        if phase == 1 and current_mag_ratio <= 0.20:
            # Save checkpoint before switching
            ckpt_path = Path(args.save_dir) / "phase1_checkpoint_14b.pt"
            print(f"\n  PHASE 1 COMPLETE at magnitude {current_mag_ratio:.3f}")
            print(f"  Saving checkpoint: {ckpt_path}", flush=True)
            model.save_pretrained(str(Path(args.save_dir) / "phase1_14b_model"))
            tokenizer.save_pretrained(str(Path(args.save_dir) / "phase1_14b_model"))
            torch.save({
                "mag_ratio": current_mag_ratio,
                "ppl": ppl, "cv": cv, "mean_norm": mean_norm,
                "round": round_num,
            }, ckpt_path)
            print(f"  Switching to PHASE 2: unfreeze all weights")
            print(f"{'='*60}")
            print(f"PHASE 2: Unfreeze everything, push to unit norm")
            print(f"{'='*60}", flush=True)
            phase = 2
            phase_switched = True
            consecutive_stuck = 0
            pid = PIDController(setpoint, kp=0.3, ki=0.01, kd=0.1)
            continue

        # Also handle stuck in phase 1
        if phase == 1 and pid_out <= 0.001 and ppl > setpoint:
            consecutive_stuck += 1
            if consecutive_stuck >= 3:
                ckpt_path = Path(args.save_dir) / "phase1_checkpoint_14b.pt"
                print(f"\n  PHASE 1 FLOOR at magnitude {current_mag_ratio:.3f}")
                print(f"  Saving checkpoint: {ckpt_path}", flush=True)
                model.save_pretrained(str(Path(args.save_dir) / "phase1_14b_model"))
                tokenizer.save_pretrained(str(Path(args.save_dir) / "phase1_14b_model"))
                torch.save({
                    "mag_ratio": current_mag_ratio,
                    "ppl": ppl, "cv": cv, "mean_norm": mean_norm,
                    "round": round_num,
                }, ckpt_path)
                print(f"  Switching to PHASE 2: unfreeze all weights")
                print(f"{'='*60}")
                print(f"PHASE 2: Unfreeze everything, push to unit norm")
                print(f"{'='*60}", flush=True)
                phase = 2
                phase_switched = True
                consecutive_stuck = 0
                pid = PIDController(setpoint, kp=0.3, ki=0.01, kd=0.1)
                continue
        else:
            consecutive_stuck = 0

        # Phase 2 termination: magnitude near zero or stuck
        if phase == 2 and pid_out <= 0.001 and ppl > setpoint:
            consecutive_stuck += 1
            if consecutive_stuck >= 5:
                print(f"\n  PHASE 2 WALL at magnitude {current_mag_ratio:.3f}")
                break
        elif phase == 2:
            consecutive_stuck = 0

        # Check if we reached nGPT (mean norm ~ 1.0, CV < 0.1)
        if mean_norm < 1.1 and cv < 0.15:
            print(f"\n  nGPT FORM REACHED! mean_norm={mean_norm:.3f} cv={cv:.4f}")
            break

        if current_mag_ratio <= 0.01:
            print(f"\n  MAGNITUDE NEAR ZERO — nGPT complete")
            break

        # Apply compression
        if pid_out > 0.001:
            shrink_factor = 1.0 - pid_out
            shrink_magnitude(model, shrink_factor)
            current_mag_ratio *= shrink_factor

        # Fine-tune
        if phase == 1:
            # Norms only
            for param in model.parameters():
                param.requires_grad_(False)
            norm_params = get_norm_params(model)
            for param in norm_params:
                param.requires_grad_(True)
            if norm_params:
                train_steps(model, train_chunks, norm_params,
                            args.ft_steps_p1, args.lr_p1)
        else:
            # All parameters
            for param in model.parameters():
                param.requires_grad_(True)
            all_params = list(model.parameters())
            train_steps(model, train_chunks, all_params,
                        args.ft_steps_p2, args.lr_p2)

    # Final summary
    final_ppl = eval_ppl(model, val_chunks)
    final_cv, final_mean_norm = row_norm_cv(model)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"RESULT: Magnitude Anneal to nGPT")
    print(f"{'='*60}")
    print(f"  Teacher PPL:        {teacher_ppl:.2f}")
    print(f"  Final PPL:          {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Initial magnitude:  {init_mag:.6f}")
    print(f"  Final mag ratio:    {current_mag_ratio:.4f}")
    print(f"  Initial CV:         {init_cv:.4f} (mean norm={init_mean_norm:.3f})")
    print(f"  Final CV:           {final_cv:.4f} (mean norm={final_mean_norm:.3f})")
    print(f"  Phase 1 rounds:     {len(results['phase1'])}")
    print(f"  Phase 2 rounds:     {len(results['phase2'])}")
    print(f"  Time:               {elapsed_h:.2f}h")

    nGPT = final_cv < 0.15 and final_mean_norm < 1.1
    print(f"\n  nGPT form: {'YES' if nGPT else 'NOT YET'}")
    if nGPT:
        print(f"  Congratulations! Model converted to nGPT geometry.")

    results["final_ppl"] = final_ppl
    results["final_mag_ratio"] = current_mag_ratio
    results["final_cv"] = final_cv
    results["final_mean_norm"] = final_mean_norm
    results["nGPT_achieved"] = nGPT
    results["elapsed_h"] = elapsed_h

    rpath = Path(args.save_dir) / "pid_magnitude_ngpt.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {rpath}")

    # Save model if nGPT achieved
    if nGPT:
        save_path = Path(args.save_dir) / "ngpt_4b"
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        print(f"  Model saved: {save_path}")


if __name__ == "__main__":
    main()
