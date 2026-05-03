"""
PID-controlled tau interpolation to nGPT form.

tau=0: original weights
tau=1: all weight rows have unit L2 norm (nGPT form)

W_eff = (1-tau)*W_master + tau*normalize(W_master)

Phase 1: Freeze master weights, train norms only, tau 0→0.2
Phase 2: Unfreeze everything, tau 0.2→1.0
Phase 2 saves checkpoint before starting.

MacBook result: tau=0.2 is helpful (LASER effect), tau=1.0 achievable with full training.
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


# -- Tau-interpolated linear --

class TauLinear(nn.Module):
    """Linear layer with tau interpolation toward unit-norm rows.
    W_eff = (1-tau)*W + tau*W_normalized
    W_normalized = W / ||W_row||_2 for each row
    """
    def __init__(self, original_linear, tau=0.0):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.data.clone())
        self.tau = tau

    def forward(self, x):
        if self.tau <= 0.0:
            return F.linear(x, self.weight, self.bias)

        # Unit-norm version of each row
        row_norms = self.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        w_unit = self.weight / row_norms

        # Interpolate
        w_eff = (1.0 - self.tau) * self.weight + self.tau * w_unit
        return F.linear(x, w_eff, self.bias)


TARGET_PROJS = ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
ATTN_PROJS = ["qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj"]


def convert_to_tau(model):
    """Replace all target linear layers with TauLinear."""
    count = 0
    for layer in model.model.layers:
        for name in TARGET_PROJS:
            if name in ATTN_PROJS:
                parent = layer.self_attn
            else:
                parent = layer.mlp
            if not hasattr(parent, name):
                continue
            orig = getattr(parent, name)
            if isinstance(orig, nn.Linear):
                tau_mod = TauLinear(orig, tau=0.0)
                setattr(parent, name, tau_mod)
                count += 1
    return count


def set_tau_all(model, tau):
    """Set tau on all TauLinear modules."""
    for module in model.modules():
        if isinstance(module, TauLinear):
            module.tau = tau


def get_mean_row_norm(model):
    """Average effective row norm across all TauLinear modules."""
    import numpy as np
    norms = []
    for module in model.modules():
        if isinstance(module, TauLinear):
            row_norms = module.weight.data.norm(dim=1)
            # Effective norm after tau interpolation
            w_unit_norms = torch.ones_like(row_norms)
            eff_norms = (1.0 - module.tau) * row_norms + module.tau * w_unit_norms
            norms.extend(eff_norms.tolist())
    arr = np.array(norms)
    return float(arr.mean()), float(arr.std() / arr.mean())


def load_data(tokenizer, seq_len=256, max_train=2_000_000, max_val=100_000):
    # Try model-specific cache first, fall back to retokenizing
    model_tag = tokenizer.name_or_path.replace("/", "_")
    cache_path = f"data/owt_tokens_{model_tag}.pt"

    if os.path.exists(cache_path):
        print(f"  Loading cached corpus from {cache_path}...", flush=True)
        tokens = torch.load(cache_path, weights_only=True)
    else:
        # Retokenize from raw text
        print(f"  No cache for this tokenizer, retokenizing from OpenWebText...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        texts = []
        count = 0
        for ex in ds:
            texts.append(ex["text"])
            count += len(ex["text"]) // 4
            if count >= (max_train + max_val) * 1.5:
                break
            if len(texts) % 10000 == 0:
                print(f"    {len(texts)} docs, ~{count/1e6:.0f}M tokens...", flush=True)
        all_text = "\n\n".join(texts)
        tokens = tokenizer(all_text, return_tensors="pt", truncation=False)["input_ids"][0]
        os.makedirs("data", exist_ok=True)
        torch.save(tokens, cache_path)
        print(f"  Saved {len(tokens)/1e6:.1f}M tokens to {cache_path}")

    print(f"  {len(tokens)/1e6:.1f}M tokens loaded")
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
        return max(0.0, min(0.05, output))  # max 5% tau step


def main():
    torch.set_num_threads(32)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p2o6e100/nGPT_800m")
    cli = ap.parse_args()

    model_name = cli.model
    target_pct = 5.0
    phase1_target_tau = 0.2
    ft_steps_p1 = 200
    ft_steps_p2 = 300
    lr_p1 = 5e-5
    lr_p2 = 2e-5
    max_rounds = 300
    save_dir = "z8_pipeline_32b/pid_results"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"TAU INTERPOLATION TO nGPT: {model_name}")
    print(f"  tau=0 (original) → tau=1 (unit norm rows)")
    print(f"  Phase 1: freeze weights, norms only, tau 0→{phase1_target_tau}")
    print(f"  Phase 2: unfreeze all, tau {phase1_target_tau}→1.0")
    print(f"  Target: {target_pct}% above teacher")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading {model_name}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").eval()

    print("Loading data...", flush=True)
    train_chunks, val_chunks = load_data(tokenizer)

    # Convert to TauLinear
    n_converted = convert_to_tau(model)
    print(f"  Converted {n_converted} projections to TauLinear")

    teacher_ppl = eval_ppl(model, val_chunks)
    setpoint = teacher_ppl * (1.0 + target_pct / 100.0)
    mean_norm, cv = get_mean_row_norm(model)

    print(f"  Teacher PPL: {teacher_ppl:.2f}")
    print(f"  PID setpoint: {setpoint:.2f}")
    print(f"  Initial mean norm: {mean_norm:.3f}, CV: {cv:.4f}")
    print(flush=True)

    pid = PIDController(setpoint)
    tau = 0.0
    phase = 1

    results = {
        "model": model_name, "teacher_ppl": teacher_ppl,
        "init_mean_norm": mean_norm, "init_cv": cv,
        "phase1": [], "phase2": [],
    }

    t_start = time.time()

    print(f"{'='*60}")
    print(f"PHASE 1: Freeze weights, train norms, tau 0→{phase1_target_tau}")
    print(f"{'='*60}")
    print(f"  {'Round':>5} | {'Phase':>5} | {'Tau':>6} | {'PPL':>8} | {'Ratio':>6} | {'MeanN':>6} | {'CV':>6} | {'PID':>5} | {'Status'}")
    print(f"  {'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*10}", flush=True)

    for round_num in range(1, max_rounds + 1):
        ppl = eval_ppl(model, val_chunks)
        ratio = ppl / teacher_ppl
        mean_norm, cv = get_mean_row_norm(model)
        pid_out = pid.update(ppl)

        if ppl <= teacher_ppl:
            status = "FREE"
        elif ratio <= 1.0 + target_pct / 100:
            status = "ON TARGET"
        else:
            status = "OVER"

        print(f"  {round_num:5d} | P{phase:>4} | {tau:5.3f} | {ppl:8.2f} | {ratio:5.2f}x | "
              f"{mean_norm:5.3f} | {cv:5.4f} | {pid_out:5.3f} | {status}", flush=True)

        entry = {
            "round": round_num, "phase": phase, "tau": round(tau, 4),
            "ppl": round(ppl, 2), "ratio": round(ratio, 4),
            "mean_norm": round(mean_norm, 4), "cv": round(cv, 4),
            "pid_output": round(pid_out, 4), "status": status,
            "elapsed_s": round(time.time() - t_start, 1),
        }
        if phase == 1:
            results["phase1"].append(entry)
        else:
            results["phase2"].append(entry)

        # Save incrementally
        with open(Path(save_dir) / "pid_tau_ngpt.json", "w") as f:
            json.dump(results, f, indent=2)

        # Phase 1→2 transition
        if phase == 1 and tau >= phase1_target_tau:
            print(f"\n  PHASE 1 COMPLETE: tau={tau:.3f}")
            print(f"  Saving checkpoint...", flush=True)
            ckpt_dir = Path(save_dir) / "phase1_tau_14b_model"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            torch.save({"tau": tau, "ppl": ppl, "mean_norm": mean_norm, "cv": cv},
                        Path(save_dir) / "phase1_tau_checkpoint.pt")
            print(f"  Checkpoint saved: {ckpt_dir}")
            print(f"\n{'='*60}")
            print(f"PHASE 2: Unfreeze everything, tau {tau:.2f}→1.0")
            print(f"{'='*60}", flush=True)
            phase = 2
            pid = PIDController(setpoint, kp=0.3, ki=0.01, kd=0.1)
            continue

        # nGPT check
        if tau >= 0.99 and cv < 0.15:
            print(f"\n  nGPT FORM REACHED! tau={tau:.3f} cv={cv:.4f}")
            break

        if tau >= 1.0:
            print(f"\n  TAU=1.0 — nGPT conversion complete")
            break

        # Apply tau step
        if pid_out > 0.001:
            tau = min(1.0, tau + pid_out)
            set_tau_all(model, tau)

        # Fine-tune
        # Train all parameters (weights must be able to move)
        for p in model.parameters():
            p.requires_grad_(True)
        all_params = [p for p in model.parameters() if p.requires_grad]
        lr_use = lr_p1 if phase == 1 else lr_p2
        ft_use = ft_steps_p1 if phase == 1 else ft_steps_p2
        train_steps(model, train_chunks, all_params, ft_use, lr_use)

    # Final
    final_ppl = eval_ppl(model, val_chunks)
    final_norm, final_cv = get_mean_row_norm(model)
    elapsed_h = (time.time() - t_start) / 3600

    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"  Teacher PPL:     {teacher_ppl:.2f}")
    print(f"  Final PPL:       {final_ppl:.2f} ({final_ppl/teacher_ppl:.2f}x)")
    print(f"  Final tau:       {tau:.3f}")
    print(f"  Final mean norm: {final_norm:.3f}")
    print(f"  Final CV:        {final_cv:.4f}")
    print(f"  Time:            {elapsed_h:.2f}h")

    nGPT = tau >= 0.95 and final_cv < 0.2
    print(f"  nGPT form: {'YES' if nGPT else 'NOT YET'}")

    results["final"] = {
        "ppl": final_ppl, "tau": tau, "mean_norm": final_norm,
        "cv": final_cv, "nGPT": nGPT, "elapsed_h": elapsed_h,
    }
    with open(Path(save_dir) / "pid_tau_ngpt.json", "w") as f:
        json.dump(results, f, indent=2)

    # Always save the model
    model_tag = model_name.split("/")[-1].replace("-", "_")
    save_path = Path(save_dir) / f"ngpt_{model_tag}_tau{tau:.2f}"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Model saved: {save_path}")


if __name__ == "__main__":
    main()
