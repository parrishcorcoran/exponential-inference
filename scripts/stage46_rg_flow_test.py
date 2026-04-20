"""
Stage 46 — Frame 3: RG flow convergence test.

Tests whether the transformer's layers behave as a renormalization group
flow toward an attractor (= final prediction). If so, per-layer predictions
should converge monotonically to the final-layer prediction.

Protocol (Qwen3-0.6B):
  1. Forward on prompts with output_hidden_states=True.
  2. Apply final RMSNorm + lm_head at every layer's last-position hidden
     state → per-layer logit distribution.
  3. Compute per-layer KL(p_final || p_layer_i) — how far layer i's
     prediction is from the final-layer prediction.
  4. Average across many generation steps.
  5. If KL decreases monotonically and late layers flatten, RG flow
     confirmed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


PROMPTS = [
    "The discovery that inference accelerates with context is",
    "Proteins fold into complex three-dimensional structures determined by their",
    "Quantum mechanics describes the behavior of matter and energy at",
    "Linear algebra provides the mathematical foundation for many",
    "Evolution by natural selection operates on heritable variation in",
    "The mitochondrion is the powerhouse of the cell and",
    "The universe expanded from an initial state of",
    "Cryptographic hash functions map arbitrary input to",
    "A black hole event horizon is the boundary at which",
    "Photosynthesis in plants converts carbon dioxide and",
]


def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager").to(device).eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--gen-tokens", type=int, default=30)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/stage46_rg_flow_test.json")
    args = p.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    print(f"device={device}")

    print(f"\n=== loading {args.model} ===")
    model, tokenizer = load_model(args.model, device)
    L = len(model.model.layers)
    final_norm = model.model.norm
    lm_head = model.lm_head
    print(f"  L={L}")

    print(f"\n=== measuring per-layer KL from final prediction ===")
    # per_layer_kl[i] accumulates across tokens
    per_layer_kl = [0.0] * L
    per_layer_argmax_agreement = [0] * L
    n_tokens = 0

    for prompt_idx, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=True)
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for step in range(args.gen_tokens):
            with torch.inference_mode():
                out = model(
                    input_ids=next_token, past_key_values=past,
                    output_hidden_states=True, use_cache=True)
            past = out.past_key_values
            hidden = out.hidden_states
            # Final logits (layer L)
            final_h = hidden[-1][:, -1, :]
            final_logits = lm_head(final_norm(final_h))[0]  # [vocab]
            p_final = F.softmax(final_logits.float(), dim=-1)
            log_p_final = F.log_softmax(final_logits.float(), dim=-1)
            final_argmax = int(final_logits.argmax(dim=-1).item())

            # Per-layer KL(p_final || p_i) = sum p_final * (log p_final - log p_i)
            for i in range(L):
                h_i = hidden[i + 1][:, -1, :]
                logits_i = lm_head(final_norm(h_i))[0]
                log_p_i = F.log_softmax(logits_i.float(), dim=-1)
                kl = (p_final * (log_p_final - log_p_i)).sum().item()
                per_layer_kl[i] += kl
                layer_argmax = int(logits_i.argmax(dim=-1).item())
                if layer_argmax == final_argmax:
                    per_layer_argmax_agreement[i] += 1

            next_token = torch.tensor([[final_argmax]], device=device)
            n_tokens += 1
            if final_argmax == tokenizer.eos_token_id:
                break
        print(f"  prompt {prompt_idx+1}/{len(PROMPTS)} done")

    kl_per_layer = [x / n_tokens for x in per_layer_kl]
    agree_per_layer = [x / n_tokens for x in per_layer_argmax_agreement]

    print(f"\n=== per-layer convergence to final ===")
    print(f"  (averaged over {n_tokens} generation steps)")
    print(f"  {'layer':>5}  {'KL vs final':>12}  {'argmax agree':>13}  note")
    for i in range(L):
        note = ""
        if i > 0:
            if kl_per_layer[i] > kl_per_layer[i-1]:
                note = "  <-- KL increased vs prev layer"
        print(f"  {i:>5}  {kl_per_layer[i]:>12.4f}  {agree_per_layer[i]:>13.3f}{note}")

    # Monotonicity check: count layers where KL increased vs prev
    monotonic_violations = sum(
        1 for i in range(1, L) if kl_per_layer[i] > kl_per_layer[i-1]
    )
    print(f"\n=== monotonicity check ===")
    print(f"  layers where KL increased vs previous: {monotonic_violations}/{L-1}")
    if monotonic_violations == 0:
        print(f"  STRICTLY MONOTONIC → strong RG flow signature")
    elif monotonic_violations <= 3:
        print(f"  MOSTLY MONOTONIC ({monotonic_violations} violations) → RG flow consistent")
    else:
        print(f"  NOT MONOTONIC ({monotonic_violations} violations) → RG flow weak/falsified")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "L": L,
            "n_tokens": n_tokens,
            "per_layer_kl_vs_final": kl_per_layer,
            "per_layer_argmax_agreement": agree_per_layer,
            "monotonic_violations": monotonic_violations,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
