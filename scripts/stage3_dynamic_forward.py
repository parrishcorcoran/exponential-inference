"""Stage 3 driver: install dynamic-rank hooks, run correctness and
quality gates.

Correctness gate:
  - Run the model at full rank through DynamicRankBitNet. Logits must
    match the unwrapped model to within atol=1e-3.

Quality gate:
  - On a held-out validation slice (the last ``val-tokens`` of the
    Stage 1 calibration stream, which the predictor never saw for
    feature/target z-scoring), compute next-token accuracy for
      (a) base model
      (b) dynamic-rank with safety_multiplier=args.safety_multiplier
  - Must be within 2% absolute accuracy of baseline. If it drops more
    than that, the script loops, increasing safety_multiplier by 1.5x
    until the gate passes (with a max of 4x) and reports the chosen
    multiplier.

Writes results/stage3_gates.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import DEFAULT_MODEL_ID, describe_backend, load_bitnet  # noqa: E402
from src.measurement.cache_hidden_states import load_meta  # noqa: E402
from src.inference.dynamic_rank import (  # noqa: E402
    DynamicRankBitNet,
    DynamicRankConfig,
    build_layer_means,
    check_full_rank_matches_base,
)
from src.routing.rank_predictor import load_predictor  # noqa: E402


def _next_token_accuracy(model, ids: torch.Tensor, batch_size: int = 2) -> float:
    """Teacher-forced next-token top-1 accuracy on ``ids`` (1D)."""
    device = next(model.parameters()).device
    chunk_size = 512
    ids = ids.to(device)
    n_chunks = ids.numel() // chunk_size
    hits = 0
    tot = 0
    for b in range(0, n_chunks, batch_size):
        batch = ids[b * chunk_size : (b + batch_size) * chunk_size]
        m = (batch.numel() // chunk_size) * chunk_size
        batch = batch[:m].view(-1, chunk_size)
        with torch.inference_mode():
            out = model(input_ids=batch)
        logits = out.logits[:, :-1]
        targets = batch[:, 1:]
        pred = logits.argmax(dim=-1)
        hits += (pred == targets).sum().item()
        tot += targets.numel()
    return hits / max(tot, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "stage1_cache"))
    p.add_argument("--predictor", default=str(REPO_ROOT / "results" / "stage2_predictor" / "rank_predictor.pt"))
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "stage3_gates.json"))
    p.add_argument("--val-tokens", type=int, default=4096,
                   help="How many tail tokens of the calibration stream "
                        "to use for the quality gate.")
    p.add_argument("--accuracy-tolerance", type=float, default=0.02)
    p.add_argument("--safety-multiplier", type=float, default=1.0)
    p.add_argument("--safety-cap", type=float, default=4.0)
    p.add_argument("--correctness-prompt",
                   default="The participation ratio measures how many directions")
    args = p.parse_args()

    print("=== backend ===")
    print(json.dumps(describe_backend(), indent=2))

    print("\n=== loading model ===", flush=True)
    loaded = load_bitnet(model_id=args.model_id, device=args.device)
    model = loaded.model
    tokenizer = loaded.tokenizer

    print("\n=== loading predictor ===")
    predictor = load_predictor(args.predictor)
    print(f"  src_layer=L{predictor.src_layer:02d}  "
          f"targets={predictor.target_layers}  "
          f"k_basis={predictor.k_basis}  "
          f"basis_k_full={predictor.basis_k_full}")

    layer_means = build_layer_means(args.cache_dir, predictor.target_layers)

    # --- correctness gate ---------------------------------------------
    print("\n=== correctness gate: full rank == base ===", flush=True)
    ids = tokenizer(args.correctness_prompt, return_tensors="pt").input_ids
    ids = ids.to(loaded.device)
    passed, max_d, mean_d = check_full_rank_matches_base(
        model, predictor, layer_means, ids
    )
    print(f"  max|Δ|={max_d:.3e}  mean|Δ|={mean_d:.3e}  passed={passed}")
    if not passed:
        print("ERROR: forward-pass at full rank does not match base. "
              "Projection math has a bug; halting.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "correctness": {
                "passed": False,
                "max_abs_diff": max_d,
                "mean_abs_diff": mean_d,
            },
        }, indent=2))
        return 2

    # --- quality gate -------------------------------------------------
    print("\n=== quality gate: next-token accuracy ===", flush=True)
    meta = load_meta(args.cache_dir)
    tokens_flat = torch.load(Path(args.cache_dir) / "tokens.pt",
                             map_location="cpu")
    val_ids = tokens_flat[-args.val_tokens :]
    base_acc = _next_token_accuracy(model, val_ids)
    print(f"  base acc={base_acc:.4f}")

    multiplier = args.safety_multiplier
    attempts = []
    while True:
        cfg = DynamicRankConfig(
            rank_floor=1,
            rank_ceiling=predictor.basis_k_full,
            safety_multiplier=multiplier,
        )
        wrapper = DynamicRankBitNet(model, predictor, layer_means, cfg)
        with wrapper:
            dyn_acc = _next_token_accuracy(model, val_ids)
        mean_ranks = wrapper.stats.mean_rank_per_layer()
        entry = {
            "safety_multiplier": multiplier,
            "dyn_acc": dyn_acc,
            "delta": dyn_acc - base_acc,
            "mean_rank_per_layer": mean_ranks,
        }
        attempts.append(entry)
        print(f"  mul={multiplier:.2f}  dyn acc={dyn_acc:.4f}  "
              f"Δ={dyn_acc - base_acc:+.4f}  mean_ranks={mean_ranks}")
        if base_acc - dyn_acc <= args.accuracy_tolerance:
            break
        if multiplier >= args.safety_cap:
            print("  hit safety cap without passing quality gate")
            break
        multiplier = min(multiplier * 1.5, args.safety_cap)

    out = {
        "model_id": args.model_id,
        "predictor": args.predictor,
        "correctness": {
            "passed": passed,
            "max_abs_diff": max_d,
            "mean_abs_diff": mean_d,
        },
        "base_accuracy": base_acc,
        "quality_tolerance": args.accuracy_tolerance,
        "attempts": attempts,
        "accepted_multiplier": attempts[-1]["safety_multiplier"]
            if (base_acc - attempts[-1]["dyn_acc"]) <= args.accuracy_tolerance
            else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")

    return 0 if out["accepted_multiplier"] is not None else 3


if __name__ == "__main__":
    sys.exit(main())
