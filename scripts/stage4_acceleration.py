"""Stage 4 driver: measure the acceleration curve.

For each prompt in ``data/prompts.json``:
  1. Generate ``max_new_tokens`` tokens with the base model (KV-cached).
  2. Generate the same number with DynamicRankBitNet active at the
     quality-gate-approved safety multiplier.

Per-step wall-clock is recorded for both modes. Results are averaged
across prompts; the output is:

  - results/acceleration_curve.png (per-position speedup, mean +/- SEM)
  - results/rank_distribution.png  (mean predicted rank vs position,
    averaged across prompts and target layers)
  - results/generation_samples/<prompt_id>_{base,dynamic}.txt
  - results/summary.json  (per-prompt and aggregate numbers,
    including integral speedup and rank-at-position snapshots)

The script is deterministic: greedy decoding, fixed warmup steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import DEFAULT_MODEL_ID, describe_backend, load_bitnet  # noqa: E402
from src.evaluation.acceleration_curve import (  # noqa: E402
    aggregate_curve,
    generate_and_time,
    speedup_curve,
)
from src.inference.dynamic_rank import (  # noqa: E402
    DynamicRankBitNet,
    DynamicRankConfig,
    build_layer_means,
)
from src.routing.rank_predictor import load_predictor  # noqa: E402


def _maybe_plot(results, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    base_curve = results["aggregate"]["base"]
    dyn_curve = results["aggregate"]["dynamic"]
    speed = results["aggregate"]["speedup"]
    positions = list(range(len(speed["ratio"])))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(positions, speed["ratio"], linewidth=1.8, color="C0")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("generation position")
    ax.set_ylabel("per-token speedup (base / dynamic)")
    ax.set_title(f"BitNet b1.58 2B — per-token speedup during generation\n"
                 f"({results['meta']['n_prompts']} prompts, greedy, "
                 f"{results['meta']['max_new_tokens']} tokens each)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "acceleration_curve.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    mr = results["aggregate"]["mean_rank_per_position"]
    ax2.plot(positions, mr, color="C2", linewidth=1.8)
    ax2.set_xlabel("generation position")
    ax2.set_ylabel("mean predicted rank (over target layers)")
    ax2.set_title("Per-token manifold rank during generation")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "rank_distribution.png", dpi=150)
    plt.close(fig2)


def _mean_across_runs(values_per_run):
    if not values_per_run:
        return []
    n = min(len(v) for v in values_per_run)
    out = []
    for i in range(n):
        vals = [v[i] for v in values_per_run if i < len(v) and v[i] == v[i]]  # NaN-safe
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "stage1_cache"))
    p.add_argument("--predictor", default=str(REPO_ROOT / "results" / "stage2_predictor" / "rank_predictor.pt"))
    p.add_argument("--prompts", default=str(REPO_ROOT / "data" / "prompts.json"))
    p.add_argument("--gates", default=str(REPO_ROOT / "results" / "stage3_gates.json"),
                   help="Stage 3 output file. The accepted safety_multiplier is "
                        "read from here unless --safety-multiplier is passed.")
    p.add_argument("--safety-multiplier", type=float, default=None)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=4)
    p.add_argument("--prompt-ids", nargs="*", default=None,
                   help="Subset of prompt ids to run (default: all)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "generation_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("=== backend ===")
    backend = describe_backend()
    print(json.dumps(backend, indent=2))

    # Resolve safety multiplier.
    multiplier = args.safety_multiplier
    if multiplier is None:
        try:
            gates = json.loads(Path(args.gates).read_text())
            if gates.get("accepted_multiplier") is not None:
                multiplier = float(gates["accepted_multiplier"])
                print(f"using safety_multiplier={multiplier} from {args.gates}")
            else:
                print("WARNING: Stage 3 did not accept a multiplier within "
                      "tolerance; defaulting to 1.0.")
                multiplier = 1.0
        except Exception as e:
            print(f"could not read gates ({e}); defaulting multiplier=1.0")
            multiplier = 1.0

    print("\n=== loading model ===", flush=True)
    loaded = load_bitnet(model_id=args.model_id, device=args.device)
    model = loaded.model
    tokenizer = loaded.tokenizer

    print("\n=== loading predictor ===")
    predictor = load_predictor(args.predictor)
    layer_means = build_layer_means(args.cache_dir, predictor.target_layers)

    def make_wrapper():
        cfg = DynamicRankConfig(
            rank_floor=1,
            rank_ceiling=predictor.basis_k_full,
            safety_multiplier=multiplier,
        )
        return DynamicRankBitNet(model, predictor, layer_means, cfg)

    prompts = json.loads(Path(args.prompts).read_text())["prompts"]
    if args.prompt_ids:
        sel = set(args.prompt_ids)
        prompts = [p for p in prompts if p["id"] in sel]
    print(f"\n=== running {len(prompts)} prompts, "
          f"{args.max_new_tokens} tokens each ===", flush=True)

    results = {
        "meta": {
            "model_id": args.model_id,
            "backend": backend,
            "n_prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "safety_multiplier": multiplier,
            "predictor_src_layer": predictor.src_layer,
            "predictor_target_layers": predictor.target_layers,
            "predictor_k_basis": predictor.k_basis,
        },
        "per_prompt": [],
    }

    base_runs = []
    dyn_runs = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt['id']}  ({prompt['kind']})")

        base_run = generate_and_time(
            model, tokenizer,
            prompt_text=prompt["text"],
            prompt_id=prompt["id"],
            max_new_tokens=args.max_new_tokens,
            device=loaded.device,
            wrapper_factory=None,
            mode_label="base",
            warmup_steps=args.warmup_steps,
        )
        (samples_dir / f"{prompt['id']}_base.txt").write_text(
            prompt["text"] + base_run.generated_text
        )

        dyn_run = generate_and_time(
            model, tokenizer,
            prompt_text=prompt["text"],
            prompt_id=prompt["id"],
            max_new_tokens=args.max_new_tokens,
            device=loaded.device,
            wrapper_factory=make_wrapper,
            mode_label="dynamic",
            warmup_steps=args.warmup_steps,
        )
        (samples_dir / f"{prompt['id']}_dynamic.txt").write_text(
            prompt["text"] + dyn_run.generated_text
        )

        base_runs.append(base_run)
        dyn_runs.append(dyn_run)
        base_tot = sum(base_run.per_step_seconds)
        dyn_tot = sum(dyn_run.per_step_seconds)
        ratio = base_tot / dyn_tot if dyn_tot > 0 else float("nan")
        print(f"  base  total={base_tot:.2f}s  ({args.max_new_tokens/base_tot:.1f} tok/s)")
        print(f"  dyn   total={dyn_tot:.2f}s  ({args.max_new_tokens/dyn_tot:.1f} tok/s)")
        print(f"  integral speedup = {ratio:.3f}x")

        results["per_prompt"].append({
            "id": prompt["id"],
            "kind": prompt["kind"],
            "prompt_tokens": base_run.prompt_token_count,
            "base_total_seconds": base_tot,
            "dynamic_total_seconds": dyn_tot,
            "integral_speedup": ratio,
            "dynamic_mean_rank_per_position": dyn_run.mean_rank_per_position,
        })

    base_curve = aggregate_curve(base_runs)
    dyn_curve = aggregate_curve(dyn_runs)
    sp = speedup_curve(base_curve, dyn_curve)

    mean_rank = _mean_across_runs([r.mean_rank_per_position for r in dyn_runs])

    agg = {
        "base": base_curve,
        "dynamic": dyn_curve,
        "speedup": sp,
        "mean_rank_per_position": mean_rank,
    }
    results["aggregate"] = agg

    def at(i, arr):
        return arr[i] if i < len(arr) else None

    snapshot_positions = [99, 499, 999, 1999]
    results["summary"] = {
        "speedup_at": {
            f"pos_{i+1}": at(i, sp["ratio"]) for i in snapshot_positions
        },
        "mean_rank_at": {
            f"pos_{i+1}": at(i, mean_rank) for i in snapshot_positions
        },
        "integral_speedup_across_prompts": (
            sum(p["integral_speedup"] for p in results["per_prompt"])
            / max(1, len(results["per_prompt"]))
        ),
    }
    print("\n=== summary ===")
    print(json.dumps(results["summary"], indent=2))

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    _maybe_plot(results, out_dir)
    print(f"\nwrote {out_dir/'summary.json'}, {out_dir/'acceleration_curve.png'}, "
          f"{out_dir/'rank_distribution.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
