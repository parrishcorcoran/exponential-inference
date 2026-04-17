"""Stage 2 driver: fit the per-token rank predictor.

Walks the prediction-source layer forward until the minimum R^2 across
target layers exceeds ``--r2-floor`` (default 0.6). Tries both a linear
regressor and a small MLP at each candidate source layer, keeps the
better one.

Inputs:  results/stage1_cache/ (from Stage 1)
Outputs: results/stage2_predictor/rank_predictor.pt
         results/stage2_predictor/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.measurement.cache_hidden_states import load_meta  # noqa: E402
from src.routing.rank_predictor import (  # noqa: E402
    fit_rank_predictor,
    save_predictor,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "stage1_cache"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "stage2_predictor"))
    p.add_argument("--src-candidates", type=int, nargs="+",
                   default=[5, 10, 15, 20],
                   help="Candidate prediction-source layers to try in order.")
    p.add_argument("--target-layers", type=int, nargs="+",
                   default=[15, 20, 25, 29])
    p.add_argument("--k-basis", type=int, default=7,
                   help="Source-layer manifold basis width. Prior BitNet "
                        "work suggests ~7 per-sequence, 14 cross-sequence.")
    p.add_argument("--basis-k-full", type=int, default=256)
    p.add_argument("--energy-threshold", type=float, default=0.95)
    p.add_argument("--r2-floor", type=float, default=0.6)
    p.add_argument("--force-source", type=int, default=None,
                   help="Skip walk-forward and fix source layer to this.")
    args = p.parse_args()

    meta = load_meta(args.cache_dir)
    print(f"cache: {args.cache_dir}  model={meta.model_id}  "
          f"layers={meta.num_hidden_layers}  d={meta.hidden_size}")

    target_layers = [t for t in args.target_layers
                     if t <= meta.num_hidden_layers]

    attempts = []
    chosen = None

    candidates = ([args.force_source] if args.force_source is not None
                  else list(args.src_candidates))

    for src in candidates:
        if src > meta.num_hidden_layers:
            continue
        # Ensure targets are strictly after source.
        tl = [t for t in target_layers if t > src]
        if len(tl) == 0:
            continue

        best_for_src = None
        for kind in ("linear", "mlp"):
            fitted, report = fit_rank_predictor(
                cache_dir=args.cache_dir,
                src_layer=src,
                target_layers=tl,
                k_basis=args.k_basis,
                basis_k_full=args.basis_k_full,
                energy_threshold=args.energy_threshold,
                model_kind=kind,
            )
            r2_min = min(report.r2.values()) if report.r2 else float("-inf")
            r2_mean = (
                sum(report.r2.values()) / len(report.r2) if report.r2 else float("-inf")
            )
            entry = {
                "src_layer": src,
                "model_kind": kind,
                "r2": report.r2,
                "mae": report.mae,
                "r2_min": r2_min,
                "r2_mean": r2_mean,
            }
            attempts.append(entry)
            print(f"  src=L{src:02d} {kind:6s}  "
                  f"R2 min={r2_min:+.3f}  mean={r2_mean:+.3f}  "
                  f"R2={ {t: round(v,2) for t, v in report.r2.items()} }")
            if best_for_src is None or r2_min > best_for_src["r2_min"]:
                best_for_src = entry
                best_for_src["fitted"] = fitted

        if best_for_src is not None and best_for_src["r2_min"] >= args.r2_floor:
            chosen = best_for_src
            print(f"  -> accepted src=L{src:02d} ({best_for_src['model_kind']})")
            break

    if chosen is None and attempts:
        # None met the floor; pick the best attempted anyway and flag.
        chosen = max(attempts, key=lambda e: e["r2_min"])
        print(f"\nWARNING: no candidate met R^2 floor {args.r2_floor:.2f}. "
              f"Falling back to best available: src=L{chosen['src_layer']:02d} "
              f"({chosen['model_kind']}) with R2_min={chosen['r2_min']:+.3f}")
        # Re-fit so we have the model object (not kept on non-accepted entries).
        if "fitted" not in chosen:
            tl = [t for t in target_layers if t > chosen["src_layer"]]
            fitted, _ = fit_rank_predictor(
                cache_dir=args.cache_dir,
                src_layer=chosen["src_layer"],
                target_layers=tl,
                k_basis=args.k_basis,
                basis_k_full=args.basis_k_full,
                energy_threshold=args.energy_threshold,
                model_kind=chosen["model_kind"],
            )
            chosen["fitted"] = fitted

    if chosen is None:
        print("ERROR: no predictor could be fit (likely empty cache).")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = save_predictor(chosen["fitted"], out_dir)

    report_path = out_dir / "report.json"
    # Strip the non-serializable fitted object before writing.
    serialisable_attempts = [
        {k: v for k, v in a.items() if k != "fitted"} for a in attempts
    ]
    report_path.write_text(json.dumps({
        "model_id": meta.model_id,
        "num_hidden_layers": meta.num_hidden_layers,
        "hidden_size": meta.hidden_size,
        "k_basis": args.k_basis,
        "energy_threshold": args.energy_threshold,
        "r2_floor": args.r2_floor,
        "chosen_src_layer": chosen["src_layer"],
        "chosen_model_kind": chosen["model_kind"],
        "chosen_r2": chosen["r2"],
        "chosen_mae": chosen["mae"],
        "target_layers": chosen["fitted"].target_layers,
        "attempts": serialisable_attempts,
        "predictor_path": str(pred_path.relative_to(REPO_ROOT)),
    }, indent=2))
    print(f"\nwrote {pred_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
