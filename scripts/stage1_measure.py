"""Stage 1 driver: cache hidden states and measure per-layer PR / TwoNN.

Runs end-to-end:
  1. Load BitNet b1.58 2B.
  2. Tokenize a ~10K-token Wikipedia calibration slice.
  3. Cache per-layer hidden states to results/stage1_cache/.
  4. Compute PR and TwoNN for each of the 30 decoder-layer outputs.
  5. Save results/stage1_manifold.json and results/stage1_manifold.png.

Compare the per-layer PR/TwoNN shape to the expected 6 -> 36 -> 16
pattern (layers 5, 15, 29) on easy tokens. The driver just reports the
full curve; interpretation happens at the checkpoint.
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
from src.measurement.cache_hidden_states import (  # noqa: E402
    cache_calibration_hidden_states,
    load_layer,
    load_meta,
)
from src.measurement.intrinsic_dim import (  # noqa: E402
    compute_pr,
    compute_twonn,
    spectrum_rank_quantile,
)


def _plot(meta, per_layer: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return

    layers = [d["layer_index"] for d in per_layer]
    pr = [d["pr"] for d in per_layer]
    td = [d["twonn"] for d in per_layer]
    r95 = [d["rank_coverage"]["r95"] for d in per_layer]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    axes[0].plot(layers, pr, marker="o", label="PR")
    axes[0].plot(layers, td, marker="s", label="TwoNN")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("intrinsic dimension")
    axes[0].set_title(f"{meta.model_id}\nintrinsic dim vs layer")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(layers, r95, marker="o", color="C2")
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("rank for 95% energy")
    axes[1].set_title("spectrum rank at 95% coverage")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default=None, help="HF cache dir")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "stage1_cache"))
    p.add_argument("--total-tokens", type=int, default=10_000)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--twonn-samples", type=int, default=4000)
    p.add_argument("--skip-cache", action="store_true",
                   help="Reuse an existing cache and only recompute metrics")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if not args.skip_cache:
        print("=== backend ===")
        print(json.dumps(describe_backend(), indent=2))

        print("\n=== loading model ===", flush=True)
        loaded = load_bitnet(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.cache_dir,
        )
        print(f"loaded {args.model_id} on {loaded.device}")

        print("\n=== caching hidden states ===", flush=True)
        meta = cache_calibration_hidden_states(
            loaded.model,
            loaded.tokenizer,
            out_dir=out_dir,
            total_tokens=args.total_tokens,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        # Free GPU before metric compute.
        del loaded
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        meta = load_meta(out_dir)

    print("\n=== per-layer measurements ===", flush=True)
    per_layer = []
    for li in range(meta.num_hidden_layers + 1):
        h = load_layer(out_dir, li)
        pr = compute_pr(h)
        td = compute_twonn(h, max_samples=args.twonn_samples)
        ranks = spectrum_rank_quantile(h)
        row = {
            "layer_index": li,
            "pr": pr,
            "twonn": td,
            "rank_coverage": ranks,
        }
        per_layer.append(row)
        print(f"  L{li:02d}  PR={pr:7.2f}   TwoNN={td:6.2f}   "
              f"r50={ranks['r50']:4d}  r90={ranks['r90']:4d}  "
              f"r95={ranks['r95']:4d}  r99={ranks['r99']:4d}")

    summary_path = REPO_ROOT / "results" / "stage1_manifold.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "model_id": meta.model_id,
        "total_tokens": meta.total_tokens,
        "chunk_size": meta.chunk_size,
        "num_hidden_layers": meta.num_hidden_layers,
        "hidden_size": meta.hidden_size,
        "per_layer": per_layer,
    }, indent=2))
    print(f"\nwrote {summary_path}")

    _plot(meta, per_layer, REPO_ROOT / "results" / "stage1_manifold.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
