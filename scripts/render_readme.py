"""Render README.md from results/*.json once the pipeline has run.

The template lives in ``README.template.md``. It contains ``{{ key }}``
placeholders; this script fills them in from:

  - results/stage0_verification.json  (backend, architecture)
  - results/stage1_manifold.json      (per-layer PR/TwoNN)
  - results/stage2_predictor/report.json (chosen source layer, R^2)
  - results/stage3_gates.json         (quality-gate accepted multiplier)
  - results/summary.json              (acceleration curve snapshots)

Run after Stage 4 completes on the real machine. Keeps README and data
in lockstep.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(v, digits: int = 3, unit: str = "") -> str:
    if v is None:
        return "_(pending)_"
    if isinstance(v, float):
        if v != v:  # NaN
            return "_(pending)_"
        return f"{v:.{digits}f}{unit}"
    return f"{v}{unit}"


def _fmt_layer_table(per_layer) -> str:
    if not per_layer:
        return "_(pending: run scripts/stage1_measure.py)_"
    header = ("| layer | PR | TwoNN | r50 | r90 | r95 | r99 |\n"
              "|------:|----:|------:|----:|----:|----:|----:|\n")
    rows = []
    for row in per_layer:
        rc = row["rank_coverage"]
        rows.append(
            f"| {row['layer_index']} | {row['pr']:.2f} | {row['twonn']:.2f} | "
            f"{rc['r50']} | {rc['r90']} | {rc['r95']} | {rc['r99']} |"
        )
    return header + "\n".join(rows)


def _fmt_r2_line(report) -> str:
    if not report:
        return "_(pending)_"
    src = report["chosen_src_layer"]
    kind = report["chosen_model_kind"]
    r2 = report["chosen_r2"]
    r2_str = ", ".join(f"L{t}: {r:.2f}" for t, r in r2.items())
    return f"source layer **L{src}**, {kind} regressor, R² per target: {r2_str}"


def _fmt_speedup_snapshots(summary) -> str:
    if not summary or "summary" not in summary:
        return "_(pending)_"
    s = summary["summary"]["speedup_at"]
    integral = summary["summary"].get("integral_speedup_across_prompts")
    out = []
    for k, v in s.items():
        pos = k.replace("pos_", "position ")
        out.append(f"- {pos}: {_fmt(v, 3, 'x')}")
    if integral is not None:
        out.append(f"- integral speedup across prompts: {_fmt(integral, 3, 'x')}")
    return "\n".join(out)


def _fmt_rank_snapshots(summary) -> str:
    if not summary or "summary" not in summary:
        return "_(pending)_"
    mr = summary["summary"]["mean_rank_at"]
    return "\n".join(
        f"- {k.replace('pos_', 'position ')}: {_fmt(v, 1)}"
        for k, v in mr.items()
    )


class Jinja2Lite(Template):
    delimiter = "@@"
    idpattern = r"[_a-zA-Z][_a-zA-Z0-9]*"


def render(out_readme: Path, template_path: Path) -> None:
    s0 = _safe_load(REPO_ROOT / "results" / "stage0_verification.json")
    s1 = _safe_load(REPO_ROOT / "results" / "stage1_manifold.json")
    s2 = _safe_load(REPO_ROOT / "results" / "stage2_predictor" / "report.json")
    s3 = _safe_load(REPO_ROOT / "results" / "stage3_gates.json")
    s4 = _safe_load(REPO_ROOT / "results" / "summary.json")

    model_id = (
        (s0 or {}).get("model_id") or
        (s1 or {}).get("model_id") or
        "microsoft/BitNet-b1.58-2B-4T"
    )
    backend = (s0 or {}).get("backend", {})
    arch = (s0 or {}).get("architecture", {})

    mapping = {
        "model_id": model_id,
        "n_layers": _fmt(arch.get("num_hidden_layers") or (s1 or {}).get("num_hidden_layers")),
        "hidden_size": _fmt(arch.get("hidden_size") or (s1 or {}).get("hidden_size")),
        "backend_name": backend.get("device_name") or "_(pending)_",
        "vram_gb": _fmt(backend.get("total_memory_gb")),
        "layer_table": _fmt_layer_table((s1 or {}).get("per_layer")),
        "predictor_line": _fmt_r2_line(s2),
        "correctness_passed": _fmt(
            (s3 or {}).get("correctness", {}).get("passed")
        ),
        "correctness_max_diff": _fmt(
            (s3 or {}).get("correctness", {}).get("max_abs_diff"), 6
        ),
        "base_accuracy": _fmt((s3 or {}).get("base_accuracy"), 4),
        "accepted_multiplier": _fmt((s3 or {}).get("accepted_multiplier"), 2),
        "speedup_snapshots": _fmt_speedup_snapshots(s4),
        "rank_snapshots": _fmt_rank_snapshots(s4),
    }

    tpl = Jinja2Lite(template_path.read_text())
    out_readme.write_text(tpl.safe_substitute(mapping))
    print(f"wrote {out_readme}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--template", default=str(REPO_ROOT / "README.template.md"))
    p.add_argument("--out", default=str(REPO_ROOT / "README.md"))
    args = p.parse_args()
    render(Path(args.out), Path(args.template))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
