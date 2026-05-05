"""Run lm-eval-harness benchmarks for a base/converted model pair.

Compares Qwen3-0.6B base against the nGPT-shape variant on a standard
suite. Output: per-task JSON deltas suitable for the model card.

Usage:
    # benchmark base Qwen3-0.6B
    MODEL=Qwen/Qwen3-0.6B TAG=base python scripts/run_benchmarks.py

    # benchmark our converted model (after baking with bake_unit_norm_and_export.py)
    MODEL=model_package/Qwen3-0.6B-nGPT TAG=ngpt python scripts/run_benchmarks.py

    # tighter sweep (override task list)
    TASKS="hellaswag,piqa" MODEL=... TAG=... python scripts/run_benchmarks.py
"""
import json
import os
import sys
from pathlib import Path

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer


MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
TAG = os.environ.get("TAG", "base")
TASKS = os.environ.get(
    "TASKS",
    "hellaswag,arc_easy,arc_challenge,piqa,winogrande,wikitext"
).split(",")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "8")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("DTYPE", "bfloat16")
LIMIT = os.environ.get("LIMIT")  # optional: integer to limit per-task examples (smoke)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "results/benchmarks"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", "Qwen/Qwen3-0.6B")  # for nGPT loading


print(f"model: {MODEL}")
print(f"tag:   {TAG}")
print(f"tasks: {TASKS}")
print(f"batch: {BATCH_SIZE}  device: {DEVICE}  dtype: {DTYPE}")
if LIMIT:
    print(f"LIMIT: {LIMIT}  (per-task example cap)")

# Detect nGPT artifact (directory containing ngpt_state_dict.pt) vs standard HF model
ngpt_marker = Path(MODEL) / "ngpt_state_dict.pt" if not MODEL.startswith(("Qwen/", "meta-")) else None
if ngpt_marker and ngpt_marker.exists():
    print(f"  → detected nGPT artifact, loading via custom NGPTLinear path")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ngpt_load import load_ngpt_model
    import torch
    dtype_t = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
    model = load_ngpt_model(Path(MODEL), BASE_CHECKPOINT, DEVICE, dtype_t)
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT, trust_remote_code=True)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype=DTYPE,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        trust_remote_code=True,
    )
else:
    lm = HFLM(
        pretrained=MODEL,
        dtype=DTYPE,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        trust_remote_code=True,
    )

results = evaluator.simple_evaluate(
    model=lm,
    tasks=TASKS,
    num_fewshot=0,  # zero-shot for the commonsense suite
    limit=int(LIMIT) if LIMIT else None,
    bootstrap_iters=1000,
)

# Strip non-serializable / non-essential pieces
trim = {
    "model": MODEL,
    "tag": TAG,
    "tasks": TASKS,
    "results": results.get("results", {}),
    "n_samples": results.get("n-samples", {}),
    "task_versions": results.get("versions", {}),
    "config": {
        "model_args": results.get("config", {}).get("model_args", str(MODEL)),
        "batch_size": str(BATCH_SIZE),
        "device": DEVICE,
        "dtype": DTYPE,
        "limit": LIMIT,
    },
}

out_path = OUTPUT_DIR / f"qwen3_06b_{TAG}.json"
with open(out_path, "w") as f:
    json.dump(trim, f, indent=2, default=str)

print()
print("=" * 70)
print(f"results saved: {out_path}")
print("=" * 70)
for task, scores in trim["results"].items():
    main = next((k for k in ("acc,none", "acc_norm,none", "word_perplexity,none",
                             "byte_perplexity,none", "bits_per_byte,none")
                 if k in scores), None)
    if main:
        v = scores[main]
        stderr_key = main.replace(",none", "_stderr,none")
        e = scores.get(stderr_key, 0.0)
        e_str = f"± {e:.4f}" if isinstance(e, (int, float)) else f"± {e}"
        print(f"  {task:20s}  {main:24s}  {v:.4f}  {e_str}")
    else:
        print(f"  {task:20s}  (no standard metric)  {scores}")
