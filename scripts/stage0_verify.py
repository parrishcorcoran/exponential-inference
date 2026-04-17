"""Stage 0 verification: load BitNet b1.58 2B and generate a test sequence.

Run this on the Strix Halo box (or wherever the weights live). It:
  1. Prints the compute backend (ROCm/CUDA/CPU, VRAM, torch version).
  2. Loads the bf16 checkpoint from HuggingFace.
  3. Generates a short deterministic completion so we can eyeball coherence.
  4. Reports the architecture summary (n_layers, hidden_size) that every
     downstream stage depends on.

Exits non-zero if any step fails so CI or shell pipelines can detect it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common.model_loader import (  # noqa: E402
    DEFAULT_MODEL_ID,
    describe_backend,
    load_bitnet,
)


TEST_PROMPT = "The discovery that inference accelerates with context is"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    print("=== backend ===")
    backend = describe_backend()
    print(json.dumps(backend, indent=2))

    print("\n=== loading model ===", flush=True)
    t0 = time.perf_counter()
    loaded = load_bitnet(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    load_s = time.perf_counter() - t0
    print(f"loaded {args.model_id} on {loaded.device} in {load_s:.1f}s")

    config = loaded.model.config
    arch = {
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }
    print("\n=== architecture ===")
    print(json.dumps(arch, indent=2))

    print("\n=== generation ===", flush=True)
    inputs = loaded.tokenizer(TEST_PROMPT, return_tensors="pt").to(loaded.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = loaded.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_s = time.perf_counter() - t0
    text = loaded.tokenizer.decode(output[0], skip_special_tokens=True)
    n_new = output.shape[1] - inputs["input_ids"].shape[1]
    print(f"generated {n_new} tokens in {gen_s:.2f}s "
          f"({n_new/gen_s:.1f} tok/s)")
    print("---")
    print(text)
    print("---")

    # Writes a small artifact so we have a record of the backend used.
    out_path = REPO_ROOT / "results" / "stage0_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "backend": backend,
        "architecture": arch,
        "model_id": args.model_id,
        "load_seconds": load_s,
        "gen_tokens": n_new,
        "gen_seconds": gen_s,
        "sample_output": text,
    }, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
