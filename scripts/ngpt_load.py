"""Load an A0 nGPT-form artifact.

Builds the model: standard Qwen3 architecture with each targeted Linear replaced
by NGPTLinear (W̃ + α split), then loads the saved state_dict from A0.

Usage:
    NGPT_DIR=model_package/Qwen3-0.6B-nGPT-form python scripts/ngpt_load.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-import NGPTLinear from the conversion script to keep one source of truth
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngpt_lossless_convert import NGPTLinear, replace_with_ngpt, TARGETS  # noqa: E402


CHECKPOINT = os.environ.get("CHECKPOINT", "Qwen/Qwen3-0.6B")
NGPT_DIR = Path(os.environ.get("NGPT_DIR", "model_package/Qwen3-0.6B-nGPT-form"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def load_ngpt_model(ngpt_dir: Path = NGPT_DIR, base_checkpoint: str = CHECKPOINT,
                    device: str = DEVICE, dtype=DTYPE):
    """Load an nGPT-form model from an A0 artifact directory.

    Returns a model with NGPTLinears in place of the targeted nn.Linears,
    parameters loaded from the saved state_dict.
    """
    sd_path = ngpt_dir / "ngpt_state_dict.pt"
    if not sd_path.exists():
        raise FileNotFoundError(f"No state_dict found at {sd_path}")

    print(f"loading base architecture: {base_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)

    n = replace_with_ngpt(model)
    print(f"  patched {n} Linear → NGPTLinear")

    print(f"loading state_dict: {sd_path}")
    sd = torch.load(sd_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  warn: {len(missing)} missing keys (showing 5): {missing[:5]}")
    if unexpected:
        print(f"  warn: {len(unexpected)} unexpected keys (showing 5): {unexpected[:5]}")

    return model


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Quick sanity check: load and verify it produces sensible logits
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
        model = load_ngpt_model()
        model.eval()
        ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=15, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id or 0)
        print(f"  → {tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True)!r}")
        # Verify rows of W_tilde are unit-norm
        sample_layer = "model.layers.5.self_attn.q_proj"
        mod = model.get_submodule(sample_layer)
        if isinstance(mod, NGPTLinear):
            row_norms = mod.weight.float().norm(dim=-1)
            print(f"  {sample_layer} W̃ row norms: "
                  f"mean={row_norms.mean():.6f}  min={row_norms.min():.6f}  "
                  f"max={row_norms.max():.6f}  (should all be ~1.0)")
            print(f"  {sample_layer} α: "
                  f"mean={mod.alpha.float().mean():.4f}  min={mod.alpha.float().min():.4f}  "
                  f"max={mod.alpha.float().max():.4f}")
    else:
        model = load_ngpt_model()
        print(f"  loaded. params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
