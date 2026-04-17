"""Loader for the BitNet b1.58 2B bf16 checkpoint.

We use the HuggingFace transformers path (microsoft/BitNet-b1.58-2B-4T)
for measurement and for modifying the forward pass. The ternarized
bitnet.cpp kernels are not needed at this stage — they produce identical
logits (within quantization tolerance) and are swapped in only for the
final wall-clock reproduction once the method is validated in PyTorch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    model_id: str


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available accelerator.

    Preference order (unless ``prefer`` forces one):
        CUDA/ROCm > CPU.
    PyTorch ROCm builds expose AMD GPUs through the same ``cuda`` API, so
    ``torch.cuda.is_available()`` returning True may mean either NVIDIA or
    AMD. ``torch.version.hip`` is set on ROCm wheels.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_bitnet(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None,
) -> LoadedModel:
    """Load tokenizer + model onto the chosen device."""
    target_device = pick_device(device)

    cache_dir = cache_dir or os.environ.get("HF_HOME") or os.environ.get("EI_CACHE_DIR")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
    ).to(target_device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=target_device,
        dtype=dtype,
        model_id=model_id,
    )


def describe_backend() -> dict:
    """Return a short description of the compute backend for logging."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass
    return info
