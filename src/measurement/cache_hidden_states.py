"""Cache per-layer hidden states for a calibration corpus.

Writes one ``.pt`` file per decoder layer under ``out_dir``, each
containing a ``(N_tokens, hidden_size)`` float16 tensor of pooled
per-token hidden states, plus a ``meta.json`` describing what was
cached.

The default corpus is a small wikitext-103 slice; the caller can
provide any iterable of strings via ``iter_texts``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from tqdm import tqdm


@dataclass
class CacheMeta:
    model_id: str
    num_hidden_layers: int
    hidden_size: int
    total_tokens: int
    chunk_size: int
    dtype: str
    source: str
    per_layer_files: List[str] = field(default_factory=list)


def _default_wikipedia_texts(n_samples: int = 200) -> List[str]:
    """Load a small calibration corpus from wikitext-103 via datasets.

    Falls back to a hard-coded tiny mini-corpus if ``datasets`` or the
    network are unavailable. The hard-coded fallback is only enough for
    a sanity run — real measurements should use the full wikitext path.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train",
                          streaming=True)
        texts = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= n_samples:
                break
        if texts:
            return texts
    except Exception:
        pass

    return [
        "The participation ratio is a measure of how many principal "
        "components contribute meaningfully to the variance of a "
        "distribution. For a set of eigenvalues lambda_i, PR is "
        "defined as the square of their sum divided by the sum of "
        "their squares. This quantity equals the rank when the "
        "spectrum is flat and approaches one when the spectrum is "
        "dominated by a single mode.",
        "Spin glasses are disordered magnetic systems whose ground "
        "states exhibit a hierarchical ultrametric structure. Recent "
        "work has argued that sufficiently well-trained neural networks "
        "settle into analogous ground states, with the replica-symmetry "
        "breaking structure visible in the spectrum of their activations.",
    ] * max(1, n_samples // 2)


def _tokenize_and_chunk(
    tokenizer,
    texts: Iterable[str],
    chunk_size: int,
    total_tokens: int,
) -> torch.Tensor:
    """Flatten ``texts`` into a single token stream, cut into fixed
    chunks of ``chunk_size``, and return ``(n_chunks, chunk_size)``.

    Stops once ``total_tokens`` tokens have been emitted.
    """
    ids: List[int] = []
    eos = tokenizer.eos_token_id
    for t in texts:
        enc = tokenizer(t, add_special_tokens=False)["input_ids"]
        ids.extend(enc)
        if eos is not None:
            ids.append(eos)
        if len(ids) >= total_tokens:
            break
    if len(ids) < chunk_size:
        raise RuntimeError(
            f"calibration corpus too small: got {len(ids)} tokens, "
            f"need at least chunk_size={chunk_size}"
        )
    n_chunks = min(len(ids), total_tokens) // chunk_size
    ids = ids[: n_chunks * chunk_size]
    return torch.tensor(ids, dtype=torch.long).view(n_chunks, chunk_size)


def cache_calibration_hidden_states(
    model,
    tokenizer,
    out_dir: str | Path,
    total_tokens: int = 10_000,
    chunk_size: int = 512,
    batch_size: int = 2,
    iter_texts: Optional[Iterable[str]] = None,
    dtype: torch.dtype = torch.bfloat16,
    source_name: str = "wikitext-103",
) -> CacheMeta:
    """Runs ``model`` on the calibration token stream and saves per-layer
    hidden states to ``out_dir``.

    Returns the :class:`CacheMeta` written to ``out_dir/meta.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = iter_texts if iter_texts is not None else _default_wikipedia_texts()
    chunks = _tokenize_and_chunk(tokenizer, texts, chunk_size, total_tokens)

    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Preallocate one tensor per layer (inclusive of the embedding
    # output at index 0). Shape (N_tokens, D). Stored as float16 to
    # keep disk footprint tractable.
    n_tokens = chunks.numel()
    per_layer: List[torch.Tensor] = [
        torch.empty((n_tokens, hidden_size), dtype=dtype)
        for _ in range(n_layers + 1)
    ]
    write_offset = 0

    n_batches = math.ceil(chunks.shape[0] / batch_size)
    with torch.inference_mode():
        for b in tqdm(range(n_batches), desc="cache hidden states"):
            batch = chunks[b * batch_size : (b + 1) * batch_size].to(device)
            out = model(
                input_ids=batch,
                output_hidden_states=True,
                use_cache=False,
            )
            hs = out.hidden_states  # tuple of (B, T, D)
            flat_n = batch.numel()
            for li, h in enumerate(hs):
                per_layer[li][write_offset : write_offset + flat_n] = (
                    h.detach().reshape(-1, hidden_size).to(dtype).cpu()
                )
            write_offset += flat_n

    per_layer_files: List[str] = []
    for li, tensor in enumerate(per_layer):
        fname = f"layer_{li:02d}.pt"
        torch.save(tensor, out_dir / fname)
        per_layer_files.append(fname)

    meta = CacheMeta(
        model_id=getattr(model.config, "_name_or_path", "unknown"),
        num_hidden_layers=n_layers,
        hidden_size=hidden_size,
        total_tokens=n_tokens,
        chunk_size=chunk_size,
        dtype=str(dtype),
        source=source_name,
        per_layer_files=per_layer_files,
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
    # Also save the token ids for Stage 2 reuse.
    torch.save(chunks.view(-1), out_dir / "tokens.pt")
    return meta


def load_layer(out_dir: str | Path, layer_idx: int) -> torch.Tensor:
    out_dir = Path(out_dir)
    return torch.load(out_dir / f"layer_{layer_idx:02d}.pt", map_location="cpu")


def load_meta(out_dir: str | Path) -> CacheMeta:
    out_dir = Path(out_dir)
    data = json.loads((out_dir / "meta.json").read_text())
    return CacheMeta(**data)
