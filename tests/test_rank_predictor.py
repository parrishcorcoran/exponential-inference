"""Tests for rank-predictor primitives.

These run without loading any transformer — they use synthetic hidden
states saved to a temp directory in the same layout as Stage 1 cache.
"""

import json
from pathlib import Path

import torch

from src.routing.rank_predictor import (
    effective_rank_per_token,
    fit_rank_predictor,
    load_predictor,
    save_predictor,
    svd_basis,
)


def test_svd_basis_orthonormal():
    torch.manual_seed(0)
    h = torch.randn(500, 16)
    b = svd_basis(h, k=4)
    gram = b.T @ b
    assert torch.allclose(gram, torch.eye(4), atol=1e-4)


def test_effective_rank_per_token_monotone_in_threshold():
    torch.manual_seed(0)
    h = torch.randn(200, 32)
    b = svd_basis(h, k=32)
    r50 = effective_rank_per_token(h, b, energy_threshold=0.5).float().mean()
    r95 = effective_rank_per_token(h, b, energy_threshold=0.95).float().mean()
    assert r50.item() <= r95.item()


def _make_fake_cache(tmp_path: Path, n_layers: int = 6, d: int = 24, n: int = 2000):
    """Write Stage 1-shaped cache with synthetic structured activations.

    Tokens split 50/50 between "easy" (rank-1) and "hard" (near
    isotropic). The early-layer signal is the same as the final-layer
    label, so a linear predictor should learn the easy/hard split and
    thereby the effective-rank target with high R^2.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    easy_mask = torch.rand(n) < 0.5  # ~50% easy
    for li in range(n_layers + 1):
        h = torch.zeros(n, d)
        # Easy tokens: big deterministic spike along coord 0,
        # negligible energy elsewhere.
        h[easy_mask, 0] = 20.0
        # Hard tokens: isotropic noise across all dims with
        # moderate std.
        hard_idx = (~easy_mask).nonzero(as_tuple=True)[0]
        h[hard_idx] = torch.randn(len(hard_idx), d) * 2.0
        # Small nuisance noise everywhere.
        h = h + torch.randn(n, d) * 0.05
        torch.save(h.half(), tmp_path / f"layer_{li:02d}.pt")
    meta = {
        "model_id": "fake",
        "num_hidden_layers": n_layers,
        "hidden_size": d,
        "total_tokens": n,
        "chunk_size": 0,
        "dtype": "torch.float16",
        "source": "synthetic",
        "per_layer_files": [f"layer_{i:02d}.pt" for i in range(n_layers + 1)],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_fit_rank_predictor_learns_something(tmp_path):
    cache = _make_fake_cache(tmp_path / "c")
    fitted, report = fit_rank_predictor(
        cache_dir=cache,
        src_layer=1,
        target_layers=[3, 5],
        k_basis=4,
        basis_k_full=24,
        model_kind="linear",
    )
    assert set(report.r2) == {3, 5}
    # On this synthetic data the signal is strong; both layers should
    # beat the "predict the mean" baseline (R^2 > 0).
    for tl, r2 in report.r2.items():
        assert r2 > 0.0, (tl, r2)


def test_predictor_roundtrip(tmp_path):
    cache = _make_fake_cache(tmp_path / "c")
    fitted, _ = fit_rank_predictor(
        cache_dir=cache,
        src_layer=1,
        target_layers=[3, 5],
        k_basis=4,
        basis_k_full=24,
        model_kind="mlp",
    )
    out_dir = tmp_path / "pred"
    path = save_predictor(fitted, out_dir)
    reloaded = load_predictor(path)
    # Predict on a fresh batch; shapes and values should line up.
    x = torch.randn(30, 24)
    y1 = fitted.predict(x)
    y2 = reloaded.predict(x)
    assert y1.shape == y2.shape == (30, 2)
    assert y1.dtype == torch.long
    # Bit-identical round trip: same tensors, same normalisation constants,
    # so the two models must agree exactly.
    assert torch.equal(y1, y2)
