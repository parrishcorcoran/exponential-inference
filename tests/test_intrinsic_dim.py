"""Unit tests for the intrinsic-dimension estimators."""

import math

import numpy as np
import torch

from src.measurement.intrinsic_dim import (
    compute_pr,
    compute_twonn,
    spectrum_rank_quantile,
)


def test_pr_flat_spectrum_equals_rank():
    # Identity covariance in d dims: PR should equal d.
    torch.manual_seed(0)
    d = 8
    x = torch.randn(5000, d)
    pr = compute_pr(x)
    assert abs(pr - d) < 0.3, pr


def test_pr_rank_one_collapses():
    # Data on a single line: PR should be close to 1.
    v = torch.randn(64)
    x = torch.linspace(-1, 1, 1000).unsqueeze(1) * v
    pr = compute_pr(x)
    assert pr < 1.1, pr


def test_twonn_recovers_low_dim_subspace():
    # Uniform in a 4d ball embedded in 32d: TwoNN ~= 4.
    rng = np.random.default_rng(0)
    n, d_true, d_amb = 3000, 4, 32
    base = rng.normal(size=(n, d_true))
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
    r = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / d_true)
    pts = base * r
    embed = rng.normal(size=(d_true, d_amb))
    x = torch.from_numpy(pts @ embed).float()
    td = compute_twonn(x, max_samples=2000)
    # TwoNN has known small-sample bias; accept a reasonable band.
    assert 3.0 <= td <= 5.5, td


def test_spectrum_rank_quantile_monotone():
    torch.manual_seed(0)
    x = torch.randn(500, 64)
    ranks = spectrum_rank_quantile(x, quantiles=(0.5, 0.9, 0.95, 0.99))
    assert ranks["r50"] <= ranks["r90"] <= ranks["r95"] <= ranks["r99"]
    assert ranks["r99"] <= 64


def test_compute_pr_handles_bfloat16():
    x = torch.randn(200, 16).to(torch.bfloat16)
    pr = compute_pr(x)
    assert math.isfinite(pr) and pr > 0
