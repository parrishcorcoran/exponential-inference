"""Per-token rank prediction from early-layer manifold position.

Pipeline:
  1. For the prediction source layer L_src, SVD the cached hidden states
     and keep the top ``k_basis`` right-singular vectors as a manifold
     basis ``V_src``. A token's "manifold position" is ``V_src^T h``.
  2. For each downstream layer L, SVD its cached hidden states to obtain
     basis ``V_L``. For every cached token at layer L, compute its
     effective rank (the smallest ``r`` such that the top-``r`` basis
     components cover a fixed energy fraction ``energy_threshold``).
  3. Fit either a ridge-regression linear model or a small MLP mapping
     manifold_position@L_src -> effective_rank@L for each downstream L.
  4. Score by held-out R^2.

This module is deliberately small and leak-free: every fit uses only
train-fold data for SVD bases, targets, and regressor parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import json
import numpy as np
import torch


def svd_basis(hidden_states: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` right singular vectors of the centred hidden-state matrix.

    Returns a ``(D, k)`` basis whose columns are orthonormal. Centring
    uses the empirical mean over the sample axis.
    """
    x = hidden_states.to(torch.float32)
    if not x.isfinite().all():
        x = torch.where(x.isfinite(), x, torch.zeros_like(x))
    x = x - x.mean(dim=0, keepdim=True)
    # Use full_matrices=False; x is tall (N >> D) for our sizes.
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[:k].T.contiguous()  # (D, k)


def effective_rank_per_token(
    hidden_states: torch.Tensor,
    basis: torch.Tensor,
    energy_threshold: float = 0.95,
) -> torch.Tensor:
    """For each row of ``hidden_states``, the rank needed to cover
    ``energy_threshold`` of its energy when projected onto ``basis``.

    ``basis`` has shape ``(D, K)``. Returns a ``(N,)`` integer tensor
    with values in ``[1, K]``. Mean and std are used upstream as the
    regression target.
    """
    x = hidden_states.to(torch.float32)
    if not x.isfinite().all():
        x = torch.where(x.isfinite(), x, torch.zeros_like(x))
    coeffs = x @ basis  # (N, K)
    energy = coeffs.pow(2)
    total = energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cum = torch.cumsum(energy, dim=1) / total
    mask = cum >= energy_threshold
    # Smallest index where cumulative energy reaches threshold.
    # If row never reaches threshold, use K.
    ranks = torch.where(
        mask.any(dim=1),
        mask.float().argmax(dim=1) + 1,
        torch.full((x.shape[0],), coeffs.shape[1]),
    )
    return ranks.to(torch.long)


@dataclass
class PredictorReport:
    src_layer: int
    k_basis: int
    energy_threshold: float
    target_layers: List[int]
    model_kind: str  # "linear" or "mlp"
    r2: Dict[int, float] = field(default_factory=dict)
    mae: Dict[int, float] = field(default_factory=dict)
    target_mean: Dict[int, float] = field(default_factory=dict)
    target_std: Dict[int, float] = field(default_factory=dict)


class LinearRankPredictor(torch.nn.Module):
    """Single linear layer: k_basis -> len(target_layers)."""

    def __init__(self, k_basis: int, n_targets: int):
        super().__init__()
        self.fc = torch.nn.Linear(k_basis, n_targets)

    def forward(self, x):  # noqa: D401
        return self.fc(x)


class MLPRankPredictor(torch.nn.Module):
    """Two-hidden-layer MLP with GELU, deliberately small."""

    def __init__(self, k_basis: int, n_targets: int, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(k_basis, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_targets),
        )

    def forward(self, x):  # noqa: D401
        return self.net(x)


def _fit_torch_regressor(
    model: torch.nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_va = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_tr)
        loss = torch.nn.functional.mse_loss(pred, y_tr)
        loss.backward()
        opt.step()
        with torch.inference_mode():
            model.eval()
            va = torch.nn.functional.mse_loss(model(x_va), y_va).item()
        if va < best_va:
            best_va = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


def _r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    y_mean = y.mean(dim=0, keepdim=True)
    ss_res = (y - pred).pow(2).sum(dim=0)
    ss_tot = (y - y_mean).pow(2).sum(dim=0).clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).mean().item())


def _mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred - y).abs().mean().item())


@dataclass
class FittedPredictor:
    src_layer: int
    k_basis: int
    energy_threshold: float
    target_layers: List[int]
    basis_src: torch.Tensor  # (D, k_basis)
    hidden_mean_src: torch.Tensor  # (D,)  training mean of source-layer states
    feature_mean: torch.Tensor  # (k_basis,)
    feature_std: torch.Tensor  # (k_basis,)
    target_mean: torch.Tensor  # (n_targets,)
    target_std: torch.Tensor  # (n_targets,)
    model: torch.nn.Module
    model_kind: str
    # Per-target precomputed bases from the calibration SVD; used by the
    # inference-time router so the downstream-layer rank axis is the
    # same one used when fitting.
    bases_target: List[torch.Tensor] = field(default_factory=list)  # each (D, K_full)
    basis_k_full: int = 0

    def predict(self, hidden_src: torch.Tensor) -> torch.Tensor:
        """Return predicted integer rank per target layer.

        Input: hidden_src with shape ``(B*T, D)`` at the source layer.
        Output: ``(B*T, n_targets)`` predicted rank values, clamped to
        ``[1, basis_k_full]`` and rounded to nearest integer.
        Centring uses the calibration-time mean so the feature axis at
        inference matches what the regressor was trained on.
        """
        device = self.basis_src.device
        dtype = self.basis_src.dtype
        x = hidden_src.to(device=device, dtype=dtype)
        x = x - self.hidden_mean_src
        feats = x @ self.basis_src
        feats = (feats - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        self.model.eval()
        with torch.inference_mode():
            y = self.model(feats)
        y = y * self.target_std + self.target_mean
        y = y.clamp(min=1.0, max=float(self.basis_k_full))
        return y.round().to(torch.long)


def fit_rank_predictor(
    cache_dir: Path | str,
    src_layer: int,
    target_layers: Sequence[int],
    k_basis: int = 7,
    basis_k_full: int = 256,
    energy_threshold: float = 0.95,
    model_kind: str = "linear",
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[FittedPredictor, PredictorReport]:
    """Train a predictor from the Stage 1 hidden-state cache.

    - ``basis_k_full`` is the width of the downstream-layer bases used
      to compute per-token effective rank targets; typically the full
      hidden_size so the rank axis is well defined up to that cap.
    - Features are centred, projected onto the top-``k_basis`` source-
      layer basis, then z-scored; targets are z-scored.
    - A fixed ``val_fraction`` tail of the cached token stream is held
      out for R^2 / MAE evaluation.
    """
    from src.measurement.cache_hidden_states import load_layer

    h_src = load_layer(cache_dir, src_layer).to(torch.float32)
    n = h_src.shape[0]
    n_val = max(1, int(n * val_fraction))
    tr_slice = slice(0, n - n_val)
    va_slice = slice(n - n_val, n)

    basis_src = svd_basis(h_src[tr_slice], k_basis)
    hidden_mean_src = h_src[tr_slice].mean(dim=0)
    centred_src = h_src - hidden_mean_src
    feats_all = centred_src @ basis_src
    feat_mean = feats_all[tr_slice].mean(dim=0)
    feat_std = feats_all[tr_slice].std(dim=0).clamp_min(1e-6)
    feats_all = (feats_all - feat_mean) / feat_std

    targets_all: List[torch.Tensor] = []
    bases_target: List[torch.Tensor] = []
    for tl in target_layers:
        h_tl = load_layer(cache_dir, tl).to(torch.float32)
        b_tl = svd_basis(h_tl[tr_slice], min(basis_k_full, h_tl.shape[1]))
        bases_target.append(b_tl)
        ranks = effective_rank_per_token(
            h_tl, b_tl, energy_threshold=energy_threshold
        )
        targets_all.append(ranks.to(torch.float32))
    y_all = torch.stack(targets_all, dim=1)

    y_mean = y_all[tr_slice].mean(dim=0)
    y_std = y_all[tr_slice].std(dim=0).clamp_min(1e-6)
    y_norm = (y_all - y_mean) / y_std

    x_tr, x_va = feats_all[tr_slice], feats_all[va_slice]
    y_tr, y_va = y_norm[tr_slice], y_norm[va_slice]

    if model_kind == "linear":
        model = LinearRankPredictor(k_basis, len(target_layers))
    elif model_kind == "mlp":
        model = MLPRankPredictor(k_basis, len(target_layers))
    else:
        raise ValueError(f"unknown model_kind={model_kind}")

    _fit_torch_regressor(model, x_tr, y_tr, x_va, y_va, seed=seed)

    with torch.inference_mode():
        pred_va = model(x_va)
    pred_va_unnorm = pred_va * y_std + y_mean
    y_va_unnorm = y_va * y_std + y_mean

    r2 = {}
    mae = {}
    for i, tl in enumerate(target_layers):
        r2[tl] = _r2(pred_va_unnorm[:, i : i + 1], y_va_unnorm[:, i : i + 1])
        mae[tl] = _mae(pred_va_unnorm[:, i : i + 1], y_va_unnorm[:, i : i + 1])

    report = PredictorReport(
        src_layer=src_layer,
        k_basis=k_basis,
        energy_threshold=energy_threshold,
        target_layers=list(target_layers),
        model_kind=model_kind,
        r2=r2,
        mae=mae,
        target_mean={tl: float(y_mean[i]) for i, tl in enumerate(target_layers)},
        target_std={tl: float(y_std[i]) for i, tl in enumerate(target_layers)},
    )

    fitted = FittedPredictor(
        src_layer=src_layer,
        k_basis=k_basis,
        energy_threshold=energy_threshold,
        target_layers=list(target_layers),
        basis_src=basis_src,
        hidden_mean_src=hidden_mean_src,
        feature_mean=feat_mean,
        feature_std=feat_std,
        target_mean=y_mean,
        target_std=y_std,
        model=model,
        model_kind=model_kind,
        bases_target=bases_target,
        basis_k_full=bases_target[0].shape[1] if bases_target else 0,
    )
    return fitted, report


def save_predictor(fitted: FittedPredictor, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "src_layer": fitted.src_layer,
        "k_basis": fitted.k_basis,
        "energy_threshold": fitted.energy_threshold,
        "target_layers": fitted.target_layers,
        "basis_src": fitted.basis_src,
        "hidden_mean_src": fitted.hidden_mean_src,
        "feature_mean": fitted.feature_mean,
        "feature_std": fitted.feature_std,
        "target_mean": fitted.target_mean,
        "target_std": fitted.target_std,
        "state_dict": fitted.model.state_dict(),
        "model_kind": fitted.model_kind,
        "bases_target": fitted.bases_target,
        "basis_k_full": fitted.basis_k_full,
    }
    path = out_dir / "rank_predictor.pt"
    torch.save(payload, path)
    return path


def load_predictor(path: str | Path) -> FittedPredictor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_targets = len(payload["target_layers"])
    if payload["model_kind"] == "linear":
        model = LinearRankPredictor(payload["k_basis"], n_targets)
    else:
        model = MLPRankPredictor(payload["k_basis"], n_targets)
    model.load_state_dict(payload["state_dict"])
    return FittedPredictor(
        src_layer=payload["src_layer"],
        k_basis=payload["k_basis"],
        energy_threshold=payload["energy_threshold"],
        target_layers=payload["target_layers"],
        basis_src=payload["basis_src"],
        hidden_mean_src=payload["hidden_mean_src"],
        feature_mean=payload["feature_mean"],
        feature_std=payload["feature_std"],
        target_mean=payload["target_mean"],
        target_std=payload["target_std"],
        model=model,
        model_kind=payload["model_kind"],
        bases_target=payload["bases_target"],
        basis_k_full=payload["basis_k_full"],
    )
