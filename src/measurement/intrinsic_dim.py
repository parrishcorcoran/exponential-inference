"""Intrinsic-dimensionality estimators.

- ``compute_pr``: participation ratio of a batch of hidden-state vectors,
  PR = (sum lambda_i)^2 / sum lambda_i^2, using the eigenvalues of the
  empirical covariance. PR is a linear / second-order estimate.
- ``compute_twonn``: TwoNN intrinsic dimensionality
  (Facco, D'Errico, Rodriguez & Laio, 2017). Non-parametric, based on
  the ratio of first- and second-nearest-neighbour distances.
- ``measure_layer_manifold``: runs a model, captures hidden states at one
  layer, returns both scalars plus the cached activations so callers
  don't redo the forward pass.

All functions accept float32 or bfloat16 inputs and promote internally
to float32 for numerical stability. Inputs are shape ``(N, D)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ManifoldMeasurement:
    layer_index: int
    n_samples: int
    hidden_dim: int
    pr: float
    twonn: float
    hidden_states: Optional[torch.Tensor] = None


def _to_float32_matrix(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    t = t.detach()
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])
    elif t.ndim == 1:
        t = t.unsqueeze(0)
    return t.to(torch.float32).cpu()


def compute_pr(hidden_states, center: bool = True) -> float:
    """Participation ratio of the covariance spectrum.

    PR = (trace(C))^2 / trace(C @ C). Equivalently, (sum lambda)^2 over
    sum lambda^2. Invariant to global scale. Range is (0, rank(C)].
    """
    x = _to_float32_matrix(hidden_states)
    if x.shape[0] < 2:
        return 0.0
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    # Use singular values of x, so lambda_i = sigma_i^2 / (N-1).
    sv = torch.linalg.svdvals(x)
    lam = sv.pow(2)
    num = lam.sum().pow(2)
    den = lam.pow(2).sum()
    if den.item() == 0.0:
        return 0.0
    return float((num / den).item())


def compute_twonn(
    hidden_states,
    discard_fraction: float = 0.1,
    max_samples: Optional[int] = 4000,
    seed: int = 0,
) -> float:
    """TwoNN intrinsic dimensionality.

    Algorithm (Facco et al. 2017):
      1. For each point, find first and second nearest neighbour
         distances r1, r2.
      2. Form mu_i = r2_i / r1_i. Under the uniform-density assumption,
         P(mu) = d * mu^(-d-1), and the empirical cdf of ``log(mu)``
         is linear in ``d * log(mu)``.
      3. Fit d by least squares through the origin on
         F_emp(mu_i) = 1 - i/N vs -log(mu_i).
      4. Discard the top ``discard_fraction`` of mu values (outliers
         that violate the uniformity assumption).

    ``max_samples`` limits the N used for the pairwise-distance matrix
    (O(N^2) memory). When the batch is larger it is subsampled
    deterministically.
    """
    x = _to_float32_matrix(hidden_states)
    n = x.shape[0]
    if n < 4:
        return 0.0

    if max_samples is not None and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        x = x[idx]
        n = x.shape[0]

    # Pairwise distances with self set to +inf so argmin skips it.
    d = torch.cdist(x, x)
    d.fill_diagonal_(float("inf"))
    # Need top-2 smallest per row.
    r_sorted, _ = torch.topk(d, k=2, largest=False)
    r1 = r_sorted[:, 0]
    r2 = r_sorted[:, 1]

    # Drop degenerate pairs (duplicates) where r1 == 0.
    mask = r1 > 0
    r1 = r1[mask]
    r2 = r2[mask]
    if r1.numel() < 4:
        return 0.0

    mu = (r2 / r1).clamp_min(1.0 + 1e-12)
    mu_sorted, _ = torch.sort(mu)

    k = int((1.0 - discard_fraction) * mu_sorted.numel())
    k = max(2, k)
    mu_trim = mu_sorted[:k]

    # Empirical cdf values F_i = i / (N+1), using i = 1..k.
    i = torch.arange(1, k + 1, dtype=torch.float64)
    f_emp = i / (mu_sorted.numel() + 1)
    y = -torch.log1p(-f_emp).to(torch.float64)
    x_fit = torch.log(mu_trim).to(torch.float64)

    # Least-squares slope through the origin: d = (x.y)/(x.x)
    num = torch.dot(x_fit, y)
    den = torch.dot(x_fit, x_fit)
    if den.item() == 0.0:
        return 0.0
    return float((num / den).item())


def spectrum_rank_quantile(hidden_states, quantiles=(0.5, 0.9, 0.95, 0.99)) -> dict:
    """Number of singular values needed to cover each energy quantile.

    Useful as a coarse 'rank' summary alongside PR and TwoNN. Operates
    on the centred matrix.
    """
    x = _to_float32_matrix(hidden_states)
    if x.shape[0] < 2:
        return {f"r{q}": 0 for q in quantiles}
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x).pow(2)
    cum = torch.cumsum(sv, dim=0) / sv.sum()
    out = {}
    for q in quantiles:
        # first index where cumulative energy >= q, 1-based rank
        above = (cum >= q).nonzero(as_tuple=False)
        out[f"r{int(q * 100):02d}"] = int(above[0].item() + 1) if above.numel() else int(sv.numel())
    return out


def measure_layer_manifold(
    model,
    tokens: torch.Tensor,
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
    return_hidden: bool = True,
    twonn_max_samples: int = 4000,
) -> ManifoldMeasurement:
    """Run ``model`` on ``tokens`` and measure PR/TwoNN at one layer.

    Uses ``output_hidden_states=True`` which returns ``n_layers+1``
    tensors; ``layer_idx`` indexes into the decoder layers (0 is the
    embedding output, 1..L are the decoder-layer outputs).
    """
    model.eval()
    with torch.inference_mode():
        out = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    h = out.hidden_states[layer_idx]  # (B, T, D)
    flat = h.reshape(-1, h.shape[-1])
    pr = compute_pr(flat)
    td = compute_twonn(flat, max_samples=twonn_max_samples)
    return ManifoldMeasurement(
        layer_index=layer_idx,
        n_samples=flat.shape[0],
        hidden_dim=flat.shape[-1],
        pr=pr,
        twonn=td,
        hidden_states=flat if return_hidden else None,
    )
