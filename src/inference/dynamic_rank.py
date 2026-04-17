"""Dynamic-rank forward pass for BitNet.

Wraps a loaded HuggingFace ``BitNetForCausalLM`` (or compatible Llama-
style causal LM) and registers per-decoder-layer forward pre-hooks.
The hooks apply a per-token rank projection

    h' = h_mean + P_r P_r^T (h - h_mean)

where ``P_r`` is the top-``r`` columns of the calibration SVD basis at
that layer, ``r`` is chosen per token by the Stage 2 predictor, and
``h_mean`` is the calibration-time mean of the layer's hidden state.

Invariants:
  - At full rank (``r = basis_k_full`` for every token), the projection
    is the identity and the forward pass matches the base model
    bit-for-bit up to floating-point noise.
  - ``r`` is always >= a configurable floor so the most aggressive
    setting still produces usable outputs.

The hook runs only on decoder layers at or after
``predictor.target_layers[0]`` (i.e., strictly after the source
layer). The source layer's hidden state is snapshotted on-the-fly in a
forward hook on the source decoder layer so we don't have to re-run
the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.routing.rank_predictor import FittedPredictor


def _decoder_layers(model) -> torch.nn.ModuleList:
    """Return the ``ModuleList`` of decoder layers for BitNet / Llama."""
    # HF layouts: model.model.layers for LlamaForCausalLM-style, BitNet
    # inherits this. Some custom forks use `model.transformer.h`.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("cannot locate decoder layer list on model")


@dataclass
class DynamicRankConfig:
    rank_floor: int = 1
    rank_ceiling: Optional[int] = None  # None => basis_k_full
    safety_multiplier: float = 1.0
    # If provided, overrides the predictor. The callable receives the
    # source-layer hidden state (flat, (N, D)) and returns a (N, L)
    # integer tensor of predicted ranks, where L = len(target_layers).
    rank_override: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    # Force-full-rank disables projection (identity). Used for
    # correctness gates and as a clean baseline comparator.
    force_full_rank: bool = False


@dataclass
class RankStats:
    """Aggregated per-forward telemetry for analysis.

    - ``ranks_per_layer`` holds one tensor per applied layer, each of
      shape ``(total_tokens_seen,)``.
    - ``mean_rank_per_layer`` is the running average rank per layer.
    """
    layers: List[int] = field(default_factory=list)
    ranks_per_layer: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

    def record(self, layer_idx: int, ranks: torch.Tensor) -> None:
        self.ranks_per_layer.setdefault(layer_idx, []).append(ranks.detach().cpu())
        if layer_idx not in self.layers:
            self.layers.append(layer_idx)

    def flat(self, layer_idx: int) -> torch.Tensor:
        xs = self.ranks_per_layer.get(layer_idx, [])
        return torch.cat(xs) if xs else torch.empty(0, dtype=torch.long)

    def mean_rank_per_layer(self) -> Dict[int, float]:
        return {li: float(self.flat(li).to(torch.float32).mean()) if self.ranks_per_layer.get(li) else 0.0
                for li in self.layers}


class DynamicRankBitNet:
    """Wrapper that installs rank-projection hooks on a loaded model.

    Use as a context manager or call ``install()`` / ``remove()``
    explicitly. Only one wrapper can be active on a given model at a
    time; ``install()`` asserts this.
    """

    def __init__(
        self,
        model,
        predictor: FittedPredictor,
        layer_means: Dict[int, torch.Tensor],
        config: Optional[DynamicRankConfig] = None,
    ):
        self.model = model
        self.predictor = predictor
        self.layer_means = layer_means
        self.config = config or DynamicRankConfig()

        self._decoder_layers = _decoder_layers(model)
        self._n_layers = len(self._decoder_layers)
        self._source_layer_idx = predictor.src_layer
        # Mapping from decoder-layer index -> index into predictor.target_layers.
        # Layer indexing convention: predictor.src_layer == 5 means the
        # hidden state AFTER decoder layer 5 (i.e. hidden_states[6] in
        # the output_hidden_states tuple). So the rank projection is
        # applied at the INPUT of decoder layers whose output index is
        # in target_layers. Equivalently, decoder-layer index =
        # target_layer_index - 1.
        self._target_to_decoder = {
            t: t - 1 for t in predictor.target_layers
        }
        self._decoder_to_target_col = {
            dec_i: col for col, (_, dec_i) in enumerate(
                sorted(self._target_to_decoder.items())
            )
        }

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._installed = False
        self._last_source_hidden: Optional[torch.Tensor] = None
        self._last_predicted_ranks: Optional[torch.Tensor] = None  # (N, L_targets)
        self.stats = RankStats()

        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype

        # Move predictor tensors onto the model device once, in the
        # model's compute dtype, so hooks avoid repeated casts.
        self._basis_src = self.predictor.basis_src.to(
            self._device, dtype=torch.float32
        )
        self._hidden_mean_src = self.predictor.hidden_mean_src.to(
            self._device, dtype=torch.float32
        )
        self._feature_mean = self.predictor.feature_mean.to(
            self._device, dtype=torch.float32
        )
        self._feature_std = self.predictor.feature_std.clamp_min(1e-6).to(
            self._device, dtype=torch.float32
        )
        self._target_mean = self.predictor.target_mean.to(
            self._device, dtype=torch.float32
        )
        self._target_std = self.predictor.target_std.to(
            self._device, dtype=torch.float32
        )
        self._bases_target = [
            b.to(self._device, dtype=torch.float32)
            for b in self.predictor.bases_target
        ]
        self._predictor_model = self.predictor.model.to(self._device)
        self._predictor_model.eval()
        self._basis_k_full = self.predictor.basis_k_full
        self._rank_ceiling = self.config.rank_ceiling or self._basis_k_full

        # Per-target-layer calibration-time hidden-state mean. Accepts a
        # dict from target_layer_index -> (D,) tensor. Defaults to zero
        # if caller didn't supply (equivalent to no re-centring).
        self._layer_means = {
            dec_i: layer_means.get(
                t, torch.zeros(self._basis_src.shape[0])
            ).to(self._device, dtype=torch.float32)
            for t, dec_i in self._target_to_decoder.items()
        }

    # ------------------------------------------------------------------
    # hook lifecycle
    # ------------------------------------------------------------------
    def install(self) -> "DynamicRankBitNet":
        if self._installed:
            raise RuntimeError("DynamicRankBitNet is already installed")
        # Capture source-layer output via a forward hook on the source
        # decoder layer. The hidden state emitted by layer ``src_layer``
        # is the same that feeds decoder layer src_layer+1.
        src_decoder_idx = self._source_layer_idx - 1
        if 0 <= src_decoder_idx < self._n_layers:
            h = self._decoder_layers[src_decoder_idx].register_forward_hook(
                self._make_source_capture_hook()
            )
            self._hook_handles.append(h)
        elif self._source_layer_idx == 0:
            # Unusual but supported: source == embeddings.
            h = self.model.model.embed_tokens.register_forward_hook(
                self._make_source_capture_hook(is_embedding=True)
            )
            self._hook_handles.append(h)
        else:
            raise ValueError(f"source layer {self._source_layer_idx} out of range")

        # Pre-hooks on every target decoder layer.
        for dec_i in sorted(self._decoder_to_target_col):
            if dec_i < 0 or dec_i >= self._n_layers:
                continue
            h = self._decoder_layers[dec_i].register_forward_pre_hook(
                self._make_projection_pre_hook(dec_i), with_kwargs=True
            )
            self._hook_handles.append(h)

        self._installed = True
        return self

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._installed = False
        self._last_source_hidden = None
        self._last_predicted_ranks = None

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, tb):
        self.remove()

    # ------------------------------------------------------------------
    # hook implementations
    # ------------------------------------------------------------------
    def _make_source_capture_hook(self, is_embedding: bool = False):
        """Capture the source-layer hidden state and precompute per-token
        predicted ranks for every target layer."""
        def hook(module, args, output):
            # Decoder layer outputs are either Tensor or tuple(Tensor, ...).
            if is_embedding:
                hidden = output
            else:
                hidden = output[0] if isinstance(output, tuple) else output
            flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
            self._last_source_hidden = flat
            self._last_predicted_ranks = self._predict_ranks(flat)
            return output
        return hook

    def _predict_ranks(self, flat_hidden: torch.Tensor) -> torch.Tensor:
        """Predict integer ranks for each token at each target layer.

        Mirrors ``FittedPredictor.predict`` but reuses tensors already
        resident on the model device.
        """
        x = flat_hidden - self._hidden_mean_src
        feats = x @ self._basis_src  # (N, k_basis)
        feats = (feats - self._feature_mean) / self._feature_std
        with torch.inference_mode():
            y = self._predictor_model(feats)
        y = y * self._target_std + self._target_mean
        if self.config.safety_multiplier != 1.0:
            y = y * self.config.safety_multiplier
        y = y.clamp(
            min=float(max(1, self.config.rank_floor)),
            max=float(self._rank_ceiling),
        )
        return y.round().to(torch.long)

    def _make_projection_pre_hook(self, decoder_idx: int):
        """Project hidden state into predicted per-token rank subspace.

        ``register_forward_pre_hook(with_kwargs=True)`` signature:
            hook(module, args, kwargs) -> (new_args, new_kwargs) or None.
        Decoder layers take ``hidden_states`` as the first positional
        arg; we replace it.
        """
        target_col = self._decoder_to_target_col[decoder_idx]
        basis = self._bases_target[target_col]  # (D, basis_k_full), float32
        layer_mean = self._layer_means[decoder_idx]

        def hook(module, args, kwargs):
            if self.config.force_full_rank:
                return None  # do nothing; identity projection
            if self._last_predicted_ranks is None:
                return None

            if "hidden_states" in kwargs:
                h = kwargs["hidden_states"]
                in_kwargs = True
            else:
                h = args[0]
                in_kwargs = False

            orig_shape = h.shape
            orig_dtype = h.dtype
            flat = h.reshape(-1, orig_shape[-1]).to(torch.float32)

            ranks = self._last_predicted_ranks[:, target_col]  # (N,)
            # Safety: length alignment. When KV-caching incremental
            # decoding is used, only the new tokens pass through. The
            # source-capture hook runs on the same subset, so lengths
            # should match. If they don't, punt to full rank.
            if ranks.shape[0] != flat.shape[0]:
                return None

            # Centre
            x = flat - layer_mean
            # Coefficients in full calibration basis (N, basis_k_full)
            coeffs = x @ basis
            # Build per-token mask keeping top-r coefficients in the
            # basis order (which is energy order on the calibration
            # set). A (N, K) float mask multiplied into coeffs zeroes
            # the discarded directions.
            K = coeffs.shape[1]
            ar = torch.arange(K, device=coeffs.device).unsqueeze(0)
            mask = (ar < ranks.unsqueeze(1)).to(coeffs.dtype)
            coeffs = coeffs * mask
            # Reconstruct: P P^T (x) = basis @ coeffs.T
            recon = coeffs @ basis.T
            new_flat = (recon + layer_mean).to(orig_dtype)
            new_h = new_flat.reshape(orig_shape)

            # Telemetry
            self.stats.record(decoder_idx, ranks)

            if in_kwargs:
                kwargs["hidden_states"] = new_h
                return args, kwargs
            else:
                return (new_h,) + args[1:], kwargs

        return hook


# ----------------------------------------------------------------------
# convenience: build layer-mean dict from Stage 1 cache
# ----------------------------------------------------------------------

def build_layer_means(cache_dir, target_layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    """Compute per-layer calibration-time mean hidden states."""
    from src.measurement.cache_hidden_states import load_layer
    out: Dict[int, torch.Tensor] = {}
    for li in target_layers:
        h = load_layer(cache_dir, li).to(torch.float32)
        out[li] = h.mean(dim=0)
    return out


# ----------------------------------------------------------------------
# correctness gate
# ----------------------------------------------------------------------

def check_full_rank_matches_base(
    model,
    predictor: FittedPredictor,
    layer_means: Dict[int, torch.Tensor],
    input_ids: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Tuple[bool, float, float]:
    """Run model at full rank through DynamicRankBitNet and compare
    logits against the unwrapped model.

    Returns (passed, max_abs_diff, mean_abs_diff).
    """
    model.eval()
    with torch.inference_mode():
        base_logits = model(input_ids=input_ids).logits
    cfg = DynamicRankConfig(force_full_rank=False,
                            rank_floor=predictor.basis_k_full,
                            rank_ceiling=predictor.basis_k_full)
    wrapper = DynamicRankBitNet(model, predictor, layer_means, cfg)
    with wrapper:
        with torch.inference_mode():
            dyn_logits = model(input_ids=input_ids).logits
    diff = (base_logits.float() - dyn_logits.float()).abs()
    max_d = float(diff.max().item())
    mean_d = float(diff.mean().item())
    passed = torch.allclose(base_logits.float(), dyn_logits.float(),
                            atol=atol, rtol=rtol)
    return passed, max_d, mean_d
