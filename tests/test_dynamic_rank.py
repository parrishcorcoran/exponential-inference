"""Test the dynamic-rank wrapper against a tiny Llama-style stand-in.

We build a minimal nn.Module that exposes the same attribute layout
``model.model.layers`` that the wrapper expects, then check:

1. Full-rank projection (rank == basis_k_full for all tokens) reproduces
   the unwrapped output bit-for-bit.
2. Reduced-rank projection actually changes the output.
3. Predicted ranks are recorded in RankStats.
"""

from __future__ import annotations

import torch

from src.inference.dynamic_rank import (
    DynamicRankBitNet,
    DynamicRankConfig,
)
from src.routing.rank_predictor import (
    FittedPredictor,
    LinearRankPredictor,
    svd_basis,
)


class TinyLayer(torch.nn.Module):
    """Mimics a Llama decoder layer: accepts hidden_states positionally."""

    def __init__(self, d: int):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.ff = torch.nn.Linear(d, d)

    def forward(self, hidden_states, *args, **kwargs):
        h = self.ln(hidden_states)
        h = self.ff(h) + hidden_states
        return (h,)


class TinyInner(torch.nn.Module):
    def __init__(self, d: int, n_layers: int, vocab: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([TinyLayer(d) for _ in range(n_layers)])

    def forward(self, input_ids=None, **kw):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        return h


class TinyLM(torch.nn.Module):
    def __init__(self, d: int = 24, n_layers: int = 4, vocab: int = 128):
        super().__init__()
        self.config = type("cfg", (), {
            "hidden_size": d,
            "num_hidden_layers": n_layers,
            "vocab_size": vocab,
            "_name_or_path": "tiny",
        })()
        self.model = TinyInner(d, n_layers, vocab)
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, **kw):
        h = self.model(input_ids=input_ids)
        return type("out", (), {"logits": self.lm_head(h)})()


def _build_predictor(d: int, k_basis: int, basis_k_full: int, src_layer: int, targets):
    torch.manual_seed(0)
    # Fabricate a predictor that always predicts rank == basis_k_full so
    # it acts as a pure-identity projection (for the correctness test).
    basis_src = svd_basis(torch.randn(500, d), k_basis)
    bases_target = [svd_basis(torch.randn(500, d), basis_k_full) for _ in targets]
    model = LinearRankPredictor(k_basis, len(targets))
    # Zero out the model so its output is just the target_mean after unnormalising.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return FittedPredictor(
        src_layer=src_layer,
        k_basis=k_basis,
        energy_threshold=0.95,
        target_layers=list(targets),
        basis_src=basis_src,
        hidden_mean_src=torch.zeros(d),
        feature_mean=torch.zeros(k_basis),
        feature_std=torch.ones(k_basis),
        target_mean=torch.full((len(targets),), float(basis_k_full)),  # full rank
        target_std=torch.ones(len(targets)),
        model=model,
        model_kind="linear",
        bases_target=bases_target,
        basis_k_full=basis_k_full,
    )


def test_full_rank_is_identity():
    torch.manual_seed(0)
    d, n_layers, vocab = 24, 4, 64
    model = TinyLM(d=d, n_layers=n_layers, vocab=vocab).eval()
    predictor = _build_predictor(
        d=d, k_basis=4, basis_k_full=d, src_layer=1, targets=[2, 3]
    )
    layer_means = {t: torch.zeros(d) for t in predictor.target_layers}
    input_ids = torch.randint(0, vocab, (1, 16))

    with torch.inference_mode():
        base = model(input_ids=input_ids).logits

    wrapper = DynamicRankBitNet(model, predictor, layer_means)
    with wrapper:
        with torch.inference_mode():
            dyn = model(input_ids=input_ids).logits

    assert torch.allclose(base.float(), dyn.float(), atol=1e-4), \
        (base - dyn).abs().max().item()


def test_low_rank_changes_output_and_records_stats():
    torch.manual_seed(0)
    d, n_layers, vocab = 24, 4, 64
    model = TinyLM(d=d, n_layers=n_layers, vocab=vocab).eval()
    predictor = _build_predictor(
        d=d, k_basis=4, basis_k_full=d, src_layer=1, targets=[2, 3]
    )
    # Force rank 1 everywhere.
    with torch.no_grad():
        predictor.target_mean.fill_(1.0)
    layer_means = {t: torch.zeros(d) for t in predictor.target_layers}
    input_ids = torch.randint(0, vocab, (1, 16))

    with torch.inference_mode():
        base = model(input_ids=input_ids).logits

    wrapper = DynamicRankBitNet(model, predictor, layer_means)
    with wrapper:
        with torch.inference_mode():
            dyn = model(input_ids=input_ids).logits

    assert not torch.allclose(base.float(), dyn.float(), atol=1e-3)
    # Stats: two target layers, 16 tokens each
    for dec_i in (1, 2):  # target 2,3 -> decoder indices 1,2
        ranks = wrapper.stats.flat(dec_i)
        assert ranks.numel() == 16, dec_i
        assert int(ranks.max()) == 1
