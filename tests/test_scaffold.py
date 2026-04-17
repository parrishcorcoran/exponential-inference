"""Cheap smoke test: the package imports and the loader module is well-formed.

No model is loaded — that happens on the Strix Halo via scripts/stage0_verify.py.
"""

import importlib

import pytest


MODULES = [
    "src",
    "src.measurement",
    "src.routing",
    "src.inference",
    "src.evaluation",
    "src.common",
    "src.common.model_loader",
]


@pytest.mark.parametrize("mod", MODULES)
def test_imports(mod):
    importlib.import_module(mod)


def test_loader_exposes_expected_symbols():
    m = importlib.import_module("src.common.model_loader")
    for name in ("load_bitnet", "describe_backend", "pick_device", "DEFAULT_MODEL_ID"):
        assert hasattr(m, name), f"model_loader missing {name}"
