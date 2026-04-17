"""Tests for the Stage 4 aggregation / speedup helpers."""

from __future__ import annotations

import math

from src.evaluation.acceleration_curve import (
    PerPromptRun,
    aggregate_curve,
    speedup_curve,
)


def _mk(t_per_step, mode="base"):
    r = PerPromptRun(prompt_id="p", prompt_text="", prompt_token_count=0, mode=mode)
    r.per_step_seconds = list(t_per_step)
    return r


def test_aggregate_mean_sem_and_counts():
    runs = [_mk([0.1, 0.2, 0.3]), _mk([0.1, 0.2, 0.4]), _mk([0.1, 0.2])]
    agg = aggregate_curve(runs)
    assert len(agg["mean_seconds"]) == 3
    # position 0: 0.1, 0.1, 0.1
    assert abs(agg["mean_seconds"][0] - 0.1) < 1e-9
    assert agg["n"] == [3, 3, 2]
    # Last position only has two samples.
    assert abs(agg["mean_seconds"][2] - 0.35) < 1e-9
    # SEM for position 0 should be 0.
    assert abs(agg["sem_seconds"][0]) < 1e-12


def test_speedup_ratio():
    base = {"mean_seconds": [1.0, 1.0, 1.0], "sem_seconds": [0, 0, 0], "n": [1, 1, 1]}
    dyn = {"mean_seconds": [1.0, 0.5, 0.25], "sem_seconds": [0, 0, 0], "n": [1, 1, 1]}
    sp = speedup_curve(base, dyn)
    assert sp["ratio"] == [1.0, 2.0, 4.0]


def test_speedup_handles_unequal_lengths():
    base = {"mean_seconds": [1.0, 1.0, 1.0, 1.0], "sem_seconds": [0]*4, "n": [1]*4}
    dyn = {"mean_seconds": [2.0, 1.0], "sem_seconds": [0, 0], "n": [1, 1]}
    sp = speedup_curve(base, dyn)
    assert sp["ratio"] == [0.5, 1.0]


def test_speedup_zero_guard():
    base = {"mean_seconds": [1.0], "sem_seconds": [0], "n": [1]}
    dyn = {"mean_seconds": [0.0], "sem_seconds": [0], "n": [1]}
    sp = speedup_curve(base, dyn)
    assert math.isnan(sp["ratio"][0])
