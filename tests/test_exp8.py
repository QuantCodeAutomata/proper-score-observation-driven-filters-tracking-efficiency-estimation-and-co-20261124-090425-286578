"""Tests for exp_8: permanent variance break and jump contamination
(Section 7.4, Table G.7)."""
from __future__ import annotations

import numpy as np

from exp.exp8 import run_break_experiment, run_jump_experiment


def test_break_experiment_small_run_is_finite_and_bounded_adapts_slower():
    """A small paired Monte Carlo run should produce finite, positive
    adaptation delays and cumulative excess losses, with the bounded
    density-power filter adapting no faster than the Gaussian log filter
    (robustness-adaptation trade-off, Section 7.4)."""
    summary = run_break_experiment(R=200, seed=1)
    for rule in ("gaussian_log", "density_power"):
        delay = summary[rule]["adaptation_delay"]
        excess = summary[rule]["cumulative_excess_loss"]
        assert np.isfinite(delay) and delay > 0
        assert np.isfinite(excess) and excess > 0
    assert summary["density_power"]["adaptation_delay"] >= summary["gaussian_log"]["adaptation_delay"]


def test_jump_experiment_small_run_is_finite():
    """A small jump-contamination run should produce finite jump-minus-clean
    RMSE changes for every filter cell, in both the frozen-fit and
    refitted arms."""
    summary = run_jump_experiment(T=400, seed=1)
    for name, stats in summary["cells"].items():
        assert np.isfinite(stats["jump_minus_clean_frozen"])
        assert np.isfinite(stats["jump_minus_clean_refitted"])


def test_jump_experiment_gaussian_log_most_sensitive_to_jumps():
    """The unbounded Gaussian-log driver should show a jump-minus-clean RMSE
    degradation no smaller than the bounded Gaussian density-power driver
    in the frozen-fit arm (Proposition 2.3)."""
    summary = run_jump_experiment(T=600, seed=2)
    log_delta = summary["cells"]["gaussian_log"]["jump_minus_clean_frozen"]
    dp_delta = summary["cells"]["gaussian_density_power"]["jump_minus_clean_frozen"]
    assert log_delta >= dp_delta - 1e-9
