"""Fast unit tests for exp/exp3.py (Section 6.3, Table 2).

All tests use small synthetic sizes (R ~ 20-50, T ~ 50-100), not the full
experiment configuration in src/config.py, so the suite runs quickly.
"""
from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import (
    bernoulli_variance_contamination_batch,
    latent_log_vol_path_batch,
)
from exp.exp3 import CONDITIONS, RULES, run_experiment, summarize
from src.config import EXP3_FILTER_OMEGA, EXP3_FILTER_PHI, EXP3_GAIN_ALPHA, EXP3_LATENT_PHI, EXP3_LATENT_SIGMA
from src.filters import filter_scale_path, filter_scale_path_batch


@pytest.mark.parametrize("rule", RULES)
def test_filter_scale_path_batch_matches_scalar(rule):
    """filter_scale_path_batch on a single replication must reproduce the
    scalar src.filters.filter_scale_path exactly (same recursion, same
    curvature normalisation)."""
    rng = np.random.default_rng(1)
    T = 80
    y = rng.standard_normal(T) * np.exp(rng.standard_normal(T) * 0.1)

    lam_scalar = filter_scale_path(
        y, EXP3_FILTER_OMEGA, EXP3_FILTER_PHI, EXP3_GAIN_ALPHA, rule,
        curvature_normalize=True,
    )
    lam_batch = filter_scale_path_batch(
        y[None, :], EXP3_FILTER_OMEGA, EXP3_FILTER_PHI, EXP3_GAIN_ALPHA, rule,
        curvature_normalize=True,
    )
    np.testing.assert_allclose(lam_batch[0], lam_scalar, rtol=1e-10, atol=1e-10)


def test_run_experiment_shapes_and_keys():
    R, T = 20, 60
    metrics = run_experiment(R, T, rules=RULES, seed=7)
    assert set(metrics.keys()) == set(CONDITIONS)
    for cond in CONDITIONS:
        assert set(metrics[cond].keys()) == set(RULES)
        for rule in RULES:
            cell = metrics[cond][rule]
            assert set(cell.keys()) == {"rmse", "bias", "max_dev", "flagged_update"}
            for arr in cell.values():
                assert arr.shape == (R,)


def test_metrics_finite_and_nonnegative_where_expected():
    R, T = 25, 70
    metrics = run_experiment(R, T, rules=RULES, seed=11)
    for cond in CONDITIONS:
        for rule in RULES:
            cell = metrics[cond][rule]
            for name, arr in cell.items():
                assert np.all(np.isfinite(arr)), f"{cond}/{rule}/{name} has non-finite values"
            assert np.all(cell["rmse"] >= 0)
            assert np.all(cell["max_dev"] >= 0)
            assert np.all((cell["flagged_update"] >= 0) & (cell["flagged_update"] <= 1))

    summary = summarize(metrics, R, T, 11)
    assert summary["R"] == R and summary["T"] == T
    for cond in CONDITIONS:
        for rule in RULES:
            for name in ("rmse", "bias", "max_dev", "flagged_update"):
                assert np.isfinite(summary[cond][rule][name])
                assert np.isfinite(summary[cond][rule][f"{name}_se"])
                assert summary[cond][rule][f"{name}_se"] >= 0
            assert summary[cond][rule]["rmse"] >= 0
            assert summary[cond][rule]["max_dev"] >= 0


def test_contaminated_gaussian_log_rmse_much_larger_than_bounded_rules():
    """Qualitative robustness property of Section 6.3: under contamination
    the unbounded Gaussian-log driver's RMSE is much larger (paper reports
    an order-of-magnitude-scale effect) than the bounded density-power,
    CRPS and MMD-RBF drivers' RMSE, on a quick small-scale paired run."""
    R, T = 50, 100
    metrics = run_experiment(R, T, rules=RULES, seed=2026)
    summary = summarize(metrics, R, T, 2026)

    gaussian_rmse = summary["contaminated"]["gaussian_log"]["rmse"]
    for rule in ("density_power", "crps", "mmd_rbf"):
        bounded_rmse = summary["contaminated"][rule]["rmse"]
        assert gaussian_rmse > 2 * bounded_rmse, (
            f"expected gaussian_log RMSE ({gaussian_rmse}) > 2x {rule} RMSE "
            f"({bounded_rmse}) under contamination"
        )


def test_paired_design_shares_randomness_across_rules():
    """The clean series fed to every rule within a replication batch must be
    identical (common latent path, innovations and mask), as required by
    the paired design."""
    R, T = 15, 40
    rng = np.random.default_rng(5)
    x = latent_log_vol_path_batch(R, T, EXP3_LATENT_PHI, EXP3_LATENT_SIGMA, rng)
    contam = bernoulli_variance_contamination_batch(x, 0.02, 6.0, rng)
    assert contam["y_clean"].shape == (R, T)
    assert contam["y_contam"].shape == (R, T)
    # Off-masked dates must be identical between clean and contaminated series.
    unmasked = ~contam["mask"]
    np.testing.assert_allclose(contam["y_clean"][unmasked], contam["y_contam"][unmasked])
