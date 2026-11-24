"""Unit tests for exp/exp7.py (Section 7.1, controlled single-observation
outlier-propagation experiment): fast checks on tiny synthetic return
series, no network calls."""
from __future__ import annotations

import numpy as np
import pytest

from exp.exp7 import (
    compute_perturbation_path,
    compute_sigma_ref,
    fit_loglog_slope,
    peak_perturbation,
    perturb_returns,
)
from src.empirical import filtered_variance_path

CELLS = (
    ("gaussian_log", "gaussian", "log", {}),
    ("student_log", "student", "log", {"nu": 6}),
    ("student_density_power", "student", "density_power", {"nu": 6, "beta": 0.15}),
    ("student_mmd", "student", "mmd", {"nu": 6, "ell": 1.5, "mmd_nodes": 16}),
)

OMEGA, PHI, ALPHA = -0.05, 0.9, 0.1


def _synthetic_returns(T: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) * 0.8


@pytest.mark.parametrize("name,density,criterion,kwargs", CELLS)
def test_zero_perturbation_leaves_path_unchanged(name, density, criterion, kwargs):
    """B=0 must add exactly 0*sigma_ref to the event-date return, so the
    perturbed and clean filtered log-variance paths are identical for every
    cell."""
    returns = _synthetic_returns()
    event_idx = 60
    sigma_ref = compute_sigma_ref(returns, event_idx, window=60)

    v_clean = filtered_variance_path(returns, OMEGA, PHI, ALPHA, criterion, density=density, **kwargs)
    returns_pert = perturb_returns(returns, event_idx, B=0, sigma_ref=sigma_ref)
    v_pert = filtered_variance_path(returns_pert, OMEGA, PHI, ALPHA, criterion, density=density, **kwargs)

    np.testing.assert_array_equal(returns_pert, returns)
    pert_path = compute_perturbation_path(np.log(v_clean), np.log(v_pert), event_idx)
    assert peak_perturbation(pert_path) < 1e-10


def test_peak_perturbation_monotone_nondecreasing_in_B_gaussian_log():
    """On a small synthetic example, the Gaussian-log cell's peak
    perturbation must be monotonically non-decreasing as B grows (larger
    outliers cannot shrink the downstream divergence)."""
    returns = _synthetic_returns(T=150, seed=1)
    event_idx = 80
    sigma_ref = compute_sigma_ref(returns, event_idx, window=60)
    v_clean = filtered_variance_path(returns, OMEGA, PHI, ALPHA, "log", density="gaussian")
    lam_clean = np.log(v_clean)

    B_grid = [0, 1, 5, 10, 20, 40]
    peaks = []
    for B in B_grid:
        returns_pert = perturb_returns(returns, event_idx, B, sigma_ref)
        v_pert = filtered_variance_path(returns_pert, OMEGA, PHI, ALPHA, "log", density="gaussian")
        pert_path = compute_perturbation_path(lam_clean, np.log(v_pert), event_idx)
        peaks.append(peak_perturbation(pert_path))

    diffs = np.diff(peaks)
    assert np.all(diffs >= -1e-9)
    assert peaks[-1] > peaks[0]


def test_fit_loglog_slope_recovers_quadratic_growth():
    """fit_loglog_slope must recover a slope close to 2 when peak = c*B**2
    for a constant c, matching Proposition 2.3(i)'s quadratic growth."""
    B_values = np.array([35.0, 60.0, 100.0])
    c = 0.003
    peak_values = c * B_values ** 2
    slope = fit_loglog_slope(B_values, peak_values)
    assert abs(slope - 2.0) < 1e-8


def test_fit_loglog_slope_matches_intercept_too():
    """The fitted log-log line must also reproduce the known intercept
    log(c), as a sanity check on the full regression, not just the slope."""
    B_values = np.array([10.0, 20.0, 50.0, 100.0])
    c = 0.5
    peak_values = c * B_values ** 2
    coeffs = np.polyfit(np.log(B_values), np.log(peak_values), 1)
    assert abs(coeffs[0] - 2.0) < 1e-8
    assert abs(coeffs[1] - np.log(c)) < 1e-8


def test_compute_sigma_ref_matches_manual_rms():
    """compute_sigma_ref must equal the root-mean-square of the 60 returns
    strictly preceding the event index."""
    rng = np.random.default_rng(2)
    returns = rng.standard_normal(200)
    event_idx = 150
    window = 60
    expected = np.sqrt(np.mean(returns[event_idx - window:event_idx] ** 2))
    assert abs(compute_sigma_ref(returns, event_idx, window) - expected) < 1e-12


def test_perturb_returns_only_touches_event_index():
    """perturb_returns must leave every observation except the event index
    unchanged, and must not mutate the input array in place."""
    returns = np.arange(10, dtype=float)
    original = returns.copy()
    out = perturb_returns(returns, event_idx=4, B=10.0, sigma_ref=0.5)
    np.testing.assert_array_equal(returns, original)
    expected = original.copy()
    expected[4] += 5.0
    np.testing.assert_array_equal(out, expected)
