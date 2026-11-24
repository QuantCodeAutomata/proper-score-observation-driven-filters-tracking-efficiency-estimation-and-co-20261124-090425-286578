"""Unit tests for src/pseudo_true.py (Section 6.2, identity 2.2)."""
from __future__ import annotations

import numpy as np
import pytest

from src.pseudo_true import pseudo_true_log_variance, pseudo_true_log_variance_closed_form


@pytest.mark.parametrize("tau", [1.0, 5.0, 20.0, 80.0])
def test_gaussian_log_numeric_matches_closed_form(tau):
    v_numeric = pseudo_true_log_variance("gaussian_log", tau, v0=1.0, epsilon=0.05)
    v_closed = pseudo_true_log_variance_closed_form(tau, v0=1.0, epsilon=0.05)
    assert abs(v_numeric - v_closed) < 1e-5, (v_numeric, v_closed)


def test_gaussian_log_tau80_value():
    """v_log*/v0 = 1 + 0.05*(80-1) = 4.95 at tau=80 (Section 6.2)."""
    v = pseudo_true_log_variance_closed_form(80.0, v0=1.0, epsilon=0.05)
    assert abs(v - 4.95) < 1e-9


def test_pseudo_true_at_tau_equal_one_is_baseline():
    """No contamination (tau=1) should leave the target at v0 for every rule."""
    for rule in ("gaussian_log", "density_power", "crps", "mmd_rbf"):
        v = pseudo_true_log_variance(rule, 1.0, v0=1.0, epsilon=0.05)
        assert abs(v - 1.0) < 1e-3, f"{rule}: v*={v}"


def test_bounded_rules_saturate_below_gaussian_log():
    """Density-power, CRPS and MMD targets stay below the (unbounded) Gaussian
    log target at large contamination variance ratio (Section 6.2)."""
    tau = 80.0
    v_log = pseudo_true_log_variance_closed_form(tau)
    for rule in ("density_power", "crps", "mmd_rbf"):
        v_rule = pseudo_true_log_variance(rule, tau, v0=1.0, epsilon=0.05)
        assert v_rule < v_log, f"{rule}: v*={v_rule} should saturate below {v_log}"
        assert v_rule < 2.0, f"{rule}: v*={v_rule} should stay well below the KL target"


def test_pseudo_true_is_monotone_in_tau():
    taus = np.array([1.0, 2.0, 5.0, 10.0, 40.0])
    for rule in ("gaussian_log", "density_power", "crps", "mmd_rbf"):
        vals = np.array([pseudo_true_log_variance(rule, t, v0=1.0, epsilon=0.05) for t in taus])
        assert np.all(np.diff(vals) >= -1e-8), f"{rule}: not monotone, vals={vals}"
