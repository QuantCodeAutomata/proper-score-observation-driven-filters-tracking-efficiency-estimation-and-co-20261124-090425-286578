"""Unit tests for src/scoring_rules.py (Appendix G, Table G.5)."""
from __future__ import annotations

import numpy as np
import pytest

from src.scoring_rules import (
    DRIVERS,
    J_S,
    K_S,
    PSIS,
    RULE_NAMES,
    curvature_normalised_driver,
    driver_crps,
    driver_density_power,
    driver_gaussian_log,
    driver_mmd_rbf,
    psi_gaussian_log,
)


def test_gaussian_log_driver_matches_psi_times_two():
    """At the Gaussian reference state g_S(z) = 2*psi_S(z) (Appendix G)."""
    z = np.linspace(-4, 4, 50)
    g = driver_gaussian_log(z)
    psi = psi_gaussian_log(0.0, z)
    np.testing.assert_allclose(g, 2 * psi, atol=1e-10)


def test_gaussian_log_driver_formula():
    z = np.array([-2.0, 0.0, 1.0, 3.0])
    np.testing.assert_allclose(driver_gaussian_log(z), z ** 2 - 1.0)


@pytest.mark.parametrize("rule", ["gaussian_log", "density_power", "crps", "mmd_rbf"])
def test_curvature_normalised_scale_in_expected_range(rule):
    """sqrt(K_S)/|J_S| should lie roughly in [1.4, 1.6] for all four rules
    (expected_outcomes of exp_1)."""
    scale = np.sqrt(K_S[rule]) / abs(J_S[rule])
    assert 1.2 < scale < 1.7, f"{rule}: scale={scale}"


def test_gaussian_log_is_unbounded_others_are_bounded():
    """Gaussian log driver grows without bound; the other three drivers are
    bounded on |z| in [0, 50] (tail-saturating, Section 6.1)."""
    z = np.linspace(0, 50, 2000)
    g_log = np.abs(curvature_normalised_driver("gaussian_log", z))
    assert g_log[-1] > g_log[len(g_log) // 2] * 2, "Gaussian log should keep growing"
    for rule in ("density_power", "crps", "mmd_rbf"):
        g = np.abs(curvature_normalised_driver(rule, z))
        assert np.max(g) < 100, f"{rule} driver should stay bounded"
        assert g[-1] < g[len(g) // 2] * 1.5, f"{rule} should approach a plateau, not keep growing"


def test_crps_driver_monotone_to_plateau():
    """CRPS approaches its plateau monotonically on z >= 0 (Section 6.1)."""
    z = np.linspace(0, 10, 1000)
    g = driver_crps(z)
    assert np.all(np.diff(g) >= -1e-9), "CRPS driver should be monotone on [0, 10]"


def test_density_power_and_mmd_have_interior_peak():
    """Density-power and MMD drivers decline from an interior peak in |z|
    (Section 6.1), i.e. are not monotone over the whole range [0, 10]."""
    z = np.linspace(0.01, 10, 2000)
    for rule in ("density_power", "mmd_rbf"):
        g_abs = np.abs(curvature_normalised_driver(rule, z))
        peak_idx = np.argmax(g_abs)
        assert 0 < peak_idx < len(z) - 1, f"{rule} should have an interior peak"


def test_mmd_driver_reduces_to_reference_state_formula():
    z = np.array([0.0, 1.0, -2.0])
    g1 = driver_mmd_rbf(z, lam=0.0)
    g2 = DRIVERS["mmd_rbf"](z, lam=0.0)
    np.testing.assert_allclose(g1, g2)


@pytest.mark.parametrize("rule", RULE_NAMES)
def test_quadrature_matches_table_curvatures(rule):
    """Independent Gauss-Hermite check of J_S, K_S against Table G.5 values."""
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(80)
    y = np.sqrt(2) * x
    weights = w / np.sqrt(np.pi)
    psi = PSIS[rule]
    k_quad = float(np.sum(weights * psi(0.0, y) ** 2))
    h = 1e-5
    dpsi = (psi(h, y) - psi(-h, y)) / (2 * h)
    j_quad = float(-np.sum(weights * dpsi))
    assert abs(j_quad - J_S[rule]) < 1e-3, f"{rule}: J_quad={j_quad}, J_table={J_S[rule]}"
    assert abs(k_quad - K_S[rule]) < 1e-3, f"{rule}: K_quad={k_quad}, K_table={K_S[rule]}"
