"""Tests for exp_9: Gaussian location filtering, Nile River illustration
(Appendix G.7)."""
from __future__ import annotations

import numpy as np

from src.filters import filter_location_path, location_update


def test_gaussian_log_location_driver_is_linear_and_unbounded():
    """The Gaussian log location driver g(w) = w grows without bound."""
    ws = np.array([0.0, 1.0, 10.0, 100.0])
    out = np.array([location_update("gaussian_log", w) for w in ws])
    assert np.allclose(out, ws)


def test_crps_and_density_power_location_drivers_are_bounded():
    """CRPS saturates to a finite plateau and density-power redescends to
    zero as |w| grows, unlike the unbounded Gaussian log driver
    (Appendix G.7)."""
    w_large = 50.0
    crps_val = location_update("crps", w_large)
    dp_val = location_update("density_power", w_large)
    log_val = location_update("gaussian_log", w_large)
    assert abs(crps_val) < abs(log_val)
    assert abs(dp_val) < abs(crps_val)
    # density-power is redescending: it must vanish for extreme residuals
    assert abs(dp_val) < 1e-3


def test_crps_driver_is_odd_and_bounded_by_one_over_curvature():
    """CRPS driver g(w) = (2*Phi(w)-1)/J_S is an odd, monotone-saturating
    function of the standardised residual."""
    from scipy.stats import norm
    j_s = 2 * norm.pdf(0.0)
    for w in (0.5, 2.0, 5.0):
        pos = location_update("crps", w)
        neg = location_update("crps", -w)
        assert np.isclose(pos, -neg)
        assert abs(pos) <= 1.0 / j_s + 1e-9


def test_outlier_displacement_bounded_filter_smaller_than_gaussian_log():
    """Injecting a single large outlier into an otherwise flat series should
    displace the bounded CRPS and density-power location filters less than
    the unbounded Gaussian log filter (Proposition 2.3 applied to location
    filtering, Appendix G.7)."""
    T = 60
    y_clean = np.zeros(T)
    y_outlier = y_clean.copy()
    y_outlier[30] = 6.0  # +6 sigma outlier with sigma=1
    sigma, omega, phi, alpha = 1.0, 0.0, 0.9, 0.3

    peak_by_rule = {}
    for rule in ("gaussian_log", "crps", "density_power"):
        path_clean = filter_location_path(y_clean, sigma, omega, phi, alpha, rule, mu_init=0.0)
        path_outlier = filter_location_path(y_outlier, sigma, omega, phi, alpha, rule, mu_init=0.0)
        peak_by_rule[rule] = float(np.max(np.abs(path_outlier - path_clean)))

    assert peak_by_rule["crps"] < peak_by_rule["gaussian_log"]
    assert peak_by_rule["density_power"] < peak_by_rule["crps"]


def test_filter_location_path_tracks_constant_level():
    """With a constant input series equal to the stationary level, the
    location filter should converge to and remain at that level."""
    T = 50
    level = 3.0
    omega, phi, alpha = level * (1 - 0.8), 0.8, 0.2
    y = np.full(T, level)
    path = filter_location_path(y, sigma=1.0, omega=omega, phi=phi, alpha=alpha,
                                 rule="gaussian_log", mu_init=0.0)
    assert np.isclose(path[-1], level, atol=1e-6)
