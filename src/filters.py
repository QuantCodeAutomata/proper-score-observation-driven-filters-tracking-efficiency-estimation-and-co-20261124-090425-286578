"""Observation-driven scale and location filters (Definition 2.2, Section 5).

The autoregressive recursion is

    lambda_{t+1} = omega + phi * lambda_t + alpha * u_S(lambda_t, y_t)

where ``u_S`` is the curvature-scaled recursion driver of the scoring rule
``S``. For the Gaussian log, density-power and CRPS rules the curvature
scaling makes u_S depend on the state only through the standardised
residual z_t = y_t * exp(-lambda_t/2), so u_S(lambda_t, y_t) = g_S(z_t);
for MMD-RBF the bandwidth is fixed in data units and u_S keeps an explicit
lambda-dependence (see ``src/scoring_rules.py``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.scoring_rules import (
    driver_crps,
    driver_density_power,
    driver_gaussian_log,
    driver_mmd_rbf,
    driver_student_log,
)


def scale_update(rule: str, lam: float, y: float, **kwargs) -> float:
    """Recursion update u_S(lambda, y) for a scale (log-variance) filter."""
    z = y * np.exp(-lam / 2)
    if rule == "gaussian_log":
        return driver_gaussian_log(z)
    if rule == "density_power":
        return driver_density_power(z, beta=kwargs.get("beta", 0.15))
    if rule == "crps":
        return driver_crps(z)
    if rule == "mmd_rbf":
        return driver_mmd_rbf(z, lam=lam, ell=kwargs.get("ell", 2.0))
    if rule == "student_log":
        return driver_student_log(z, nu=kwargs.get("nu", 6))
    raise ValueError(f"Unknown scoring rule: {rule}")


def filter_scale_path(
    y: np.ndarray,
    omega: float,
    phi: float,
    alpha: float,
    rule: str,
    lambda_init: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Run the scale filter forward over an observed path ``y``.

    Returns the filtered log-variance path ``lambda`` of the same length as
    ``y``, with ``lambda[0]`` the one-step-ahead prediction for ``y[0]``
    (initialised at the unconditional mean omega/(1-phi) unless
    ``lambda_init`` is given).
    """
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = lambda_init if lambda_init is not None else omega / (1 - phi)
    for t in range(T):
        u = scale_update(rule, lam[t], y[t], **kwargs)
        lam[t + 1] = omega + phi * lam[t] + alpha * u
    return lam[:T]


def filter_scale_path_frozen(
    y: np.ndarray, omega: float, phi: float, alpha: float, rule: str, lambda_init: float, **kwargs
) -> np.ndarray:
    """Like :func:`filter_scale_path` but returns the length-``T`` filtered
    path lambda_1..lambda_T (post-update values), used when parameters are
    frozen and the recursion must be continued from a given state."""
    T = len(y)
    lam_prev = lambda_init
    out = np.empty(T)
    for t in range(T):
        u = scale_update(rule, lam_prev, y[t], **kwargs)
        lam_prev = omega + phi * lam_prev + alpha * u
        out[t] = lam_prev
    return out


# ---------------------------------------------------------------------------
# Location filters (Appendix G.7): y_t = mu_t + sigma * eps_t
# ---------------------------------------------------------------------------
def location_update(rule: str, w: float, beta: float = 0.15) -> float:
    """Curvature-normalised driver g_S(w) for the standardised residual
    w = (y - mu) / sigma of a Gaussian location model.

    * Gaussian log: g(w) = w (unbounded, linear).
    * CRPS: g(w) = 2*Phi(w) - 1 (bounded, monotone saturating).
    * density-power: g(w) = (1+beta) c_beta w exp(-beta w^2 / 2) / [(1+beta) c_beta]
      i.e. normalised to unit curvature at w=0 (redescending, vanishes at
      large |w|). Both the CRPS and density-power location drivers are
      given curvature-normalised scaling G_S = J_S^{-1} so that each
      driver has unit slope at w = 0, matching the paper's curvature
      normalisation principle (Section 6.3) applied here to the location
      case described qualitatively in Appendix G.7.
    """
    from scipy.stats import norm

    if rule == "gaussian_log":
        return w
    if rule == "crps":
        j_s = 2 * norm.pdf(0.0)  # d/dw[2*Phi(w)-1] at w=0
        return (2 * norm.cdf(w) - 1) / j_s
    if rule == "density_power":
        c_beta = (2 * np.pi) ** (-beta / 2)
        raw = (1 + beta) * c_beta * w * np.exp(-beta * w ** 2 / 2)
        j_s = (1 + beta) * c_beta  # d/dw[raw] at w=0
        return raw / j_s
    raise ValueError(f"Unknown location rule: {rule}")


def filter_location_path(
    y: np.ndarray, sigma: float, omega: float, phi: float, alpha: float, rule: str,
    mu_init: float, beta: float = 0.15,
) -> np.ndarray:
    """Run the Gaussian location filter forward over an observed path ``y``."""
    T = len(y)
    mu_prev = mu_init
    out = np.empty(T)
    for t in range(T):
        w = (y[t] - mu_prev) / sigma
        u = location_update(rule, w, beta=beta)
        mu_prev = omega + phi * mu_prev + alpha * u
        out[t] = mu_prev
    return out
