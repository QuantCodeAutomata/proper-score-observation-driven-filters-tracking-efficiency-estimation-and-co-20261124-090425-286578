"""Pseudo-true variance targets under Gaussian variance contamination (Section 6.2).

Under Q_{eps,tau} = (1-eps) N(0, v0) + eps N(0, tau*v0), the pseudo-true
variance target v*(tau) solves the first-order condition
E_Q[psi_S(lambda, Y)] = 0 for lambda = log(v). For the Gaussian log score
this has the closed form v_log* = v0 * (1 + eps*(tau-1)) (identity 2.2);
for density-power, CRPS and MMD-RBF the root is found numerically.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq

from src.scoring_rules import PSIS


@lru_cache(maxsize=8)
def _standard_gh_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal Gauss-Hermite nodes/weights, cached (expensive eigenproblem)."""
    x, w = hermgauss(n_nodes)
    return np.sqrt(2) * x, w / np.sqrt(np.pi)


def gauss_hermite_nodes(n_nodes: int, mean: float, sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for E[f(Y)] with Y ~ N(mean, sd^2), via probabilists'
    rescaling of the physicists' Gauss-Hermite quadrature."""
    x0, weights = _standard_gh_nodes(n_nodes)
    nodes = mean + sd * x0
    return nodes, weights


def mixture_expectation(
    func, v0: float, tau: float, epsilon: float, n_nodes: int
) -> float:
    """E_Q[func(y)] for Q = (1-eps) N(0,v0) + eps N(0, tau*v0), func vectorised."""
    nodes1, w1 = gauss_hermite_nodes(n_nodes, 0.0, np.sqrt(v0))
    nodes2, w2 = gauss_hermite_nodes(n_nodes, 0.0, np.sqrt(tau * v0))
    e1 = np.sum(w1 * func(nodes1))
    e2 = np.sum(w2 * func(nodes2))
    return (1 - epsilon) * e1 + epsilon * e2


def pseudo_true_log_variance(
    rule: str, tau: float, v0: float = 1.0, epsilon: float = 0.05, n_nodes: int = 96,
    root_scan_points: int = 6000,
) -> float:
    """Solve E_Q[psi_S(lambda, Y)] = 0 for lambda = log(v), return v* = exp(lambda*)."""
    psi = PSIS[rule]

    # Nodes/weights of the two mixture components do not depend on lambda, so
    # the whole root-scan grid is evaluated in one vectorised broadcast
    # instead of one Python call per grid point (the latter is numerically
    # exact but was the bottleneck observed in the paper's 6000-point scan).
    nodes1, w1 = gauss_hermite_nodes(n_nodes, 0.0, np.sqrt(v0))
    nodes2, w2 = gauss_hermite_nodes(n_nodes, 0.0, np.sqrt(tau * v0))

    lo, hi = np.log(v0 * 1e-4), np.log(v0 * max(tau, 1.0) * 1e4)
    grid = np.linspace(lo, hi, root_scan_points)

    def objective_vec(lam_grid: np.ndarray) -> np.ndarray:
        lam_col = lam_grid[:, None]
        e1 = psi(lam_col, nodes1[None, :]) @ w1
        e2 = psi(lam_col, nodes2[None, :]) @ w2
        return (1 - epsilon) * e1 + epsilon * e2

    def objective(lam: float) -> float:
        return mixture_expectation(lambda y: psi(lam, y), v0, tau, epsilon, n_nodes)

    vals = objective_vec(grid)
    sign_change = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_change) == 0:
        raise RuntimeError(f"No sign change found for rule={rule}, tau={tau}")
    idx = sign_change[0]
    root_lam = brentq(objective, grid[idx], grid[idx + 1], xtol=1e-10)
    return float(np.exp(root_lam))


def pseudo_true_log_variance_closed_form(tau: float, v0: float = 1.0, epsilon: float = 0.05) -> float:
    """Closed-form KL (Gaussian log score) pseudo-true variance, Section 6.2."""
    return v0 * (1 + epsilon * (tau - 1))
