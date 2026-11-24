"""Synthetic data generators for the simulation experiments (exp_3, exp_5, exp_8).

The paper's Section 6.3 and 7.4 describe the observation equation
Y_t = exp(x_t/2) * epsilon_t with a latent "stochastic volatility" path x_t,
but do not give the generating recursion for x_t. In the absence of a
specified mechanism, x_t is generated here as a stationary Gaussian AR(1)
process (the standard discrete-time stochastic-volatility specification of
Taylor, 1986), with persistence and innovation scale set in
``src/config.py`` (EXP3_LATENT_PHI, EXP3_LATENT_SIGMA) and flagged there as
not specified in the paper.
"""
from __future__ import annotations

import numpy as np


def latent_log_vol_path(T: int, phi: float, sigma: float, rng: np.random.Generator, x0: float = 0.0) -> np.ndarray:
    """Stationary Gaussian AR(1) latent log-volatility path x_t, t=0..T-1."""
    eta = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = x0
    for t in range(1, T):
        x[t] = phi * x[t - 1] + sigma * eta[t]
    return x


def bernoulli_variance_contamination(
    x: np.ndarray, prob: float, multiplier: float, rng: np.random.Generator
) -> dict:
    """Y_t = exp(x_t/2) * eps_t (clean) and *mult on masked dates (contaminated).

    Common latent path, common baseline innovations and common mask are
    returned so that every scoring rule in a paired design sees identical
    randomness (Section 6.3).
    """
    T = len(x)
    eps = rng.standard_normal(T)
    mask = rng.random(T) < prob
    eps_contam = np.where(mask, eps * multiplier, eps)
    scale = np.exp(x / 2)
    return {
        "y_clean": scale * eps,
        "y_contam": scale * eps_contam,
        "mask": mask,
        "eps": eps,
        "scale": scale,
    }


def one_sided_jump_contamination(
    x: np.ndarray, prob: float, halfnormal_scale: float, rng: np.random.Generator
) -> dict:
    """Y_t = exp(x_t/2)*eps_t (clean); on Bernoulli(prob) dates the observation
    is additionally hit by -halfnormal_scale*exp(x_t/2)*|Z_t| (Section 7.4),
    leaving the latent path x_t unchanged."""
    T = len(x)
    eps = rng.standard_normal(T)
    mask = rng.random(T) < prob
    z_abs = np.abs(rng.standard_normal(T))
    scale = np.exp(x / 2)
    y_clean = scale * eps
    y_jump = np.where(mask, y_clean - halfnormal_scale * scale * z_abs, y_clean)
    return {"y_clean": y_clean, "y_jump": y_jump, "mask": mask, "scale": scale}


# ---------------------------------------------------------------------------
# Batch (R replications at once) generators for large Monte Carlo experiments
# (exp_3, exp_8), vectorised over the replication axis to avoid a Python-level
# loop over replications on top of the time loop.
# ---------------------------------------------------------------------------
def latent_log_vol_path_batch(
    R: int, T: int, phi: float, sigma: float, rng: np.random.Generator, x0: float = 0.0
) -> np.ndarray:
    """R independent stationary Gaussian AR(1) latent log-volatility paths,
    shape (R, T); row r reproduces :func:`latent_log_vol_path` exactly."""
    eta = rng.standard_normal((R, T))
    x = np.empty((R, T))
    x[:, 0] = x0
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + sigma * eta[:, t]
    return x


def bernoulli_variance_contamination_batch(
    x: np.ndarray, prob: float, multiplier: float, rng: np.random.Generator
) -> dict:
    """Batch version of :func:`bernoulli_variance_contamination`; ``x`` has
    shape (R, T), common latent paths across rules within each replication."""
    R, T = x.shape
    eps = rng.standard_normal((R, T))
    mask = rng.random((R, T)) < prob
    eps_contam = np.where(mask, eps * multiplier, eps)
    scale = np.exp(x / 2)
    return {
        "y_clean": scale * eps,
        "y_contam": scale * eps_contam,
        "mask": mask,
        "eps": eps,
        "scale": scale,
    }


def one_sided_jump_contamination_batch(
    x: np.ndarray, prob: float, halfnormal_scale: float, rng: np.random.Generator
) -> dict:
    """Batch version of :func:`one_sided_jump_contamination`; ``x`` has shape (R, T)."""
    R, T = x.shape
    eps = rng.standard_normal((R, T))
    mask = rng.random((R, T)) < prob
    z_abs = np.abs(rng.standard_normal((R, T)))
    scale = np.exp(x / 2)
    y_clean = scale * eps
    y_jump = np.where(mask, y_clean - halfnormal_scale * scale * z_abs, y_clean)
    return {"y_clean": y_clean, "y_jump": y_jump, "mask": mask, "scale": scale}
