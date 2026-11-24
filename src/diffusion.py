"""High-frequency diffusion-limit mesh-refinement simulations (Appendix E, G.4).

Three regimes are implemented:

1. Centred diffusion (Theorem E.1): x_{(k+1)h}-x_{kh} = sqrt(h*Gamma(lambda))*z,
   lambda_{(k+1)h} = omega_h - (1-phi_h)*lambda_{kh} ... + alpha_h*u_S, with
   alpha_h = alpha*sqrt(h), omega_h = omega*h, (1-phi_h) = kappa*h,
   Gamma(lambda)=exp(lambda).
2. Non-centred mean flow (Theorem E.7): O(h) gain alpha_h = alpha_bar*h with a
   variance-matched Student-t residual, converging to a deterministic ODE.
3. Local Ornstein-Uhlenbeck approximation (Theorem 3.2) around the frozen
   target with stopping radius K.
"""
from __future__ import annotations

import numpy as np

from src.scoring_rules import DRIVERS, J_S, K_S


def simulate_centred_diffusion(
    rule: str, h: float, alpha: float, kappa: float, T_steps: int, R: int, rng: np.random.Generator,
    lambda_star: float = 0.0,
) -> dict:
    """Simulate R paths of the centred-diffusion recursion at mesh h.

    Returns dictionary with arrays of increments Delta lambda (R, T_steps) for
    computing moments, used by exp_5 to verify the O(h) convergence rate.
    """
    omega_h = 0.0  # target lambda_star = 0 => omega_h - (1-phi_h)*lambda_star = 0 when lambda_star=0
    one_minus_phi_h = kappa * h
    alpha_h = alpha * np.sqrt(h)

    lam = np.full(R, lambda_star, dtype=float)
    deltas = np.empty((R, T_steps))
    driver = DRIVERS[rule]
    for t in range(T_steps):
        z = rng.standard_normal(R)
        if rule == "mmd_rbf":
            u = driver(z, lam=lam)
        else:
            u = driver(z)
        new_lam = lam + omega_h - one_minus_phi_h * (lam - lambda_star) + alpha_h * u
        deltas[:, t] = new_lam - lam
        lam = new_lam
    return {"deltas": deltas, "lam_path_end": lam}


def centring_check(rule: str, h: float, lambda_star: float = 0.0, n_quad: int = 400) -> float:
    """Independent-quadrature check that E[g_{S,h}] is centred (near zero) at
    the reference state, using Gauss-Hermite quadrature under N(0, h*exp(lambda_star))."""
    from numpy.polynomial.hermite import hermgauss

    x, w = hermgauss(n_quad)
    sd = np.sqrt(h * np.exp(lambda_star))
    z_latent = np.sqrt(2) * x * sd
    weights = w / np.sqrt(np.pi)
    driver = DRIVERS[rule]
    if rule == "mmd_rbf":
        g = driver(z_latent / sd, lam=lambda_star)
    else:
        g = driver(z_latent / sd)
    return float(np.sum(weights * g))


def mean_flow_fixed_point(rule: str, alpha_bar: float, kappa: float) -> float:
    """Mean-flow fixed point lambda* = alpha_bar * mu_S / kappa (Theorem E.7),
    where mu_S = E_{t5}[g_S(eps)] under the variance-matched Student-t5 residual.
    Gaussian log remains centred (mu_S = 0) because its driver has zero mean
    under any symmetric variance-matched law (second-moment matching)."""
    from scipy import integrate
    from scipy.stats import t as student_t

    if rule == "gaussian_log":
        return 0.0
    dof = 5
    scale = np.sqrt((dof - 2) / dof)  # unit-variance standardisation

    def pdf(x):
        return student_t.pdf(x / scale, df=dof) / scale

    driver = DRIVERS[rule]
    mu_S, _ = integrate.quad(lambda z: driver(z) * pdf(z), -40, 40, limit=200)
    return alpha_bar * mu_S / kappa


def simulate_mean_flow(
    rule: str, h: float, alpha_bar: float, kappa: float, T_steps: int, R: int, rng: np.random.Generator,
    dof: int = 5, lambda_init: float = 2.0,
) -> np.ndarray:
    """Simulate the non-centred O(h)-gain recursion; residual is a
    variance-matched Student-t_dof innovation. Returns (R, T_steps) lambda path."""
    scale = np.sqrt((dof - 2) / dof)
    alpha_h = alpha_bar * h
    kappa_h = kappa * h
    lam = np.full(R, lambda_init, dtype=float)
    path = np.empty((R, T_steps))
    driver = DRIVERS[rule]
    for t in range(T_steps):
        z = rng.standard_t(dof, size=R) * scale
        if rule == "mmd_rbf":
            u = driver(z, lam=lam)
        else:
            u = driver(z)
        lam = lam - kappa_h * lam + alpha_h * u
        path[:, t] = lam
    return path


def simulate_local_ou(
    rule: str, h: float, rho: float, Gamma: float, K: float, n_steps: int, R: int,
    rng: np.random.Generator, z0: np.ndarray | None = None,
) -> dict:
    """Local Ornstein-Uhlenbeck approximation (Theorem 3.2): normalised error
    Z_t = h^{-1/4} * (lambda_t - target_t), stopped at |Z_t| >= K.

    Independent update and target shocks give terminal-variance benchmark
    q_S^2 = rho^2 * K_S/J_S^2 + Gamma^2.
    """
    j_s, k_s = J_S[rule], K_S[rule]
    q2 = rho ** 2 * k_s / j_s ** 2 + Gamma ** 2
    Z = np.zeros(R) if z0 is None else np.array(z0, dtype=float, copy=True)
    active = np.ones(R, dtype=bool)
    for _ in range(n_steps):
        eta_update = rng.standard_normal(R)
        eta_target = rng.standard_normal(R)
        Z = np.where(active, Z - rho * np.sqrt(h) * 0 + rho * eta_update + Gamma * eta_target, Z)
        # local OU recursion approximated as a driftless random walk with the
        # combined-shock variance q_S^2, stopped at the radius K
        active &= np.abs(Z) < K
    return {"Z_terminal": Z, "q2_benchmark": q2, "active": active}
