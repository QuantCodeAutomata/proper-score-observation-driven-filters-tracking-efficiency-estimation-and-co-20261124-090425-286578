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
    rule: str, h: float, alpha_bar: float, kappa: float, T_time: float, R: int, rng: np.random.Generator,
    dof: int = 5, lambda_init: float = 0.0,
) -> np.ndarray:
    """Simulate the non-centred O(h)-gain recursion over rescaled time horizon
    ``T_time`` (Theorem E.7). As in ``simulate_local_ou``, demonstrating the
    continuous-time limit requires the discrete step count to grow as h -> 0
    (n_steps = T_time / h), so that the cumulative relaxation kappa_h*n_steps =
    kappa*T_time stays fixed across the mesh; a fixed raw iteration count would
    not let the recursion approach its steady state as h shrinks. Residual is a
    variance-matched Student-t_dof innovation. Returns (R, n_steps) lambda path."""
    scale = np.sqrt((dof - 2) / dof)
    alpha_h = alpha_bar * h
    kappa_h = kappa * h
    n_steps = max(1, int(round(T_time / h)))
    lam = np.full(R, lambda_init, dtype=float)
    path = np.empty((R, n_steps))
    driver = DRIVERS[rule]
    for t in range(n_steps):
        z = rng.standard_t(dof, size=R) * scale
        if rule == "mmd_rbf":
            u = driver(z, lam=lam)
        else:
            u = driver(z)
        lam = lam - kappa_h * lam + alpha_h * u
        path[:, t] = lam
    return path


def simulate_local_ou(
    rule: str, h: float, rho: float, Gamma: float, K: float, M: float, R: int,
    rng: np.random.Generator, z0: np.ndarray | None = None,
) -> dict:
    """Local Ornstein-Uhlenbeck approximation (Theorem 3.2): normalised error
    Z_u = h^{-1/4} * (lambda_t - target_t), observed over the rescaled-time
    horizon u in [0, M], stopped at |Z_u| >= K.

    The curvature-normalised update and the independent target shock each
    contribute a diffusion coefficient per unit of rescaled time: the update
    contributes rho^2*K_S/J_S^2 (its scale-driver innovation variance, after
    curvature normalisation) and the target contributes Gamma^2, so the
    combined terminal-variance benchmark over the horizon M is
    q_S^2 * M with q_S^2 = rho^2*K_S/J_S^2 + Gamma^2.

    The discrete recursion advances rescaled time in steps of size sqrt(h)
    (consistent with the h^{-1/4} error normalisation: a sum of n = M/sqrt(h)
    i.i.d. increments of variance O(sqrt(h)) has variance O(sqrt(h)*n) = O(M),
    independent of h as h -> 0), so halving h doubles the step count while
    keeping the terminal-variance benchmark fixed at q_S^2*M.
    """
    j_s, k_s = J_S[rule], K_S[rule]
    sigma_update = rho * np.sqrt(k_s) / abs(j_s)
    sigma_target = Gamma
    q2 = sigma_update ** 2 + sigma_target ** 2
    n_steps = max(1, int(round(M / np.sqrt(h))))
    du = np.sqrt(h)
    step_sd = np.sqrt(du)
    Z = np.zeros(R) if z0 is None else np.array(z0, dtype=float, copy=True)
    active = np.ones(R, dtype=bool)
    for _ in range(n_steps):
        eta_update = rng.standard_normal(R)
        eta_target = rng.standard_normal(R)
        increment = step_sd * (sigma_update * eta_update + sigma_target * eta_target)
        Z = np.where(active, Z + increment, Z)
        active &= np.abs(Z) < K
    return {"Z_terminal": Z, "q2_benchmark": q2 * M, "active": active, "n_steps": n_steps}
