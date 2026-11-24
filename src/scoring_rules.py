"""Proper scoring rules for scale (log-variance) filtering.

Implements the Gaussian, density-power, CRPS, MMD-RBF and Student-t
scoring rules of Appendix G ("Scores and updates", Table G.5). The
working density is the scale family p(y | lambda) with variance
v = exp(lambda) and standardised residual z = y * exp(-lambda/2).

For every rule we expose:
  * ``score``        - the per-observation proper score S(P_lambda, y);
  * ``psi``           - the raw innovation psi_S = -d/dlambda S (Eq. 2.6);
  * ``driver``         - the curvature-normalised recursion driver g_S(z),
                        i.e. the "update" u_S entering the autoregressive
                        recursion lambda_{t+1} = omega + phi*lambda_t + alpha*u_S;
  * ``J``, ``K``       - the reference-state curvature and innovation
                        variance of Table G.5 (exact values reported in
                        the paper, reproduced here as module constants).

Appendix G shows that, at the Gaussian reference state (lambda = 0),
g_S(z) = 2 * psi_S(z) for every rule, and that the curvature scaling
G_S(lambda) (G_log = 2, G_beta(lambda) = 2*exp(beta*lambda/2),
G_CRPS(lambda) = 2*exp(-lambda/2), G_MMD = 2) makes the recursion update
u_S(lambda, z) = g_S(z) depend on the current state only through the
standardised residual z for the Gaussian log, density-power and CRPS
rules; the MMD-RBF update keeps an explicit lambda-dependence because
its bandwidth ell is fixed in data units rather than standardised.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import norm, t as student_t

# ---------------------------------------------------------------------------
# Catalogue constants (paper Appendix G, "Tuning constants" and Table G.5)
# ---------------------------------------------------------------------------
BETA_DP = 0.15          # density-power exponent (paper default)
ELL_MMD = 2.0            # MMD-RBF bandwidth in standardised units (paper default)
NU_STUDENT = 6           # Student-t benchmark degrees of freedom (paper default)

C_BETA = (2 * np.pi) ** (-BETA_DP / 2)                       # == 0.871
KAPPA_BETA = BETA_DP * C_BETA * (1 + BETA_DP) ** (-0.5)       # == 0.122
D0_MMD = ELL_MMD ** 2 + 1
B_ELL = 2 * ELL_MMD * (ELL_MMD ** 2 + 2) ** (-1.5)            # == 0.272

# Reference-state (lambda = 0, v0 = 1) curvature J_S and innovation variance
# K_S, reported to three decimals in Table G.5 of Appendix G.
J_S = {
    "gaussian_log": 0.500000,
    "density_power": 0.357206,
    "crps": 0.070524,
    "mmd_rbf": 0.034,
}
K_S = {
    "gaussian_log": 0.500000,
    "density_power": 0.268493,
    "crps": 0.012311,
    "mmd_rbf": 0.002,
}

RULE_NAMES = ("gaussian_log", "density_power", "crps", "mmd_rbf")


def a0_mmd(z: np.ndarray) -> np.ndarray:
    """a_0(z) = ell * D0^{-1/2} * exp(-z^2 / (2 D0)) (reference-state MMD kernel term)."""
    return ELL_MMD * D0_MMD ** (-0.5) * np.exp(-z ** 2 / (2 * D0_MMD))


# ---------------------------------------------------------------------------
# Curvature-normalised recursion drivers g_S(z) (Table G.5 driver column)
# ---------------------------------------------------------------------------
def driver_gaussian_log(z: np.ndarray) -> np.ndarray:
    """g_S(z) = z^2 - 1 (unbounded)."""
    return z ** 2 - 1.0


def driver_density_power(z: np.ndarray, beta: float = BETA_DP) -> np.ndarray:
    """g_S(z) = (1+beta) c_beta exp(-beta z^2/2) (z^2 - 1) + kappa_beta."""
    c_beta = (2 * np.pi) ** (-beta / 2)
    kappa_beta = beta * c_beta * (1 + beta) ** (-0.5)
    return (1 + beta) * c_beta * np.exp(-beta * z ** 2 / 2) * (z ** 2 - 1.0) + kappa_beta


def driver_crps(z: np.ndarray) -> np.ndarray:
    """g_S(z) = pi^{-1/2} - sqrt(2/pi) exp(-z^2/2)."""
    return np.pi ** (-0.5) - np.sqrt(2 / np.pi) * np.exp(-z ** 2 / 2)


def driver_mmd_rbf(z: np.ndarray, lam: float | np.ndarray = 0.0, ell: float = ELL_MMD) -> np.ndarray:
    """u_S(lambda, z); reduces to the reference-state driver at lambda = 0.

    r_lambda = ell * exp(-lambda/2), D_lambda = r_lambda^2 + 1.
    """
    r = ell * np.exp(-np.asarray(lam) / 2)
    D = r ** 2 + 1
    term1 = 2 * (r / np.sqrt(D)) * np.exp(-z ** 2 / (2 * D)) * (z ** 2 / D ** 2 - 1 / D)
    term2 = 2 * r / (r ** 2 + 2) ** 1.5
    return term1 + term2


def driver_student_log(z: np.ndarray, nu: float = NU_STUDENT) -> np.ndarray:
    """g_S(z) = (nu+1) z^2 / ((nu-2) + z^2) - 1."""
    return (nu + 1) * z ** 2 / ((nu - 2) + z ** 2) - 1.0


DRIVERS = {
    "gaussian_log": driver_gaussian_log,
    "density_power": driver_density_power,
    "crps": driver_crps,
    "mmd_rbf": driver_mmd_rbf,
    "student_log": driver_student_log,
}


# ---------------------------------------------------------------------------
# Raw proper scores S(lambda, y) and raw innovations psi_S = -d/dlambda S
# (used for root-finding of pseudo-true targets in exp_2 and exp_4)
# ---------------------------------------------------------------------------
def score_gaussian_log(lam: float, y: np.ndarray) -> np.ndarray:
    v = np.exp(lam)
    return 0.5 * lam + 0.5 * y ** 2 / v


def psi_gaussian_log(lam: float, y: np.ndarray) -> np.ndarray:
    z = y * np.exp(-lam / 2)
    return 0.5 * (z ** 2 - 1.0)


def score_density_power(lam: float, y: np.ndarray, beta: float = BETA_DP) -> np.ndarray:
    v = np.exp(lam)
    z = y / np.sqrt(v)
    c_beta = (2 * np.pi) ** (-beta / 2)
    return c_beta * v ** (-beta / 2) * ((1 + beta) ** (-0.5) - (1 + 1 / beta) * np.exp(-beta * z ** 2 / 2))


def psi_density_power(lam: float, y: np.ndarray, beta: float = BETA_DP) -> np.ndarray:
    z = y * np.exp(-lam / 2)
    return 0.5 * driver_density_power(z, beta) * np.exp(-beta * lam / 2)


def score_crps(lam: float, y: np.ndarray) -> np.ndarray:
    v = np.exp(lam)
    z = y / np.sqrt(v)
    return np.sqrt(v) * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - np.pi ** (-0.5))


def psi_crps(lam: float, y: np.ndarray) -> np.ndarray:
    z = y * np.exp(-lam / 2)
    return 0.5 * driver_crps(z) * np.exp(lam / 2)


def score_mmd_rbf(lam: float, y: np.ndarray, ell: float = ELL_MMD) -> np.ndarray:
    v = np.exp(lam)
    return 1 - 2 * ell / np.sqrt(ell ** 2 + v) * np.exp(-y ** 2 / (2 * (ell ** 2 + v))) + ell / np.sqrt(ell ** 2 + 2 * v)


def psi_mmd_rbf(lam: float, y: np.ndarray, ell: float = ELL_MMD) -> np.ndarray:
    z = y * np.exp(-lam / 2)
    return 0.5 * driver_mmd_rbf(z, lam, ell)


SCORES = {
    "gaussian_log": score_gaussian_log,
    "density_power": score_density_power,
    "crps": score_crps,
    "mmd_rbf": score_mmd_rbf,
}
PSIS = {
    "gaussian_log": psi_gaussian_log,
    "density_power": psi_density_power,
    "crps": psi_crps,
    "mmd_rbf": psi_mmd_rbf,
}


def curvature_normalised_driver(rule: str, z: np.ndarray) -> np.ndarray:
    """g_S^circ(z) = g_S(z) / J_S, the curvature-normalised driver of Section 6.1."""
    if rule == "mmd_rbf":
        g = driver_mmd_rbf(z, lam=0.0)
    else:
        g = DRIVERS[rule](z)
    return g / J_S[rule]


# ---------------------------------------------------------------------------
# CRPS for a Student-t working density (Appendix G, Eq. G.1, Jordan et al. 2019)
# ---------------------------------------------------------------------------
def crps_student_t(v: np.ndarray, y: np.ndarray, nu: float = NU_STUDENT) -> np.ndarray:
    """CRPS_{t_nu}(v, y) per Eq. (G.1); v is the predictive variance."""
    from scipy.special import beta as beta_fn

    z = y / np.sqrt(v)
    f_nu = student_t.pdf(z, df=nu)
    F_nu = student_t.cdf(z, df=nu)
    d_nu = nu - 2
    c_nu = (2 * np.sqrt(nu - 2) / (nu - 1)) * beta_fn(0.5, nu - 0.5) / beta_fn(0.5, nu / 2) ** 2
    return np.sqrt(v) * (z * (2 * F_nu - 1) + 2 * f_nu * (d_nu + z ** 2) / (nu - 1) - c_nu)
