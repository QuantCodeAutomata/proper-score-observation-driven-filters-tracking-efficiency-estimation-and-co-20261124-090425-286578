"""Population tracking efficiency for scale-equivariant scoring-rule cells
(Section 3.3-3.4, Corollary 3.7, Table 3).

For an innovation law F (variance-standardised to unit variance) and a
scale-equivariant scoring rule S, define:

  * the target shift delta_S(F), solving E_F[psi_S(eps, delta)] = 0;
  * the curvature J_F = R''_F(delta_S(F)) and innovation variance
    K_F = E_F[psi_S(eps, delta_S(F))^2];
  * kappa_S(F) = K_F / J_F^2;
  * the Fisher information for the log-variance score, I_F;
  * the tracking efficiency eff_S(F) = 1 / (I_F * kappa_S(F)) (Corollary 3.7(ii));
  * the minimal frozen tracking-risk multiplier eff_S(F)^{-1/2}.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.optimize import brentq
from scipy.stats import norm, t as student_t

QUAD_LIMIT = 200


def _innovation_density(law: str):
    """Variance-standardised density and log-derivative score function."""
    if law == "gaussian":
        return norm.pdf, None
    dof = int(law.split("_")[1])
    scale = np.sqrt((dof - 2) / dof)  # variance standardisation: Var(t_nu/scale)=1

    def pdf(x):
        return student_t.pdf(x / scale, df=dof) / scale

    return pdf, dof


def fisher_information(law: str) -> float:
    """Fisher information for the log-variance (half the scale) score.

    I_F = E_F[s_F^2], s_F = -1/2 - eps*f'(eps)/(2 f(eps)).
    Gaussian: I_F = 1/2. Student-t_nu (variance standardised): I_F = nu/(2(nu+3)).
    """
    if law == "gaussian":
        return 0.5
    dof = int(law.split("_")[1])
    return dof / (2 * (dof + 3))


def _psi_scale_equivariant(rule: str, eps: np.ndarray, delta: float, beta: float, nu_work: int) -> np.ndarray:
    """Raw innovation psi_S(eps, delta) for a scale-equivariant cell at shift delta.

    ``eps`` is the (unit-variance) innovation; delta is log(v) in the working
    density P_{delta,theta}. Formulas from Appendix G (Scores and updates).
    """
    z = eps * np.exp(-delta / 2)
    if rule == "gaussian_log":
        return 0.5 * (z ** 2 - 1.0)
    if rule == "student_log":
        return 0.5 * ((nu_work + 1) * z ** 2 / ((nu_work - 2) + z ** 2) - 1.0)
    if rule == "density_power":
        c_beta = (2 * np.pi) ** (-beta / 2)
        kappa_beta = beta * c_beta * (1 + beta) ** (-0.5)
        g = (1 + beta) * c_beta * np.exp(-beta * z ** 2 / 2) * (z ** 2 - 1.0) + kappa_beta
        return 0.5 * g * np.exp(-beta * delta / 2)
    if rule == "crps":
        g = np.pi ** (-0.5) - np.sqrt(2 / np.pi) * np.exp(-z ** 2 / 2)
        return 0.5 * g * np.exp(delta / 2)
    if rule == "student_crps":
        from src.scoring_rules import crps_student_t
        v = np.exp(delta)
        scale2 = v * (nu_work - 2) / nu_work
        h = 1e-5
        s_plus = crps_student_t(scale2 * np.exp(h), eps, nu=nu_work)
        s_minus = crps_student_t(scale2 * np.exp(-h), eps, nu=nu_work)
        return -(s_plus - s_minus) / (2 * h)
    if rule == "student_density_power":
        v = np.exp(delta)
        h = 1e-5
        s_plus = _student_density_power_score(v * np.exp(h), eps, beta, nu_work)
        s_minus = _student_density_power_score(v * np.exp(-h), eps, beta, nu_work)
        return -(s_plus - s_minus) / (2 * h)
    raise ValueError(f"Unknown rule {rule}")


def _student_density_power_score(v: np.ndarray, y: np.ndarray, beta: float, nu: int) -> np.ndarray:
    """Density-power score with a Student-t_nu working density, variance v."""
    scale2 = v * (nu - 2) / nu
    z2 = y ** 2 / scale2
    f = student_t.pdf(y / np.sqrt(scale2), df=nu) / np.sqrt(scale2)
    norm_term = integrate.quad(
        lambda x: (student_t.pdf(x / np.sqrt(scale2), df=nu) / np.sqrt(scale2)) ** (1 + beta),
        -40 * np.sqrt(scale2), 40 * np.sqrt(scale2), limit=QUAD_LIMIT,
    )[0]
    return -(1 + 1 / beta) * f ** beta + norm_term


def _expectation_over_law(func, law: str) -> float:
    """E_F[func(eps)]. Student-t innovation laws with low degrees of freedom have
    power-law tails (pdf ~ |x|^{-(nu+1)}), so a finite truncation such as +/-40
    discards a non-negligible share of E[eps^4] as nu -> 4 (observed bias up to
    ~8% at nu=5 against the closed form (nu-1)/(2(nu-4)) for the log-score
    update variance). Integrating to +/-inf lets scipy's adaptive quadrature
    handle the tail via its own variable transform instead of a fixed cutoff."""
    pdf, _ = _innovation_density(law)
    val, _ = integrate.quad(lambda x: func(x) * pdf(x), -np.inf, np.inf, limit=QUAD_LIMIT)
    return val


def target_shift(rule: str, law: str, beta: float = 0.15, nu_work: int = 6) -> float:
    """Solve E_F[psi_S(eps, delta)] = 0 for delta (Corollary 3.7)."""
    if rule == "gaussian_log":
        return 0.0  # Bartlett identity: the log score target is always the latent log-variance

    def objective(delta: float) -> float:
        return _expectation_over_law(lambda e: _psi_scale_equivariant(rule, e, delta, beta, nu_work), law)

    lo, hi = -3.0, 3.0
    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo * f_hi > 0:
        # widen bracket if needed
        lo, hi = -6.0, 6.0
        f_lo, f_hi = objective(lo), objective(hi)
    return brentq(objective, lo, hi, xtol=1e-8)


def curvature_and_variance(rule: str, law: str, delta: float, beta: float = 0.15, nu_work: int = 6) -> tuple[float, float]:
    """J_F = d/d(delta) (-E_F[psi_S]) at delta (numerical derivative);
    K_F = E_F[psi_S(eps,delta)^2]."""
    h = 1e-4

    def mean_psi(d):
        return _expectation_over_law(lambda e: _psi_scale_equivariant(rule, e, d, beta, nu_work), law)

    J_F = -(mean_psi(delta + h) - mean_psi(delta - h)) / (2 * h)
    K_F = _expectation_over_law(lambda e: _psi_scale_equivariant(rule, e, delta, beta, nu_work) ** 2, law)
    return J_F, K_F


def tracking_efficiency(rule: str, law: str, beta: float = 0.15, nu_work: int = 6) -> dict:
    """eff_S(F) = 1/(I_F * kappa_S(F)); returns a dict of all intermediate
    quantities for a given cell (rule) and innovation law F."""
    I_F = fisher_information(law)
    if rule == "gaussian_log" and law.startswith("t_"):
        dof = int(law.split("_")[1])
        if dof <= 4:
            return {"delta": 0.0, "J_F": np.nan, "K_F": np.inf, "kappa": np.inf,
                     "I_F": I_F, "efficiency": 0.0, "risk_multiplier": np.inf}
    delta = target_shift(rule, law, beta, nu_work)
    J_F, K_F = curvature_and_variance(rule, law, delta, beta, nu_work)
    kappa = K_F / J_F ** 2
    efficiency = 1.0 / (I_F * kappa)
    risk_multiplier = efficiency ** (-0.5)
    return {"delta": delta, "J_F": J_F, "K_F": K_F, "kappa": kappa,
            "I_F": I_F, "efficiency": efficiency, "risk_multiplier": risk_multiplier}


def gaussian_log_efficiency_student(dof: int) -> float:
    """Closed-form Gaussian-log efficiency under t_nu innovations, Section 6.4:
    eff_log(t_nu) = (nu-4)(nu+3) / (nu (nu-1)) for nu > 4, else 0."""
    if dof <= 4:
        return 0.0
    return (dof - 4) * (dof + 3) / (dof * (dof - 1))
