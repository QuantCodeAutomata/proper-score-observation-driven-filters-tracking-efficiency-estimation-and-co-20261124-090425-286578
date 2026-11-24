"""Shared empirical pipeline: fitting, proxies, losses, and benchmarks for the
equity-index application and the SPX counterfactual (Section 7, Appendix G.1).

Public API:
  * ``fit_filter(y, rule, density, n_starts, beta, ell, nu)`` -> dict with
    omega, phi, alpha, success, nll
  * ``filtered_variance_path(y, omega, phi, alpha, rule, density, beta, ell, nu)``
  * ``qlike(variance_pred, proxy)``
  * ``parkinson_proxy(high, low)``, ``garman_klass_proxy(open_, high, low, close)``
  * ``var_exceedance_frequency(y, variance_pred, level, density, nu)``
  * ``pit_values(y, variance_pred, density, nu)``
  * ``fit_garch11(y, n_starts)``, ``fit_ewma(y, decay, init_window)``
  * ``mmd_bandwidth(returns, max_subsample)``
  * ``sample_criterion(y, omega, phi, alpha, rule, density, beta, ell, nu)``
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

from src.scoring_rules import crps_student_t


# ---------------------------------------------------------------------------
# Working-density proper scores, as a function of predictive variance v and
# observation y (Appendix G: Gaussian and Student-t_nu criteria).
# ---------------------------------------------------------------------------
def _score_gaussian(v: np.ndarray, y: np.ndarray, criterion: str, beta: float, ell: float) -> np.ndarray:
    z = y / np.sqrt(v)
    if criterion == "log":
        return 0.5 * np.log(v) + 0.5 * y ** 2 / v
    if criterion == "density_power":
        c_beta = (2 * np.pi) ** (-beta / 2)
        return c_beta * v ** (-beta / 2) * ((1 + beta) ** (-0.5) - (1 + 1 / beta) * np.exp(-beta * z ** 2 / 2))
    if criterion == "crps":
        return np.sqrt(v) * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - np.pi ** (-0.5))
    if criterion == "mmd":
        return (1 - 2 * ell / np.sqrt(ell ** 2 + v) * np.exp(-y ** 2 / (2 * (ell ** 2 + v)))
                + ell / np.sqrt(ell ** 2 + 2 * v))
    raise ValueError(criterion)


def _score_student(v: np.ndarray, y: np.ndarray, criterion: str, beta: float, ell: float, nu: int) -> np.ndarray:
    """Student-t_nu working density, predictive variance v (Appendix G.1 for CRPS)."""
    scale2 = v * (nu - 2) / nu
    z = y / np.sqrt(scale2)
    if criterion == "log":
        return 0.5 * np.log(v) + 0.5 * (nu + 1) * np.log(1 + z ** 2 / (nu - 2))
    if criterion == "crps":
        return crps_student_t(v, y, nu=nu)
    if criterion == "density_power":
        f = student_t.pdf(z, df=nu) / np.sqrt(scale2)
        c_nu_beta = _student_dp_norm_const(nu, beta)
        return -(1 + 1 / beta) * f ** beta + c_nu_beta * v ** (-beta / 2)
    if criterion == "mmd":
        return _mmd_student_quadrature(v, y, ell, nu)
    raise ValueError(criterion)


_DP_NORM_CACHE: dict = {}


def _student_dp_norm_const(nu: int, beta: float) -> float:
    """integral f_{t_nu,scale=1}^(1+beta) dx, scaled to unit variance; cached."""
    key = (nu, beta)
    if key in _DP_NORM_CACHE:
        return _DP_NORM_CACHE[key]
    from scipy import integrate

    scale2 = (nu - 2) / nu
    val, _ = integrate.quad(
        lambda x: (student_t.pdf(x / np.sqrt(scale2), df=nu) / np.sqrt(scale2)) ** (1 + beta),
        -60, 60, limit=200,
    )
    _DP_NORM_CACHE[key] = val
    return val


_GL_NODES_CACHE: dict = {}


def _mmd_student_quadrature(v: np.ndarray, y: np.ndarray, ell: float, nu: int, n_nodes: int = 256) -> np.ndarray:
    """Student-t MMD-RBF score via a 256-node Gauss-Legendre rule after a
    probability-integral-transform change of variables (Section 7, Appendix G.1)."""
    if n_nodes not in _GL_NODES_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
        u = 0.5 * (nodes + 1) * (1 - 2e-6) + 1e-6  # u in (0,1), avoid exact 0/1
        w = weights * 0.5
        _GL_NODES_CACHE[n_nodes] = (u, w)
    u, w = _GL_NODES_CACHE[n_nodes]
    scale2 = v * (nu - 2) / nu
    scale2 = np.atleast_1d(scale2)
    y = np.atleast_1d(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        s2 = scale2[i] if len(scale2) > 1 else scale2[0]
        x = student_t.ppf(u, df=nu) * np.sqrt(s2)
        kxy = np.exp(-(x - y[i]) ** 2 / (2 * ell ** 2))
        kxx = np.exp(-(x[:, None] - x[None, :]) ** 2 / (2 * ell ** 2))
        e_kxy = np.sum(w * kxy)
        e_kxx = np.sum(w[:, None] * w[None, :] * kxx)
        out[i] = 1 - 2 * e_kxy + e_kxx
    return out


def score(v: np.ndarray, y: np.ndarray, criterion: str, density: str,
          beta: float = 0.15, ell: float = 2.0, nu: int = 6) -> np.ndarray:
    """Per-observation proper score under ``density`` in {"gaussian","student"}."""
    if density == "gaussian":
        return _score_gaussian(v, y, criterion, beta, ell)
    if density == "student":
        return _score_student(v, y, criterion, beta, ell, nu)
    raise ValueError(density)


# ---------------------------------------------------------------------------
# Filter recursion and sample criterion
# ---------------------------------------------------------------------------
def filtered_variance_path(
    y: np.ndarray, omega: float, phi: float, alpha: float, rule: str,
    beta: float = 0.15, ell: float = 2.0, nu: int = 6,
) -> np.ndarray:
    """Run the recursion lambda_{t+1}=omega+phi*lambda_t+alpha*u_S(lambda_t,y_t)
    for the Gaussian-working-density drivers; returns the predictive variance
    path v_t = exp(lambda_t), aligned with y (v_t predicts y_t)."""
    from src.filters import scale_update

    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1 - phi)
    for t in range(T):
        u = scale_update(rule, lam[t], y[t], beta=beta, ell=ell, nu=nu)
        lam[t + 1] = omega + phi * lam[t] + alpha * u
    return np.exp(lam[:T])


def sample_criterion(
    y: np.ndarray, params: np.ndarray, criterion: str, density: str,
    beta: float = 0.15, ell: float = 2.0, nu: int = 6,
) -> float:
    """Q_hat_T(vartheta) = (1/T) sum S(P_{lambda_t}, y_t), recursion started
    at lambda_1 = omega/(1-phi)."""
    omega, phi, alpha = params
    if not (-0.999 < phi < 0.999) or alpha <= 0:
        return 1e10
    rule = {"log": "gaussian_log", "density_power": "density_power",
            "crps": "crps", "mmd": "mmd_rbf"}[criterion]
    v = filtered_variance_path(y, omega, phi, alpha, rule, beta=beta, ell=ell, nu=nu)
    if density == "student":
        # refit uses the Student-t predictive variance scaling for the score,
        # while the recursion itself remains on the Gaussian working density
        # (Section 7: predictive density and updating criterion are varied
        # independently in the factorial design).
        pass
    s = score(v, y, criterion, density, beta=beta, ell=ell, nu=nu)
    if not np.all(np.isfinite(s)):
        return 1e10
    return float(np.mean(s))


def fit_filter(
    y: np.ndarray, criterion: str, density: str, n_starts: int = 8,
    beta: float = 0.15, ell: float = 2.0, nu: int = 6, seed: int = 0,
) -> dict:
    """Estimate (omega, phi, alpha) by minimising the sample criterion with
    L-BFGS-B over ``n_starts`` fixed starting values (Section 7)."""
    rng = np.random.default_rng(seed)
    starts = []
    phis = np.linspace(0.80, 0.98, n_starts)
    for i in range(n_starts):
        phi0 = phis[i]
        omega0 = np.log(np.var(y)) * (1 - phi0)
        alpha0 = 0.05 + 0.02 * (i % 3)
        starts.append((omega0, phi0, alpha0))

    best = None
    for x0 in starts:
        res = minimize(
            lambda p: sample_criterion(y, p, criterion, density, beta, ell, nu),
            x0=np.array(x0), method="L-BFGS-B",
            bounds=[(-5, 5), (-0.999, 0.999), (1e-4, 2.0)],
            options={"maxiter": 200},
        )
        if best is None or res.fun < best.fun:
            best = res
    omega, phi, alpha = best.x
    return {"omega": float(omega), "phi": float(phi), "alpha": float(alpha),
            "nll": float(best.fun), "success": bool(best.success)}


# ---------------------------------------------------------------------------
# Loss functions and proxies
# ---------------------------------------------------------------------------
def qlike(variance_pred: np.ndarray, proxy: np.ndarray) -> float:
    """QLIKE^fd(q,v) = mean(log(v) + q/v) (Section 7)."""
    return float(np.mean(np.log(variance_pred) + proxy / variance_pred))


def parkinson_proxy(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) range-based variance proxy, in squared-log-return units."""
    return (np.log(high / low) ** 2) / (4 * np.log(2))


def garman_klass_proxy(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Garman-Klass (1980) OHLC variance proxy."""
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    return 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2


def var_exceedance_frequency(y: np.ndarray, variance_pred: np.ndarray, level: float, density: str, nu: int = 6) -> float:
    """hat_p = mean(1{y_t < VaR_t(p)}), VaR_t(p) = q(p)*sqrt(v_t) (Section 7)."""
    if density == "gaussian":
        q = norm.ppf(level)
    else:
        scale = np.sqrt((nu - 2) / nu)
        q = student_t.ppf(level, df=nu) * scale
    var_t = q * np.sqrt(variance_pred)
    return float(np.mean(y < var_t))


def pit_values(y: np.ndarray, variance_pred: np.ndarray, density: str, nu: int = 6) -> np.ndarray:
    """u_t = F(y_t / sqrt(v_t)) under the fitted working density (Section 7)."""
    z = y / np.sqrt(variance_pred)
    if density == "gaussian":
        return norm.cdf(z)
    scale = np.sqrt((nu - 2) / nu)
    return student_t.cdf(z / scale, df=nu)


def mmd_bandwidth(returns: np.ndarray, max_subsample: int = 1024) -> float:
    """Median pairwise absolute distance on a deterministic evenly spaced
    subsample of at most ``max_subsample`` training returns (Section 7)."""
    n = len(returns)
    if n > max_subsample:
        idx = np.linspace(0, n - 1, max_subsample).round().astype(int)
        sub = returns[idx]
    else:
        sub = returns
    diffs = np.abs(sub[:, None] - sub[None, :])
    iu = np.triu_indices(len(sub), k=1)
    return float(np.median(diffs[iu]))


# ---------------------------------------------------------------------------
# Benchmarks: GARCH(1,1) and EWMA
# ---------------------------------------------------------------------------
def fit_garch11(y: np.ndarray, n_starts: int = 2, a_max: float = 0.3, b_max: float = 0.98,
                sum_max: float = 0.999, density: str = "gaussian", nu: int = 6) -> dict:
    """Gaussian or Student-t GARCH(1,1), fit by Gaussian / Student quasi-likelihood."""
    var_y = np.var(y)

    def neg_loglik(p):
        a0, a1, b1 = p
        if a1 < 0 or b1 < 0 or a1 > a_max or b1 > b_max or a1 + b1 > sum_max or a0 <= 0:
            return 1e10
        T = len(y)
        v = np.empty(T)
        v[0] = var_y
        for t in range(1, T):
            v[t] = a0 + a1 * y[t - 1] ** 2 + b1 * v[t - 1]
        if density == "gaussian":
            ll = -0.5 * np.log(v) - 0.5 * y ** 2 / v
        else:
            scale2 = v * (nu - 2) / nu
            ll = student_t.logpdf(y / np.sqrt(scale2), df=nu) - 0.5 * np.log(scale2)
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -float(np.mean(ll))

    starts = [(var_y * 0.05, 0.05, 0.90), (var_y * 0.1, 0.1, 0.80)][:n_starts]
    best = None
    for x0 in starts:
        res = minimize(neg_loglik, x0=np.array(x0), method="Nelder-Mead",
                        options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    a0, a1, b1 = best.x
    T = len(y)
    v = np.empty(T)
    v[0] = var_y
    for t in range(1, T):
        v[t] = a0 + a1 * y[t - 1] ** 2 + b1 * v[t - 1]
    return {"a0": float(a0), "a1": float(a1), "b1": float(b1), "variance_path": v}


def fit_ewma(y: np.ndarray, decay: float = 0.94, init_window: int = 50) -> np.ndarray:
    """EWMA variance recursion v_t = decay*v_{t-1} + (1-decay)*y_{t-1}^2,
    initialised with the sample variance of the first ``init_window`` returns."""
    T = len(y)
    v = np.empty(T)
    v[0] = np.var(y[:init_window])
    for t in range(1, T):
        v[t] = decay * v[t - 1] + (1 - decay) * y[t - 1] ** 2
    return v
