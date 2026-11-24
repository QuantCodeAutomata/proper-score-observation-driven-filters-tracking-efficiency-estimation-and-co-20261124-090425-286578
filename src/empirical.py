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


def _score_student(v: np.ndarray, y: np.ndarray, criterion: str, beta: float, ell: float, nu: int,
                    n_nodes: int = 64) -> np.ndarray:
    """Student-t_nu working density, predictive variance v (Appendix G.1 for CRPS)."""
    scale2 = v * (nu - 2) / nu
    z = y / np.sqrt(scale2)
    if criterion == "log":
        return 0.5 * np.log(v) + 0.5 * (nu + 1) * np.log(1 + z ** 2 / (nu - 2))
    if criterion == "crps":
        return crps_student_t(scale2, y, nu=nu)
    if criterion == "density_power":
        f = _t_pdf_fast(z, nu) / np.sqrt(scale2)
        c_nu_beta = _student_dp_norm_const(nu, beta)
        return -(1 + 1 / beta) * f ** beta + c_nu_beta * v ** (-beta / 2)
    if criterion == "mmd":
        return _mmd_student_quadrature(v, y, ell, nu, n_nodes=n_nodes)
    raise ValueError(criterion)


_T_PDF_CONST_CACHE: dict = {}


def _t_pdf_const(nu: int) -> float:
    """Normalising constant of the standard Student-t_nu density, cached per
    nu to avoid scipy.stats dispatch overhead inside the recursion loop."""
    if nu not in _T_PDF_CONST_CACHE:
        from scipy.special import gammaln

        _T_PDF_CONST_CACHE[nu] = float(np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(nu * np.pi))
    return _T_PDF_CONST_CACHE[nu]


def _t_pdf_fast(x: np.ndarray, nu: int) -> np.ndarray:
    """Standard Student-t_nu density evaluated by direct formula (no
    scipy.stats object overhead); used in the hot recursion loop."""
    return _t_pdf_const(nu) * (1.0 + x * x / nu) ** (-(nu + 1) / 2)


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


_T_QUANTILE_CACHE: dict = {}


def _mmd_student_quadrature(v: np.ndarray, y: np.ndarray, ell: float, nu: int, n_nodes: int = 256,
                             chunk: int = 200) -> np.ndarray:
    """Student-t MMD-RBF score via an n_nodes-node Gauss-Legendre rule after a
    probability-integral-transform change of variables (Section 7, Appendix
    G.1). Vectorised and processed in chunks of observations to keep peak
    memory bounded, since the quadrature uses one n_nodes x n_nodes kernel
    matrix per observation. The paper uses a 256-node rule for the reported
    scores; a smaller ``n_nodes`` is used inside the optimiser for tractable
    fitting time and gives the same score at tighter tolerance (Appendix G.1
    reports convergence already at a few dozen nodes)."""
    if n_nodes not in _GL_NODES_CACHE:
        nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
        u = 0.5 * (nodes + 1) * (1 - 2e-6) + 1e-6  # u in (0,1), avoid exact 0/1
        w = weights * 0.5
        _GL_NODES_CACHE[n_nodes] = (u, w)
    u, w = _GL_NODES_CACHE[n_nodes]
    key = (n_nodes, nu)
    if key not in _T_QUANTILE_CACHE:
        _T_QUANTILE_CACHE[key] = student_t.ppf(u, df=nu)  # quantiles are state-independent
    tq = _T_QUANTILE_CACHE[key]

    scale2 = np.atleast_1d(v * (nu - 2) / nu).astype(float)
    y = np.atleast_1d(y).astype(float)
    if len(scale2) == 1 and len(y) > 1:
        scale2 = np.full(len(y), scale2[0])
    sigma = np.sqrt(scale2)
    out = np.empty(len(y))
    for start in range(0, len(y), chunk):
        sl = slice(start, start + chunk)
        x = tq[None, :] * sigma[sl, None]            # (n_chunk, n_nodes)
        kxy = np.exp(-(x - y[sl, None]) ** 2 / (2 * ell ** 2))
        e_kxy = kxy @ w
        diff = x[:, :, None] - x[:, None, :]           # (n_chunk, n_nodes, n_nodes)
        kxx = np.exp(-diff ** 2 / (2 * ell ** 2))
        e_kxx = np.einsum("i,nij,j->n", w, kxx, w)
        out[sl] = 1 - 2 * e_kxy + e_kxx
    return out


def score(v: np.ndarray, y: np.ndarray, criterion: str, density: str,
          beta: float = 0.15, ell: float = 2.0, nu: int = 6, mmd_nodes: int = 64) -> np.ndarray:
    """Per-observation proper score under ``density`` in {"gaussian","student"}.
    ``mmd_nodes`` controls the Gauss-Legendre quadrature order used only by
    the Student-t MMD-RBF cell; pass 256 for the paper's reported final
    scores (Appendix G.1) and a smaller value for the optimiser's objective."""
    if density == "gaussian":
        return _score_gaussian(v, y, criterion, beta, ell)
    if density == "student":
        return _score_student(v, y, criterion, beta, ell, nu, n_nodes=mmd_nodes)
    raise ValueError(density)


# ---------------------------------------------------------------------------
# Filter recursion and sample criterion
# ---------------------------------------------------------------------------
def _recursion_driver(
    lam: float, y: float, criterion: str, density: str,
    beta: float, ell: float, nu: int, mmd_nodes: int,
) -> float:
    """Raw innovation u_S(lambda,y) = -dS/dlambda driving the recursion for
    cell (criterion, density); closed form where Appendix G gives one
    (Gaussian log/density-power/CRPS/MMD via ``src.filters.scale_update``,
    Student-t6 log via the Student-t influence function), else a central
    finite difference of the cell's own ``score`` in lambda = log v (Eq. 2.6,
    psi_S = -partial_lambda S), so every (density, criterion) cell of the
    Section 7 factorial updates its own filtered state rather than a shared
    Gaussian recursion. The Student-t6 MMD cell uses a reduced-node
    Gauss-Legendre quadrature (``mmd_nodes``) inside this hot loop for
    tractable optimisation time; final reported scores use the full 256-node
    rule via ``score()``."""
    if density == "gaussian":
        from src.filters import scale_update

        rule = {"log": "gaussian_log", "density_power": "density_power",
                "crps": "crps", "mmd": "mmd_rbf"}[criterion]
        return scale_update(rule, lam, y, beta=beta, ell=ell, nu=nu)
    if density == "student" and criterion == "log":
        from src.scoring_rules import driver_student_log

        scale2 = (nu - 2) / nu
        z = y / np.sqrt(np.exp(lam) * scale2)
        return 0.5 * driver_student_log(z, nu=nu)
    # Numerical innovation psi_S = -dS/dlambda for cells without a closed
    # form wired in above (Student-t6 density-power, CRPS, MMD).
    h = 1e-5
    if criterion == "mmd":
        s_p = _score_student(np.exp(lam + h), np.array([y]), "mmd", beta, ell, nu, n_nodes=mmd_nodes)[0]
        s_m = _score_student(np.exp(lam - h), np.array([y]), "mmd", beta, ell, nu, n_nodes=mmd_nodes)[0]
    else:
        s_p = _score_student(np.exp(lam + h), np.array([y]), criterion, beta, ell, nu)[0]
        s_m = _score_student(np.exp(lam - h), np.array([y]), criterion, beta, ell, nu)[0]
    return -(s_p - s_m) / (2 * h)


def filtered_variance_path(
    y: np.ndarray, omega: float, phi: float, alpha: float, criterion: str,
    density: str = "gaussian", beta: float = 0.15, ell: float = 2.0, nu: int = 6,
    mmd_nodes: int = 64,
) -> np.ndarray:
    """Run the recursion lambda_{t+1}=omega+phi*lambda_t+alpha*u_S(lambda_t,y_t)
    for the (criterion, density) cell; returns the predictive variance path
    v_t = exp(lambda_t), aligned with y (v_t predicts y_t)."""
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = omega / (1 - phi)
    for t in range(T):
        u = _recursion_driver(lam[t], y[t], criterion, density, beta, ell, nu, mmd_nodes)
        lam[t + 1] = omega + phi * lam[t] + alpha * u
    return np.exp(lam[:T])


def sample_criterion(
    y: np.ndarray, params: np.ndarray, criterion: str, density: str,
    beta: float = 0.15, ell: float = 2.0, nu: int = 6, mmd_nodes: int = 64,
) -> float:
    """Q_hat_T(vartheta) = (1/T) sum S(P_{lambda_t}, y_t), recursion started
    at lambda_1 = omega/(1-phi)."""
    omega, phi, alpha = params
    if not (-0.999 < phi < 0.999) or alpha <= 0:
        return 1e10
    v = filtered_variance_path(y, omega, phi, alpha, criterion, density=density,
                                beta=beta, ell=ell, nu=nu, mmd_nodes=mmd_nodes)
    s = score(v, y, criterion, density, beta=beta, ell=ell, nu=nu, mmd_nodes=mmd_nodes)
    if not np.all(np.isfinite(s)):
        return 1e10
    return float(np.mean(s))


def fit_filter(
    y: np.ndarray, criterion: str, density: str, n_starts: int = 8,
    beta: float = 0.15, ell: float = 2.0, nu: int = 6, seed: int = 0,
    mmd_nodes: int = 64,
) -> dict:
    """Estimate (omega, phi, alpha) by minimising the sample criterion with
    L-BFGS-B over ``n_starts`` fixed starting values (Section 7). ``mmd_nodes``
    sets the Gauss-Legendre order used while optimising the Student-t MMD-RBF
    cell; the full 256-node rule (Appendix G.1) is reserved for the final
    score evaluation via ``score()`` since it is too costly inside the
    optimiser's finite-difference gradient loop."""
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
            lambda p: sample_criterion(y, p, criterion, density, beta, ell, nu, mmd_nodes=mmd_nodes),
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
