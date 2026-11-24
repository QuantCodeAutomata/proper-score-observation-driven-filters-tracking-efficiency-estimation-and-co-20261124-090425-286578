"""exp_9: Gaussian location filtering, Nile River annual flow illustration
(Appendix G.7).

Fits Gaussian location filters (y_t = mu_t + sigma*eps_t) with Gaussian log,
CRPS, and density-power scoring rules on the Nile annual flow series
(1871-1970), training on 1871-1944 and freezing parameters thereafter. Two
panels are produced: (a) the frozen-fit filtered location path across the
documented 1899 downward level shift (Cobb, 1978), and (b) the displacement
induced by a common +6*sigma_log outlier injected at 1945, relative to each
method's clean path.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.filters import filter_location_path

RULES = ["gaussian_log", "crps", "density_power"]
RULE_LABELS = {"gaussian_log": "Gaussian log", "crps": "CRPS", "density_power": "density-power"}


def _fit_location_filter(y: np.ndarray, rule: str, n_starts: int, beta: float, seed: int = 0) -> dict:
    """Fit (omega, phi, alpha, sigma) for the Gaussian location model by
    minimising the in-sample Gaussian log-likelihood proxy criterion: the
    location recursion's own rule drives the state, while sigma and the
    Gaussian log-likelihood of the implied residuals are used to score the
    fit, since the three rules differ only in how omega/phi/alpha are
    updated, not in the reported residual density (Appendix G.7)."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    phis = np.linspace(0.3, 0.95, n_starts)
    starts = []
    for i in range(n_starts):
        phi0 = phis[i]
        mu0 = float(np.mean(y))
        omega0 = mu0 * (1 - phi0)
        alpha0 = 0.3 + 0.1 * (i % 3)
        sigma0 = float(np.std(y - mu0))
        starts.append((omega0, phi0, alpha0, sigma0))

    def neg_ll(p):
        omega, phi, alpha, sigma = p
        if not (-0.999 < phi < 0.999) or alpha <= 0 or sigma <= 0:
            return 1e10
        mu_path = filter_location_path(y, sigma, omega, phi, alpha, rule, mu_init=float(np.mean(y)), beta=beta)
        resid = y - np.concatenate(([mu_path[0]], mu_path[:-1]))
        ll = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * resid ** 2 / sigma ** 2
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -float(np.mean(ll))

    best = None
    for x0 in starts:
        res = minimize(neg_ll, x0=np.array(x0), method="Nelder-Mead",
                        options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    omega, phi, alpha, sigma = best.x
    return {"omega": float(omega), "phi": float(phi), "alpha": float(alpha), "sigma": float(sigma)}


def run() -> dict:
    nile = sm.datasets.nile.load_pandas().data
    years = nile["year"].astype(int).to_numpy()
    flow = nile["volume"].astype(float).to_numpy()
    assert len(flow) == 100, "Nile series must have 100 annual observations (1871-1970)"

    train_mask = years <= config.EXP9_TRAIN_END_YEAR
    y_train = flow[train_mask]

    fits = {}
    clean_paths = {}
    for rule in RULES:
        fit = _fit_location_filter(y_train, rule, config.EXP9_LBFGS_N_STARTS, config.EXP9_BETA_DP)
        fits[rule] = fit
        mu_init = float(np.mean(y_train))
        clean = filter_location_path(flow, fit["sigma"], fit["omega"], fit["phi"], fit["alpha"], rule,
                                      mu_init=mu_init, beta=config.EXP9_BETA_DP)
        clean_paths[rule] = clean

    sigma_log = fits["gaussian_log"]["sigma"]
    outlier_idx = int(np.where(years == config.EXP9_OUTLIER_YEAR)[0][0])
    flow_pert = flow.copy()
    flow_pert[outlier_idx] = flow[outlier_idx] + config.EXP9_OUTLIER_SIGMA_MULT * sigma_log

    pert_paths = {}
    displacement = {}
    peak_1946 = {}
    half_decay_year = {}
    for rule in RULES:
        fit = fits[rule]
        mu_init = float(np.mean(y_train))
        pert = filter_location_path(flow_pert, fit["sigma"], fit["omega"], fit["phi"], fit["alpha"], rule,
                                     mu_init=mu_init, beta=config.EXP9_BETA_DP)
        pert_paths[rule] = pert
        disp = pert - clean_paths[rule]
        displacement[rule] = disp
        idx_1946 = int(np.where(years == 1946)[0][0])
        peak = disp[idx_1946:]
        peak_val = float(np.max(np.abs(peak)))
        peak_1946[rule] = peak_val / sigma_log
        # First year after 1946 where |displacement| drops below half the 1946 peak.
        post = np.abs(disp[idx_1946:])
        half = 0.5 * np.abs(disp[idx_1946])
        below = np.where(post < half)[0]
        half_decay_year[rule] = int(years[idx_1946 + below[0]]) if len(below) else None

    # --- Panel (a): clean filtered location paths across the 1899 shift ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    ax = axes[0]
    ax.plot(years, flow, color="gray", alpha=0.5, label="Nile flow")
    for rule in RULES:
        ax.plot(years, clean_paths[rule], label=RULE_LABELS[rule])
    ax.axvline(1899, color="black", linestyle="--", linewidth=0.8, label="1899 level shift")
    ax.axvline(1944, color="red", linestyle=":", linewidth=0.8, label="freeze (1944)")
    ax.set_title("Panel (a): filtered location across the 1899 downward level shift")
    ax.set_xlabel("year")
    ax.set_ylabel("flow (10^8 m^3)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for rule in RULES:
        ax.plot(years, displacement[rule] / sigma_log, label=RULE_LABELS[rule])
    ax.axvline(1945, color="black", linestyle="--", linewidth=0.8, label="injected outlier (1945)")
    ax.set_title("Panel (b): displacement from a common +6*sigma_log outlier at 1945")
    ax.set_xlabel("year")
    ax.set_ylabel("displacement / sigma_log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/exp9_nile_location.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_obs": int(len(flow)),
        "train_years": [int(years[0]), config.EXP9_TRAIN_END_YEAR],
        "test_years": [config.EXP9_TRAIN_END_YEAR + 1, int(years[-1])],
        "fits": fits,
        "sigma_log": sigma_log,
        "peak_displacement_1946_over_sigma_log": peak_1946,
        "first_half_decay_year": half_decay_year,
        "expected_outcomes": {
            "peak_1946_over_sigma_log": {"gaussian_log": 1.615, "crps": 0.465, "density_power": 0.066},
            "first_half_decay_year": {"gaussian_log": 1949, "crps": 1948, "density_power": 1949},
        },
    }
    with open("results/exp9_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "fits"}, indent=2))
