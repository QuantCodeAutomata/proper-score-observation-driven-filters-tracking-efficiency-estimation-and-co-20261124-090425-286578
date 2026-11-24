"""exp_5: High-frequency diffusion-limit mesh refinement (Appendix E, G.4;
Tables G.7, G.8).

Verifies three continuous-time limit theorems by mesh-refinement Monte Carlo
simulation, for the Gaussian log, density-power and CRPS scoring rules only
(``mmd_rbf`` and ``student_log`` are not part of this experiment):

* Part A -- Centred diffusion (Theorem E.1): the O(h) rate of convergence of
  the second-moment of the recursion increment Delta lambda to
  alpha^2 * Xi_S, and the vanishing of the fourth-moment ("Lindeberg") term,
  checked against an independent quadrature centring diagnostic.
* Part B -- Non-centred mean flow (Theorem E.7): the empirical terminal
  cross-sectional mean of the O(h)-gain recursion converging, as h shrinks,
  to the analytical mean-flow fixed point.
* Part C -- Local Ornstein-Uhlenbeck approximation (Theorem 3.2): the
  terminal-variance ratio of the normalised local error against its
  analytical benchmark, plus a companion-arm Kolmogorov-Smirnov check of
  mesh-insensitivity of the terminal law.

All Monte Carlo mechanics (recursions, quadrature) live in ``src/diffusion.py``;
this module only orchestrates the simulations, computes the diagnostics
described in the paper, and produces the plots/summary.

Run as ``python -m exp.exp5`` (reduced replication counts) or
``python -m exp.exp5 --full`` (paper replication counts; same h-grids and
step counts in both modes).
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from src.config import (
    EXP5_ALPHA,
    EXP5_ALPHA_BAR,
    EXP5_H_GRID_CENTRED,
    EXP5_KAPPA,
    EXP5_MEANFLOW_T5_DOF,
    EXP5_OU_GAMMA,
    EXP5_OU_H_COMPANION,
    EXP5_OU_H_GRID,
    EXP5_OU_K,
    EXP5_OU_M,
    EXP5_OU_RHO,
    EXP5_R_CENTRED_PAPER,
    EXP5_R_CENTRED_RUN,
    EXP5_R_MEANFLOW_PAPER,
    EXP5_R_MEANFLOW_RUN,
    EXP5_R_OU_PAPER,
    EXP5_R_OU_RUN,
    EXP5_SEED,
    EXP5_T_CENTRED,
    EXP5_T_MEANFLOW,
    EXP5_XI,
)
from src.diffusion import (
    centring_check,
    mean_flow_fixed_point,
    simulate_centred_diffusion,
    simulate_local_ou,
    simulate_mean_flow,
)

RESULTS_DIR = "results"
RULES = ("gaussian_log", "density_power", "crps")

# Discrepancies below this floor are treated as numerical noise around zero
# and excluded from the log-log slope regression (log(0) is undefined).
_DISCREPANCY_FLOOR = 1e-10

# numpy.polynomial.hermite.hermgauss (used inside centring_check) overflows
# to NaN weights at its default n_quad=400 node count on this numpy build;
# 300 nodes is the largest value confirmed finite and is used throughout
# instead of the library default.
_CENTRING_N_QUAD = 300


def loglog_slope(h_grid: np.ndarray, discrepancy: np.ndarray) -> float:
    """Fit ``log(discrepancy) = slope * log(h) + const`` by OLS, excluding
    points where the discrepancy has underflowed to numerical noise."""
    h_grid = np.asarray(h_grid, dtype=float)
    discrepancy = np.asarray(discrepancy, dtype=float)
    mask = np.isfinite(discrepancy) & (discrepancy > _DISCREPANCY_FLOOR)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(h_grid[mask]), np.log(discrepancy[mask]), 1)
    return float(slope)


def run_part_a(
    rules=RULES,
    R: int = EXP5_R_CENTRED_RUN,
    rng: np.random.Generator | None = None,
    h_grid=EXP5_H_GRID_CENTRED,
    T_steps: int = EXP5_T_CENTRED,
    alpha: float = EXP5_ALPHA,
    kappa: float = EXP5_KAPPA,
) -> dict:
    """Theorem E.1 centred-diffusion mesh-refinement check.

    For each rule and mesh h: simulate the recursion, estimate the O(h)
    second-moment h^{-1}*E[(Delta lambda)^2] and its discrepancy against the
    analytical limit alpha^2*Xi_S, the (vanishing) fourth-moment analogue,
    the equivalent realised-quadratic-variation estimator, and an
    independent-quadrature centring check of E[g_{S,h}].
    """
    if rng is None:
        rng = np.random.default_rng(EXP5_SEED)
    h_grid = list(h_grid)
    out = {}
    for rule in rules:
        target = alpha ** 2 * EXP5_XI[rule]
        second_moment_est, discrepancy = [], []
        fourth_moment_over_h = []
        qv_over_h, qv_discrepancy = [], []
        centring_vals = []
        for h in h_grid:
            sim = simulate_centred_diffusion(rule, h, alpha, kappa, T_steps, R, rng)
            deltas = sim["deltas"]

            m2 = float(np.mean(deltas ** 2) / h)
            second_moment_est.append(m2)
            discrepancy.append(abs(m2 - target))

            m4 = float(np.mean(deltas ** 4) / h)
            fourth_moment_over_h.append(m4)

            qv_per_rep = np.sum(deltas ** 2, axis=1) / T_steps
            qv_mean = float(np.mean(qv_per_rep) / h)
            qv_over_h.append(qv_mean)
            qv_discrepancy.append(abs(qv_mean - target))

            centring_vals.append(centring_check(rule, h, n_quad=_CENTRING_N_QUAD))

        out[rule] = {
            "h_grid": h_grid,
            "alpha2_xi_target": target,
            "second_moment_estimate": second_moment_est,
            "discrepancy": discrepancy,
            "fourth_moment_over_h": fourth_moment_over_h,
            "qv_over_h_estimate": qv_over_h,
            "qv_discrepancy": qv_discrepancy,
            "centring_check": centring_vals,
            "max_abs_centring_check": float(np.max(np.abs(centring_vals))),
            "loglog_slope": loglog_slope(h_grid, discrepancy),
        }
    return out


def run_part_b(
    rules=RULES,
    R: int = EXP5_R_MEANFLOW_RUN,
    rng: np.random.Generator | None = None,
    h_grid=EXP5_H_GRID_CENTRED,
    T_steps: int = EXP5_T_MEANFLOW,
    alpha_bar: float = EXP5_ALPHA_BAR,
    kappa: float = EXP5_KAPPA,
    dof: int = EXP5_MEANFLOW_T5_DOF,
) -> dict:
    """Theorem E.7 non-centred mean-flow mesh-refinement check: the empirical
    terminal cross-sectional mean at the finest mesh should approach the
    analytical mean-flow fixed point."""
    if rng is None:
        rng = np.random.default_rng(EXP5_SEED)
    h_grid = list(h_grid)
    out = {}
    for rule in rules:
        fixed_point_estimate = []
        for h in h_grid:
            path = simulate_mean_flow(rule, h, alpha_bar, kappa, T_steps, R, rng, dof=dof)
            fixed_point_estimate.append(float(path[:, -1].mean()))
        analytical = mean_flow_fixed_point(rule, alpha_bar, kappa)
        limit_estimate = fixed_point_estimate[-1]  # h_grid is decreasing in h
        out[rule] = {
            "h_grid": h_grid,
            "fixed_point_estimate": fixed_point_estimate,
            "fixed_point_analytical": analytical,
            "fixed_point_limit_estimate": limit_estimate,
            "abs_diff_limit_vs_analytical": abs(limit_estimate - analytical),
        }
    return out


def run_part_c(
    rules=RULES,
    R: int = EXP5_R_OU_RUN,
    rng: np.random.Generator | None = None,
    h_grid=EXP5_OU_H_GRID,
    n_steps: int = EXP5_OU_M,
    rho: float = EXP5_OU_RHO,
    Gamma: float = EXP5_OU_GAMMA,
    K: float = EXP5_OU_K,
    h_companion: float = EXP5_OU_H_COMPANION,
) -> dict:
    """Theorem 3.2 local Ornstein-Uhlenbeck mesh-refinement check: terminal
    variance ratio of the active (non-stopped) paths against the analytical
    benchmark q_S^2, plus a companion-arm KS test of mesh-insensitivity of
    the terminal law at an ultra-fine companion mesh."""
    if rng is None:
        rng = np.random.default_rng(EXP5_SEED)
    h_grid = list(h_grid)
    out = {}
    for rule in rules:
        variance_ratio, empirical_variance, q2_benchmark, active_fraction = [], [], [], []
        smallest_h_Z_active = None
        for h in h_grid:
            sim = simulate_local_ou(rule, h, rho, Gamma, K, n_steps, R, rng)
            Z, active, q2 = sim["Z_terminal"], sim["active"], sim["q2_benchmark"]
            Z_active = Z[active]
            emp_var = float(np.var(Z_active, ddof=1))
            variance_ratio.append(emp_var / q2)
            empirical_variance.append(emp_var)
            q2_benchmark.append(q2)
            active_fraction.append(float(np.mean(active)))
            smallest_h_Z_active = Z_active  # h_grid is decreasing: last iter = smallest h

        z0 = rng.uniform(-1, 1, size=R)
        companion = simulate_local_ou(rule, h_companion, rho, Gamma, K, n_steps, R, rng, z0=z0)
        companion_Z_active = companion["Z_terminal"][companion["active"]]
        ks_stat, ks_pvalue = ks_2samp(companion_Z_active, smallest_h_Z_active)

        out[rule] = {
            "h_grid": h_grid,
            "variance_ratio_by_h": variance_ratio,
            "empirical_variance_by_h": empirical_variance,
            "q2_benchmark": q2_benchmark[0],
            "active_fraction_by_h": active_fraction,
            "terminal_variance_ratio": variance_ratio[-1],
            "h_companion": h_companion,
            "ks_statistic_companion_vs_smallest_h": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
        }
    return out


def _plot_part_a(part_a: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for rule in RULES:
        d = part_a[rule]
        ax.loglog(d["h_grid"], d["discrepancy"], marker="o",
                   label=f"{rule} (slope={d['loglog_slope']:.2f})")
    ax.set_xlabel("mesh h")
    ax.set_ylabel(r"$|h^{-1}E[(\Delta\lambda)^2] - \alpha^2\Xi_S|$")
    ax.set_title("Centred diffusion: second-moment discrepancy vs mesh (Thm E.1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_part_b(part_b: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for rule in RULES:
        d = part_b[rule]
        line, = ax.plot(d["h_grid"], d["fixed_point_estimate"], marker="o", label=rule)
        ax.axhline(d["fixed_point_analytical"], color=line.get_color(), linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("mesh h (log scale)")
    ax.set_ylabel(r"empirical fixed point $E[\lambda_T]$")
    ax.set_title("Non-centred mean flow: empirical fixed point vs mesh (Thm E.7)\n"
                  "(dashed = analytical fixed point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_part_c(part_c: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ratios = [part_c[rule]["terminal_variance_ratio"] for rule in RULES]
    ax.bar(RULES, ratios, color=["C0", "C1", "C2"])
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("terminal variance ratio (empirical / $q_S^2$)")
    ax.set_title("Local OU approximation: terminal variance ratio by rule (Thm 3.2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(full: bool = False) -> dict:
    rng = np.random.default_rng(EXP5_SEED)
    R_centred = EXP5_R_CENTRED_PAPER if full else EXP5_R_CENTRED_RUN
    R_meanflow = EXP5_R_MEANFLOW_PAPER if full else EXP5_R_MEANFLOW_RUN
    R_ou = EXP5_R_OU_PAPER if full else EXP5_R_OU_RUN

    part_a = run_part_a(RULES, R_centred, rng)
    part_b = run_part_b(RULES, R_meanflow, rng)
    part_c = run_part_c(RULES, R_ou, rng)

    _plot_part_a(part_a, f"{RESULTS_DIR}/exp5_centred_diffusion.png")
    _plot_part_b(part_b, f"{RESULTS_DIR}/exp5_mean_flow.png")
    _plot_part_c(part_c, f"{RESULTS_DIR}/exp5_local_ou.png")

    summary = {
        "seed": EXP5_SEED,
        "R_centred": R_centred,
        "R_meanflow": R_meanflow,
        "R_ou": R_ou,
        "full_mode": full,
        "rules": {
            rule: {"part_A": part_a[rule], "part_B": part_b[rule], "part_C": part_c[rule]}
            for rule in RULES
        },
    }
    with open(f"{RESULTS_DIR}/exp5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="use paper replication counts (EXP5_R_*_PAPER) instead of the reduced *_RUN counts")
    args = parser.parse_args()
    run(full=args.full)
