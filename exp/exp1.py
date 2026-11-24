"""exp_1: Curvature-normalised driver shape comparison across scoring rules.

Reproduces the driver-shape comparison of Section 6.1 and Appendix G
(Table G.5): the curvature-normalised driver g_S^circ(z) = g_S(z)/J_S for
the Gaussian log, density-power, CRPS and MMD-RBF scoring rules on the
Gaussian working density at the reference state (lambda = 0, v0 = 1).

Run as ``python -m exp.exp1``.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss

from src.config import EXP1_N_DISPLAY, EXP1_Z_ABS_RANGE, EXP1_Z_RANGE
from src.scoring_rules import DRIVERS, J_S, K_S, RULE_NAMES, curvature_normalised_driver

RESULTS_DIR = "results"


def classify_tail(rule: str, z_max: float = 10.0, tail_window: float = 1.0) -> dict:
    """Classify the tail behaviour of g_S^circ on |z| in [0, z_max].

    The "redescending" vs. "tail-saturating" distinction is a property of the
    true asymptotic limit as z -> infinity (as in Appendix G and Section 6.1:
    a redescending driver returns to zero, a tail-saturating one approaches a
    finite nonzero plateau), not of the ratio between the value at z_max and
    an interior peak -- density-power has a tall interior peak (Table G.5)
    followed by a slow decay to its nonzero plateau kappa_beta/J_S, which a
    peak-relative threshold would misclassify as redescending.

    * "unbounded" if |g^circ| keeps growing without bound for z -> infinity;
    * "tail_saturating" if the asymptotic limit is finite and nonzero;
    * "redescending" if the asymptotic limit is (numerically) zero.
    """
    z_grid = np.linspace(0.01, z_max, 2000)
    g = curvature_normalised_driver(rule, z_grid)
    g_abs = np.abs(g)

    # Probe the true z -> infinity behaviour far beyond the display range.
    z_far = np.array([1.0e2, 1.0e3, 1.0e4])
    g_far = np.abs(curvature_normalised_driver(rule, z_far))
    is_unbounded = g_far[-1] > 10.0 * max(g_far[0], 1e-12)
    asymptotic_limit = float(np.abs(curvature_normalised_driver(rule, np.array([1.0e6]))[0]))

    if is_unbounded:
        return {"tail": "unbounded", "tail_limit": None, "peak_z": None, "monotone_to_plateau": None}

    peak_idx = int(np.argmax(g_abs))
    peak_z = float(z_grid[peak_idx])
    # Redescending: the asymptotic plateau is itself (numerically) zero.
    is_redescending = asymptotic_limit < 1.0e-3 * max(g_abs[peak_idx], 1e-12)
    monotone = bool(np.all(np.diff(g_abs[: peak_idx + 1]) >= -1e-9)) if peak_idx > 0 else True
    monotone_to_plateau = bool(np.all(np.diff(g_abs[peak_idx:]) >= -1e-6))
    return {
        "tail": "redescending" if is_redescending else "tail_saturating",
        "tail_limit": float(g[-1]),
        "asymptotic_limit": asymptotic_limit,
        "peak_z": peak_z,
        "monotone_to_plateau": monotone_to_plateau,
    }


def quadrature_check(rule: str, n_nodes: int = 80) -> dict:
    """Independent Gauss-Hermite quadrature check of J_S and K_S at the
    Gaussian reference state (lambda=0, v0=1), under Y ~ N(0,1) fixed.

    K_S = E_Y[psi_S(0,Y)^2]. J_S = R''_S(0) = -E_Y[d/dlambda psi_S(lambda,Y)]|_{lambda=0},
    obtained by a central finite difference of psi_S in its lambda argument
    (holding the realised y fixed), independent of the closed-form driver
    algebra used elsewhere in this module.
    """
    from src.scoring_rules import PSIS

    x, w = hermgauss(n_nodes)
    y = np.sqrt(2) * x  # quadrature nodes for Y ~ N(0,1)
    weights = w / np.sqrt(np.pi)
    psi = PSIS[rule]
    psi_vals = psi(0.0, y)
    k_quad = float(np.sum(weights * psi_vals ** 2))
    h = 1e-5
    psi_plus = psi(h, y)
    psi_minus = psi(-h, y)
    dpsi_dlambda = (psi_plus - psi_minus) / (2 * h)
    j_quad = float(-np.sum(weights * dpsi_dlambda))
    return {"J_quad": j_quad, "K_quad": k_quad, "J_table": J_S[rule], "K_table": K_S[rule]}


def run() -> dict:
    z_signed = np.linspace(EXP1_Z_RANGE[0], EXP1_Z_RANGE[1], EXP1_N_DISPLAY)
    z_abs = np.linspace(max(EXP1_Z_ABS_RANGE[0], 1e-3), EXP1_Z_ABS_RANGE[1], 2000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    signed_curves, magnitude_curves = {}, {}
    for rule in RULE_NAMES:
        g_signed = curvature_normalised_driver(rule, z_signed)
        g_abs = curvature_normalised_driver(rule, z_abs)
        signed_curves[rule] = g_signed.tolist()
        magnitude_curves[rule] = np.abs(g_abs).tolist()
        axes[0].plot(z_signed, g_signed, label=rule)
        axes[1].loglog(z_abs, np.abs(g_abs), label=rule)
    axes[0].set_xlabel("z")
    axes[0].set_ylabel(r"$g_S^\circ(z)$")
    axes[0].set_title("Signed curvature-normalised driver")
    axes[0].legend()
    axes[0].axhline(0, color="grey", linewidth=0.5)
    axes[1].set_xlabel("|z|")
    axes[1].set_ylabel(r"$|g_S^\circ(z)|$ (log scale)")
    axes[1].set_title("Magnitude, log-log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_driver_shapes.png", dpi=150)
    plt.close(fig)

    classifications = {rule: classify_tail(rule) for rule in RULE_NAMES}
    quad_checks = {rule: quadrature_check(rule) for rule in RULE_NAMES}
    scales = {rule: float(np.sqrt(K_S[rule]) / abs(J_S[rule])) for rule in RULE_NAMES}

    summary = {
        "J_S": J_S,
        "K_S": K_S,
        "curvature_normalised_scale_sqrtK_over_J": scales,
        "tail_classification": classifications,
        "quadrature_check": quad_checks,
        "redescending_rules": [r for r, c in classifications.items() if c["tail"] == "redescending"],
    }
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
