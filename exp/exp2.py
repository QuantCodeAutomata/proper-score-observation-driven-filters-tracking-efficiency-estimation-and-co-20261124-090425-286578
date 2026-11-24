"""exp_2: Pseudo-true target under variance contamination.

Reproduces Section 6.2: the rule-specific pseudo-true variance target
v*(tau) under the Gaussian variance-contamination law
Q_{eps,tau} = (1-eps) N(0, v0) + eps N(0, tau*v0), eps = 0.05, for the
Gaussian log, density-power, CRPS and MMD-RBF scoring rules, verifying the
closed-form Gaussian-log identity v_log*/v0 = 1 + eps*(tau-1).

Run as ``python -m exp.exp2``.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EXP2_EPSILON, EXP2_GH_NODES, EXP2_N_TAU, EXP2_ROOT_SCAN_POINTS, EXP2_TAU_MAX, EXP2_TAU_MIN, EXP2_V0
from src.pseudo_true import pseudo_true_log_variance, pseudo_true_log_variance_closed_form
from src.scoring_rules import RULE_NAMES

RESULTS_DIR = "results"


def run() -> dict:
    taus = np.linspace(EXP2_TAU_MIN, EXP2_TAU_MAX, EXP2_N_TAU)
    taus[0] = max(taus[0], 1e-6)  # tau=0 is a degenerate point mass; avoid it exactly

    targets = {rule: np.empty(EXP2_N_TAU) for rule in RULE_NAMES}
    for rule in RULE_NAMES:
        for i, tau in enumerate(taus):
            targets[rule][i] = pseudo_true_log_variance(
                rule, tau, v0=EXP2_V0, epsilon=EXP2_EPSILON,
                n_nodes=EXP2_GH_NODES, root_scan_points=EXP2_ROOT_SCAN_POINTS,
            )

    kl_line = pseudo_true_log_variance_closed_form(taus, v0=EXP2_V0, epsilon=EXP2_EPSILON)
    closed_form_check = np.array([
        pseudo_true_log_variance_closed_form(t, EXP2_V0, EXP2_EPSILON) for t in taus
    ])
    max_root_discrepancy = float(np.max(np.abs(targets["gaussian_log"] - closed_form_check)) / EXP2_V0)

    fig, ax = plt.subplots(figsize=(7, 5))
    for rule in RULE_NAMES:
        ax.plot(taus, targets[rule] / EXP2_V0, label=rule)
    ax.plot(taus, kl_line / EXP2_V0, "k--", label="KL line 1+eps(tau-1)")
    ax.set_xlabel(r"$\tau$ (contamination variance ratio)")
    ax.set_ylabel(r"$v^*/v_0$")
    ax.set_title("Pseudo-true variance target under contamination (eps=0.05)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_pseudo_true_targets.png", dpi=150)
    plt.close(fig)

    summary = {
        "epsilon": EXP2_EPSILON,
        "v0": EXP2_V0,
        "tau_grid_min_max_n": [float(taus[0]), float(taus[-1]), EXP2_N_TAU],
        "v_star_over_v0_at_tau_80": {
            rule: float(np.interp(80.0, taus, targets[rule] / EXP2_V0)) for rule in RULE_NAMES
        },
        "v_star_over_v0_at_tau_max": {
            rule: float(targets[rule][-1] / EXP2_V0) for rule in RULE_NAMES
        },
        "max_gaussian_log_root_discrepancy": max_root_discrepancy,
        "gaussian_log_matches_kl_closed_form": bool(max_root_discrepancy < 1e-5),
    }
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
