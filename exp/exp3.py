"""exp_3: Finite-sample filtering under contamination, paired Monte Carlo
(Section 6.3, Table 2).

For each of 5 recursion drivers (Gaussian log, density-power, CRPS,
MMD-RBF, and a Student-t6 log-score benchmark -- the benchmark uses the
Student-t6 driver but the same Gaussian-based data-generating process as
the other four) the curvature-normalised scale filter

    lambda_{t+1} = omega + phi*lambda_t + alpha*u_S(y_t, lambda_t)

is run over a latent AR(1) log-volatility path x_t under the observation
equation Y_t = exp(x_t/2)*eps_t, both in a "clean" condition and in a
condition where a Bernoulli(p) fraction of innovations is multiplied by 6
(variance ratio 36). Within each replication the latent path, baseline
innovations and contamination mask are shared across all 5 rules (a paired
design), but independent across replications.

Run as ``python -m exp.exp3`` for the reduced run (R=EXP3_R_RUN) used here,
or ``python -m exp.exp3 --full`` for the paper's R=EXP3_R_PAPER=20000
replications -- the full run needs substantially more memory and time and
is not executed by default in this environment.
"""
from __future__ import annotations

import argparse
import gc
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import (
    bernoulli_variance_contamination_batch,
    latent_log_vol_path_batch,
)
from src.config import (
    EXP3_CONTAM_MULT,
    EXP3_CONTAM_PROB,
    EXP3_FILTER_OMEGA,
    EXP3_FILTER_PHI,
    EXP3_FLAG_THRESHOLD,
    EXP3_GAIN_ALPHA,
    EXP3_LATENT_PHI,
    EXP3_LATENT_SIGMA,
    EXP3_R_PAPER,
    EXP3_R_RUN,
    EXP3_SEED,
    EXP3_T_RUN,
)
from src.filters import filter_scale_path_batch, scale_update

RESULTS_DIR = "results"
RULES = ("gaussian_log", "density_power", "crps", "mmd_rbf", "student_log")
CONDITIONS = ("clean", "contaminated")
METRIC_NAMES = ("rmse", "bias", "max_dev", "flagged_update")


def _metrics_for_rule(Y: np.ndarray, x: np.ndarray, rule: str) -> dict:
    """Filter ``Y`` (shape (R, T)) with ``rule`` and return the 4 per-
    replication metric arrays (each shape (R,)) against the latent path
    ``x``. All (R, T) intermediates (lam, err, u) are local and are
    released when this function returns."""
    lam = filter_scale_path_batch(
        Y, EXP3_FILTER_OMEGA, EXP3_FILTER_PHI, EXP3_GAIN_ALPHA, rule,
        curvature_normalize=True,
    )
    err = lam - x
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    bias = np.mean(err, axis=1)
    max_dev = np.max(np.abs(err), axis=1)
    u = scale_update(rule, lam, Y, curvature_normalize=True)
    flagged_update = np.mean(np.abs(u) > EXP3_FLAG_THRESHOLD, axis=1)
    return {"rmse": rmse, "bias": bias, "max_dev": max_dev, "flagged_update": flagged_update}


def run_experiment(R: int, T: int, rules=RULES, seed: int = EXP3_SEED) -> dict:
    """Core paired Monte Carlo loop.

    Generates one common latent path / baseline innovations / contamination
    mask batch of shape (R, T), shared by every rule (paired design), then
    filters the clean and contaminated series one rule at a time so that at
    most a handful of (R, T) arrays are alive simultaneously.

    Returns a nested dict: ``metrics[condition][rule][metric] -> (R,) array``.
    """
    rng = np.random.default_rng(seed)
    x = latent_log_vol_path_batch(R, T, EXP3_LATENT_PHI, EXP3_LATENT_SIGMA, rng)
    contam = bernoulli_variance_contamination_batch(x, EXP3_CONTAM_PROB, EXP3_CONTAM_MULT, rng)
    y_by_condition = {"clean": contam["y_clean"], "contaminated": contam["y_contam"]}
    del contam

    metrics = {cond: {} for cond in CONDITIONS}
    for rule in rules:
        for cond in CONDITIONS:
            metrics[cond][rule] = _metrics_for_rule(y_by_condition[cond], x, rule)
        gc.collect()
    return metrics


def summarize(metrics: dict, R: int, T: int, seed: int) -> dict:
    """Collapse the per-replication metric arrays to across-replication
    mean and standard error (std/sqrt(R)) for each rule x condition cell."""
    summary = {"R": R, "T": T, "seed": seed}
    for cond in CONDITIONS:
        summary[cond] = {}
        for rule, rule_metrics in metrics[cond].items():
            cell = {}
            for name, arr in rule_metrics.items():
                arr = np.asarray(arr, dtype=float)
                cell[name] = float(np.mean(arr))
                cell[f"{name}_se"] = float(np.std(arr) / np.sqrt(len(arr)))
            summary[cond][rule] = cell
    return summary


def print_summary_table(summary: dict, rules=RULES) -> None:
    header = f"{'rule':<14}{'cond':<13}{'rmse':>10}{'bias':>10}{'max_dev':>10}{'flag%':>9}"
    print(header)
    print("-" * len(header))
    for rule in rules:
        for cond in CONDITIONS:
            cell = summary[cond][rule]
            print(
                f"{rule:<14}{cond:<13}{cell['rmse']:>10.4f}{cell['bias']:>10.4f}"
                f"{cell['max_dev']:>10.4f}{100 * cell['flagged_update']:>8.3f}%"
            )


def plot_summary(summary: dict, path: str, rules=RULES, metric: str = "rmse") -> None:
    x_pos = np.arange(len(rules))
    width = 0.35
    clean_vals = [summary["clean"][r][metric] for r in rules]
    contam_vals = [summary["contaminated"][r][metric] for r in rules]
    clean_se = [summary["clean"][r][f"{metric}_se"] for r in rules]
    contam_se = [summary["contaminated"][r][f"{metric}_se"] for r in rules]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, clean_vals, width, yerr=clean_se, label="clean", capsize=3)
    ax.bar(x_pos + width / 2, contam_vals, width, yerr=contam_se, label="contaminated", capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rules, rotation=20)
    ax.set_xlabel("scoring rule")
    ax.set_ylabel(metric.upper() if metric == "rmse" else metric)
    ax.set_title("Finite-sample filtering under contamination: " + metric.upper() + " by rule (Section 6.3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(R: int = EXP3_R_RUN, T: int = EXP3_T_RUN) -> dict:
    metrics = run_experiment(R, T, rules=RULES, seed=EXP3_SEED)
    summary = summarize(metrics, R, T, EXP3_SEED)

    with open(f"{RESULTS_DIR}/exp3_contamination_filtering.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_summary(summary, f"{RESULTS_DIR}/exp3_summary.png")

    print_summary_table(summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Use the paper's full R=EXP3_R_PAPER replications instead of the "
             "reduced EXP3_R_RUN count (needs substantially more memory/time; "
             "not run by default in this environment).",
    )
    args = parser.parse_args()
    run_R = EXP3_R_PAPER if args.full else EXP3_R_RUN
    run(R=run_R, T=EXP3_T_RUN)
