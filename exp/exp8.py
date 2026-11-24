"""exp_8: Controlled permanent variance break and jump contamination
experiments (Section 7.4, Table G.7).

Part A (permanent break) holds the Gaussian working density fixed and runs
the curvature-normalised scale recursion
    lambda_{t+1} = omega + phi*lambda_t + alpha*u_S(y_t,lambda_t)/J_S
with omega=0, phi=1 (a pure adaptive filter, log-variance started at zero)
over a path whose true variance jumps once from
EXP8_BREAK_VAR_PRE to EXP8_BREAK_VAR_POST, for the Gaussian log score and the
bounded density-power score, in a paired Monte Carlo design.

Part B (jump contamination) fits the Section 7 estimation pipeline
(``src.empirical.fit_filter`` / ``filtered_variance_path``) on a single
latent diffusive-scale path contaminated by one-sided negative jumps on a
Bernoulli(EXP8_JUMP_PROB) subset of dates, comparing a frozen-fit arm
(clean-training parameters applied to both the clean and jump test paths)
and a refitted arm (parameters estimated on the jump-contaminated training
sample), for the Gaussian log/density-power and Student-t6 log/density-power
cells. The paper does not specify a replication count for this design, so a
single realisation is used (deviation recorded in results/RESULTS.md).

Run as ``python -m exp.exp8`` for the reduced run, or
``python -m exp.exp8 --full`` for the paper's R=EXP8_BREAK_R_PAPER=20000
permanent-break replications.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import latent_log_vol_path, one_sided_jump_contamination
from src.config import (
    EXP8_BREAK_ADAPT_FRACTION,
    EXP8_BREAK_GAIN,
    EXP8_BREAK_R_PAPER,
    EXP8_BREAK_R_RUN,
    EXP8_BREAK_T_POST,
    EXP8_BREAK_T_PRE,
    EXP8_BREAK_TRAILING_WINDOW,
    EXP8_BREAK_VAR_POST,
    EXP8_BREAK_VAR_PRE,
    EXP8_FILTER_GAIN,
    EXP8_JUMP_BOUND,
    EXP8_JUMP_HALFNORMAL_SCALE,
    EXP8_JUMP_PROB,
    EXP8_JUMP_T_PAPER,
    EXP8_JUMP_T_RUN,
    EXP8_JUMP_LBFGS_N_STARTS,
    EXP8_JUMP_TRAIN_FRACTION,
    EXP8_LATENT_PHI,
    EXP8_LATENT_SIGMA,
    EXP8_SEED,
)
from src.empirical import fit_filter, filtered_variance_path
from src.filters import filter_scale_path_batch, scale_update

RESULTS_DIR = "results"
BREAK_RULES = ("gaussian_log", "density_power")
JUMP_CELLS = (("log", "gaussian"), ("density_power", "gaussian"),
              ("log", "student"), ("density_power", "student"))


# ---------------------------------------------------------------------------
# Part A: permanent variance break
# ---------------------------------------------------------------------------
def run_break_experiment(R: int, seed: int = EXP8_SEED) -> dict:
    """Paired Monte Carlo over ``R`` replications of the break design."""
    T_pre, T_post = EXP8_BREAK_T_PRE, EXP8_BREAK_T_POST
    T = T_pre + T_post
    rng = np.random.default_rng(seed)
    std = np.where(np.arange(T) < T_pre, np.sqrt(EXP8_BREAK_VAR_PRE), np.sqrt(EXP8_BREAK_VAR_POST))
    Y = rng.standard_normal((R, T)) * std[None, :]

    log_v_target = np.log(EXP8_BREAK_VAR_POST)
    target_level = EXP8_BREAK_ADAPT_FRACTION * log_v_target
    v_star = EXP8_BREAK_VAR_POST

    out = {}
    for rule in BREAK_RULES:
        lam = filter_scale_path_batch(
            Y, omega=0.0, phi=1.0, alpha=EXP8_BREAK_GAIN, rule=rule,
            lambda_init=0.0, curvature_normalize=True,
        )
        post = lam[:, T_pre:]
        w = EXP8_BREAK_TRAILING_WINDOW
        trailing = np.array([
            np.convolve(row, np.ones(w) / w, mode="valid") for row in post
        ])  # shape (R, T_post - w + 1), trailing[:, k] = mean(post[k:k+w])
        reaches = trailing >= target_level
        delay = np.full(R, float(T_post), dtype=float)
        for r in range(R):
            idx = np.argmax(reaches[r]) if reaches[r].any() else -1
            if idx >= 0:
                delay[r] = idx + w  # index of the trailing window's last date, 1-based post-break date
        v_t = np.exp(post)
        qlike_excess = 2.0 * (v_star / v_t + post - 1 - np.log(v_star))
        cum_excess = qlike_excess.sum(axis=1)

        out[rule] = {
            "adaptation_delay": float(np.mean(delay)),
            "adaptation_delay_se": float(np.std(delay) / np.sqrt(R)),
            "cumulative_excess_loss": float(np.mean(cum_excess)),
            "cumulative_excess_loss_se": float(np.std(cum_excess) / np.sqrt(R)),
        }
    out["density_power_minus_log"] = {
        "adaptation_delay_diff": out["density_power"]["adaptation_delay"] - out["gaussian_log"]["adaptation_delay"],
        "cumulative_excess_loss_diff": out["density_power"]["cumulative_excess_loss"] - out["gaussian_log"]["cumulative_excess_loss"],
    }
    out["R"], out["T_pre"], out["T_post"] = R, T_pre, T_post
    return out


# ---------------------------------------------------------------------------
# Part B: jump contamination
# ---------------------------------------------------------------------------
def run_jump_experiment(T: int, seed: int = EXP8_SEED) -> dict:
    rng = np.random.default_rng(seed)
    x = latent_log_vol_path(T, EXP8_LATENT_PHI, EXP8_LATENT_SIGMA, rng)
    sim = one_sided_jump_contamination(x, EXP8_JUMP_PROB, EXP8_JUMP_HALFNORMAL_SCALE, rng)
    y_clean, y_jump = sim["y_clean"], sim["y_jump"]

    n_train = int(EXP8_JUMP_TRAIN_FRACTION * T)
    train_clean, test_clean = y_clean[:n_train], y_clean[n_train:]
    train_jump, test_jump = y_jump[:n_train], y_jump[n_train:]
    x_test = x[n_train:]

    results = {}
    crossings = {}
    for criterion, density in JUMP_CELLS:
        name = f"{density}_{criterion}"
        frozen_fit = fit_filter(train_clean, criterion, density, seed=0)
        refit = fit_filter(train_jump, criterion, density, seed=1)

        def rmse(y_test: np.ndarray, params: dict) -> float:
            v = filtered_variance_path(
                y_test, params["omega"], params["phi"], params["alpha"], criterion, density=density,
            )
            lam = np.log(v)
            return float(np.sqrt(np.mean((lam - x_test) ** 2)))

        rmse_frozen_clean = rmse(test_clean, frozen_fit)
        rmse_frozen_jump = rmse(test_jump, frozen_fit)
        rmse_refit_clean = rmse(test_clean, refit)
        rmse_refit_jump = rmse(test_jump, refit)
        results[name] = {
            "frozen_fit": frozen_fit, "refitted_fit": refit,
            "rmse_clean_frozen": rmse_frozen_clean, "rmse_jump_frozen": rmse_frozen_jump,
            "rmse_clean_refitted": rmse_refit_clean, "rmse_jump_refitted": rmse_refit_jump,
            "jump_minus_clean_frozen": rmse_frozen_jump - rmse_frozen_clean,
            "jump_minus_clean_refitted": rmse_refit_jump - rmse_refit_clean,
        }
        if name == "gaussian_log":
            lam_frozen_full = np.log(filtered_variance_path(
                y_jump, frozen_fit["omega"], frozen_fit["phi"], frozen_fit["alpha"], criterion, density=density,
            ))
            lam_refit_full = np.log(filtered_variance_path(
                y_jump, refit["omega"], refit["phi"], refit["alpha"], criterion, density=density,
            ))
            crossings["frozen"] = int(np.sum(np.abs(lam_frozen_full) > EXP8_JUMP_BOUND))
            crossings["refitted"] = int(np.sum(np.abs(lam_refit_full) > EXP8_JUMP_BOUND))
    return {"cells": results, "gaussian_log_bound_crossings": crossings, "T": T, "n_train": n_train}


def plot_break(break_summary: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rules = BREAK_RULES
    delays = [break_summary[r]["adaptation_delay"] for r in rules]
    delay_se = [break_summary[r]["adaptation_delay_se"] for r in rules]
    losses = [break_summary[r]["cumulative_excess_loss"] for r in rules]
    loss_se = [break_summary[r]["cumulative_excess_loss_se"] for r in rules]
    axes[0].bar(rules, delays, yerr=delay_se, capsize=4)
    axes[0].set_title("Adaptation delay (dates)")
    axes[1].bar(rules, losses, yerr=loss_se, capsize=4)
    axes[1].set_title("Cumulative oracle excess QLIKE")
    fig.suptitle("Permanent variance break (Section 7.4)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_jump(jump_summary: dict, path: str) -> None:
    names = list(jump_summary["cells"].keys())
    frozen = [jump_summary["cells"][n]["jump_minus_clean_frozen"] for n in names]
    refit = [jump_summary["cells"][n]["jump_minus_clean_refitted"] for n in names]
    x_pos = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos - width / 2, frozen, width, label="frozen-fit")
    ax.bar(x_pos + width / 2, refit, width, label="refitted")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("jump-minus-clean RMSE")
    ax.set_title("Jump contamination: RMSE change by cell (Section 7.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(full: bool = False) -> dict:
    R = EXP8_BREAK_R_PAPER if full else EXP8_BREAK_R_RUN
    T_jump = EXP8_JUMP_T_PAPER if full else EXP8_JUMP_T_RUN

    break_summary = run_break_experiment(R)
    jump_summary = run_jump_experiment(T_jump)

    summary = {"break": break_summary, "jump": jump_summary, "full": full}
    with open(f"{RESULTS_DIR}/exp8_break_jump.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_break(break_summary, f"{RESULTS_DIR}/exp8_break.png")
    plot_jump(jump_summary, f"{RESULTS_DIR}/exp8_jump.png")

    print(f"Permanent break (R={R}):")
    for rule in BREAK_RULES:
        cell = break_summary[rule]
        print(f"  {rule:<16} delay={cell['adaptation_delay']:.3f} (se {cell['adaptation_delay_se']:.3f})"
              f"  cum_excess={cell['cumulative_excess_loss']:.3f} (se {cell['cumulative_excess_loss_se']:.3f})")
    print("Jump contamination:")
    for name, cell in jump_summary["cells"].items():
        print(f"  {name:<20} jump-minus-clean frozen={cell['jump_minus_clean_frozen']:.4f}"
              f"  refitted={cell['jump_minus_clean_refitted']:.4f}")
    print(f"  gaussian_log bound crossings: {jump_summary['gaussian_log_bound_crossings']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the paper's R=20000 break replications")
    args = parser.parse_args()
    run(full=args.full)
