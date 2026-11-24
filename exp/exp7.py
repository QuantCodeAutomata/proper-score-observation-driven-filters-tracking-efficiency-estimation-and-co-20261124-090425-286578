"""exp_7: Controlled single-observation outlier propagation, SPX
counterfactual (Section 7.1).

Four (density, criterion) cells -- Gaussian log, Student-t6 log, Student-t6
density-power, Student-t6 MMD-RBF -- are fit once on a 2012-2016 training
window of S&P 500 daily log returns, then frozen. The frozen recursion is run
forward through 2017 to get each cell's "clean" filtered log-variance path.
On one event date the actual return is replaced by (actual + B*sigma_ref) for
a grid of perturbation sizes B, and the downstream divergence of the
perturbed path from the clean path is measured (Section 7.1, Proposition
2.3). ``^GSPC`` is rejected by Massive aggregates without an index
entitlement, so SPY is used instead (``src/config.py``), as for exp_6.

Run as ``python -m exp.exp7`` for the reduced run (3 L-BFGS-B starts per
cell), or ``python -m exp.exp7 --full`` for the paper's 8 starts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_ohlcv
from src.config import (
    EXP6_BETA_DP,
    EXP6_LBFGS_N_STARTS,
    EXP7_B_GRID,
    EXP7_EVENT_DATE,
    EXP7_FORWARD_END,
    EXP7_SIGMA_REF_WINDOW,
    EXP7_SLOPE_B,
    EXP7_TICKER,
    EXP7_TICKER_PAPER,
    EXP7_TRAIN_END,
    EXP7_TRAIN_START,
)
from src.empirical import fit_filter, filtered_variance_path, mmd_bandwidth

RESULTS_DIR = Path("results")

# Reduced-for-time-budget counterpart of EXP6_LBFGS_N_STARTS=8 (the paper's
# start count, Section 7): this one experiment fits only four cells on a
# single market, so 3 starts keeps the default run fast; --full restores 8.
N_STARTS_RUN = 3

# Reduced Gauss-Legendre quadrature order for the Student-t MMD-RBF cell,
# used both while fitting and for the final frozen-parameter evaluation
# (Appendix G.1 reports convergence already at a few dozen nodes; 256 is not
# needed for this single-market, single-event experiment).
MMD_NODES = 16

CELLS: tuple[tuple[str, str, str, dict], ...] = (
    ("gaussian_log", "gaussian", "log", {}),
    ("student_log", "student", "log", {"nu": 6}),
    ("student_density_power", "student", "density_power", {"nu": 6, "beta": EXP6_BETA_DP}),
    ("student_mmd", "student", "mmd", {"nu": 6, "mmd_nodes": MMD_NODES}),  # "ell" filled in at run time
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_exp7.py)
# ---------------------------------------------------------------------------
def log_returns_pct(close: np.ndarray) -> np.ndarray:
    """Close-to-close log returns in per cent, 100*diff(log(close))."""
    return 100.0 * np.diff(np.log(close))


def training_window_mask(dates: np.ndarray, train_start: str, train_end: str) -> np.ndarray:
    """Boolean mask selecting dates in [train_start, train_end] inclusive."""
    start = np.datetime64(pd.Timestamp(train_start))
    end = np.datetime64(pd.Timestamp(train_end))
    d = dates.astype("datetime64[ns]")
    return (d >= start) & (d <= end)


def demean_by_training(returns: np.ndarray, n_train: int) -> tuple[np.ndarray, float]:
    """Subtract the TRAINING-sample mean from the whole series (Section 7.1):
    the training slice is ``returns[:n_train]``."""
    train_mean = float(np.mean(returns[:n_train]))
    return returns - train_mean, train_mean


def find_event_index(dates: np.ndarray, event_date: str) -> int:
    """Integer index of ``event_date`` in ``dates`` (exact match); falls back
    to the first date strictly after it if the exact date is not a trading
    day in the series."""
    target = np.datetime64(pd.Timestamp(event_date))
    d = dates.astype("datetime64[ns]")
    hit = np.where(d == target)[0]
    if len(hit):
        return int(hit[0])
    after = np.where(d > target)[0]
    if len(after):
        return int(after[0])
    raise ValueError(f"event_date {event_date} is after the last available date")


def compute_sigma_ref(returns: np.ndarray, event_idx: int, window: int = EXP7_SIGMA_REF_WINDOW) -> float:
    """Root mean squared return over the ``window`` sessions strictly
    preceding the event date (Section 7.1)."""
    trailing = returns[max(0, event_idx - window):event_idx]
    return float(np.sqrt(np.mean(trailing ** 2)))


def perturb_returns(returns: np.ndarray, event_idx: int, B: float, sigma_ref: float) -> np.ndarray:
    """A copy of ``returns`` with the event-date observation replaced by
    actual + B*sigma_ref (Section 7.1); no other date is touched and no
    state projection is applied."""
    out = returns.copy()
    out[event_idx] = out[event_idx] + B * sigma_ref
    return out


def compute_perturbation_path(lam_clean: np.ndarray, lam_pert: np.ndarray, event_idx: int) -> np.ndarray:
    """Log-variance perturbation path lam_pert - lam_clean, from the event
    date onward (Section 7.1)."""
    return lam_pert[event_idx:] - lam_clean[event_idx:]


def peak_perturbation(perturbation_path: np.ndarray) -> float:
    """Peak absolute divergence over the downstream window."""
    return float(np.max(np.abs(perturbation_path)))


def fit_loglog_slope(B_values: np.ndarray, peak_values: np.ndarray) -> float:
    """Slope of log(peak) on log(B) by OLS (Section 7.1, Proposition 2.3(i)):
    quadratic growth in B corresponds to a slope near 2."""
    coeffs = np.polyfit(np.log(np.asarray(B_values, dtype=float)), np.log(np.asarray(peak_values, dtype=float)), 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(full: bool = False) -> dict[str, Any]:
    start_time = time.time()
    n_starts = EXP6_LBFGS_N_STARTS if full else N_STARTS_RUN
    print(f"exp7: ticker={EXP7_TICKER} (paper ticker {EXP7_TICKER_PAPER})")
    print(f"exp7: train=[{EXP7_TRAIN_START},{EXP7_TRAIN_END}] event={EXP7_EVENT_DATE} "
          f"forward_end={EXP7_FORWARD_END} n_starts={n_starts} full={full}")

    bars_df, meta = load_ohlcv([EXP7_TICKER], EXP7_TRAIN_START, EXP7_FORWARD_END, timeframe="1d")
    detail = next((d for d in meta["ticker_details"] if d.get("ticker") == EXP7_TICKER), {})
    print(f"exp7: ticker_details for {EXP7_TICKER}: data_source={detail.get('data_source')} "
          f"warnings={meta['warnings'].get(EXP7_TICKER)} failure={meta['failed'].get(EXP7_TICKER)}")
    pd.DataFrame([{
        "ticker": EXP7_TICKER,
        "data_source": detail.get("data_source"),
        "warnings": meta["warnings"].get(EXP7_TICKER),
        "failure": meta["failed"].get(EXP7_TICKER),
        "realized_start": detail.get("window", [None, None])[0] if detail.get("window") else None,
        "realized_end": detail.get("window", [None, None])[1] if detail.get("window") else None,
    }]).to_csv(RESULTS_DIR / "exp7_ticker_details.csv", index=False)
    if meta["failed"]:
        raise RuntimeError(f"exp7: ticker load failed: {meta['failed']}")

    rows = bars_df[bars_df["ticker"] == EXP7_TICKER].sort_values("date")
    close = rows["close"].to_numpy(dtype=float)
    dates = rows["date"].to_numpy()[1:]
    returns = log_returns_pct(close)

    train_mask = training_window_mask(dates, EXP7_TRAIN_START, EXP7_TRAIN_END)
    n_train = int(train_mask.sum())
    if not np.array_equal(train_mask, np.r_[np.ones(n_train, dtype=bool), np.zeros(len(dates) - n_train, dtype=bool)]):
        raise ValueError("training window is not a contiguous prefix of the loaded, date-sorted series")
    returns, train_mean = demean_by_training(returns, n_train)
    train_returns = returns[:n_train]
    print(f"exp7: n_obs={len(returns)} n_train={n_train} n_forward={len(returns) - n_train} train_mean={train_mean:.6f}")

    event_idx = find_event_index(dates, EXP7_EVENT_DATE)
    sigma_ref = compute_sigma_ref(returns, event_idx, EXP7_SIGMA_REF_WINDOW)
    print(f"exp7: event_date={pd.Timestamp(dates[event_idx]).date()} event_idx={event_idx} sigma_ref={sigma_ref:.6f}")

    ell_mmd = mmd_bandwidth(train_returns)
    print(f"exp7: frozen MMD bandwidth (training only) ell={ell_mmd:.6f}")

    peak_table: dict[str, dict[int, float]] = {}
    path_b100: dict[str, np.ndarray] = {}
    fits: dict[str, dict] = {}

    for name, density, criterion, kwargs in CELLS:
        cell_kwargs = dict(kwargs)
        if criterion == "mmd":
            cell_kwargs["ell"] = ell_mmd
        fit = fit_filter(train_returns, criterion, density, n_starts=n_starts, seed=0, **cell_kwargs)
        fits[name] = fit
        print(f"exp7: fit {name}: omega={fit['omega']:.4f} phi={fit['phi']:.4f} alpha={fit['alpha']:.4f} "
              f"success={fit['success']}")

        v_clean = filtered_variance_path(returns, fit["omega"], fit["phi"], fit["alpha"], criterion,
                                          density=density, **cell_kwargs)
        lam_clean = np.log(v_clean)

        peaks: dict[int, float] = {}
        for B in EXP7_B_GRID:
            returns_pert = perturb_returns(returns, event_idx, B, sigma_ref)
            v_pert = filtered_variance_path(returns_pert, fit["omega"], fit["phi"], fit["alpha"], criterion,
                                             density=density, **cell_kwargs)
            lam_pert = np.log(v_pert)
            pert_path = compute_perturbation_path(lam_clean, lam_pert, event_idx)
            peaks[B] = peak_perturbation(pert_path)
            if B == 100:
                path_b100[name] = pert_path
        peak_table[name] = peaks

    # Log-log slope of peak perturbation vs B for the Gaussian-log cell.
    slope_B = np.array(EXP7_SLOPE_B, dtype=float)
    slope_peaks = np.array([peak_table["gaussian_log"][int(b)] for b in EXP7_SLOPE_B])
    slope = fit_loglog_slope(slope_B, slope_peaks)
    slope_record = {
        "slope": slope,
        "points": [{"B": int(b), "peak": float(p)} for b, p in zip(slope_B, slope_peaks)],
    }
    (RESULTS_DIR / "exp7_gaussian_log_slope.json").write_text(json.dumps(slope_record, indent=2))

    peak_df = pd.DataFrame(peak_table)
    peak_df.index.name = "B"
    peak_df.to_csv(RESULTS_DIR / "exp7_peak_perturbation.csv")

    downstream_dates = pd.to_datetime(dates[event_idx:])
    plot_perturbation_paths(path_b100, peak_table, downstream_dates, str(RESULTS_DIR / "exp7_perturbation_paths.png"))

    elapsed = time.time() - start_time
    print(f"exp7: elapsed {elapsed:.1f}s")
    print("exp7: summary (peak perturbation at B=100):")
    summary = {name: peak_table[name][100] for name, *_ in CELLS}
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")
    print(f"exp7: gaussian_log log-log slope (B in {EXP7_SLOPE_B}): {slope:.4f}")

    return {
        "elapsed_seconds": elapsed,
        "peak_table": peak_table,
        "gaussian_log_slope": slope,
        "fits": fits,
        "data_source": detail.get("data_source"),
    }


def plot_perturbation_paths(
    path_b100: dict[str, np.ndarray], peak_table: dict[str, dict[int, float]],
    downstream_dates: pd.DatetimeIndex, path: str,
) -> None:
    """2x3 grid: one subplot per cell with the B=100 perturbation path over
    downstream dates, plus one log-log plot of peak perturbation vs B for all
    four cells (Section 7.1)."""
    names = list(path_b100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    flat_axes = axes.flat
    for ax, name in zip(flat_axes, names):
        ax.plot(downstream_dates, path_b100[name], color="firebrick")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("date")
        ax.set_ylabel("log-variance perturbation")
        ax.tick_params(axis="x", rotation=30)

    ax_loglog = next(flat_axes)
    for name in names:
        peaks = peak_table[name]
        b_vals = sorted(peaks)
        ax_loglog.loglog(b_vals, [peaks[b] for b in b_vals], marker="o", label=name)
    ax_loglog.set_xlabel("B (perturbation size, units of sigma_ref)")
    ax_loglog.set_ylabel("peak |log-variance perturbation|")
    ax_loglog.set_title("Peak perturbation vs B (log-log)")
    ax_loglog.legend(fontsize=8)

    for ax in flat_axes:
        ax.axis("off")

    fig.suptitle("exp7: single-observation outlier propagation, SPY (Section 7.1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="use the paper's L-BFGS-B start count (8) instead of the reduced default (3)")
    args = parser.parse_args()
    run(full=args.full)
