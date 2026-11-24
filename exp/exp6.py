"""exp_6: Empirical equity index application (Section 7, Table G.10).

Fits the 2 (density: Gaussian, Student-t6) x 4 (criterion: log,
density_power, crps, mmd) factorial of observation-driven volatility filters
on a panel of equity-index proxies, out of sample against GARCH(1,1) and
EWMA benchmarks (Section 7). Twelve paper tickers (``EXP6_TICKERS_PAPER``,
Table G.10) are proxied by US-listed ETFs (``EXP6_TICKERS``) because the
Massive index codes require an entitlement this environment does not have
(see ``src/config.py``); a further reduced ticker list and sample start date
(``EXP6_TICKERS_RUN``, ``EXP6_START_DATE_RUN``) and L-BFGS-B start count
(``EXP6_LBFGS_N_STARTS_RUN``) are used by default to keep the Student-t
MMD-RBF cell's finite-difference-gradient fit tractable within the time
available; ``--full`` restores the paper's universe, start date and start
count.

Run as ``python -m exp.exp6`` for the reduced run, or
``python -m exp.exp6 --full`` for the paper's full run.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import DataloaderError, load_ohlcv
from src.config import (
    EXP6_BETA_DP,
    EXP6_END_DATE,
    EXP6_EWMA_DECAY,
    EXP6_EWMA_INIT_WINDOW,
    EXP6_GARCH_A_MAX,
    EXP6_GARCH_B_MAX,
    EXP6_GARCH_N_STARTS,
    EXP6_GARCH_SUM_MAX,
    EXP6_LBFGS_N_STARTS,
    EXP6_LBFGS_N_STARTS_RUN,
    EXP6_MMD_BW_SENSITIVITY,
    EXP6_MMD_SUBSAMPLE_MAX,
    EXP6_NU_STUDENT,
    EXP6_NU_SENSITIVITY_GRID,
    EXP6_START_DATE,
    EXP6_START_DATE_RUN,
    EXP6_START_DATE_STOXX50,
    EXP6_TICKERS,
    EXP6_TICKERS_PAPER,
    EXP6_TICKERS_RUN,
    EXP6_TRAIN_FRACTION,
    EXP6_VAR_LEVELS,
)
from src.empirical import (
    fit_ewma,
    fit_filter,
    fit_garch11,
    filtered_variance_path,
    garman_klass_proxy,
    mmd_bandwidth,
    parkinson_proxy,
    pit_values,
    qlike,
    score,
    var_exceedance_frequency,
)

RESULTS_DIR = Path("results")
CRITERIA = ("log", "density_power", "crps", "mmd")
DENSITIES = ("gaussian", "student")
CELLS = [(density, criterion) for density in DENSITIES for criterion in CRITERIA]
FEZ_TICKER = "FEZ"

# The Student-t MMD-RBF cell's quadrature order inside the L-BFGS-B
# finite-difference-gradient loop is reduced to keep fitting tractable
# (Section 7); the paper's Appendix G.1 reports convergence already at a few
# dozen nodes. The final per-market evaluation (score / filtered_variance_path
# on frozen parameters) uses a larger but still reduced order: the paper's
# Appendix G.1 specifies 256 nodes for the reported final scores, but 64 is
# used here for execution-time reasons.
MMD_NODES_FIT = 12
MMD_NODES_FINAL = 64


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_exp6.py)
# ---------------------------------------------------------------------------
def log_returns_pct(close: np.ndarray) -> np.ndarray:
    """Close-to-close log returns in per cent, 100*diff(log(close))."""
    return 100.0 * np.diff(np.log(close))


def n_train_count(n: int, train_fraction: float = EXP6_TRAIN_FRACTION) -> int:
    """70/30 split by index position on dates: n_train = round(train_fraction*n)."""
    return int(round(train_fraction * n))


def demean_by_training(returns: np.ndarray, n_train: int) -> tuple[np.ndarray, float]:
    """Subtract the TRAINING-sample mean from the whole series (Section 7)."""
    train_mean = float(np.mean(returns[:n_train]))
    return returns - train_mean, train_mean


def build_cross_score_table(
    v_by_cell: dict[str, np.ndarray],
    y: np.ndarray,
    density_by_cell: dict[str, str],
    eval_criteria: tuple[str, ...] = CRITERIA,
    beta: float = EXP6_BETA_DP,
    ell: float = 2.0,
    nu: int = EXP6_NU_STUDENT,
    mmd_nodes: int = MMD_NODES_FINAL,
) -> pd.DataFrame:
    """Row = a fitted cell's frozen variance path, column = mean out-of-sample
    score under each evaluation criterion (Section 7): reproduces the paper's
    claim that a cell's own training criterion does not generally give the
    smallest out-of-sample mean under its own score."""
    rows: dict[str, dict[str, float]] = {}
    for cell_name, v in v_by_cell.items():
        density = density_by_cell[cell_name]
        rows[cell_name] = {
            f"eval_{criterion}": float(
                np.mean(score(v, y, criterion, density, beta=beta, ell=ell, nu=nu, mmd_nodes=mmd_nodes))
            )
            for criterion in eval_criteria
        }
    return pd.DataFrame(rows).T


def roll_garch_variance(y_full: np.ndarray, n_train: int, a0: float, a1: float, b1: float) -> np.ndarray:
    """Continue the GARCH(1,1) recursion v_t = a0 + a1*y_{t-1}^2 + b1*v_{t-1}
    across the train/test boundary without restarting state, seeded as in
    ``fit_garch11`` (Section 7)."""
    T = len(y_full)
    v = np.empty(T)
    v[0] = np.var(y_full[:n_train])
    for t in range(1, T):
        v[t] = a0 + a1 * y_full[t - 1] ** 2 + b1 * v[t - 1]
    return v


def equal_weight_mean(values: list[float]) -> float:
    """Equal (1/n_markets) weighted mean, ignoring the degenerate empty case."""
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_active_data(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, dict]:
    """Load OHLCV bars for ``tickers``, using ``EXP6_START_DATE_STOXX50`` for
    the EURO STOXX 50 proxy ticker FEZ (Section 7: ^STOXX50E starts 2007)."""
    main_tickers = [t for t in tickers if t != FEZ_TICKER]
    fez_tickers = [t for t in tickers if t == FEZ_TICKER]

    frames, details, failed, warnings_by_ticker, non_massive = [], [], {}, {}, {}
    if main_tickers:
        bars, meta = load_ohlcv(main_tickers, start_date, end_date, timeframe="1d")
        frames.append(bars)
        details += meta["ticker_details"]
        failed.update(meta["failed"])
        warnings_by_ticker.update(meta["warnings"])
        non_massive.update(meta["non_massive_source"])
    if fez_tickers:
        bars, meta = load_ohlcv(fez_tickers, EXP6_START_DATE_STOXX50, end_date, timeframe="1d")
        frames.append(bars)
        details += meta["ticker_details"]
        failed.update(meta["failed"])
        warnings_by_ticker.update(meta["warnings"])
        non_massive.update(meta["non_massive_source"])
    bars_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    meta = {
        "ticker_details": details,
        "failed": failed,
        "warnings": warnings_by_ticker,
        "non_massive_source": non_massive,
    }
    return bars_df, meta


# ---------------------------------------------------------------------------
# Per-market pipeline
# ---------------------------------------------------------------------------
def prepare_market(bars_df: pd.DataFrame, ticker: str) -> dict[str, Any] | None:
    """Build demeaned log returns and the test-period volatility proxies for
    ``ticker`` (Section 7); returns None if there is too little data."""
    rows = bars_df[bars_df["ticker"] == ticker].sort_values("date")
    if len(rows) < 100:
        return None
    close = rows["close"].to_numpy(dtype=float)
    open_ = rows["open"].to_numpy(dtype=float)
    high = rows["high"].to_numpy(dtype=float)
    low = rows["low"].to_numpy(dtype=float)
    dates = rows["date"].to_numpy()

    returns = log_returns_pct(close)
    n_train = n_train_count(len(returns))
    if n_train < 50 or len(returns) - n_train < 50:
        return None
    returns, train_mean = demean_by_training(returns, n_train)

    # OHLC proxies aligned with the returns index: return y_t uses close_t
    # vs close_{t-1}, so the same-day OHLC (index 1..N-1) is the matching
    # intraday proxy for date t.
    open_, high, low, close = open_[1:], high[1:], low[1:], close[1:]
    dates = dates[1:]

    parkinson = parkinson_proxy(high, low)
    gk = garman_klass_proxy(open_, high, low, close)
    sq_return = returns ** 2

    return {
        "ticker": ticker,
        "returns": returns,
        "n_train": n_train,
        "train_mean": train_mean,
        "dates": dates,
        "parkinson": parkinson,
        "garman_klass": gk,
        "sq_return": sq_return,
        "row_count": int(len(rows)),
        "date_min": str(pd.Timestamp(dates.min()).date()) if len(dates) else None,
        "date_max": str(pd.Timestamp(dates.max()).date()) if len(dates) else None,
    }


def fit_market_cells(
    market: dict[str, Any], n_starts: int,
) -> tuple[dict[str, dict], dict[str, np.ndarray], float, list[str]]:
    """Fit all 8 (density, criterion) cells on the training sample, then run
    the frozen recursion over train+test to obtain each cell's test-period
    variance path (Section 7). A cell whose fit or recursion raises is
    skipped (logged, not raised) so one failing cell does not drop the whole
    market. Returns (fits, test-variance-paths, frozen MMD bandwidth,
    skipped-cell messages)."""
    returns, n_train = market["returns"], market["n_train"]
    train_returns = returns[:n_train]
    ell = mmd_bandwidth(train_returns, max_subsample=EXP6_MMD_SUBSAMPLE_MAX)

    fits: dict[str, dict] = {}
    v_test: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for density, criterion in CELLS:
        name = f"{density}_{criterion}"
        try:
            fit = fit_filter(
                train_returns, criterion, density, n_starts=n_starts,
                beta=EXP6_BETA_DP, ell=ell, nu=EXP6_NU_STUDENT, seed=0,
                mmd_nodes=MMD_NODES_FIT,
            )
            v_full = filtered_variance_path(
                returns, fit["omega"], fit["phi"], fit["alpha"], criterion,
                density=density, beta=EXP6_BETA_DP, ell=ell, nu=EXP6_NU_STUDENT,
                mmd_nodes=MMD_NODES_FINAL,
            )
        except Exception as exc:  # noqa: BLE001 - one cell's fit failed, skip only that cell
            skipped.append(f"{name}: {exc}")
            continue
        fits[name] = fit
        v_test[name] = v_full[n_train:]
    return fits, v_test, ell, skipped


def fit_market_benchmarks(market: dict[str, Any]) -> dict[str, np.ndarray]:
    """Gaussian/Student-t6 GARCH(1,1) and EWMA(0.94) benchmarks, fit on the
    training sample and rolled across train+test without restarting state
    (Section 7)."""
    returns, n_train = market["returns"], market["n_train"]
    train_returns = returns[:n_train]

    out: dict[str, np.ndarray] = {}
    garch_n = fit_garch11(train_returns, n_starts=EXP6_GARCH_N_STARTS, a_max=EXP6_GARCH_A_MAX,
                           b_max=EXP6_GARCH_B_MAX, sum_max=EXP6_GARCH_SUM_MAX, density="gaussian")
    out["garch_gaussian"] = roll_garch_variance(returns, n_train, garch_n["a0"], garch_n["a1"], garch_n["b1"])[n_train:]

    garch_t = fit_garch11(train_returns, n_starts=EXP6_GARCH_N_STARTS, a_max=EXP6_GARCH_A_MAX,
                           b_max=EXP6_GARCH_B_MAX, sum_max=EXP6_GARCH_SUM_MAX, density="student",
                           nu=EXP6_NU_STUDENT)
    out["garch_student"] = roll_garch_variance(returns, n_train, garch_t["a0"], garch_t["a1"], garch_t["b1"])[n_train:]

    out["ewma"] = fit_ewma(returns, decay=EXP6_EWMA_DECAY, init_window=EXP6_EWMA_INIT_WINDOW)[n_train:]
    return out


def evaluate_cell(
    v_test: np.ndarray, market: dict[str, Any], density: str, nu: int = EXP6_NU_STUDENT,
) -> dict[str, float]:
    """QLIKE under the three test-period proxies and 1%/5% VaR exceedance
    frequencies for one frozen variance path (Section 7). ``market``'s proxy
    arrays are stored full-length (aligned with the full returns series), so
    they are sliced to the test period here, matching ``v_test``."""
    n_train = market["n_train"]
    y_test = market["returns"][n_train:]
    out = {
        "qlike_sq_return": qlike(v_test, market["sq_return"][n_train:]),
        "qlike_parkinson": qlike(v_test, market["parkinson"][n_train:]),
        "qlike_garman_klass": qlike(v_test, market["garman_klass"][n_train:]),
    }
    for level in EXP6_VAR_LEVELS:
        out[f"exceedance_{level:.2f}"] = var_exceedance_frequency(y_test, v_test, level, density, nu=nu)
    return out


# ---------------------------------------------------------------------------
# Sensitivity analyses (single representative market)
# ---------------------------------------------------------------------------
def run_nu_sensitivity(market: dict[str, Any], n_starts: int) -> pd.DataFrame:
    """Student-t nu in EXP6_NU_SENSITIVITY_GRID for the log and crps
    criteria, on the representative market (Appendix G.6)."""
    returns, n_train = market["returns"], market["n_train"]
    train_returns = returns[:n_train]
    y_test = returns[n_train:]
    rows = []
    for criterion in ("log", "crps"):
        for nu in EXP6_NU_SENSITIVITY_GRID:
            fit = fit_filter(train_returns, criterion, "student", n_starts=n_starts,
                              beta=EXP6_BETA_DP, nu=nu, seed=0, mmd_nodes=MMD_NODES_FIT)
            v_full = filtered_variance_path(returns, fit["omega"], fit["phi"], fit["alpha"], criterion,
                                             density="student", beta=EXP6_BETA_DP, nu=nu, mmd_nodes=MMD_NODES_FINAL)
            v_test = v_full[n_train:]
            rows.append({
                "criterion": criterion, "nu": nu,
                "qlike_sq_return": qlike(v_test, market["sq_return"][n_train:]),
                "mean_own_score": float(np.mean(score(v_test, y_test, criterion, "student", beta=EXP6_BETA_DP,
                                                        nu=nu, mmd_nodes=MMD_NODES_FINAL))),
                "exceedance_0.01": var_exceedance_frequency(y_test, v_test, 0.01, "student", nu=nu),
                "exceedance_0.05": var_exceedance_frequency(y_test, v_test, 0.05, "student", nu=nu),
            })
    return pd.DataFrame(rows)


def run_mmd_bandwidth_sensitivity(market: dict[str, Any], ell0: float, n_starts: int) -> pd.DataFrame:
    """MMD bandwidth at EXP6_MMD_BW_SENSITIVITY multiples of the frozen
    training bandwidth, both densities (Appendix G.6)."""
    returns, n_train = market["returns"], market["n_train"]
    train_returns = returns[:n_train]
    y_test = returns[n_train:]
    rows = []
    for density in DENSITIES:
        for mult in EXP6_MMD_BW_SENSITIVITY:
            ell = ell0 * mult
            fit = fit_filter(train_returns, "mmd", density, n_starts=n_starts,
                              beta=EXP6_BETA_DP, ell=ell, nu=EXP6_NU_STUDENT, seed=0, mmd_nodes=MMD_NODES_FIT)
            v_full = filtered_variance_path(returns, fit["omega"], fit["phi"], fit["alpha"], "mmd",
                                             density=density, beta=EXP6_BETA_DP, ell=ell, nu=EXP6_NU_STUDENT,
                                             mmd_nodes=MMD_NODES_FINAL)
            v_test = v_full[n_train:]
            rows.append({
                "density": density, "bandwidth_multiplier": mult, "bandwidth": ell,
                "qlike_sq_return": qlike(v_test, market["sq_return"][n_train:]),
                "mean_mmd_score": float(np.mean(score(v_test, y_test, "mmd", density, beta=EXP6_BETA_DP,
                                                        ell=ell, nu=EXP6_NU_STUDENT, mmd_nodes=MMD_NODES_FINAL))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pit_histograms(pit_by_cell: dict[str, np.ndarray], path: str) -> None:
    """2x4 grid of PIT histograms pooled across markets, one per cell, with
    the reference U(0,1) density line (Section 7)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, (density, criterion) in zip(axes.flat, CELLS):
        name = f"{density}_{criterion}"
        u = pit_by_cell.get(name, np.array([]))
        if len(u):
            ax.hist(u, bins=20, range=(0, 1), density=True, color="steelblue", alpha=0.8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Pooled out-of-sample PIT histograms by cell (Section 7)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run(full: bool = False) -> dict[str, Any]:
    """Run the Section 7 factorial over the active ticker list and write all
    outputs under ``results/`` (see module docstring for the --full switch)."""
    start_time = time.time()
    tickers = EXP6_TICKERS if full else EXP6_TICKERS_RUN
    start_date = EXP6_START_DATE if full else EXP6_START_DATE_RUN
    n_starts = EXP6_LBFGS_N_STARTS if full else EXP6_LBFGS_N_STARTS_RUN

    print(f"exp6: tickers={tickers}")
    print(f"exp6: start_date={start_date} end_date={EXP6_END_DATE} n_starts={n_starts} full={full}")
    print(f"exp6: paper tickers for reference: {EXP6_TICKERS_PAPER}")

    bars_df, meta = load_active_data(tickers, start_date, EXP6_END_DATE)
    if meta["failed"]:
        print(f"exp6: tickers that FAILED to load: {meta['failed']}")
    for ticker, source in meta["non_massive_source"].items():
        print(f"exp6: ticker {ticker} loaded from non-massive source: {source}")
    for ticker, warn in meta["warnings"].items():
        print(f"exp6: ticker {ticker} warnings: {warn}")

    ticker_detail_rows = []
    markets: dict[str, dict] = {}
    skipped: list[str] = []
    for ticker in tickers:
        requested_start = EXP6_START_DATE_STOXX50 if ticker == FEZ_TICKER else start_date
        detail = next((d for d in meta["ticker_details"] if d.get("ticker") == ticker), None)
        market = None
        try:
            if ticker in meta["failed"]:
                raise DataloaderError(meta["failed"][ticker])
            market = prepare_market(bars_df, ticker)
            if market is None:
                raise ValueError("too little data after the train/test split")
        except (DataloaderError, ValueError) as exc:
            skipped.append(f"{ticker}: {exc}")
            print(f"exp6: SKIPPING {ticker}: {exc}")
        ticker_detail_rows.append({
            "ticker": ticker,
            "requested_start": requested_start,
            "requested_end": EXP6_END_DATE,
            "realized_start": market["date_min"] if market else None,
            "realized_end": market["date_max"] if market else None,
            "row_count": market["row_count"] if market else 0,
            "data_source": detail.get("data_source") if detail else None,
            "warnings": meta["warnings"].get(ticker),
            "failure": meta["failed"].get(ticker),
        })
        if market is not None:
            markets[ticker] = market
            print(f"exp6: {ticker} rows={market['row_count']} dates=[{market['date_min']}, {market['date_max']}] "
                  f"n_train={market['n_train']} n_test={len(market['returns']) - market['n_train']}")

    pd.DataFrame(ticker_detail_rows).to_csv(RESULTS_DIR / "exp6_ticker_details.csv", index=False)

    cell_eval_records: dict[str, list[dict]] = {f"{d}_{c}": [] for d, c in CELLS}
    bench_eval_records: dict[str, list[dict]] = {"garch_gaussian": [], "garch_student": [], "ewma": []}
    cross_score_records: dict[str, list[dict]] = {f"{d}_{c}": [] for d, c in CELLS}
    pit_pool: dict[str, list[np.ndarray]] = {f"{d}_{c}": [] for d, c in CELLS}
    cell_skips: list[str] = []
    first_market_name = None

    for ticker, market in markets.items():
        if first_market_name is None:
            first_market_name = ticker
        fits, v_test_by_cell, ell, fit_skips = fit_market_cells(market, n_starts)
        for msg in fit_skips:
            cell_skips.append(f"{msg} for {ticker}")
            print(f"exp6: SKIPPING {msg} for {ticker}")
        if not v_test_by_cell:
            print(f"exp6: SKIPPING all cells for {ticker}: no cell fit succeeded")
            continue

        y_test = market["returns"][market["n_train"]:]
        density_by_cell = {f"{d}_{c}": d for d, c in CELLS}
        for name, v_test in v_test_by_cell.items():
            density = density_by_cell[name]
            try:
                cell_eval_records[name].append(evaluate_cell(v_test, market, density))
                pit_pool[name].append(pit_values(y_test, v_test, density, nu=EXP6_NU_STUDENT))
                cross_score_records[name].append({
                    f"eval_{criterion}": float(np.mean(
                        score(v_test, y_test, criterion, density, beta=EXP6_BETA_DP, ell=ell,
                              nu=EXP6_NU_STUDENT, mmd_nodes=MMD_NODES_FINAL)
                    ))
                    for criterion in CRITERIA
                })
            except Exception as exc:  # noqa: BLE001 - one cell's evaluation failed, skip only that cell
                cell_skips.append(f"{name} for {ticker}: {exc}")
                print(f"exp6: SKIPPING {name} for {ticker}: {exc}")

        try:
            bench_v = fit_market_benchmarks(market)
            bench_eval_records["garch_gaussian"].append(evaluate_cell(bench_v["garch_gaussian"], market, "gaussian"))
            bench_eval_records["garch_student"].append(evaluate_cell(bench_v["garch_student"], market, "student"))
            bench_eval_records["ewma"].append(evaluate_cell(bench_v["ewma"], market, "gaussian"))
        except Exception as exc:  # noqa: BLE001
            cell_skips.append(f"benchmarks for {ticker}: {exc}")
            print(f"exp6: SKIPPING benchmarks for {ticker}: {exc}")

    # Factorial + benchmark summary table, equal-weighted across markets.
    summary_rows = {}
    for name, records in {**cell_eval_records, **bench_eval_records}.items():
        if not records:
            continue
        summary_rows[name] = {
            "mean_qlike_sq_return": equal_weight_mean([r["qlike_sq_return"] for r in records]),
            "mean_qlike_parkinson": equal_weight_mean([r["qlike_parkinson"] for r in records]),
            "mean_qlike_garman_klass": equal_weight_mean([r["qlike_garman_klass"] for r in records]),
            "mean_exceedance_0.01": equal_weight_mean([r["exceedance_0.01"] for r in records]),
            "mean_exceedance_0.05": equal_weight_mean([r["exceedance_0.05"] for r in records]),
            "n_markets": len(records),
        }
    factorial_summary = pd.DataFrame(summary_rows).T
    factorial_summary.to_csv(RESULTS_DIR / "exp6_factorial_summary.csv")

    cross_score_cols = {f"eval_{c}": c for c in CRITERIA}
    cross_rows = {
        name: {col: equal_weight_mean([r[col] for r in records]) for col in cross_score_cols}
        for name, records in cross_score_records.items() if records
    }
    cross_score = pd.DataFrame(cross_rows).T
    cross_score.to_csv(RESULTS_DIR / "exp6_cross_score.csv")

    # Sensitivity grids on the first successfully loaded market only.
    nu_sensitivity = pd.DataFrame()
    mmd_sensitivity = pd.DataFrame()
    if first_market_name is not None:
        rep_market = markets[first_market_name]
        try:
            nu_sensitivity = run_nu_sensitivity(rep_market, n_starts)
        except Exception as exc:  # noqa: BLE001
            cell_skips.append(f"nu sensitivity: {exc}")
            print(f"exp6: SKIPPING nu sensitivity: {exc}")
        try:
            rep_ell = mmd_bandwidth(rep_market["returns"][:rep_market["n_train"]], max_subsample=EXP6_MMD_SUBSAMPLE_MAX)
            mmd_sensitivity = run_mmd_bandwidth_sensitivity(rep_market, rep_ell, n_starts)
        except Exception as exc:  # noqa: BLE001
            cell_skips.append(f"mmd bandwidth sensitivity: {exc}")
            print(f"exp6: SKIPPING mmd bandwidth sensitivity: {exc}")
    nu_sensitivity.to_csv(RESULTS_DIR / "exp6_sensitivity_nu.csv", index=False)
    mmd_sensitivity.to_csv(RESULTS_DIR / "exp6_sensitivity_mmd_bandwidth.csv", index=False)

    pooled_pit = {name: (np.concatenate(arrs) if arrs else np.array([])) for name, arrs in pit_pool.items()}
    plot_pit_histograms(pooled_pit, str(RESULTS_DIR / "exp6_pit_histograms.png"))

    elapsed = time.time() - start_time
    print(f"exp6: elapsed {elapsed:.1f}s, markets used: {list(markets)}, skipped tickers: {skipped}")
    print(f"exp6: skipped cells/steps: {cell_skips}")
    print("exp6: equal-market factorial + benchmark summary:")
    print(factorial_summary.to_string())

    return {
        "elapsed_seconds": elapsed,
        "tickers_used": list(markets),
        "skipped_tickers": skipped,
        "skipped_cells": cell_skips,
        "factorial_summary": factorial_summary,
        "cross_score": cross_score,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="run the paper's full ticker list, 2000 start date and L-BFGS-B start count")
    args = parser.parse_args()
    run(full=args.full)
