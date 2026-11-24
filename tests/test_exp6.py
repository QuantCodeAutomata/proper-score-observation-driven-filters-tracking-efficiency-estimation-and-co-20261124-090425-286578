"""Unit tests for exp/exp6.py (Section 7, empirical equity index
application): fast checks on tiny synthetic return series, no network
calls."""
from __future__ import annotations

import math

import numpy as np

from exp.exp6 import (
    build_cross_score_table,
    demean_by_training,
    n_train_count,
)
from src.empirical import qlike


def test_demean_by_training_subtracts_exact_training_mean():
    """demean_by_training must subtract exactly the mean of the training
    slice (indices [:n_train]) from the whole series, train and test alike."""
    returns = np.array([1.0, 2.0, 3.0, 4.0, 10.0, -10.0])
    n_train = 4
    train_mean = returns[:n_train].mean()
    demeaned, reported_mean = demean_by_training(returns, n_train)
    assert reported_mean == train_mean
    np.testing.assert_allclose(demeaned, returns - train_mean)
    # The demeaned training slice must now average to exactly zero.
    assert abs(demeaned[:n_train].mean()) < 1e-12


def test_qlike_finite_and_not_negative_infinite_for_sane_variance_path():
    """QLIKE must be a finite real number (not +/-inf, not NaN) for a sane,
    strictly positive variance path and a bounded proxy."""
    rng = np.random.default_rng(0)
    variance_pred = np.full(200, 1.5) + rng.uniform(-0.2, 0.2, size=200)
    proxy = rng.chisquare(df=1, size=200) * 1.2
    value = qlike(variance_pred, proxy)
    assert math.isfinite(value)
    assert value > -math.inf


def test_train_test_split_uses_round_of_train_fraction():
    """n_train_count must equal round(0.70*n) exactly, by index position, for
    a range of sample sizes (even and odd n)."""
    for n in (10, 11, 100, 101, 999, 1000, 2503):
        expected = int(round(0.70 * n))
        assert n_train_count(n, 0.70) == expected
        # The split must be strictly within range for non-trivial n.
        assert 0 <= n_train_count(n, 0.70) <= n


def test_train_test_split_default_fraction_matches_config():
    """The default train_fraction of n_train_count matches
    EXP6_TRAIN_FRACTION=0.70 from src/config.py."""
    from src.config import EXP6_TRAIN_FRACTION

    assert EXP6_TRAIN_FRACTION == 0.70
    n = 777
    assert n_train_count(n) == int(round(0.70 * n))


def test_cross_score_table_is_square_and_finite_on_tiny_synthetic_example():
    """build_cross_score_table must produce a square (4x4) table of finite
    entries when given four synthetic cells and four evaluation criteria."""
    rng = np.random.default_rng(1)
    T = 64
    y = rng.standard_normal(T) * 0.8
    criteria = ("log", "density_power", "crps", "mmd")
    v_by_cell = {f"gaussian_{c}": np.full(T, 0.7) + rng.uniform(-0.05, 0.05, size=T) for c in criteria}
    density_by_cell = {name: "gaussian" for name in v_by_cell}

    table = build_cross_score_table(v_by_cell, y, density_by_cell, eval_criteria=criteria, ell=1.0)

    assert table.shape[0] == table.shape[1] == 4
    assert np.all(np.isfinite(table.to_numpy()))


def test_cross_score_table_rows_match_input_cells():
    """The table's row index must match the fitted cells passed in, in order."""
    rng = np.random.default_rng(2)
    T = 32
    y = rng.standard_normal(T) * 0.5
    v_by_cell = {
        "gaussian_log": np.full(T, 0.5),
        "student_log": np.full(T, 0.5),
    }
    density_by_cell = {"gaussian_log": "gaussian", "student_log": "student"}
    table = build_cross_score_table(v_by_cell, y, density_by_cell, eval_criteria=("log", "crps"), nu=6)
    assert list(table.index) == ["gaussian_log", "student_log"]
    assert list(table.columns) == ["eval_log", "eval_crps"]
    assert np.all(np.isfinite(table.to_numpy()))
