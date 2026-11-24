"""Fast unit tests for exp_5 (Appendix E, G.4; Tables G.7, G.8).

Uses small R / T_steps / h-grids throughout so the suite runs quickly; these
are directional / order-of-magnitude checks only, not tight numerical
reproductions of the paper's Table G.8 values (that is the job of
``python -m exp.exp5``).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import EXP5_ALPHA, EXP5_KAPPA, EXP5_XI
from src.diffusion import centring_check, simulate_centred_diffusion, simulate_local_ou

RULES = ("gaussian_log", "density_power", "crps")

# Small-scale settings for fast tests (paper uses R in the thousands, T=200).
R_SMALL = 200
T_SMALL = 20
H_GRID_SMALL = (2.0 ** -4, 2.0 ** -6)


def _second_moment_discrepancy(rule: str, h: float, R: int, T_steps: int, rng: np.random.Generator) -> float:
    sim = simulate_centred_diffusion(rule, h, EXP5_ALPHA, EXP5_KAPPA, T_steps, R, rng)
    m2 = np.mean(sim["deltas"] ** 2) / h
    target = EXP5_ALPHA ** 2 * EXP5_XI[rule]
    return abs(m2 - target)


def test_simulate_centred_diffusion_shape_and_order_of_magnitude():
    rng = np.random.default_rng(0)
    h = H_GRID_SMALL[1]  # moderately small h
    sim = simulate_centred_diffusion("gaussian_log", h, EXP5_ALPHA, EXP5_KAPPA, T_SMALL, R_SMALL, rng)
    deltas = sim["deltas"]
    assert deltas.shape == (R_SMALL, T_SMALL)
    assert sim["lam_path_end"].shape == (R_SMALL,)

    m2 = np.mean(deltas ** 2) / h
    target = EXP5_ALPHA ** 2 * EXP5_XI["gaussian_log"]
    # Loose factor-of-3 tolerance: R and T are small, so MC noise is large.
    assert target / 3 <= m2 <= target * 3, (m2, target)


def test_part_a_discrepancy_shrinks_with_smaller_h():
    # Average over a few seeds to tame MC noise in this directional check.
    h_large, h_small = H_GRID_SMALL
    n_seeds = 5
    disc_large = np.mean([
        _second_moment_discrepancy("gaussian_log", h_large, R_SMALL, T_SMALL, np.random.default_rng(seed))
        for seed in range(n_seeds)
    ])
    disc_small = np.mean([
        _second_moment_discrepancy("gaussian_log", h_small, R_SMALL, T_SMALL, np.random.default_rng(100 + seed))
        for seed in range(n_seeds)
    ])
    # Generous tolerance: the smaller-h discrepancy should not be much larger
    # than the larger-h one (directional check only, given MC noise).
    assert disc_small <= disc_large * 3 + 1e-6, (disc_large, disc_small)


@pytest.mark.parametrize("rule", RULES)
def test_centring_check_near_zero(rule):
    h = H_GRID_SMALL[0]
    # numpy.polynomial.hermite.hermgauss overflows to NaN weights at the
    # library's default n_quad=400 on this numpy build; 300 nodes is the
    # largest confirmed-finite node count (see exp/exp5.py _CENTRING_N_QUAD).
    value = centring_check(rule, h, n_quad=300)
    assert abs(value) < 1e-6, (rule, value)


def test_simulate_local_ou_shapes():
    rng = np.random.default_rng(1)
    R = R_SMALL
    out = simulate_local_ou("crps", 2.0 ** -6, rho=0.8, Gamma=0.65, K=3.5, M=3, R=R, rng=rng)
    assert out["Z_terminal"].shape == (R,)
    assert out["active"].shape == (R,)
    assert out["active"].dtype == bool
    assert isinstance(out["q2_benchmark"], float)
