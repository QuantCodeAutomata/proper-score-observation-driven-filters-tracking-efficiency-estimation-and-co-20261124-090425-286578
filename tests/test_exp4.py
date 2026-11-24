"""Unit tests for exp/exp4.py and src/tracking_efficiency.py (Section 6.4,
Table 3): the tracking efficiency frontier for scale-equivariant cells."""
from __future__ import annotations

import math

from src.config import BETA_DP, NU_STUDENT
from src.tracking_efficiency import (
    gaussian_log_efficiency_student,
    target_shift,
    tracking_efficiency,
)

from exp.exp4 import LAWS, RULES, build_table


def test_student_log_correctly_specified_efficiency_near_one():
    """Corollary 3.7: when the working nu matches the true innovation nu
    (both 6), the Student-log cell should be (near-)efficient."""
    result = tracking_efficiency("student_log", "t_6", beta=BETA_DP, nu_work=NU_STUDENT)
    assert abs(result["efficiency"] - 1.0) < 0.05, result


def test_gaussian_log_correctly_specified_efficiency_near_one():
    """Gaussian log score under Gaussian innovations is the correctly
    specified baseline case: eff_log(gaussian) = 1."""
    result = tracking_efficiency("gaussian_log", "gaussian", beta=BETA_DP, nu_work=NU_STUDENT)
    assert abs(result["efficiency"] - 1.0) < 0.02, result


def test_gaussian_log_breaks_down_for_low_dof():
    """Gaussian log score has an infinite-variance update second moment for
    innovations with nu <= 4: efficiency collapses to 0 and the frozen
    tracking-risk multiplier diverges."""
    result = tracking_efficiency("gaussian_log", "t_4", beta=BETA_DP, nu_work=NU_STUDENT)
    assert result["efficiency"] == 0.0
    assert math.isinf(result["risk_multiplier"])


def test_gaussian_log_efficiency_student_closed_form_arithmetic():
    """eff_log(t_nu) = (nu-4)(nu+3) / (nu (nu-1)) for nu > 4 (Section 6.4)."""
    dof = 10
    expected = (dof - 4) * (dof + 3) / (dof * (dof - 1))
    assert gaussian_log_efficiency_student(dof) == expected


def test_target_shift_gaussian_log_is_zero():
    """Bartlett identity: the log-score target is always the latent
    log-variance, so delta_log(F) = 0 for every innovation law F."""
    delta = target_shift("gaussian_log", "t_5", beta=BETA_DP, nu_work=NU_STUDENT)
    assert delta == 0.0


def test_efficiency_in_unit_interval_for_finite_cells():
    """eff_S(F) lies in [0, 1] for every finite (rule, law) cell of the 6x6
    grid used in exp/exp4.py."""
    table = build_table()
    for rule in RULES:
        for law in LAWS:
            eff = table[rule][law]["efficiency"]
            if math.isnan(eff) or math.isinf(eff):
                continue
            assert -1e-8 <= eff <= 1.0 + 1e-6, f"{rule}, {law}: eff={eff}"
