"""exp_4: Tracking efficiency frontier for scale-equivariant cells.

Reproduces Section 6.4 / Table 3: the population tracking efficiency
eff_S(F), the minimal frozen tracking-risk multiplier eff_S(F)^{-1/2} and
the target shift delta_S(F), for six scale-equivariant scoring-rule cells
("gaussian_log", "student_log", "density_power", "crps", "student_crps",
"student_density_power") against six variance-standardised innovation laws
("gaussian", "t_10", "t_6", "t_5", "t_4", "t_3").

Also verifies the closed-form Gaussian-log efficiency formula (Section 6.4)
against the numerical quadrature result, and the Corollary 3.7 sanity check
that eff_{student_log}(t_6) ~= 1 at the correctly-specified working nu=6.

Run as ``python -m exp.exp4``.
"""
from __future__ import annotations

import functools
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import t as student_t

import src.tracking_efficiency as _te
from src.config import BETA_DP, EXP4_STUDENT_DOF, NU_STUDENT
from src.tracking_efficiency import (
    fisher_information,
    gaussian_log_efficiency_student,
    tracking_efficiency,
)

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Pure performance cache (no math change): src.tracking_efficiency's
# ``_student_density_power_score`` recomputes a v-only-dependent normalising
# quadrature (``norm_term``) once per residual value ``y`` even though, for a
# fixed working variance v, that integral does not depend on y at all. That
# makes every expectation integral over the innovation law re-run the same
# expensive quad hundreds of times. Memoising norm_term by (v, beta, nu)
# returns bit-identical values (same formula, same quad call) while cutting
# the redundant recomputation, which is otherwise too slow to finish the 6x6
# grid in reasonable time. The original function is left untouched on disk.
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _norm_term_cached(v: float, beta: float, nu: int) -> float:
    scale2 = v * (nu - 2) / nu
    return integrate.quad(
        lambda x: (student_t.pdf(x / np.sqrt(scale2), df=nu) / np.sqrt(scale2)) ** (1 + beta),
        -40 * np.sqrt(scale2), 40 * np.sqrt(scale2), limit=_te.QUAD_LIMIT,
    )[0]


def _student_density_power_score_cached(v, y, beta: float, nu: int):
    scale2 = v * (nu - 2) / nu
    z2 = y ** 2 / scale2
    f = student_t.pdf(y / np.sqrt(scale2), df=nu) / np.sqrt(scale2)
    norm_term = _norm_term_cached(float(v), beta, nu)
    return -(1 + 1 / beta) * f ** beta + norm_term


_te._student_density_power_score = _student_density_power_score_cached

RULES = (
    "gaussian_log",
    "student_log",
    "density_power",
    "crps",
    "student_crps",
    "student_density_power",
)
LAWS = ("gaussian",) + tuple(f"t_{dof}" for dof in EXP4_STUDENT_DOF)

FIELDS = ("delta", "J_F", "K_F", "kappa", "I_F", "efficiency", "risk_multiplier")


def build_table() -> dict:
    """Compute tracking_efficiency(rule, law, ...) for the full 6x6 grid.

    Returns a nested dict {rule: {law: {field: value}}}.
    """
    table = {}
    for rule in RULES:
        table[rule] = {}
        for law in LAWS:
            table[rule][law] = tracking_efficiency(
                rule, law, beta=BETA_DP, nu_work=NU_STUDENT
            )
    return table


def to_dataframe(table: dict) -> pd.DataFrame:
    """Flatten the nested table into a rule-indexed DataFrame with one
    column per (field, law) combination, e.g. "efficiency_t_6"."""
    data = {}
    for law in LAWS:
        for field in FIELDS:
            data[f"{field}_{law}"] = [table[rule][law][field] for rule in RULES]
    return pd.DataFrame(data, index=pd.Index(RULES, name="rule"))


def _json_safe(obj):
    """Replace non-finite floats with the strings 'inf'/'-inf'/'nan' so the
    table can be serialised with the standard json module."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    return obj


def plot_frontier(table: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = list(range(len(LAWS)))
    for rule in RULES:
        y = [table[rule][law]["efficiency"] for law in LAWS]
        ax.plot(x, y, marker="o", label=rule)
    ax.set_xticks(x)
    ax.set_xticklabels(LAWS)
    ax.set_xlabel("innovation law F (increasing tail heaviness to the right)")
    ax.set_ylabel(r"tracking efficiency $\mathrm{eff}_S(F)$")
    ax.set_title("Tracking efficiency frontier across scale-equivariant cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run() -> dict:
    table = build_table()
    df = to_dataframe(table)
    df.to_csv(f"{RESULTS_DIR}/exp4_efficiency_table.csv")

    plot_frontier(table, f"{RESULTS_DIR}/exp4_efficiency_frontier.png")

    # Closed-form cross-check for the Gaussian-log cell under Student-t laws.
    closed_form_check = {}
    for dof in EXP4_STUDENT_DOF:
        law = f"t_{dof}"
        numeric = table["gaussian_log"][law]["efficiency"]
        closed_form = gaussian_log_efficiency_student(dof)
        closed_form_check[law] = {
            "numeric": numeric,
            "closed_form": closed_form,
            "matches": bool(
                (dof <= 4 and numeric == 0.0 and closed_form == 0.0)
                or (dof > 4 and abs(numeric - closed_form) < 1e-3)
            ),
        }

    # Corollary 3.7 sanity check: student_log is efficient (eff ~= 1) when
    # the working nu matches the true nu (both 6 here).
    student6_t6_efficiency = table["student_log"]["t_6"]["efficiency"]
    sanity_check_student6_t6 = bool(abs(student6_t6_efficiency - 1.0) < 1e-2)

    summary = {
        "sanity_check_student6_t6": sanity_check_student6_t6,
        "student_log_t6_efficiency": student6_t6_efficiency,
        "gaussian_log_closed_form_check": closed_form_check,
        "fisher_information": {law: fisher_information(law) for law in LAWS},
        "table": _json_safe(table),
    }
    with open(f"{RESULTS_DIR}/exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(df.round(6).to_string())
    print(json.dumps(
        {
            "sanity_check_student6_t6": sanity_check_student6_t6,
            "student_log_t6_efficiency": student6_t6_efficiency,
            "gaussian_log_closed_form_check": closed_form_check,
        },
        indent=2,
    ))
    return summary


if __name__ == "__main__":
    run()
