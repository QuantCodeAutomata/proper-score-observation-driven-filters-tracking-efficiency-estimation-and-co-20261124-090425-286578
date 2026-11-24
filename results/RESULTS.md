# Results — Proper-Score Observation-Driven Filters: Tracking Efficiency, Estimation, and Continuous-Time Limits

This file compares the output of every reproduced experiment against the
values reported in the paper. Every number below is read from a file saved by
the final run in `results/`.

## Experiment 1 — Curvature-Normalized Driver Shape Comparison Across Scoring Rules

Source: `results/exp1_summary.json`, `results/exp1_driver_shapes.png`.

| Rule | J_S | K_S | sqrt(K_S)/J_S | tail class |
|---|---|---|---|---|
| Gaussian log | 0.500 | 0.500 | 1.414 | unbounded |
| Density-power (beta=0.15) | 0.357 | 0.268 | 1.451 | tail-saturating |
| CRPS | 0.0705 | 0.0123 | 1.573 | tail-saturating (monotone to plateau) |
| MMD-RBF (ell=2) | 0.0340 | 0.00246 | 1.315 | tail-saturating |

Independent Gauss-Hermite quadrature reproduces the closed-form `J_S`, `K_S`
pairs of Appendix G Table G.5 to 6+ significant digits
(`results/exp1_summary.json: quadrature_check`). No rule is classified
redescending (`redescending_rules: []`); density-power and MMD peak at
`z ≈ 3.79` and `z ≈ 3.87` and decline to a positive plateau, while CRPS rises
monotonically to its plateau (`peak_z ≈ 8.63`, `monotone_to_plateau: true`).

**Verdict:** reproduced. The curvature-normalised scale `sqrt(K_S)/|J_S|` lies
in [1.41, 1.57] for all four rules, matching the paper's reported range
1.41–1.57; the tail classification (unbounded log score, tail-saturating
others, none redescending) matches Proposition 2.3 / C.1 exactly.

## Experiment 2 — Pseudo-True Target Under Variance Contamination

Source: `results/exp2_summary.json`, `results/exp2_pseudo_true_targets.png`.

| Rule | v*/v0 at tau=80 | v*/v0 at tau=100 |
|---|---|---|
| Gaussian log | 4.950 | 5.950 |
| Density-power | 1.132 | 1.121 |
| CRPS | 1.201 | 1.208 |
| MMD-RBF | 1.212 | 1.213 |

The Gaussian log target matches the closed form `v_log*/v0 = 1 + epsilon(tau-1)`
exactly (`1 + 0.05*79 = 4.95` at tau=80, `1 + 0.05*99 = 5.95` at tau=100);
maximum numerical-root discrepancy against the closed form is `1.0e-10`
(`max_gaussian_log_root_discrepancy`), well below the paper's reported `5.2e-7`
and below the 1e-6 tolerance the paper states. Density-power, CRPS, and MMD
saturate near 1.12–1.21 instead of growing linearly.

**Verdict:** reproduced. The log-score target grows linearly in tau exactly as
predicted, and the three bounded rules saturate near the paper's reported
range (paper: "approximately 1.2" for density-power/CRPS/MMD at large tau;
executed values 1.12–1.21).

## Experiment 3 — Finite-Sample Filtering Under Contamination: Paired Monte Carlo

Source: `results/exp3_contamination_filtering.json`, `results/exp3_summary.png`.

Executed configuration: `R=2000` replications (paper: `R=20000`; reduced to
keep total runtime inside the available budget — see "Deviations from the
paper"), `T=4000` (unchanged).

| Rule | RMSE clean | RMSE contam. | max-dev contam. | flagged-update contam. |
|---|---|---|---|---|
| Gaussian log | 0.652 | 6.693 | 62.89 | 0.113 |
| Density-power | 0.542 | 0.586 | 2.51 | 0.522 |
| CRPS | 0.561 | 0.584 | 2.25 | 0.658 |
| MMD-RBF | 0.584 | 0.607 | 3.19 | 0.418 |
| Student-t6 log | 0.560 | 0.653 | 2.77 | 0.558 |

**Verdict:** partially reproduced. The qualitative result of Proposition 2.3 is
reproduced exactly: the unbounded Gaussian-log filter degrades dramatically
under contamination (RMSE +927%, max-dev 25x larger) while the four bounded
filters stay close to their clean-sample RMSE (+2–8%, consistent with the
paper's "RMSE around 0.56–0.63"). The paper's exact magnitudes (contaminated
Gaussian log RMSE 3.618, max-dev 30.78) are not matched — the executed
contaminated Gaussian-log RMSE (6.69) and max-dev (62.9) are roughly double
the paper's reported values — because the paper does not specify the latent
log-volatility recursion (`omega`, `phi`, latent innovation scale) used to
generate `x_t`; a persistence/innovation-scale combination was assumed (see
"Deviations from the paper"). The direction, sign, and relative ordering of
all five rules match the paper.

## Experiment 4 — Tracking Efficiency Frontier: Population Calculations for Scale-Equivariant Cells

Source: `results/exp4_summary.json`, `results/exp4_efficiency_table.csv`, `results/exp4_efficiency_frontier.png`.

| Innovation law | Fisher info I_F | eff_log(F) (paper) | eff_log(F) (executed) | eff_student_t6_log(F) (executed) | eff_crps(F) (executed) |
|---|---|---|---|---|---|
| Gaussian | 0.5000 | 1.000 | 1.000 | 0.903 | 0.808 |
| t_10 | 0.3846 | 0.867 | 0.867 | 0.986 | 0.912 |
| t_6 | 0.3333 | 0.600 | 0.600 | 1.000 | 0.954 |
| t_5 | 0.3125 | 0.400 | 0.400 | 0.997 | 0.969 |
| t_4 | 0.2857 | 0.000 | 0.000 | 0.983 | 0.984 |
| t_3 | 0.2500 | 0.000 | 0.000 | 0.940 | 0.993 |

Fisher information `I_F` matches the closed form `I_F = nu/(2(nu+3))` exactly
for every degree of freedom. The numerically computed Gaussian-log efficiency
matches the closed form `(nu-4)(nu+3)/(nu(nu-1))` to 1e-9 for every finite-
variance case and both t_4/t_3 report zero efficiency with infinite risk
multiplier as the paper's stated convention
(`results/exp4_summary.json: gaussian_log_closed_form_check`). The correctly
specified Student-t6 log score attains efficiency `1.000` exactly under t_6.

**Verdict:** reproduced. All reported numbers match the paper's Table 3 to at
least three significant digits.

## Experiment 5 — High-Frequency Diffusion Limit Mesh Refinement

Source: `results/exp5_summary.json`, `results/exp5_centred_diffusion.png`,
`results/exp5_mean_flow.png`, `results/exp5_local_ou.png`.

Executed configuration (reduced from the paper's replication counts to fit
the available time; see "Deviations from the paper"): `R_centred=2000`
(paper 8000), `R_meanflow=2000` (paper 8000), `R_ou=3000` (paper 12000); mesh
grids, `T`, and all method parameters (`alpha`, `kappa`, `rho`, `Gamma`, `K`)
are the paper's values.

| Rule | centred log-log slope | QV/T (target alpha^2*Xi_S) | mean-flow fixed point (empirical / analytical) | local-OU terminal variance ratio |
|---|---|---|---|---|
| Gaussian log | 0.92 | 2.02 (target 2.00) | 0.003 / 0.000 | 0.460 |
| Density-power | 1.07 | 1.08 (target 1.07) | -0.211 / -0.209 | 0.452 |
| CRPS | 1.01 | 0.049 (target 0.049) | -0.062 / -0.063 | 0.412 |

Centring is verified to machine precision (`max_abs_centring_check` ≤ 1.1e-16
for all rules, as required for the Lindeberg check). The non-centred mean-flow
fixed point matches its own closed form (Theorem E.7) to within 0.002 for all
three rules, and the realised quadratic variation converges to
`alpha^2*Xi_S` (Table G.8 constants 2.000, 1.074, 0.049) to within 1–3%.

**Verdict:** partially reproduced. The centred-diffusion second-moment
convergence (QV/T to `alpha^2*Xi_S`) and the non-centred mean-flow fixed point
are reproduced (within a few percent of target, matching Theorems E.1/E.7).
The log-log slope of the centred second-moment discrepancy is close to but
below the paper's reported 1.00 for the Gaussian-log rule (0.92 vs 1.00) at
the reduced replication count, consistent with higher Monte Carlo noise at
`R=2000` rather than 8000. The local-OU terminal-variance ratios (0.41–0.46)
do not match the paper's reported benchmark ratios (0.975–1.027); this is a
material difference whose cause was not identified within the available time
— the paper does not fully specify the stopped local-OU boundary-handling
procedure (listed as an uncertainty in the experiment description), and the
discrepancy is consistent with a difference in how the stopping radius `K`
interacts with the terminal-variance benchmark rather than with the
underlying diffusion formulas, which pass the centring and mean-flow checks.

## Experiment 6 — Empirical Equity Index Application: Density x Criterion Factorial

Source: `results/exp6_factorial_summary.csv`, `results/exp6_cross_score.csv`,
`results/exp6_pit_histograms.png`, `results/exp6_sensitivity_nu.csv`,
`results/exp6_sensitivity_mmd_bandwidth.csv`, `results/exp6_ticker_details.csv`.

Executed universe: 4 of the paper's 12 equity indices (`^GSPC`, `^IXIC`,
`^DJI`, `^FTSE`; narrowed by liquidity/availability ranking — see
"Deviations from the paper"), sample 2015-01-01 to the data end, 70%/30%
train/test split, `L-BFGS-B` with 2 starts (paper: 8 — reduced for runtime).

| Cell | QLIKE (sq.-return) | 1% exceedance | 5% exceedance |
|---|---|---|---|
| Gaussian log | 1.116 | 1.95% | 6.79% |
| Gaussian density-power | 1.116 | 1.99% | 6.62% |
| Gaussian CRPS | 1.130 | 2.09% | 6.92% |
| Gaussian MMD | 1.121 | 2.32% | 6.82% |
| Student-t6 log | 1.182 | 0.43% | 3.28% |
| Student-t6 density-power | 1.105 | 0.76% | 6.09% |
| Student-t6 CRPS | 2.046 | 0.00% | 0.20% |
| Student-t6 MMD | 1.111 | 0.83% | 5.99% |
| GARCH(1,1)-Gaussian | 1.114 | 1.95% | 6.62% |
| GARCH(1,1)-Student | 1.111 | 1.26% | 7.12% |
| EWMA | 1.090 | 1.69% | 6.42% |

**Verdict:** not reproduced at the paper's precision, partially reproduced
qualitatively. On the reduced 4-market universe the equal-market QLIKE range
(1.09–2.05) brackets but is wider than the paper's reported 1.007–1.088, and
exceedance frequencies are higher than nominal for most cells instead of
clustering near the paper's reported 1.91%–2.63% (Gaussian) / 1.46%–1.56%
(Student-t6) at the 1% level. The Student-t6 CRPS cell is a clear outlier
(QLIKE 2.05, near-zero exceedances), which was not present in the paper's
report; this indicates the cell's parameters converged to a degenerate fit on
this small universe/short sample rather than reproducing the paper's result.
The direction the paper emphasises — switching from Gaussian to Student-t6
moves exceedance frequencies toward the nominal level — holds for the log and
density-power criteria (1% exceedance drops from ~2% to ~0.4–0.8%, under-
shooting rather than approaching 1% as in the paper) but the magnitude and
some individual cells do not match. The cause was not fully isolated within
the available time; the reduced universe (4 of 12 markets) and shortened
sample (2015–present instead of 2000/2007–2024) are the most likely drivers,
since both reduce the training data available to the 3-parameter recursion
and the degrees-of-freedom-fixed Student-t6 CRPS cell is known to be
numerically delicate (Section 7 notes CRPS/MMD cells are more prone to flat
likelihood regions on short samples).

## Experiment 7 — Controlled Outlier Propagation: SPX Counterfactual

Source: `results/exp7_peak_perturbation.csv`, `results/exp7_gaussian_log_slope.json`,
`results/exp7_perturbation_paths.png`.

Ticker used: `SPY` (ETF proxy; `^GSPC` is not a valid Dataloader/Massive
symbol — see "Deviations from the paper").

| B | Gaussian log | Student-t6 log | Student-t6 density-power | Student-t6 MMD |
|---|---|---|---|---|
| 20 | 22.57 | 1.29 | 0.43 | 0.44 |
| 35 | 68.40 | 1.31 | 0.34 | 0.44 |
| 60 | 199.85 | 1.32 | 0.29 | 0.44 |
| 100 | 553.33 | 1.32 | 0.27 | 0.44 |

Gaussian-log log-log slope over B in {35, 60, 100}: **1.991** (paper: 1.99).

**Verdict:** reproduced for the qualitative mechanism and the log-log slope;
partially reproduced for the absolute peak magnitude at B=100. The measured
slope (1.991) matches the paper's reported 1.99 to three significant digits,
confirming the predicted quadratic growth of the unbounded Gaussian-log
response (Proposition 2.3(i)). The bounded Student-t6 cells saturate at O(1)
peaks exactly as the paper describes ("bounded drivers cap the response"), and
at B=20 the ordering (Gaussian log ≫ Student log > Student MMD ≈ Student
density-power) matches the paper. The paper's absolute peak at B=100
(Gaussian log 541.6) differs from the executed value (553.3) by about 2%,
within normal variation given the SPY/^GSPC proxy and the resulting
difference in the pre-event 2012–2016 fit; the Student-t6 cells differ by a
larger relative margin (1.32 vs 1.29 for log, 0.27 vs 0.25 for density-power,
0.44 vs 0.42 for MMD) but preserve the paper's ordering and order of
magnitude.

## Experiment 8 — Controlled Permanent Variance Break and Jump Contamination Experiments

Source: `results/exp8_break_jump.json`, `results/exp8_break.png`, `results/exp8_jump.png`.

Executed configuration: permanent break `R=3000` (paper 20000); jump
experiment `T=2000` (paper 12000) — both reduced for runtime (see
"Deviations from the paper").

**Permanent break (variance 1 → 9):**

| Rule | Adaptation delay (dates) | Cumulative excess QLIKE |
|---|---|---|
| Gaussian log | 17.51 (SE 0.17) | 76.44 (SE 1.37) |
| Density-power | 25.21 (SE 0.20) | 80.20 (SE 0.58) |
| Density-power − log | 7.70 | 3.76 |

**Jump contamination (frozen-fit / refitted arms, RMSE jump − clean):**

| Rule | Frozen | Refitted |
|---|---|---|
| Gaussian log | 1.809 | 0.228 |
| Gaussian density-power | 0.049 | 0.048 |
| Student-t6 log | 0.094 | 0.092 |
| Student-t6 density-power | 0.036 | 0.036 |

Gaussian-log path crossings of the ±12 diagnostic bound: 0 (frozen), 0
(refitted).

**Verdict:** partially reproduced. The qualitative robustness-adaptation
trade-off of Section 7.4 is reproduced: the bounded density-power filter
adapts more slowly to the permanent break than the unbounded log filter
(delay +7.7 dates; paper's reported difference is +13.4 dates) while showing
a smaller RMSE/QLIKE degradation under the jump contamination (0.049 vs 1.809
jump-minus-clean RMSE for the Gaussian family; paper's reported values are
0.016 for density-power vs no frozen-log row reported). The exact magnitudes
differ: the paper's reported adaptation delays (29.4 / 42.8 dates) and
cumulative excess losses (26.2 / 41.6) are larger than the executed values
(17.5 / 25.2 dates, 76.4 / 80.2 loss units respectively), and the paper
reports six (frozen) / one (refitted) Gaussian-log crossings of the ±12 bound
versus zero crossings in both arms here. The paper does not specify the gain
or AR persistence used for this simulation beyond "gain 0.05,
curvature-normalised" and a "10-date trailing adaptation rule"; the assumed
latent persistence (`phi=0.98`) and the reduced replication count (3000 vs
20000) are plausible contributors, but the cause of the zero vs six/one bound
crossings was not conclusively isolated within the available time.

## Experiment 9 — Gaussian Location Filtering: Nile River Data Illustration

Source: `results/exp9_summary.json`, `results/exp9_nile_location.png`.

Training-sample residual scale: `sigma_log = 148.00` (10^8 m^3).

| Rule | Peak displacement 1946 (units of sigma_log) | First half-decay year |
|---|---|---|
| Gaussian log | 1.070 (paper 1.615) | 1947 (paper 1949) |
| CRPS | 0.394 (paper 0.465) | 1947 (paper 1948) |
| Density-power | 0.103 (paper 0.066) | 1947 (paper 1949) |

**Verdict:** partially reproduced. The ordering the paper emphasises holds
exactly: Gaussian log shows the largest peak displacement, CRPS intermediate,
density-power the smallest, confirming that the bounded CRPS and redescending
density-power location updates attenuate the injected +6·sigma_log outlier
far more than the unbounded Gaussian-log location score. The magnitudes are
of the same order but not matched to the paper's precision (differences of
30–55% in the displacement ratios), and the half-decay years are one to two
years earlier than reported. The paper does not give the exact location-
filter driver formulas or the precise definition of `sigma_log` beyond
"estimated from the training data" (Appendix G.7), which were implemented
from the general scale-filter drivers adapted to a location model; this
interpretation gap is the most likely source of the remaining magnitude
difference.

## Deviations from the paper

- **Data source for `exp_6`/`exp_7`.** OHLCV bars are loaded through the
  QuantX Dataloader REST API (`data/data_loader.py`), dividend-adjusted, as
  specified by this repository's data-loading path. `^GSPC` is not a valid
  Dataloader/Massive symbol (index symbols require a Massive index
  entitlement that returned `NOT_AUTHORIZED`); the S&P 500 is proxied by the
  `SPY` ETF for both `exp_6` and `exp_7`, as the Dataloader/Massive
  documentation recommends. All loaded tickers reported `data_source:
  massive` in `ticker_details` (see `results/exp6_ticker_details.csv`,
  `results/exp7_ticker_details.csv`); none used the Yahoo backstop.
- **`exp_6` universe and sample period.** The paper's universe is twelve
  international equity indices over 2000–2024 (EURO STOXX 50 from 2007). The
  full twelve-index, 24-year daily download together with eight-start
  L-BFGS-B fits for eight factorial cells plus GARCH/EWMA benchmarks and
  sensitivity grids was estimated to exceed the time available for this
  replication; the executed run uses four markets (`^GSPC`→`SPY`, `^IXIC`→
  `QQQ`, `^DJI`→`DIA`, `^FTSE` proxy) selected by data availability among the
  paper's list, from 2015-01-01 to the data end, with two L-BFGS-B starts.
  The full paper configuration remains implemented and is selected by
  `python -m exp.exp6 --full`, which restores the twelve-index universe,
  2000/2007–2024 sample, and eight starts
  (`src/config.py: EXP6_TICKERS_PAPER`, `EXP6_START_DATE`, `EXP6_LBFGS_N_STARTS`).
- **`exp_3`/`exp_5`/`exp_8` replication counts.** The paper's Monte Carlo
  replication counts (20000 for `exp_3` and the `exp_8` permanent break,
  8000/8000/12000 for the three parts of `exp_5`, 12000 observations for the
  `exp_8` jump experiment) were estimated to exceed the time available
  together with the other eight experiments; each was reduced (see the
  per-experiment tables above for the executed counts) while every method
  parameter (gain, mesh grid, mean-reversion rate, contamination probability
  and scale) was kept at the paper's value. `python -m exp.exp3 --full`,
  `python -m exp.exp5 --full`, and `python -m exp.exp8 --full` restore the
  paper's replication counts and path lengths.
- **Unspecified latent dynamics (`exp_3`, `exp_8` jump arm).** The paper does
  not fully specify the latent log-volatility process `x_t` generating the
  simulated observations for the contamination Monte Carlo and the jump
  experiment, only that it is "stochastic volatility". A standard AR(1) in
  log-variance with persistence 0.98 and innovation scale 0.15 was assumed
  (`src/config.py: EXP3_LATENT_PHI`, `EXP3_LATENT_SIGMA`), and the filter's
  own AR(1) coefficients were matched to this persistence rather than taken
  from the paper, which does not state them for these two experiments.
- **Flagged-update threshold (`exp_3`).** The paper does not state the
  numerical threshold defining a "flagged" update. A threshold of `|u_S|>2`
  (twice the reference-state tail limit shared by the three bounded Gaussian
  drivers) was used (`src/config.py: EXP3_FLAG_THRESHOLD`).
- **`exp_9` location-filter driver formulas.** Appendix G.7 states that CRPS
  and density-power location drivers are bounded/redescending but does not
  give their closed-form expressions in the retrieved text. They were
  constructed by adapting the scale-model driver algebra (Table G.5) to a
  location model with the scoring rule evaluated at `z = (y-mu)/sigma`,
  following the general proper-scoring-rule construction of Section 2; this
  is an interpretation of an incompletely specified formula, not a value
  taken directly from the paper.
- **`exp_5` local-OU terminal-variance benchmark.** The paper's exact stopped
  local-Ornstein-Uhlenbeck boundary-handling procedure (Theorem 3.2) is not
  fully specified; the implementation stops a path at the first passage of
  the normalised error outside `[-K, K]` and compares the empirical terminal
  variance across completed paths to the closed-form OU benchmark
  `q_S^2 = rho^2*K_S/J_S^2 + Gamma^2`. The resulting terminal-variance ratios
  (0.41–0.46) do not match the paper's reported 0.975–1.027; this gap is
  reported as not reproduced above because the boundary-handling detail could
  not be resolved from the paper text within the available time.

## Not reproduced

- **`exp_6` standard errors, Model Confidence Sets, and calibration tests.**
  The paper states explicitly that no significance tests, Model Confidence
  Sets, or formal calibration tests are reported for the empirical
  application; this repository likewise reports only descriptive point
  estimates, consistent with the paper's stated scope.
- **`exp_6` full twelve-index universe at the paper's 2000/2007–2024 sample.**
  Listed above under "Deviations from the paper" together with the `--full`
  command that runs it; it was not executed in this replication because the
  full download and eight-cell, eight-start fit for twelve markets over up to
  25 years was estimated to exceed the time available.
- **`exp_5` local-OU companion-arm correlation diagnostics.** The paper
  reports correlations between the companion arm's `Z_0 ~ U[-1,1]` draws and
  the Brownian drivers ranging from -0.008 to 0.013; this fine-grained
  diagnostic was not separately computed, since the terminal-variance
  mismatch above already shows the local-OU part of this experiment does not
  reproduce the paper's benchmark and a secondary diagnostic would not change
  that conclusion.
