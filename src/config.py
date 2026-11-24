"""Experiment parameters for the proper-score observation-driven filter study.

Every module-level constant is used by at least one experiment script in
``exp/``. Each is annotated with its meaning, unit, and its source (a paper
section/table, or "not specified in the paper" when the paper leaves the
value open and a conservative default was chosen for execution).
"""

# ---------------------------------------------------------------------------
# Shared scoring-rule catalogue constants (Appendix G, "Tuning constants")
# ---------------------------------------------------------------------------
BETA_DP = 0.15     # density-power exponent (dimensionless), Appendix G tuning constants
ELL_MMD = 2.0       # MMD-RBF bandwidth in standardised residual units, Appendix G
NU_STUDENT = 6      # Student-t benchmark degrees of freedom, Appendix G tuning constants

# ---------------------------------------------------------------------------
# exp_1: Curvature-normalised driver shapes (Section 6.1, Table G.5, G.7)
# ---------------------------------------------------------------------------
EXP1_Z_RANGE = (-5.0, 5.0)      # signed-response display range, Table G.7
EXP1_Z_ABS_RANGE = (0.0, 10.0)  # magnitude display range (log scale), Table G.7
EXP1_N_DISPLAY = 401             # number of display points, Table G.7

# ---------------------------------------------------------------------------
# exp_2: Pseudo-true target under variance contamination (Section 6.2)
# ---------------------------------------------------------------------------
EXP2_EPSILON = 0.05       # contamination probability, Section 6.2
EXP2_TAU_MAX = 100.0       # maximum contamination variance ratio tau, Section 6.2 / Fig G.3
EXP2_TAU_MIN = 0.0         # minimum tau, Section 6.2
EXP2_N_TAU = 200           # number of tau grid points for the plot (not specified in the paper; chosen for a smooth curve)
EXP2_GH_NODES = 96         # Gauss-Hermite nodes per mixture component, Table G.7
EXP2_ROOT_SCAN_POINTS = 6000  # root-scan resolution, Table G.7
EXP2_V0 = 1.0              # baseline variance v0, Section 6.2

# ---------------------------------------------------------------------------
# exp_3: Finite-sample filtering under contamination (Section 6.3, Table 2, G.7)
# ---------------------------------------------------------------------------
EXP3_T_PAPER = 4000          # path length, Table G.7
EXP3_R_PAPER = 20000         # Monte Carlo replications, Table G.7
EXP3_T_RUN = 4000            # executed path length (unchanged: paper value)
EXP3_R_RUN = 2000            # executed replication count (reduced from the paper's 20000 to fit the time available; --full restores EXP3_R_PAPER)
EXP3_SEED = 20260430         # simulation seed, Table G.7 note
EXP3_CONTAM_PROB = 0.02      # Bernoulli contamination probability, Section 6.3
EXP3_CONTAM_MULT = 6.0       # innovation multiplier on contaminated dates, Section 6.3 (variance ratio 36)
EXP3_GAIN_ALPHA = 0.1        # filter gain alpha, Section 6.3 / Table G.7
EXP3_FLAG_THRESHOLD = 2.0    # |u_S| threshold defining a "flagged" update (not specified in the paper; set at twice the reference-state tail limit of the bounded drivers)
# Latent log-volatility AR(1) process x_{t+1} = PHI_X*x_t + SIGMA_X*eta_t.
# The paper states only that x_t is "latent stochastic volatility" (Section 6.3)
# without giving the generating recursion; the AR(1) calibration below follows
# the standard discrete-time stochastic-volatility specification (Taylor, 1986)
# and is recorded here as an assumption, not a paper value.
EXP3_LATENT_PHI = 0.98       # latent AR(1) persistence (not specified in the paper; standard SV calibration)
EXP3_LATENT_SIGMA = 0.15     # latent AR(1) innovation scale (not specified in the paper; standard SV calibration)
EXP3_FILTER_OMEGA = 0.0      # filter intercept omega (not specified in the paper; matched to the latent unconditional mean 0)
EXP3_FILTER_PHI = 0.98       # filter autoregressive coefficient (not specified in the paper; matched to the latent persistence)

# ---------------------------------------------------------------------------
# exp_4: Tracking efficiency frontier (Section 6.4, Table 3)
# ---------------------------------------------------------------------------
EXP4_STUDENT_DOF = (10, 6, 5, 4, 3)   # Student-t innovation degrees of freedom grid, Table 3
EXP4_QUAD_LIMIT = 200                  # scipy.integrate.quad subdivision limit (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_5: High-frequency diffusion limit mesh refinement (Appendix G.4, Table G.7, G.8)
# ---------------------------------------------------------------------------
EXP5_ALPHA = 1.0            # diffusion-limit gain constant, Table G.7
EXP5_KAPPA = 0.5             # mean-reversion rate, Table G.7
EXP5_H_GRID_CENTRED = tuple(2.0 ** (-k) for k in range(1, 8))   # h in {2^-1,...,2^-7}, Table G.7
EXP5_R_CENTRED_PAPER = 8000   # replications, centred diffusion, Table G.7
EXP5_T_CENTRED = 200          # steps, centred diffusion, Table G.7
EXP5_R_CENTRED_RUN = 2000     # executed replications (reduced to fit the time available; --full restores EXP5_R_CENTRED_PAPER)

EXP5_ALPHA_BAR = 1.0          # mean-flow O(h) gain constant, Table G.7
EXP5_T_MEANFLOW = 30          # steps, non-centred mean flow, Table G.7
EXP5_R_MEANFLOW_PAPER = 8000  # replications, mean flow, Table G.7
EXP5_R_MEANFLOW_RUN = 2000    # executed replications (reduced; --full restores EXP5_R_MEANFLOW_PAPER)
EXP5_MEANFLOW_T5_DOF = 5      # variance-matched Student-t degrees of freedom, Table G.7

EXP5_OU_RHO = 0.8             # local-OU update gain constant rho, Table G.7
EXP5_OU_GAMMA = 0.65          # local-OU target-shock scale Gamma, Table G.7
EXP5_OU_K = 3.5               # stopping radius K, Table G.7
EXP5_OU_H_GRID = tuple(2.0 ** (-k) for k in range(4, 13))  # h in {2^-4,...,2^-12}, Table G.7
EXP5_OU_H_COMPANION = 2.0 ** (-16)  # companion-arm mesh, Table G.7
EXP5_OU_M = 3                  # M parameter (per-mesh path count), Table G.7
EXP5_R_OU_PAPER = 12000        # replications, local OU, Table G.7
EXP5_R_OU_RUN = 3000            # executed replications (reduced; --full restores EXP5_R_OU_PAPER)
EXP5_SEED = 20260430

# Centred-diffusion diagnostic constants Xi_S (Table G.8)
EXP5_XI = {"gaussian_log": 2.000000, "density_power": 1.073970, "crps": 0.049243}

# ---------------------------------------------------------------------------
# exp_6: Empirical equity index application (Section 7, Table G.10)
# ---------------------------------------------------------------------------
# Twelve international equity indices, Table G.10. Yahoo Finance tickers as
# given in the paper's data description; ^AXJ0 is not a valid Yahoo symbol
# for the ASX 200 and is loaded as ^AXJO (see RESULTS.md deviations).
EXP6_TICKERS_PAPER = [
    "^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI", "^FCHI",
    "^STOXX50E", "^N225", "^HSI", "^GSPTSE", "^AXJO", "^BVSP",
]
# Index symbols such as ^GSPC are not accepted by Massive aggregates, and the
# Massive index codes (e.g. I:SPX) return HTTP 502 NOT_AUTHORIZED without an
# index entitlement. Each headline index is therefore replaced by the
# corresponding US-listed exchange-traded fund that tracks it, as the
# market-data access rule for unentitled index symbols directs; this is
# recorded as a deviation in RESULTS.md. Order matches EXP6_TICKERS_PAPER.
EXP6_TICKERS = [
    "SPY",   # S&P 500 (^GSPC)
    "QQQ",   # proxy for NASDAQ Composite (^IXIC); QQQ tracks the Nasdaq-100
    "DIA",   # Dow Jones Industrial Average (^DJI)
    "EWU",   # proxy for FTSE 100 (^FTSE); iShares MSCI United Kingdom
    "EWG",   # proxy for DAX (^GDAXI); iShares MSCI Germany
    "EWQ",   # proxy for CAC 40 (^FCHI); iShares MSCI France
    "FEZ",   # EURO STOXX 50 (^STOXX50E)
    "EWJ",   # proxy for Nikkei 225 (^N225); iShares MSCI Japan
    "EWH",   # proxy for Hang Seng (^HSI); iShares MSCI Hong Kong
    "EWC",   # proxy for S&P/TSX (^GSPTSE); iShares MSCI Canada
    "EWA",   # proxy for ASX 200 (^AXJO); iShares MSCI Australia
    "EWZ",   # proxy for Bovespa (^BVSP); iShares MSCI Brazil
]
EXP6_START_DATE = "2000-01-01"          # sample start, Section 7 ("2000-2024")
EXP6_START_DATE_STOXX50 = "2007-01-01"  # EURO STOXX 50 starts 2007, Section 7
EXP6_END_DATE = "2024-12-31"            # sample end, Section 7
EXP6_TRAIN_FRACTION = 0.70               # training-sample fraction, Section 7
EXP6_NU_STUDENT = 6                      # fixed Student-t shape for the factorial, Section 7
EXP6_BETA_DP = 0.15                      # density-power exponent, Section 7
EXP6_MMD_SUBSAMPLE_MAX = 1024            # deterministic evenly spaced subsample size for the MMD bandwidth, Section 7
EXP6_LBFGS_N_STARTS = 8                  # number of L-BFGS-B starting values, Section 7 (exact values not specified in the paper)
EXP6_EWMA_DECAY = 0.94                   # EWMA decay, Section 7
EXP6_EWMA_INIT_WINDOW = 50               # EWMA initial-variance window (days), Section 7
EXP6_GARCH_N_STARTS = 2                  # GARCH(1,1) starting values, Section 7
EXP6_GARCH_A_MAX = 0.3                   # GARCH ARCH-coefficient upper bound, Section 7
EXP6_GARCH_B_MAX = 0.98                  # GARCH GARCH-coefficient upper bound, Section 7
EXP6_GARCH_SUM_MAX = 0.999                # GARCH stationarity bound a+b, Section 7
EXP6_VAR_LEVELS = (0.01, 0.05)            # VaR exceedance levels, Section 7
EXP6_MMD_BW_SENSITIVITY = (0.5, 1.0, 2.0)  # bandwidth sensitivity multipliers, Appendix G.6
EXP6_NU_SENSITIVITY_GRID = (4, 5, 6, 8, 10)  # Student-t shape sensitivity grid, Appendix G.6
# Reduced-for-time counterpart of EXP6_LBFGS_N_STARTS=8: the Student-t
# MMD-RBF cell's finite-difference-gradient L-BFGS-B fit is too slow at the
# paper's start count within the time available; --full restores EXP6_LBFGS_N_STARTS.
EXP6_LBFGS_N_STARTS_RUN = 2
# Reduced-for-time counterpart of EXP6_TICKERS=12 markets: the paper's
# list (Table G.10) is a fixed set with no ranking rule, so the first four
# markets in list order (SPY, QQQ, DIA, EWU = US, US, US, UK) are kept as the
# largest/most liquid subset; --full restores EXP6_TICKERS.
EXP6_TICKERS_RUN = EXP6_TICKERS[:4]
# Reduced-for-time counterpart of EXP6_START_DATE="2000-01-01": a
# 10-year window instead of 24 keeps each per-market fit's series length
# tractable; --full restores EXP6_START_DATE.
EXP6_START_DATE_RUN = "2015-01-01"

# ---------------------------------------------------------------------------
# exp_7: Controlled outlier propagation, SPX counterfactual (Section 7.1)
# ---------------------------------------------------------------------------
# ^GSPC is rejected by Massive aggregates and the Massive index code I:SPX
# returns HTTP 502 NOT_AUTHORIZED without an index entitlement; SPY (the
# S&P 500 ETF) is used instead, per the same substitution rule as EXP6_TICKERS.
EXP7_TICKER_PAPER = "^GSPC"
EXP7_TICKER = "SPY"
EXP7_TRAIN_START = "2012-01-01"   # pre-event training window start, Section 7.1
EXP7_TRAIN_END = "2016-12-31"     # pre-event training window end, Section 7.1
EXP7_EVENT_DATE = "2017-07-03"    # event date, Section 7.1
EXP7_FORWARD_END = "2017-12-31"   # forward-filter horizon end (not specified beyond "through 2017")
EXP7_SIGMA_REF_WINDOW = 60        # trailing window for sigma_ref, Section 7.1
EXP7_B_GRID = (1, 10, 20, 30, 35, 60, 80, 100)  # perturbation sizes, Section 7.1
EXP7_SLOPE_B = (35, 60, 100)      # B values used for the Gaussian-log log-log slope, Section 7.1

# ---------------------------------------------------------------------------
# exp_8: Permanent variance break and jump contamination (Section 7.4, Table G.7)
# ---------------------------------------------------------------------------
EXP8_BREAK_R_PAPER = 20000       # replications, permanent break, Table G.7
EXP8_BREAK_R_RUN = 3000           # executed replications (reduced; --full restores EXP8_BREAK_R_PAPER)
EXP8_BREAK_T_PRE = 100            # pre-break dates, Table G.7
EXP8_BREAK_T_POST = 300           # post-break dates, Table G.7
EXP8_BREAK_VAR_PRE = 1.0          # pre-break variance, Section 7.4
EXP8_BREAK_VAR_POST = 9.0         # post-break variance, Section 7.4
EXP8_BREAK_GAIN = 0.05            # gain, Section 7.4 / Table G.7
EXP8_BREAK_TRAILING_WINDOW = 10   # trailing-mean window for the adaptation rule, Section 7.4 / Table G.7
EXP8_BREAK_ADAPT_FRACTION = 0.90  # fraction of new log-variance defining adaptation, Section 7.4
EXP8_SEED = 20260430

EXP8_JUMP_T_PAPER = 12000         # path length, Table G.7
EXP8_JUMP_T_RUN = 2000            # executed path length (reduced; --full restores EXP8_JUMP_T_PAPER)
EXP8_JUMP_TRAIN_FRACTION = 0.60   # training fraction, Table G.7
EXP8_JUMP_LBFGS_N_STARTS = 3      # L-BFGS-B starting values for the jump-experiment fits (reduced from EXP6_LBFGS_N_STARTS=8 for tractability; not specified in the paper)
EXP8_JUMP_PROB = 0.02             # one-sided jump probability, Table G.7
EXP8_JUMP_HALFNORMAL_SCALE = 5.0  # half-normal jump scale (in units of exp(x_t/2)), Table G.7
EXP8_JUMP_BOUND = 12.0            # +/- diagnostic bound on filtered log-variance, Section 7.4
# Latent diffusive-scale process for the jump experiment: AR(1), same
# specification and same "not specified in the paper" status as EXP3_LATENT_*.
EXP8_LATENT_PHI = 0.98
EXP8_LATENT_SIGMA = 0.15
EXP8_FILTER_OMEGA = 0.0
EXP8_FILTER_PHI = 0.98
EXP8_FILTER_GAIN = 0.1            # gain alpha used for the jump-contamination filters (matched to EXP3_GAIN_ALPHA)

# ---------------------------------------------------------------------------
# exp_9: Gaussian location filtering, Nile River data (Appendix G.7)
# ---------------------------------------------------------------------------
EXP9_TRAIN_END_YEAR = 1944         # training/freeze boundary, Appendix G.7
EXP9_OUTLIER_YEAR = 1945           # injected-outlier year, Appendix G.7
EXP9_OUTLIER_SIGMA_MULT = 6.0      # outlier size in units of sigma_log, Appendix G.7
EXP9_BETA_DP = 0.15                # density-power exponent for the location filter, Appendix G tuning constants
EXP9_LBFGS_N_STARTS = 8            # L-BFGS-B starting values, consistent with Section 7's fitting procedure
