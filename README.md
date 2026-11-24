# Proper-Score Observation-Driven Filters: Tracking Efficiency, Estimation, and Continuous-Time Limits

Replication code for the paper *"Proper-Score Observation-Driven Filters: Tracking
Efficiency, Estimation, and Continuous-Time Limits"*.

This repository reproduces nine experiments from the paper: analytical driver
shapes and pseudo-true targets for four proper scoring rules (Gaussian log,
density-power, CRPS, MMD-RBF), a paired Monte Carlo filtering study under
contamination, the population tracking-efficiency frontier, high-frequency
diffusion-limit mesh refinement, an empirical equity-index factorial
application, a controlled SPX outlier-propagation counterfactual, a permanent
variance-break / jump-contamination study, and a Nile River location-filtering
illustration.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for exp_6, exp_7
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic/analytical fixtures
python -m pytest                         # full suite, including live market-data tests
```

## Experiments

Each experiment is run as a module from the repository root and writes its
artifacts directly into `results/`:

```bash
python -m exp.exp1   # curvature-normalised driver shapes
python -m exp.exp2   # pseudo-true variance targets under contamination
python -m exp.exp3   # paired Monte Carlo filtering under contamination (--full for the paper's R=20000)
python -m exp.exp4   # tracking-efficiency frontier
python -m exp.exp5   # diffusion-limit mesh refinement (--full for the paper's replication counts)
python -m exp.exp6   # empirical equity-index factorial (requires API keys; --full for the paper's 2000-2024 window)
python -m exp.exp7   # SPX controlled outlier-propagation counterfactual (requires API keys)
python -m exp.exp8   # permanent variance break and jump contamination (--full for the paper's replication counts)
python -m exp.exp9   # Nile River Gaussian location filtering
```

`reproduce.sh` runs the full test suite followed by every experiment in its
default (time-bounded) configuration.

## Results

All generated tables, metrics and plots are written to `results/`. See
`results/RESULTS.md` for the comparison against the paper's reported values,
methodology deviations and items not reproduced.

## Data

Daily OHLCV bars for the equity-index application (`exp_6`) and the SPX
counterfactual (`exp_7`) are loaded through the QuantX Dataloader REST API
(`data/data_loader.py`); downloaded bars are never committed. All other
experiments use deterministic analytical calculations or synthetic data
generated in `data/generate_synthetic.py`. The Nile River series used in
`exp_9` is the standard 1871-1970 annual-flow record bundled with
`statsmodels.datasets.nile`.
