# Repository guide

Replication of *"Proper-Score Observation-Driven Filters: Tracking Efficiency,
Estimation, and Continuous-Time Limits"* (nine experiments; see
README.md and `results/RESULTS.md`).

## Layout

- `src/` — shared library modules: scoring rules and drivers
  (`scoring_rules.py`), the observation-driven filter recursion
  (`filters.py`), pseudo-true contamination targets (`pseudo_true.py`),
  the tracking-efficiency frontier (`tracking_efficiency.py`), diffusion-limit
  mesh refinement (`diffusion.py`), the empirical fitting/evaluation pipeline
  (`empirical.py`), and all experiment parameters (`config.py`, module-level
  constants only).
- `exp/` — one script per experiment, `python -m exp.expN`, calling only the
  `src/` APIs.
- `data/` — `data_loader.py` loads OHLCV bars through the QuantX Dataloader
  REST API (`exp_6`, `exp_7`); `generate_synthetic.py` generates the latent
  processes and contamination masks used by `exp_3`, `exp_5`, `exp_8`.
  Downloaded data is never committed.
- `tests/` — pytest tests for `src/` and the experiments.
- `results/` — all generated tables, metrics and plots, plus `RESULTS.md`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # default test run
python -m pytest                         # include live market-data tests
python -m exp.exp6 --full                # paper's full 2000-2024 universe
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` constants, each annotated with its meaning, unit and paper
  source (or "not specified in the paper").
- Market data is loaded only through `data/data_loader.py`; synthetic data
  only through `data/generate_synthetic.py`. No other module fetches or
  generates data.
- `DATALOADER_API_KEY` / `MASSIVE_API_KEY` are read from an uncommitted
  `.env` file (see `.env.example`); they never appear in code, logs or
  results.
- Tests that load real market data are marked `market_data` and are skipped
  automatically when `DATALOADER_API_KEY` is not set.
- Re-running an experiment command overwrites its files in `results/`; the
  numbers and verdicts in `results/RESULTS.md` are then checked against the
  new files and updated by hand.
