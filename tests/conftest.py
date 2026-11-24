"""Shared pytest fixtures and marker registration."""
from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load real market data through data/data_loader.py"
    )


@pytest.fixture(scope="session")
def equity_slice(request: pytest.FixtureRequest):
    """A short, cached slice of two equity indices for market-data tests.

    Two tickers, two years, loaded once per session through
    ``data.data_loader.load_ohlcv`` and cached via the pytest cache so
    repeated test runs do not re-download it.
    """
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set: market-data tests are skipped")
    cache_key = "equity_slice/spy_ewu_2019_2020"
    cached = request.config.cache.get(cache_key, None)
    if cached is not None:
        import pandas as pd

        return pd.DataFrame(cached)

    from data.data_loader import load_ohlcv

    # SPY and EWU track the S&P 500 and FTSE 100 respectively (see
    # src/config.py EXP6_TICKERS): ^GSPC/^FTSE index codes are not served by
    # Massive without an index entitlement, so the equity-index application
    # uses the corresponding exchange-traded funds throughout.
    bars, meta = load_ohlcv(
        tickers=["SPY", "EWU"],
        start_date="2019-01-01",
        end_date="2020-12-31",
        timeframe="1d",
    )
    request.config.cache.set(cache_key, bars.to_dict(orient="list"))
    return bars
