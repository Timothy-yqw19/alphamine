"""Synthetic fixtures.

Tests run against a small in-memory panel so they finish in a second and never
depend on the 400 MB download.  The real file is exercised by ``alphamine check``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphamine.data import build_panel


def synthetic_raw(
    n_dates: int = 260,
    n_instruments: int = 40,
    *,
    seed: int = 0,
    start: str = "2020-01-01",
    index_codes: bool = True,
) -> pd.DataFrame:
    """A long ``(datetime, instrument)`` frame in the shape of daily_pv.h5."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    instruments = [f"SH{600000 + i:06d}" for i in range(n_instruments)]
    if index_codes:
        instruments = ["SH000300", "SZ399300", "BJ834765", *instruments]

    index = pd.MultiIndex.from_product(
        [dates, instruments], names=["datetime", "instrument"]
    )
    n = len(index)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    frame = pd.DataFrame(
        {
            "$open": close * (1 + rng.normal(0, 0.005, n)),
            "$high": close * (1 + abs(rng.normal(0, 0.01, n))),
            "$low": close * (1 - abs(rng.normal(0, 0.01, n))),
            "$close": close,
            "$volume": rng.lognormal(12, 1, n),
            "$factor": rng.uniform(0.01, 1.0, n),
        },
        index=index,
    )
    # A brand new listing appears 20 business days before the end of the window:
    # nothing before that, clean data after.
    newest = instruments[-1]
    listed_from = dates[-20]
    row_date = frame.index.get_level_values(0)
    head = (frame.index.get_level_values(1) == newest) & (row_date < listed_from)
    frame.loc[head, ["$open", "$high", "$low", "$close", "$volume", "$factor"]] = np.nan
    return frame


@pytest.fixture
def raw() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture
def panel(raw: pd.DataFrame):
    return build_panel(raw, min_listing_days=60)
