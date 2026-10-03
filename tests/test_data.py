"""Universe rules and label construction."""

from __future__ import annotations

import numpy as np
import pytest

from alphamine.data import build_panel, is_stock
from tests.conftest import synthetic_raw


def test_index_and_bse_codes_are_excluded_by_default():
    assert is_stock("SH600000")
    assert is_stock("SZ300750")
    assert is_stock("SH688981")
    assert not is_stock("SH000300")
    assert not is_stock("SZ399300")
    assert not is_stock("SH900901")
    assert not is_stock("BJ834765")
    assert is_stock("BJ834765", include_bse=True)
    assert is_stock("BJ920000", include_bse=True)


def test_bse_codes_are_six_digits():
    """Regression: ``BJ(?:4|8|9)\\d{4}`` matched "BJ834765" but then rejected it."""

    for code in ("BJ430047", "BJ834765", "BJ920045"):
        assert not is_stock(code)
        assert is_stock(code, include_bse=True)


def test_panel_drops_indices_and_bse(panel):
    assert "SH000300" not in panel.instruments
    assert "SZ399300" not in panel.instruments
    assert "BJ834765" not in panel.instruments
    # 40 synthetic stocks minus the newest listing, which never reaches 60 days.
    assert panel.shape[1] == 39


def test_new_listing_is_held_out_for_min_listing_days(panel):
    # The fixture's newest listing has only 20 traded days, so with a 60-day
    # grace period it never becomes tradable and is dropped from the panel.
    assert "SH600039" not in panel.instruments


def test_listing_age_holds_out_the_first_traded_days():
    raw = synthetic_raw()
    relaxed = build_panel(raw, min_listing_days=10)
    newest = "SH600039"
    assert newest in relaxed.instruments
    tradable = relaxed.tradable[newest]
    # 20 traded days and a 10-day grace period, counted inclusively: age 10..20.
    assert int(tradable.sum()) == 11
    assert not tradable.iloc[-20:-11].any(), "the first nine days are held out"
    assert tradable.iloc[-11:].all()


def test_start_window_does_not_reset_listing_age():
    """Regression: filtering to ``start`` before counting age deletes real history."""

    raw = synthetic_raw(n_dates=200, start="2018-01-01")
    full = build_panel(raw, min_listing_days=60)
    windowed = build_panel(raw, start="2018-06-01", min_listing_days=60)
    first_date = windowed.dates[0]
    assert windowed.tradable.loc[first_date].sum() > 0
    assert windowed.tradable.loc[first_date].sum() == pytest.approx(
        full.tradable.loc[first_date].sum()
    )


def test_forward_return_matches_its_definition(panel):
    close = panel.fields["close"]
    horizon = 5
    label = panel.forward_return(horizon)
    row = 100
    for column in list(panel.instruments[:5]):
        expected = close[column].iloc[row + horizon] / close[column].iloc[row + 1] - 1
        assert label[column].iloc[row] == pytest.approx(expected, rel=1e-5)


def test_forward_return_is_shifted_so_it_cannot_use_the_trade_price(panel):
    """label(t) uses close(t+1) and close(t+h); a spike at t must not leak in."""

    horizon = 5
    close = panel.fields["close"]
    label = panel.forward_return(horizon)
    row = 100
    instrument = panel.instruments[0]
    manual = close[instrument].iloc[row + horizon] / close[instrument].iloc[row + 1] - 1
    assert label[instrument].iloc[row] == pytest.approx(manual, rel=1e-5)
    assert label[instrument].iloc[row] != pytest.approx(
        close[instrument].iloc[row + horizon] / close[instrument].iloc[row] - 1, rel=1e-5
    )
