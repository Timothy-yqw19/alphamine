"""Point-in-time index universes.

``liquid300`` is a turnover *proxy* for the liquid segment; these tests cover the
real thing - explicit membership intervals read from a published CSI file.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphamine.data import (
    index_mask,
    index_membership,
    restrict_universe,
    universe_mask,
)

MEMBERSHIP = (
    "symbol,name,opt-in,opt-out\n"
    "SH600000,\u6d66\u53d1\u94f6\u884c,2020-02-03,2020-04-30\n"
    "SH600001,\u90af\u90f8\u94f6\u884c,2020-03-02,\n"
    "SH999999,\u4e0d\u5728\u9762\u677f\u91cc,2020-01-01,\n"
)


@pytest.fixture
def membership_file(tmp_path):
    path = tmp_path / "csi300.csv"
    # written with a BOM, exactly like the published file
    path.write_text(MEMBERSHIP, encoding="utf-8-sig")
    return path


def test_index_membership_reads_the_published_format(membership_file):
    frame = index_membership("csi300", path=membership_file)
    assert list(frame.columns)[:2] == ["symbol", "name"]
    assert len(frame) == 3
    assert frame["opt_in"].dtype.kind == "M"
    # an empty opt-out means "still a member"
    assert pd.isna(frame.loc[frame["symbol"] == "SH600001", "opt_out"]).all()


def test_index_mask_marks_only_the_membership_interval(panel, membership_file):
    mask = index_mask(panel, "csi300", path=membership_file)
    assert list(mask.columns) == list(panel.instruments)

    def at(day, symbol):
        return bool(mask.loc[pd.Timestamp(day), symbol])

    assert not at("2020-01-31", "SH600000")  # before opt-in
    assert at("2020-02-03", "SH600000")  # opt-in day counts
    assert at("2020-04-30", "SH600000")  # opt-out is the last day, inclusive
    assert not at("2020-05-01", "SH600000")  # out the day after

    assert not at("2020-02-28", "SH600001")
    assert at("2020-03-02", "SH600001")
    assert at(panel.dates[-1].strftime("%Y-%m-%d"), "SH600001")  # still open-ended


def test_index_mask_skips_instruments_the_panel_does_not_have(panel, membership_file):
    mask = index_mask(panel, "csi300", path=membership_file)
    assert "SH999999" not in mask.columns
    # only the two real members ever get marked
    assert mask.sum(axis=1).max() <= 2


def test_restrict_universe_to_index_keeps_only_members(panel, membership_file, monkeypatch):
    monkeypatch.setattr("alphamine.data.INDEX_DIR", membership_file.parent)
    restricted = restrict_universe(panel, ("index", "csi300"))

    assert set(restricted.instruments) == {"SH600000", "SH600001"}
    assert restricted.shape[1] == 2
    assert restricted.shape[0] == panel.shape[0]
    assert restricted.meta["universe_slice"] == ("index", "csi300")
    assert restricted.meta["restricted_from"] == panel.shape[1]
    # the restriction is applied to the panel, not merely to a scoring mask
    assert restricted.tradable.to_numpy().sum() > 0
    assert not restricted.tradable.to_numpy().any(axis=1).all()


def test_universe_mask_still_handles_the_turnover_slices(panel):
    top = universe_mask(panel, ("top", 5))
    bottom = universe_mask(panel, ("bottom", 5))
    assert top.sum(axis=1).max() <= 5
    assert bottom.sum(axis=1).max() <= 5
    # the two ends do not overlap on any date
    assert not (top & bottom).to_numpy().any()


def test_universe_mask_rejects_an_unknown_side(panel):
    with pytest.raises(ValueError, match="top.*bottom.*index"):
        universe_mask(panel, ("sideways", 300))


def test_missing_membership_file_says_what_to_do(panel, tmp_path, monkeypatch):
    monkeypatch.setattr("alphamine.data.INDEX_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="README"):
        index_mask(panel, "csi300")
