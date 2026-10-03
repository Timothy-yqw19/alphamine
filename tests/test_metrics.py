"""IC, ICIR, turnover and split bookkeeping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphamine.config import Config, Split
from alphamine.eval import rank_ic_series, score_factor, split_dates, turnover
from alphamine.eval.metrics import decay_curve


SYNTHETIC_SPLITS = (
    Split("train", "2020-01-01", "2020-08-31"),
    Split("valid", "2020-09-01", None),
)


def test_rank_ic_is_one_for_a_perfect_signal(panel):
    label = panel.forward_return(5)
    ic = rank_ic_series(label, label, panel.tradable, min_cross=10)
    assert ic.dropna().min() == pytest.approx(1.0, abs=1e-6)


def test_rank_ic_is_minus_one_for_the_negated_signal(panel):
    label = panel.forward_return(5)
    ic = rank_ic_series(-label, label, panel.tradable, min_cross=10)
    assert ic.dropna().max() == pytest.approx(-1.0, abs=1e-6)


def test_rank_ic_drops_dates_with_too_few_names(panel):
    label = panel.forward_return(5)
    ic = rank_ic_series(label, label, panel.tradable, min_cross=10_000)
    assert ic.dropna().empty


def test_noise_has_near_zero_ic(panel):
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(
        rng.normal(size=panel.shape), index=panel.dates, columns=panel.instruments
    )
    ic = rank_ic_series(noise, panel.forward_return(5), panel.tradable, min_cross=10)
    assert abs(ic.mean()) < 0.2


def test_split_dates_leave_room_for_the_label(panel):
    horizon = 5
    for split in Config().splits:
        dates = split_dates(panel, split, horizon)
        if len(dates) == 0:
            continue
        position = panel.dates.get_loc(dates[-1])
        assert position + horizon < len(panel.dates)


def test_split_ranges_do_not_overlap():
    splits = (
        Split("a", "2015-01-01", "2016-12-31"),
        Split("b", "2017-01-01", "2018-12-31"),
    )
    cfg = Config(splits=splits)
    assert cfg.split_by_name("a").end < cfg.split_by_name("b").start


def test_turnover_of_a_constant_factor_is_zero(panel):
    constant = pd.DataFrame(
        np.ones(panel.shape, dtype="float32"),
        index=panel.dates,
        columns=panel.instruments,
    )
    assert turnover(constant, panel.tradable, min_cross=10) == pytest.approx(0.0)


def test_score_factor_flags_an_empty_signal_as_degenerate(panel):
    empty = pd.DataFrame(
        np.nan, index=panel.dates, columns=panel.instruments, dtype="float32"
    )
    assert score_factor(panel, empty, splits=Config().splits, min_cross=10).status == (
        "degenerate"
    )


def test_score_factor_gives_a_perfect_ic_for_the_label_itself(panel):
    label = panel.forward_return(5)
    score = score_factor(
        panel, label, splits=SYNTHETIC_SPLITS, min_cross=10, label=label
    )
    assert score.status == "ok"
    assert score.train_rank_ic == pytest.approx(1.0, abs=1e-6)


def test_decay_curve_skips_the_degenerate_one_day_horizon(panel):
    """``close(t+1)/close(t+1) - 1`` is identically zero, so horizon 1 is undefined."""

    horizon_one = panel.forward_return(1)
    assert np.nanmax(np.abs(horizon_one.to_numpy())) == pytest.approx(0.0)
    curve = decay_curve(
        panel,
        panel.forward_return(5),
        start="2020-01-01",
        end=None,
        min_cross=10,
    )
    assert 1 not in curve.index
    assert curve.index.min() == 2


def test_decay_curve_agrees_with_the_split_clean_scorer(panel):
    """Regression: the window must drop the tail whose labels cross ``end``.

    ``decay_curve`` used to filter by date alone, so a 20-day curve ending on
    2020-12-31 scored labels that are realised in 2021 - inside the test period.
    It now drops the last ``horizon`` dates exactly as ``split_dates`` does,
    which makes it agree with ``score_factor`` over the same window.
    """

    horizon = 5
    start, end = "2020-01-01", "2020-08-31"
    factor = panel.fields["close"]
    label = panel.forward_return(horizon)

    curve = decay_curve(
        panel, factor, horizons=(horizon,), start=start, end=end, min_cross=10
    )
    score = score_factor(
        panel,
        factor,
        horizon=horizon,
        splits=(Split("valid", start, end),),
        min_cross=10,
        label=label,
        split_names=("valid",),
    )
    assert curve[horizon] == pytest.approx(score.valid_rank_ic)


def test_decay_curve_does_not_use_labels_beyond_its_window(panel):
    """The last date scored at horizon ``h`` must still have a realised label."""

    horizon = 5
    end_position = panel.dates.get_loc(pd.Timestamp("2020-08-31"))
    curve = decay_curve(
        panel,
        panel.fields["close"],
        horizons=(horizon,),
        start="2020-01-01",
        end="2020-08-31",
        min_cross=10,
    )
    last_scored = panel.dates[end_position - horizon]
    assert last_scored + pd.tseries.offsets.BDay(horizon) <= panel.dates[end_position]
    assert np.isfinite(curve[horizon])
