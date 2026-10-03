"""Comparison and selection operators.

These exist so the 101 Formulaic Alphas can be transcribed - they lean on
``(cond ? a : b)``, which arithmetic alone cannot express.  They are registered
but kept out of the default sampling pool, so the random search space is
unchanged and every published ablation number still holds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphamine.expr import OPERATORS, Engine, default_pool, parse

NEW = ("gt", "lt", "ge", "le", "where")


def test_registered_but_never_sampled():
    for name in NEW:
        assert name in OPERATORS
        assert name not in default_pool(), f"{name} must not enter the search space"


def test_comparisons_are_zero_one_indicators(panel):
    close, open_ = panel.fields["close"], panel.fields["open"]
    live = (close.notna() & open_.notna()).to_numpy()

    for text, expected in [
        ("gt(close, open)", close > open_),
        ("lt(close, open)", close < open_),
        ("ge(close, open)", close >= open_),
        ("le(close, open)", close <= open_),
    ]:
        got = Engine(panel).evaluate(parse(text)).to_numpy()[live]
        assert np.isin(got, [0.0, 1.0]).all(), text
        assert np.array_equal(got, expected.to_numpy()[live].astype("float32")), text


def test_a_missing_input_is_missing_not_false():
    """A NaN price must not silently read as "condition is false".

    Tested on a hand-built frame: the synthetic panel fixture has no missing
    values (names that never trade are dropped), so it cannot exercise this.
    """

    import operator

    from alphamine.expr.ops import _comparison

    a = pd.DataFrame({"x": [1.0, np.nan, -1.0]})
    b = pd.DataFrame({"x": [0.0, 0.0, 0.0]})
    out = _comparison(operator.gt)(a, b)["x"].tolist()
    assert out[0] == 1.0
    assert np.isnan(out[1])  # not 0.0
    assert out[2] == 0.0


def test_where_selects_the_right_branch(panel):
    close, open_ = panel.fields["close"], panel.fields["open"]
    live = (close.notna() & open_.notna()).to_numpy()

    out = Engine(panel).evaluate(parse("where(gt(close, open), close, open)")).to_numpy()
    expected = np.where(close.to_numpy() > open_.to_numpy(), close.to_numpy(), open_.to_numpy())
    assert np.allclose(out[live], expected[live], equal_nan=True)


def test_where_propagates_a_missing_condition():
    from alphamine.expr.ops import _where

    cond = pd.DataFrame({"x": [1.0, np.nan, 0.0]})
    a = pd.DataFrame({"x": [10.0, 10.0, 10.0]})
    b = pd.DataFrame({"x": [0.0, 0.0, 0.0]})
    out = _where(cond, a, b)["x"].tolist()
    assert out[0] == 10.0  # condition true -> a
    assert np.isnan(out[1])  # unknown condition -> unknown, not a silent branch
    assert out[2] == 0.0  # condition false -> b


def test_comparison_against_a_scalar_works_both_ways(panel):
    """``gt(close, 0)`` and ``gt(0, close)`` both broadcast."""

    close = panel.fields["close"]
    live = close.notna().to_numpy()
    engine = Engine(panel)
    forward = engine.evaluate(parse("gt(close, 0)")).to_numpy()[live]
    reversed_ = engine.evaluate(parse("gt(0, close)")).to_numpy()[live]
    assert np.isin(forward, [0.0, 1.0]).all()
    assert np.array_equal(forward, 1.0 - reversed_)


def test_nested_where_covers_alpha009(panel):
    """Alpha#9's shape: OR two conditions, then choose between two branches."""

    text = (
        "where(max2(gt(ts_min(delta(close, 1), 5), 0), lt(ts_max(delta(close, 1), 5), 0)), "
        "delta(close, 1), neg(delta(close, 1)))"
    )
    out = Engine(panel).evaluate(parse(text))
    assert out.shape == panel.shape
    assert np.isfinite(out.to_numpy()).any()
