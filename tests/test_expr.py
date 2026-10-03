"""Parsing, canonicalisation, evaluation and sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphamine.expr import (
    OPERATORS,
    Engine,
    canonical,
    default_pool,
    depth,
    parse,
    random_formula,
    size,
    to_rpn,
)
from alphamine.expr.parse import ParseError


def test_canonical_collapses_commutative_order():
    assert canonical(parse("add(close, volume)")) == canonical(parse("add(volume, close)"))
    assert canonical(parse("close + volume")) == canonical(parse("volume + close"))
    # Subtraction is not commutative and must not be sorted.
    assert canonical(parse("sub(close, volume)")) != canonical(parse("sub(volume, close)"))
    # ts_corr's trailing window argument must not take part in the sort.
    same = canonical(parse("ts_corr(close, volume, 20)"))
    assert same == canonical(parse("ts_corr(volume, close, 20)"))
    assert same != canonical(parse("ts_corr(close, volume, 60)"))


def test_infix_and_function_syntax_agree():
    assert canonical(parse("mul(ts_mean(volume, 20), 0.5)")) == canonical(
        parse("ts_mean(volume, 20) * 0.5")
    )
    assert canonical(parse("-close")) == "neg(close)"
    assert canonical(parse("-5")) == "-5"


def test_depth_size_and_rpn():
    node = parse("div(ts_mean(volume, 20), ts_mean(volume, 60))")
    assert depth(node) == 3
    assert size(node) == 7
    assert to_rpn(node) == [
        "volume",
        "20",
        "ts_mean",
        "volume",
        "60",
        "ts_mean",
        "div",
    ]


@pytest.mark.parametrize(
    "bad",
    [
        "ts_mean(volume)",
        "ts_mean(volume, close)",
        "nope(close)",
        "close +",
        "close, volume",
        "add(close)",
    ],
)
def test_bad_formulas_raise(bad):
    with pytest.raises(ParseError):
        parse(bad)


def test_engine_computes_ts_mean(panel):
    engine = Engine(panel, cache_mb=16)
    got = engine.evaluate(parse("ts_mean(close, 5)"))
    expected = panel.fields["close"].rolling(5, min_periods=4).mean()
    pd.testing.assert_frame_equal(
        got.astype("float64"), expected.astype("float64"), atol=1e-3
    )


def test_engine_caches_shared_subtrees(panel):
    engine = Engine(panel, cache_mb=64)
    engine.evaluate(parse("div(ts_mean(volume, 20), ts_mean(volume, 60))"))
    before = engine.stats.hits
    engine.evaluate(parse("sub(ts_mean(volume, 20), ts_mean(volume, 60))"))
    assert engine.stats.hits > before, "shared subterms should hit the cache"


def test_cross_sectional_operators_accept_a_series_denominator(panel):
    """Regression: _safe_div received a Series and raised on its truth value."""

    engine = Engine(panel, cache_mb=16)
    for text in ("zscore(close)", "scale(volume)", "cs_median(close)"):
        assert engine.evaluate(parse(text)).notna().to_numpy().any()


def test_every_default_operator_returns_something_finite(panel):
    engine = Engine(panel, cache_mb=16)
    variables = ["close", "volume", "adv20", "returns", "vwap_proxy"]
    for name in default_pool():
        spec = OPERATORS[name]
        args: list[str] = []
        for index, kind in enumerate(spec.sig):
            if kind == "S":
                args.append(variables[index % len(variables)])
            elif kind == "W":
                args.append(str(spec.windows()[-1]))
            else:
                args.append("0.5")
        text = f"{name}({', '.join(args)})"
        out = engine.evaluate(parse(text))
        assert out.notna().to_numpy().any(), f"{text} produced nothing finite"


def test_sampler_is_deterministic_for_a_given_seed():
    assert canonical(random_formula(3, max_depth=4)) == canonical(
        random_formula(3, max_depth=4)
    )
    assert canonical(random_formula(4, max_depth=4)) != canonical(
        random_formula(3, max_depth=4)
    )


def test_sampler_respects_per_operator_window_caps():
    rng = np.random.default_rng(0)
    for _ in range(300):
        text = canonical(random_formula(rng, max_depth=5))
        for name in ("ts_argmax", "ts_argmin"):
            if text.startswith(name + "("):
                assert "250)" not in text and "120)" not in text


def test_ts_median_is_available_but_not_sampled_by_default():
    """It costs ~3 s per evaluation, so it is opt-in for search-space ablations."""

    assert "ts_median" in OPERATORS
    assert "ts_median" not in default_pool()
