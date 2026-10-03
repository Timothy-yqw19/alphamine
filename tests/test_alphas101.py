"""The 101-Alpha transcription.

These run on the synthetic panel, so they are fast and need no download.  They
guard the hand transcription against syntax rot and against a formula quietly
referring to a field this dataset does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from alphamine.alphas101 import ALPHAS_101, formulas
from alphamine.expr import Engine, parse

#: The ids ``scripts/count_101_alphas.py`` reports as expressible from OHLCV.
EXPECTED = [
    1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    28, 29, 30, 33, 34, 37, 38, 39, 43, 44, 45, 51, 52, 53, 54, 101,
]


def test_the_worklist_is_the_one_the_classifier_reports():
    assert sorted(ALPHAS_101) == EXPECTED
    assert len(ALPHAS_101) == 35
    assert [n for n, _ in formulas()] == EXPECTED


def test_none_of_them_reach_for_a_field_we_do_not_have():
    """The 35 are the OHLCV-expressible ones - no vwap, no industry."""

    for number, text in formulas():
        assert "vwap" not in text.lower(), number
        assert "IndClass" not in text, number
        assert "market_cap" not in text, number


@pytest.mark.parametrize("number", EXPECTED)
def test_every_alpha_parses(number):
    assert parse(ALPHAS_101[number]) is not None


@pytest.mark.parametrize("number", EXPECTED)
def test_every_alpha_evaluates(number, panel):
    out = Engine(panel).evaluate(parse(ALPHAS_101[number]))
    assert out.shape == panel.shape
    assert np.isfinite(out.to_numpy()).any(), f"alpha{number:03d} is finite nowhere"


def test_the_conditional_ones_actually_use_where():
    """The six alphas needing `(cond ? a : b)` must contain a `where`."""

    for number in (1, 7, 9, 21, 23, 51):
        assert "where(" in ALPHAS_101[number], number
