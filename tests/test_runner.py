"""Budget bookkeeping."""

from __future__ import annotations

import pandas as pd

from alphamine.runner import budget_curve, sample_random_formulas


def test_sampled_formulas_are_distinct_and_reproducible():
    a = sample_random_formulas(50, seed=0)
    b = sample_random_formulas(50, seed=0)
    assert a == b
    assert len(set(a)) == len(a) == 50
    assert sample_random_formulas(50, seed=1) != a


def test_budget_curve_is_non_decreasing_and_skips_nan():
    results = pd.DataFrame({"valid_rank_ic": [0.01, float("nan"), 0.03, 0.02]})
    curve = budget_curve(results)
    assert curve["evaluations"].tolist() == [1, 2, 3, 4]
    assert curve["best"].tolist() == [0.01, 0.01, 0.03, 0.03]
