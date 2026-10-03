"""The equal-budget leaderboard's scoring rules.

``curve_at`` is where the whole R3 comparison is decided, so it is tested
directly: the two metrics must disagree in the way P1 showed they can.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alphamine.leaderboard import Entry, curve_at, render, score_entries, summarise


def _frame(train, valid):
    return pd.DataFrame(
        {"formula": ["f"] * len(train), "train_rank_ic": train, "valid_rank_ic": valid}
    )


def test_the_two_metrics_answer_different_questions():
    # The third row has the worst training IC but the best validation IC.
    frame = _frame([0.05, 0.04, 0.01], [0.10, 0.20, 0.90])

    best_of_n, selected = curve_at(frame, 3)
    assert best_of_n == pytest.approx(0.90)  # max on validation - not selectable
    assert selected == pytest.approx(0.10)  # what picking on training would get


def test_curve_at_respects_the_budget():
    frame = _frame([0.05, 0.04, 0.09], [0.10, 0.20, 0.90])
    assert curve_at(frame, 2) == pytest.approx((0.20, 0.10))
    assert curve_at(frame, 3) == pytest.approx((0.90, 0.90))


def test_curve_at_tolerates_a_run_with_no_finite_scores():
    frame = _frame([float("nan")], [float("nan")])
    best, selected = curve_at(frame, 1)
    assert best != best and selected != selected  # both NaN


def test_selecting_on_validation_would_flatter_a_lucky_row():
    """The exact failure P1 caught: a row that is only good out of sample."""

    frame = _frame([0.01, 0.02], [0.99, 0.00])
    best_of_n, selected = curve_at(frame, 2)
    assert best_of_n > selected
    assert best_of_n - selected == pytest.approx(0.99)


def test_score_and_summarise_across_seeds():
    entries = [
        Entry("random", 0, _frame([0.05, 0.04], [0.10, 0.20])),
        Entry("random", 1, _frame([0.03, 0.02], [0.30, 0.40])),
        Entry("gp", 0, _frame([0.05, 0.09], [0.10, 0.50])),
        Entry("gp", 1, _frame([0.03, 0.04], [0.30, 0.40])),
    ]
    scores = score_entries(entries, budgets=(1, 2))
    assert len(scores) == 8

    summary = summarise(scores)
    at_two = summary[summary["budget"] == 2].set_index("method")
    # random selected at budget 2: seed 0 -> best train 0.05 -> valid 0.10;
    # seed 1 -> best train 0.03 -> valid 0.30. Mean 0.20, not 0.225.
    assert at_two.loc["random", "selected_mean"] == pytest.approx(0.20)
    # gp selected at budget 2: seed 0 -> train 0.09 -> valid 0.50; seed 1 -> 0.40.
    assert at_two.loc["gp", "selected_mean"] == pytest.approx(0.45)
    assert at_two.loc["random", "n_seeds"] == 2


def test_render_reports_a_paired_difference():
    entries = [
        Entry("random", 0, _frame([0.05, 0.04], [0.10, 0.20])),
        Entry("gp", 0, _frame([0.05, 0.06], [0.10, 0.30])),
    ]
    scores = score_entries(entries, budgets=(2,))
    text = render(scores, summarise(scores))
    assert "Paired per-seed differences" in text
    # the second method run is the challenger, so the pair reads "gp minus random"
    assert "`gp` minus `random`" in text
    assert "| 0 | 2 | +0.2000 |" in text
    assert "Mean paired difference: **+0.2000**." in text


# --------------------------------------------------------------------------
# R2: the label axis
# --------------------------------------------------------------------------


def test_shuffled_labels_reach_the_config_and_vary_by_seed(monkeypatch):
    """Each arm must get its own shuffled market, or the seeds are correlated."""

    from alphamine.config import Config
    from alphamine.leaderboard import run_entry

    seen = []
    monkeypatch.setattr(
        "alphamine.leaderboard.evaluate_batch",
        lambda formulas, cfg, **kw: (seen.append(cfg), _frame([0.0], [0.0]))[1],
    )

    run_entry("random", 3, budget=1, cfg=Config(), labels="shuffled", progress=False)
    run_entry("random", 4, budget=1, cfg=Config(), labels="shuffled", progress=False)
    run_entry("random", 3, budget=1, cfg=Config(), labels="real", progress=False)

    assert seen[0].shuffle_labels is True
    assert seen[1].shuffle_labels is True
    assert seen[0].shuffle_seed != seen[1].shuffle_seed
    assert seen[2].shuffle_labels is False


def test_unknown_label_arm_is_rejected():
    from alphamine.config import Config
    from alphamine.leaderboard import run_entry

    with pytest.raises(ValueError, match="real, shuffled"):
        run_entry("random", 0, budget=1, cfg=Config(), labels="nonsense", progress=False)


def test_net_is_real_minus_shuffled():
    from alphamine.leaderboard import net_by_seed

    entries = [
        Entry("random", 0, _frame([0.1], [0.10]), labels="real"),
        Entry("random", 0, _frame([0.1], [0.04]), labels="shuffled"),
        Entry("gp", 0, _frame([0.1], [0.12]), labels="real"),
        Entry("gp", 0, _frame([0.1], [0.05]), labels="shuffled"),
    ]
    net = net_by_seed(score_entries(entries, budgets=(1,))).set_index("method")
    assert net.loc["random", "net"] == pytest.approx(0.06)
    assert net.loc["gp", "net"] == pytest.approx(0.07)


def test_net_is_empty_without_both_arms():
    from alphamine.leaderboard import net_by_seed

    scores = score_entries([Entry("random", 0, _frame([0.1], [0.1]))], budgets=(1,))
    assert net_by_seed(scores).empty


def test_render_reports_the_net_when_both_arms_are_present():
    entries = [
        Entry("random", 0, _frame([0.1], [0.10]), labels="real"),
        Entry("random", 0, _frame([0.1], [0.04]), labels="shuffled"),
        Entry("gp", 0, _frame([0.1], [0.12]), labels="real"),
        Entry("gp", 0, _frame([0.1], [0.05]), labels="shuffled"),
    ]
    scores = score_entries(entries, budgets=(1,))
    text = render(scores, summarise(scores))
    assert "### labels: `real`" in text
    assert "### labels: `shuffled`" in text
    assert "The net of each method" in text
    assert "| `gp` | 1 | +0.0700 +/-" in text
    assert "Paired per-seed differences of nets (`gp` minus `random`):" in text
    assert "Mean paired difference: **+0.0100**." in text
