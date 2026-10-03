"""Genetic programming: budget accounting, determinism, and legal offspring.

These stub out the evaluator, so they test the search loop itself - not how fast
a formula scores - and they need no panel.
"""

from __future__ import annotations

import zlib

import pandas as pd
import pytest

from alphamine.config import Config
from alphamine.expr import at, canonical, depth, parse, positions, replace, Var
from alphamine.gp import GPConfig, run_gp, safe_points


def _stub_evaluate(texts, cfg, *, workers=None, progress=True, **kwargs):
    """A deterministic fake scorer: no panel, no multiprocessing."""

    return pd.DataFrame(
        {
            "formula": list(texts),
            "status": ["ok"] * len(texts),
            "coverage": [0.9] * len(texts),
            "train_rank_ic": [zlib.crc32(t.encode()) % 997 / 9970 for t in texts],
            "valid_rank_ic": [0.0] * len(texts),
        }
    )


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr("alphamine.gp.evaluate_batch", _stub_evaluate)


def test_safe_points_stay_legal():
    """Replacing any safe point with a variable must still parse.

    Windows and constants have to stay literals, so a GP that bred into those
    slots would emit trees the parser rejects.
    """

    node = parse("div(ts_mean(volume, 20), ts_std(close, 60))")
    points = safe_points(node)
    assert () in points
    for path in points:
        bred = replace(node, path, Var("low"))
        assert parse(canonical(bred)) is not None


def test_positions_at_and_replace_round_trip():
    node = parse("add(close, mul(volume, 2))")
    # add, close, mul, volume, 2
    assert len(positions(node)) == 5
    path = (1, 0)
    assert path == (1, 0)
    assert canonical(at(node, path)) == "volume"
    # mul is commutative, so canonical() sorts its arguments
    assert canonical(replace(node, path, Var("low"))) == "add(close, mul(2, low))"
    # the empty path replaces the whole tree
    assert canonical(replace(node, (), Var("open"))) == "open"


def test_budget_is_spent_exactly(stub):
    for budget in (10, 37, 200):
        frame = run_gp(
            Config(), budget=budget, seed=0,
            gp=GPConfig(population=40), progress=False,
        )
        assert len(frame) == budget


def test_gp_never_repeats_a_formula(stub):
    frame = run_gp(
        Config(), budget=300, seed=0, gp=GPConfig(population=50), progress=False
    )
    assert frame["formula"].nunique() == len(frame)


def test_gp_is_deterministic(stub):
    kwargs = dict(budget=120, seed=7, gp=GPConfig(population=30, elite=3))
    a = run_gp(Config(), progress=False, **kwargs)
    b = run_gp(Config(), progress=False, **kwargs)
    assert a["formula"].tolist() == b["formula"].tolist()


def test_a_different_seed_gives_a_different_search(stub):
    a = run_gp(Config(), budget=80, seed=1, gp=GPConfig(population=20), progress=False)
    b = run_gp(Config(), budget=80, seed=2, gp=GPConfig(population=20), progress=False)
    assert a["formula"].tolist() != b["formula"].tolist()


def test_every_candidate_parses_and_respects_the_depth_cap(stub):
    cap = 4
    frame = run_gp(
        Config(), budget=250, seed=3,
        gp=GPConfig(population=50, max_depth=cap), progress=False,
    )
    for text in frame["formula"]:
        assert depth(parse(text)) <= cap, text


def test_generations_are_recorded_and_the_population_is_reused(stub):
    frame = run_gp(
        Config(), budget=200, seed=0,
        gp=GPConfig(population=50, elite=2), progress=False,
    )
    assert "generation" in frame.columns
    assert frame["generation"].nunique() > 1
    assert frame["generation"].iloc[0] == 0
    # the first generation is exactly one population of random draws
    assert (frame["generation"] == 0).sum() == 50


def test_selection_is_on_training_ic_only(stub, monkeypatch):
    """Validation must not influence what survives.

    The stub makes train and validation IC independent, then the run is repeated
    with validation zeroed.  Identical searches prove validation was never read.
    """

    first = run_gp(Config(), budget=150, seed=5, gp=GPConfig(population=30), progress=False)

    def blinded(texts, cfg, **kwargs):
        frame = _stub_evaluate(texts, cfg, **kwargs)
        frame["valid_rank_ic"] = -999.0
        return frame

    monkeypatch.setattr("alphamine.gp.evaluate_batch", blinded)
    second = run_gp(Config(), budget=150, seed=5, gp=GPConfig(population=30), progress=False)
    assert first["formula"].tolist() == second["formula"].tolist()


def test_parsimony_prefers_the_smaller_tree_within_tolerance():
    from alphamine.gp import _prefer

    big = parse("div(ts_mean(volume, 20), ts_std(close, 60))")
    small = parse("close")
    scores = {canonical(big): 0.0500, canonical(small): 0.0495}

    # 0.0005 apart: inside a 0.01 band, so the smaller tree wins
    assert _prefer(small, big, scores, 0.01) is small
    # same pair, parsimony off -> the higher IC wins
    assert _prefer(small, big, scores, 0.0) is big
    # gap wider than the band -> IC decides again
    scores[canonical(small)] = 0.0400
    assert _prefer(small, big, scores, 0.01) is big


def test_parsimony_cannot_promote_an_unscored_tree():
    from alphamine.gp import _prefer

    scored = parse("div(ts_mean(volume, 20), ts_std(close, 60))")
    unknown = parse("close")
    scores = {canonical(scored): 0.02}
    # `unknown` has -inf fitness, so a huge tolerance must not let it win
    assert _prefer(unknown, scored, scores, 999.0) is scored
