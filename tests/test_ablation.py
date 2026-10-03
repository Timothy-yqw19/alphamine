"""Search-space ablation plumbing."""

from __future__ import annotations

from alphamine.ablation import DEFAULT_VARIANTS, PRICE_VARIABLES, _without, variant_config
from alphamine.config import Config
from alphamine.data import liquidity_mask, restrict_universe
from alphamine.expr import OPERATORS, default_pool


def test_without_removes_exactly_one_category():
    no_cs = _without({"cs"})
    assert no_cs, "the operator set must not become empty"
    assert all(OPERATORS[name].category != "cs" for name in no_cs)
    assert all(
        name in no_cs for name in default_pool() if OPERATORS[name].category != "cs"
    )
    assert any(OPERATORS[name].category == "cs" for name in default_pool())


def test_variants_cover_the_four_ingredients():
    names = {v.name for v in DEFAULT_VARIANTS}
    assert names == {
        "baseline",
        "liquid300",
        "no-cross-section",
        "no-volume",
        "no-time-series",
    }
    baseline = next(v for v in DEFAULT_VARIANTS if v.name == "baseline")
    liquid = next(v for v in DEFAULT_VARIANTS if v.name == "liquid300")
    # The universe pair must differ only in the universe, or the comparison
    # would confound grammar with universe.
    assert baseline.ops == liquid.ops is None
    assert baseline.variables == liquid.variables is None


def test_variant_config_only_changes_the_universe():
    base = Config()
    liquid = next(v for v in DEFAULT_VARIANTS if v.name == "liquid300")
    cfg = variant_config(base, liquid)
    assert cfg.universe_top_n == 300
    assert cfg.horizon == base.horizon
    assert cfg.splits == base.splits
    assert base.universe_top_n is None, "the base config must not be mutated"


def test_no_volume_variant_has_no_volume_terms():
    assert "volume" not in PRICE_VARIABLES
    assert "adv20" not in PRICE_VARIABLES


def test_liquidity_mask_keeps_at_most_top_n_names_per_date(panel):
    mask = liquidity_mask(panel, 10)
    counts = mask.sum(axis=1)
    assert counts.max() <= 10
    assert counts.sum() > 0


def test_restrict_universe_shrinks_the_panel_and_keeps_ranks_inside(panel):
    small = restrict_universe(panel, 12)
    assert small.shape[1] <= panel.shape[1]
    assert small.tradable.sum(axis=1).max() <= 12
    assert small.meta["universe_top_n"] == 12
    assert small.meta["restricted_from"] == panel.shape[1]
    # Every retained field is sliced consistently.
    for frame in small.fields.values():
        assert frame.shape == small.tradable.shape
