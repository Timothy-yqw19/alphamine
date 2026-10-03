"""Search-space ablation plumbing."""

from __future__ import annotations

import pytest

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
        "illiquid300",
        "csi300",
        "no-cross-section",
        "no-volume",
        "no-time-series",
    }
    baseline = next(v for v in DEFAULT_VARIANTS if v.name == "baseline")
    liquid = next(v for v in DEFAULT_VARIANTS if v.name == "liquid300")
    illiquid = next(v for v in DEFAULT_VARIANTS if v.name == "illiquid300")
    index = next(v for v in DEFAULT_VARIANTS if v.name == "csi300")
    # The universe variants must differ only in the universe, or the comparison
    # would confound grammar with universe.
    assert baseline.ops == liquid.ops is None
    assert baseline.variables == liquid.variables is None
    assert illiquid.ops is None and illiquid.variables is None
    assert index.ops is None and index.variables is None
    assert liquid.universe == ("top", 300)
    assert illiquid.universe == ("bottom", 300)
    # csi300 is real membership, not a turnover slice: it is the universe whose
    # numbers are comparable with published CSI 300 results.
    assert index.universe == ("index", "csi300")


def test_variant_config_only_changes_the_universe():
    base = Config()
    liquid = next(v for v in DEFAULT_VARIANTS if v.name == "liquid300")
    cfg = variant_config(base, liquid)
    assert cfg.universe_slice == ("top", 300)
    assert cfg.horizon == base.horizon
    assert cfg.splits == base.splits
    assert base.universe_slice is None, "the base config must not be mutated"


def test_csi300_variant_switches_the_universe_to_index_membership():
    base = Config()
    index = next(v for v in DEFAULT_VARIANTS if v.name == "csi300")
    cfg = variant_config(base, index)
    assert cfg.universe_slice == ("index", "csi300")
    assert cfg.horizon == base.horizon
    assert base.universe_slice is None, "the base config must not be mutated"


def test_no_volume_variant_has_no_volume_terms():
    assert "volume" not in PRICE_VARIABLES
    assert "adv20" not in PRICE_VARIABLES


def test_liquidity_mask_keeps_at_most_top_n_names_per_date(panel):
    mask = liquidity_mask(panel, ("top", 10))
    counts = mask.sum(axis=1)
    assert counts.max() <= 10
    assert counts.sum() > 0


def test_bottom_selection_takes_the_other_end(panel):
    top = liquidity_mask(panel, ("top", 10))
    bottom = liquidity_mask(panel, ("bottom", 10))
    overlapping = (top & bottom).sum().sum()
    assert bottom.sum().sum() > 0
    assert overlapping == 0, "the two ends of the ranking must not overlap"
    assert top.sum().sum() == bottom.sum().sum()


def test_unknown_selection_side_is_rejected(panel):
    with pytest.raises(ValueError, match="top.*bottom"):
        liquidity_mask(panel, ("middle", 10))


def test_restrict_universe_shrinks_the_panel_and_keeps_ranks_inside(panel):
    small = restrict_universe(panel, ("top", 12))
    assert small.shape[1] <= panel.shape[1]
    assert small.tradable.sum(axis=1).max() <= 12
    assert small.meta["universe_slice"] == ("top", 12)
    assert small.meta["restricted_from"] == panel.shape[1]
    # Every retained field is sliced consistently.
    for frame in small.fields.values():
        assert frame.shape == small.tradable.shape
