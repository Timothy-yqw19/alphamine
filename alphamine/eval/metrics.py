"""How a formula is scored.

The headline number everywhere in this project is the **daily cross-sectional
rank IC**: on each date, rank the names by the factor and by the forward return,
then correlate the two rankings.  Averaging that over a split gives one number
that is comparable across formulas, methods and papers.

Two details keep the numbers honest:

* the factor and the label are aligned on their *common* support inside the
  tradable universe, so a formula that only fires on 5% of names is not credited
  for the days it happens to hit;
* the last ``horizon`` rows of every split are dropped, because the label at row
  ``t`` reads ``close(t+h)`` and would otherwise reach into the next split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import Split


@dataclass
class Score:
    """One formula's result.  ``status`` is 'ok', 'degenerate' or 'error'."""

    status: str = "ok"
    message: str = ""

    #: Fraction of tradable (date, instrument) cells where the factor is finite.
    coverage: float = float("nan")
    #: Mean number of names in the cross-section after intersecting with the label.
    n_names: float = float("nan")

    train_ic: float = float("nan")
    train_rank_ic: float = float("nan")
    train_icir: float = float("nan")
    train_days: int = 0

    valid_ic: float = float("nan")
    valid_rank_ic: float = float("nan")
    valid_icir: float = float("nan")
    valid_days: int = 0

    test_rank_ic: float = float("nan")
    test_days: int = 0

    #: Mean one-day turnover of the rank-weighted long/short book.  Only filled
    #: in when scoring asks for it, because it costs a second pass over the panel.
    turnover: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def split_dates(panel, split: Split, horizon: int) -> pd.DatetimeIndex:
    """Dates belonging to ``split``, minus the tail whose labels cross the boundary."""

    idx = panel.dates
    mask = idx >= pd.Timestamp(split.start)
    if split.end is not None:
        mask &= idx <= pd.Timestamp(split.end)
    selected = idx[mask]
    if len(selected) <= horizon:
        return selected[:0]
    return selected[: len(selected) - horizon]


def _align(
    factor: pd.DataFrame, label: pd.DataFrame, mask: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to the cells where both are finite and tradable."""

    common = mask & factor.notna() & label.notna()
    return factor.where(common), label.where(common)


def _xs_corr(a: pd.DataFrame, b: pd.DataFrame, min_cross: int, *, rank: bool) -> pd.Series:
    """Per-date cross-sectional Pearson (or Spearman) correlation, vectorised."""

    if rank:
        a = a.rank(axis=1)
        b = b.rank(axis=1)
    am = a.sub(a.mean(axis=1), axis=0)
    bm = b.sub(b.mean(axis=1), axis=0)
    numerator = (am * bm).sum(axis=1)
    denominator = np.sqrt((am**2).sum(axis=1) * (bm**2).sum(axis=1))
    out = numerator / denominator.replace(0, np.nan)
    out[a.notna().sum(axis=1) < min_cross] = np.nan
    return out.astype("float64")


def rank_ic_series(
    factor: pd.DataFrame,
    label: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    min_cross: int = 100,
    rank: bool = True,
) -> pd.Series:
    """Daily cross-sectional IC of ``factor`` against ``label``."""

    a, b = _align(factor, label, mask)
    return _xs_corr(a, b, min_cross, rank=rank)


def turnover_series(
    factor: pd.DataFrame, mask: pd.DataFrame, *, min_cross: int = 100
) -> pd.Series:
    """One-day turnover of a rank-weighted, gross-normalised long/short book."""

    weights = factor.where(mask).rank(axis=1, pct=True) - 0.5
    gross = weights.abs().sum(axis=1)
    weights = weights.div(gross.replace(0, np.nan), axis=0)
    delta = (weights - weights.shift(1)).abs().sum(axis=1) * 0.5
    delta[weights.notna().sum(axis=1) < min_cross] = np.nan
    return delta.astype("float64")


def turnover(factor: pd.DataFrame, mask: pd.DataFrame, *, min_cross: int = 100) -> float:
    return float(turnover_series(factor, mask, min_cross=min_cross).mean())


def _mean_icir(ic: pd.Series) -> tuple[float, float]:
    clean = ic.dropna()
    if len(clean) < 2:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    icir = mean / std if std > 0 else float("nan")
    return mean, icir


def score_factor(
    panel,
    factor: pd.DataFrame,
    *,
    horizon: int = 5,
    splits: tuple[Split, ...],
    min_cross: int = 100,
    label: pd.DataFrame | None = None,
    with_turnover: bool = False,
    with_pearson: bool = False,
    split_names: tuple[str, ...] = ("train", "valid", "test"),
) -> Score:
    """Score one factor on every requested split.

    Rank IC is always computed; it is the headline metric and the one robust to
    the fat tails in A-share returns.  Pearson IC costs a second correlation
    pass per split and is off by default - on the full panel it is about a
    third of the per-formula cost, which matters when the whole point is to
    spend a large evaluation budget.
    """

    label = panel.forward_return(horizon) if label is None else label
    mask = panel.tradable
    result = Score()

    finite = int((factor.notna() & mask).to_numpy().sum())
    total = int(mask.to_numpy().sum())
    result.coverage = finite / total if total else float("nan")
    if finite < min_cross * 5:
        result.status = "degenerate"
        result.message = "factor is finite almost nowhere"
        return result

    a, b = _align(factor, label, mask)
    result.n_names = float(a.notna().sum(axis=1).replace(0, np.nan).mean())

    lookups = {s.name: s for s in splits}
    for name in split_names:
        split = lookups.get(name)
        if split is None:
            continue
        dates = split_dates(panel, split, horizon)
        if len(dates) == 0:
            continue
        sl = slice(dates[0], dates[-1])
        ic_rank = _xs_corr(a.loc[sl], b.loc[sl], min_cross, rank=True)
        mean_rank, icir = _mean_icir(ic_rank)

        if name == "train":
            if with_pearson:
                ic_pearson = _xs_corr(a.loc[sl], b.loc[sl], min_cross, rank=False)
                result.train_ic, _ = _mean_icir(ic_pearson)
            result.train_rank_ic = mean_rank
            result.train_icir = icir
            result.train_days = int(ic_rank.notna().sum())
        elif name == "valid":
            if with_pearson:
                ic_pearson = _xs_corr(a.loc[sl], b.loc[sl], min_cross, rank=False)
                result.valid_ic, _ = _mean_icir(ic_pearson)
            result.valid_rank_ic = mean_rank
            result.valid_icir = icir
            result.valid_days = int(ic_rank.notna().sum())
            if with_turnover:
                result.turnover = turnover(
                    factor.loc[sl], mask.loc[sl], min_cross=min_cross
                )
        elif name == "test":
            result.test_rank_ic = mean_rank
            result.test_days = int(ic_rank.notna().sum())

    result = _check_degenerate(result)
    return result


def _check_degenerate(result: Score) -> Score:
    """Flag formulas that never produced a usable cross-section."""

    if result.status != "ok":
        return result
    if not np.isfinite(result.train_rank_ic) and not np.isfinite(result.valid_rank_ic):
        result.status = "degenerate"
        result.message = "no date had enough names to compute an IC"
    return result


def decay_curve(
    panel,
    factor: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = tuple(range(2, 21)),
    start: str = "2019-01-01",
    end: str | None = "2020-12-31",
    min_cross: int = 100,
) -> pd.Series:
    """Rank IC of the same signal at increasing forward horizons.

    A price/volume factor that decays to nothing in three days tells a very
    different story from one that still pays after a month.

    Horizons start at 2 on purpose: with the label ``close(t+h)/close(t+1) - 1``
    a horizon of 1 is identically zero, so its IC is undefined rather than
    small.
    """

    dates = panel.dates
    mask = dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    selected = dates[mask]
    out = {}
    for horizon in horizons:
        label = panel.forward_return(horizon)
        ic = rank_ic_series(factor, label, panel.tradable, min_cross=min_cross)
        out[horizon] = float(ic.loc[selected[0] : selected[-1]].mean())
    return pd.Series(out, name="rank_ic")
