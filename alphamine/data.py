"""The daily price/volume panel: loading, universe rules and labels.

Source: ``QuantaAlpha/qlib_csi300`` on Hugging Face (Apache-2.0), file
``daily_pv.h5``, key ``data``.  A ``(datetime, instrument)`` MultiIndex with
``$open $high $low $close $volume $factor``.

Two facts about the file were verified by hand and matter for correctness:

* ``$close`` is **already adjusted** for splits and dividends.  Large moves do
  not cluster in the June/July dividend season, which unadjusted A-share prices
  would show.  Returns must therefore be computed from ``$close`` directly.
* ``$close / $factor`` recovers the **raw quoted price** (checked against real
  Sheng Pu Development Bank closes: 13.25 in 2008, 6.62 in 2023).  Use it if you
  ever need tick-size or price-limit rules.

The raw file is *not* a clean equity panel.  It contains index levels
(``SH000300``, ``SZ399300``), Beijing Stock Exchange listings, newly listed
stocks with a first-day pop, and 3.1% missing values.  :func:`load_panel`
applies explicit universe rules so that every downstream number is defensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

#: A-share *stock* codes: exchange prefix plus six digits.
#: SH main board 600/601/603/605 and STAR 688/689; SZ main board 000/001/002/003
#: and ChiNext 300/301/302.  Index codes (SH000xxx, SZ399xxx) and B-shares
#: (SH900xxx) deliberately do not match.
_A_SHARE_CODE = re.compile(r"^(SH(?:60|68)|SZ(?:00|30))\d{4}$")

#: Beijing Stock Exchange.  Separate because it trades under a 30% price limit
#: with far thinner liquidity, so most A-share studies exclude it.
_BSE_CODE = re.compile(r"^BJ(?:4|8|9)\d{5}$")


def is_stock(instrument: str, *, include_bse: bool = False) -> bool:
    """True for an individual stock listing, False for indices and funds."""

    if _A_SHARE_CODE.match(instrument):
        return True
    return include_bse and bool(_BSE_CODE.match(instrument))


@dataclass
class Panel:
    """A date x instrument panel of price/volume fields plus a tradability mask.

    ``fields`` values are float32 ``DataFrame``s indexed by date, columns by
    instrument.  Keeping everything float32 halves memory and is more than
    enough precision for an IC in the 0.01-0.05 range.
    """

    fields: dict[str, pd.DataFrame]
    tradable: pd.DataFrame
    listing_age: pd.DataFrame
    meta: dict
    _label_cache: dict[int, pd.DataFrame] | None = None

    # -- basics -------------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index  # type: ignore[return-value]

    @property
    def instruments(self) -> pd.Index:
        return self.fields["close"].columns

    @property
    def shape(self) -> tuple[int, int]:
        return self.fields["close"].shape

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n_dates, n_inst = self.shape
        return (
            f"Panel({n_dates} dates x {n_inst} instruments, "
            f"fields={sorted(self.fields)}, "
            f"{self.dates[0].date()}..{self.dates[-1].date()})"
        )

    # -- labels -------------------------------------------------------------
    def forward_return(self, horizon: int) -> pd.DataFrame:
        """``close(t+h) / close(t+1) - 1`` - signal at t, trade at t+1, exit at t+h."""

        if self._label_cache is None:
            self._label_cache = {}
        if horizon not in self._label_cache:
            close = self.fields["close"]
            label = close.shift(-horizon) / close.shift(-1) - 1.0
            self._label_cache[horizon] = _finite(label)
        return self._label_cache[horizon]

    # -- masks --------------------------------------------------------------
    def date_mask(self, start: str | None, end: str | None) -> pd.Series:
        idx = self.dates
        mask = pd.Series(True, index=idx)
        if start is not None:
            mask &= idx >= pd.Timestamp(start)
        if end is not None:
            mask &= idx <= pd.Timestamp(end)
        return mask


def _finite(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN and force float32."""

    return df.replace([np.inf, -np.inf], np.nan).astype("float32")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the HDF5 panel.  The key is always ``data`` in this dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it with:\n"
            "  curl -L -o data/daily_pv.h5 "
            "https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/daily_pv.h5"
        )
    return pd.read_hdf(path, key="data")


def build_panel(
    raw: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    min_listing_days: int = 60,
    include_bse: bool = False,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> Panel:
    """Turn the long MultiIndex frame into a filtered wide panel.

    Universe rules applied here, in order:

    1. keep only individual stock codes (drop index levels, funds, B-shares, and
       by default the Beijing exchange);
    2. keep only rows where the stock actually traded (``volume > 0`` and a
       finite close) - this also removes suspension days;
    3. drop each stock's first ``min_listing_days`` traded days.

    The listing-age count is computed on the **full history**, before the
    ``start``/``end`` window is applied.  Counting from ``start`` instead would
    label every stock in the panel as newly listed and silently delete the first
    months of every study window.
    """

    instruments = [
        c for c in raw.index.levels[1].unique() if is_stock(c, include_bse=include_bse)
    ]
    keep = raw.index.get_level_values(1).isin(instruments)
    raw = raw[keep]

    wide: dict[str, pd.DataFrame] = {}
    for field in fields:
        col = f"${field}"
        if col not in raw.columns:
            raise KeyError(f"{col} missing from the dataset; have {list(raw.columns)}")
        wide[field] = _finite(raw[col].unstack(0).T)

    close, volume = wide["close"], wide["volume"]
    traded = close.notna() & (close > 0) & volume.notna() & (volume > 0)

    # Cumulative count of traded days = how long the stock has been alive.
    # A mid-life suspension does not reset it.
    listing_age = traded.cumsum()
    tradable = traded & (listing_age >= min_listing_days)

    in_window = pd.Series(True, index=close.index)
    if start is not None:
        in_window &= close.index >= pd.Timestamp(start)
    if end is not None:
        in_window &= close.index <= pd.Timestamp(end)

    wide = {name: frame.loc[in_window] for name, frame in wide.items()}
    traded = traded.loc[in_window]
    tradable = tradable.loc[in_window]
    listing_age = listing_age.loc[in_window]

    # Instruments that never become tradable inside the window carry no
    # information and only cost memory.
    live = tradable.any(axis=0)
    tradable = tradable.loc[:, live]
    listing_age = listing_age.loc[:, live]
    for name in wide:
        wide[name] = wide[name].loc[:, live]

    meta = {
        "n_dates": int(wide["close"].shape[0]),
        "n_instruments": int(wide["close"].shape[1]),
        "n_instruments_in_source": int(len(instruments)),
        "start": str(wide["close"].index[0].date()),
        "end": str(wide["close"].index[-1].date()),
        "min_listing_days": min_listing_days,
        "include_bse": include_bse,
        "mean_universe_size": float(tradable.sum(axis=1).mean()),
    }
    return Panel(fields=wide, tradable=tradable, listing_age=listing_age, meta=meta)


def load_panel(cfg: Config | None = None) -> Panel:
    """Load and filter the panel according to ``cfg``."""

    cfg = cfg or Config()
    raw = load_raw(cfg.data_path)
    panel = build_panel(
        raw,
        start=cfg.start,
        end=cfg.end,
        min_listing_days=cfg.min_listing_days,
        include_bse=cfg.include_bse,
    )
    if cfg.universe_slice:
        panel = restrict_universe(panel, cfg.universe_slice)
    return panel


def liquidity_rank(panel: Panel, window: int = 20) -> pd.DataFrame:
    """Cross-sectional rank of trailing turnover; 1 is the most traded name.

    The dataset has no turnover field.  ``close * volume`` is used instead, and
    the ranking was checked against known large caps: on 2023-06-30 the top 300
    contains CATL (7th), Kweichow Moutai (14th), Wuliangye (21st), BYD (22nd)
    and Ping An Insurance (35th).  That is a turnover ranking, not a market-cap
    ranking, so it is not CSI 300 membership - but it is the large, liquid
    segment, which is what the comparison needs.
    """

    turnover = (
        (panel.fields["close"] * panel.fields["volume"])
        .rolling(window, min_periods=max(2, window // 2))
        .mean()
        .where(panel.tradable)
    )
    return turnover.rank(axis=1, ascending=False, method="first")


def liquidity_mask(
    panel: Panel, selection: tuple[str, int], window: int = 20
) -> pd.DataFrame:
    """Keep the ``n`` most (or least) traded names each date."""

    side, n = selection
    if side not in {"top", "bottom"}:
        raise ValueError(f"selection side must be 'top' or 'bottom', got {side!r}")
    ranks = liquidity_rank(panel, window)
    if side == "top":
        keep = ranks.le(n)
    else:
        # ``live`` is indexed by date, so it has to be aligned explicitly.
        # Without ``axis=0`` pandas aligns it against the instrument columns and
        # the comparison silently selects nothing.
        live = ranks.notna().sum(axis=1)
        keep = ranks.gt(live.sub(n), axis=0)
    return (keep & panel.tradable).fillna(False)


def restrict_universe(
    panel: Panel, selection: tuple[str, int], window: int = 20
) -> Panel:
    """Return a panel scored only on one end of the turnover ranking.

    The restriction is applied to the panel itself, not just to the scoring
    mask, so cross-sectional operators rank inside the restricted universe.
    That is the honest version of the experiment: a study run on large caps
    would compute its ranks on large caps.
    """

    keep = liquidity_mask(panel, selection, window)
    live = keep.any(axis=0)
    fields = {name: frame.loc[:, live] for name, frame in panel.fields.items()}
    meta = dict(panel.meta)
    side, n = selection
    meta.update(
        universe_slice=(side, n),
        mean_universe_size=float(keep.loc[:, live].sum(axis=1).mean()),
        restricted_from=panel.shape[1],
    )
    return Panel(
        fields=fields,
        tradable=keep.loc[:, live],
        listing_age=panel.listing_age.loc[:, live],
        meta=meta,
    )


def field_variables(panel: Panel) -> dict[str, pd.DataFrame]:
    """The set of named series an expression may reference.

    ``vwap`` is *not* in the source dataset; ``vwap_proxy`` is the conventional
    ``(high + low + close) / 3`` stand-in and is labelled as such wherever it is
    used, because it is not the same variable the 101 Alphas were written for.
    """

    close = panel.fields["close"]
    high, low, volume = panel.fields["high"], panel.fields["low"], panel.fields["volume"]
    return {
        "open": panel.fields["open"],
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "returns": _finite(close / close.shift(1) - 1.0),
        "vwap_proxy": _finite((high + low + close) / 3.0),
        "adv20": _finite(volume.rolling(20, min_periods=5).mean()),
    }
