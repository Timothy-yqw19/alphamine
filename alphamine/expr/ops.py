"""The operator set.

Three families, following the study plan:

* **arith** - pointwise arithmetic;
* **ts**    - time-series operators, applied per instrument along the date axis;
* **cs**    - cross-sectional operators, applied per date across instruments.

Everything works on ``float32`` DataFrames indexed by date with one column per
instrument.  Scalar constants are plain floats and rely on broadcasting.

Rolling operators pandas implements natively (``mean``, ``std``, ``rank``,
``corr``, ...) are used directly.  The rest (``argmax``, ``argmin``,
``product``, ``decay_linear``) go through the chunked sliding-window helper so
that temporary arrays stay bounded no matter how large the panel is.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

#: Window lengths the search may draw from.  Keeping this list small and
#: economically sensible is itself a search-space decision (idea R5).
WINDOWS: tuple[int, ...] = (2, 3, 5, 10, 15, 20, 30, 40, 60, 120, 250)

#: Constants the search may draw from.
CONSTANTS: tuple[float, ...] = (-1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)

#: Cap on the temporary arrays used by one chunked rolling reduction, in bytes.
CHUNK_BUDGET = 256 * 1024 * 1024


@dataclass(frozen=True)
class OpSpec:
    """One operator: signature, category and implementation.

    ``sig`` has one character per argument:

    ``S`` series, ``K`` scalar constant, ``W`` window length (an integer literal).

    ``max_window`` bounds the window this operator is allowed to see.  It exists
    because a fixed *formula* budget is only a fair proxy for a fixed *compute*
    budget if no single operator dominates the cost.  Measured on the full
    2012-2026 panel, ``ts_argmax(volume, 250)`` costs 8.9 s while
    ``ts_mean(volume, 250)`` costs 0.15 s - a 60x spread that would let a search
    win on wall-clock alone.  Caps keep every operator under about a second.

    ``in_default_pool`` marks operators the random sampler draws from.
    ``ts_median`` is excluded: both the pandas and the numpy implementation cost
    ~3 s per evaluation at every window, and no fast rolling-median is available.
    It stays in the registry so an explicit search-space ablation (idea R5) can
    switch it back on.
    """

    name: str
    category: str
    sig: str
    fn: Callable[..., pd.DataFrame]
    commutative: bool = False
    max_window: int | None = None
    in_default_pool: bool = True

    @property
    def arity(self) -> int:
        return len(self.sig)

    def windows(self) -> tuple[int, ...]:
        if self.max_window is None:
            return WINDOWS
        return tuple(w for w in WINDOWS if w <= self.max_window)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).astype("float32")


def _safe_div(a, b):
    """Division that yields NaN rather than infinity for a zero denominator.

    A ``Series`` denominator comes from a row-wise reduction such as
    ``a.std(axis=1)`` and is indexed by *date*, so it has to be aligned with
    ``axis=0``.  Dividing without it aligns on columns and silently produces an
    all-NaN frame - which is exactly what ``zscore`` and ``scale`` used to do.
    """

    if isinstance(b, pd.Series):
        return a.div(b.where(b.abs() > 1e-12), axis=0)
    if isinstance(b, pd.DataFrame):
        return a / b.where(b.abs() > 1e-12)
    if abs(b) < 1e-12:
        return a * np.nan
    return a / b


def _log(x):
    """Signed log so negative inputs stay finite: ``sign(x) * log1p(|x|)``."""

    return np.sign(x) * np.log1p(np.abs(x))


def _rolling_vector(
    x: pd.DataFrame, window: int, reduce_fn: Callable[[np.ndarray], np.ndarray]
) -> pd.DataFrame:
    """Apply ``reduce_fn`` to every trailing window of length ``window``.

    ``reduce_fn`` receives an array shaped ``(n_windows, chunk_cols, window)``
    and reduces the last axis.  Columns are processed in chunks sized to fit
    :data:`CHUNK_BUDGET` bytes, so peak memory does not scale with the panel.
    """

    arr = np.ascontiguousarray(x.to_numpy(dtype=np.float32))
    n_dates, n_cols = arr.shape
    out = np.full((n_dates, n_cols), np.nan, dtype=np.float32)
    if n_dates < window:
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    bytes_per_col = max(1, n_dates * window * 4)
    chunk = max(1, min(n_cols, CHUNK_BUDGET // bytes_per_col))
    for lo in range(0, n_cols, chunk):
        hi = min(n_cols, lo + chunk)
        view = sliding_window_view(arr[:, lo:hi], window, axis=0)
        out[window - 1 :, lo:hi] = reduce_fn(view)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _argmax_windows(view: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(view), view, -np.inf)
    idx = filled.argmax(axis=-1)
    return np.where(np.isfinite(view).all(axis=-1), idx, np.nan)


def _argmin_windows(view: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(view), view, np.inf)
    idx = filled.argmin(axis=-1)
    return np.where(np.isfinite(view).all(axis=-1), idx, np.nan)


def _product_windows(view: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(view).all(axis=-1), np.prod(view, axis=-1), np.nan)


def _decay_linear_windows(view: np.ndarray) -> np.ndarray:
    weights = np.arange(1, view.shape[-1] + 1, dtype=np.float32)
    weights = weights / weights.sum()
    return np.where(np.isfinite(view).all(axis=-1), view @ weights, np.nan)


def _rank_windows(view: np.ndarray) -> np.ndarray:
    """Percentile rank of the window's last element.

    Equivalent to ``Rolling.rank(pct=True, method='max')``: ties count in full,
    so a flat window scores 1.0.  Measured 5x faster than the pandas path at
    window 20 (0.50 s versus 2.59 s) and 3x at window 60.
    """

    last = view[..., -1:]
    rank = (view <= last).sum(axis=-1) / np.float32(view.shape[-1])
    return np.where(np.isfinite(view).all(axis=-1), rank, np.nan)


def _min_periods(window: int) -> int:
    """Require most of the window to be present.

    A stock that has traded for three days has no 250-day mean; returning NaN
    there is what keeps newly listed names out of the early ICs.
    """

    return max(2, int(np.ceil(window * 0.7)))


def _rolling(x, window):
    return x.rolling(int(window), min_periods=_min_periods(int(window)))


def _indicator(result, a, b):
    """A comparison as 1.0/0.0, and NaN wherever either input is NaN.

    The 101 Alphas build conditional expressions out of comparisons.  Returning
    NaN rather than 0 for a missing input matters: a missing price silently
    reading as "False" would paste one branch of a formula over names that have
    no data at all.

    Either side may be a scalar (``gt(close, 0)``), in which case it cannot
    contribute missingness and is skipped.
    """

    if not isinstance(result, pd.DataFrame):
        return result
    live = pd.DataFrame(True, index=result.index, columns=result.columns)
    for side in (a, b):
        if isinstance(side, pd.DataFrame) and side.shape == result.shape:
            live &= side.notna()
    return result.astype("float32").where(live)


def _comparison(op):
    """Build a comparison operator from :mod:`operator`.

    ``operator`` handles every combination we need - frame vs frame, frame vs
    scalar and scalar vs frame - so the direction of the operands is preserved.
    """

    def apply(a, b):
        return _indicator(op(a, b), a, b)

    return apply


def _where(cond, a, b):
    """``cond ? a : b``, where ``cond`` is a 0/1 indicator.

    A NaN condition propagates: both branches are multiplied by NaN, so the
    result is NaN rather than a silent choice of one branch.  Either branch may
    be a scalar.
    """

    c = cond.clip(lower=0.0, upper=1.0) if isinstance(cond, pd.DataFrame) else cond
    return c * a + (1.0 - c) * b


def _build_operators() -> dict[str, OpSpec]:
    ops: list[OpSpec] = [
        # -- arithmetic ------------------------------------------------------
        OpSpec("add", "arith", "SS", lambda a, b: a + b, commutative=True),
        OpSpec("sub", "arith", "SS", lambda a, b: a - b),
        OpSpec("mul", "arith", "SS", lambda a, b: a * b, commutative=True),
        OpSpec("div", "arith", "SS", _safe_div),
        OpSpec("max2", "arith", "SS", np.maximum, commutative=True),
        OpSpec("min2", "arith", "SS", np.minimum, commutative=True),
        OpSpec("neg", "arith", "S", lambda a: -a),
        OpSpec("abs", "arith", "S", lambda a: abs(a)),
        OpSpec("log", "arith", "S", _log),
        OpSpec("sign", "arith", "S", np.sign),
        OpSpec("sqrt", "arith", "S", lambda a: np.sqrt(abs(a))),
        OpSpec("inv", "arith", "S", lambda a: _safe_div(1.0, a)),
        OpSpec("square", "arith", "S", lambda a: a * a),
        OpSpec("addconst", "arith", "SK", lambda a, k: a + k),
        OpSpec("mulconst", "arith", "SK", lambda a, k: a * k),
        # Comparisons and selection.  These exist so the 101 Formulaic Alphas
        # can be transcribed - they use `(cond ? a : b)` heavily and it is not
        # expressible with arithmetic alone.  They are deliberately **not** in
        # the default sampling pool: adding operators would enlarge the search
        # space and shift every published ablation number.  `ts_median` is
        # excluded the same way.
        OpSpec("gt", "arith", "SS", _comparison(operator.gt), in_default_pool=False),
        OpSpec("lt", "arith", "SS", _comparison(operator.lt), in_default_pool=False),
        OpSpec("ge", "arith", "SS", _comparison(operator.ge), in_default_pool=False),
        OpSpec("le", "arith", "SS", _comparison(operator.le), in_default_pool=False),
        OpSpec("where", "arith", "SSS", _where, in_default_pool=False),
        # -- time series -----------------------------------------------------
        OpSpec("delay", "ts", "SW", lambda a, w: a.shift(int(w))),
        OpSpec("delta", "ts", "SW", lambda a, w: a - a.shift(int(w))),
        OpSpec("ts_mean", "ts", "SW", lambda a, w: _rolling(a, w).mean()),
        OpSpec("ts_sum", "ts", "SW", lambda a, w: _rolling(a, w).sum()),
        OpSpec("ts_std", "ts", "SW", lambda a, w: _rolling(a, w).std()),
        OpSpec("ts_var", "ts", "SW", lambda a, w: _rolling(a, w).var()),
        OpSpec("ts_min", "ts", "SW", lambda a, w: _rolling(a, w).min()),
        OpSpec("ts_max", "ts", "SW", lambda a, w: _rolling(a, w).max()),
        OpSpec(
            "ts_median",
            "ts",
            "SW",
            lambda a, w: _rolling(a, w).median(),
            max_window=60,
            in_default_pool=False,
        ),
        OpSpec("ts_skew", "ts", "SW", lambda a, w: _rolling(a, w).skew()),
        OpSpec(
            "ts_rank",
            "ts",
            "SW",
            lambda a, w: _rolling_vector(a, int(w), _rank_windows),
            max_window=60,
        ),
        OpSpec(
            "ts_argmax",
            "ts",
            "SW",
            lambda a, w: _rolling_vector(a, int(w), _argmax_windows),
            max_window=30,
        ),
        OpSpec(
            "ts_argmin",
            "ts",
            "SW",
            lambda a, w: _rolling_vector(a, int(w), _argmin_windows),
            max_window=30,
        ),
        OpSpec(
            "ts_product",
            "ts",
            "SW",
            lambda a, w: _rolling_vector(a, int(w), _product_windows),
            max_window=60,
        ),
        OpSpec(
            "decay_linear",
            "ts",
            "SW",
            lambda a, w: _rolling_vector(a, int(w), _decay_linear_windows),
            max_window=120,
        ),
        OpSpec(
            "ts_zscore",
            "ts",
            "SW",
            lambda a, w: _safe_div(a - _rolling(a, w).mean(), _rolling(a, w).std()),
        ),
        OpSpec(
            "ts_corr", "ts", "SSW", lambda a, b, w: _rolling(a, w).corr(b), commutative=True
        ),
        OpSpec(
            "ts_cov", "ts", "SSW", lambda a, b, w: _rolling(a, w).cov(b), commutative=True
        ),
        # -- cross section ---------------------------------------------------
        OpSpec("rank", "cs", "S", lambda a: a.rank(axis=1, pct=True)),
        OpSpec(
            "zscore",
            "cs",
            "S",
            lambda a: _safe_div(
                a.sub(a.mean(axis=1), axis=0), a.std(axis=1).replace(0, np.nan)
            ),
        ),
        OpSpec("demean", "cs", "S", lambda a: a.sub(a.mean(axis=1), axis=0)),
        OpSpec(
            "scale",
            "cs",
            "S",
            lambda a: _safe_div(a, a.abs().sum(axis=1).replace(0, np.nan)),
        ),
        OpSpec("cs_median", "cs", "S", lambda a: a.sub(a.median(axis=1), axis=0)),
    ]
    return {op.name: op for op in ops}


OPERATORS: dict[str, OpSpec] = _build_operators()


def get_op(name: str) -> OpSpec:
    try:
        return OPERATORS[name]
    except KeyError as exc:
        raise KeyError(f"unknown operator {name!r}; known: {sorted(OPERATORS)}") from exc


def available_windows() -> tuple[int, ...]:
    return WINDOWS


def by_category(category: str) -> list[OpSpec]:
    return [op for op in OPERATORS.values() if op.category == category]


def default_pool() -> tuple[str, ...]:
    """Operator names the random sampler draws from by default."""

    return tuple(op.name for op in OPERATORS.values() if op.in_default_pool)
