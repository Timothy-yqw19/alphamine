"""Expression evaluation with a shared subtree cache.

Subtree caching is the single highest-leverage optimisation in the project.
Random search, GP and the LLM loop all propose formulas that reuse subterms
(``ts_mean(volume, 20)`` shows up everywhere), and caching them means the
evaluation *budget* really is a budget of distinct ideas rather than of compute.

The cache is keyed by the canonical string and bounded in bytes with LRU
eviction, because one full panel is ~70 MB and an unbounded cache would exhaust
memory within a few hundred formulas.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import Panel, field_variables
from .nodes import Const, Node, Op, Var, canonical
from .ops import get_op


@dataclass
class CacheStats:
    """Counters for a run's log, so cache effects are visible rather than assumed."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stored: int = 0


def sanitize(result):
    """Force float32 and turn infinities into NaN.

    The fast path - a finite float32 frame - returns without copying, which
    matters because this runs after every single operator.
    """

    if not isinstance(result, pd.DataFrame):
        return float(result)
    arr = result.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = arr.astype(np.float32, copy=False)
    if arr is result.to_numpy(copy=False) and result.dtypes.iloc[0] == np.float32:
        return result
    return pd.DataFrame(arr, index=result.index, columns=result.columns)


class Engine:
    """Evaluates expression trees against one :class:`~alphamine.data.Panel`."""

    def __init__(
        self,
        panel: Panel,
        variables: dict[str, pd.DataFrame] | None = None,
        cache_mb: float = 400.0,
    ) -> None:
        self.panel = panel
        self.variables = field_variables(panel) if variables is None else variables
        self.cache_bytes = int(cache_mb * 1024 * 1024)
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._bytes = 0
        self.stats = CacheStats()

    # -- cache --------------------------------------------------------------
    def _cache_get(self, key: str) -> pd.DataFrame | None:
        hit = self._cache.get(key)
        if hit is None:
            self.stats.misses += 1
            return None
        self._cache.move_to_end(key)
        self.stats.hits += 1
        return hit

    def _cache_put(self, key: str, value: pd.DataFrame) -> None:
        nbytes = int(value.shape[0]) * int(value.shape[1]) * 4
        while self._bytes + nbytes > self.cache_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._bytes -= int(evicted.shape[0]) * int(evicted.shape[1]) * 4
            self.stats.evictions += 1
        if nbytes > self.cache_bytes:
            return
        self._cache[key] = value
        self._bytes += nbytes
        self.stats.stored += 1

    def clear_cache(self) -> None:
        self._cache.clear()
        self._bytes = 0

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, node: Node) -> pd.DataFrame:
        """Evaluate ``node``; returns a date x instrument float32 frame."""

        result = self._eval(node)
        if not isinstance(result, pd.DataFrame):
            # A formula that reduces to a constant carries no cross-sectional
            # information; represent that honestly as an empty signal.
            result = pd.DataFrame(
                np.float32(result),
                index=self.panel.dates,
                columns=self.panel.instruments,
            )
        return result

    def _eval(self, node: Node):
        if isinstance(node, Var):
            try:
                return self.variables[node.name]
            except KeyError as exc:
                raise KeyError(
                    f"unknown variable {node.name!r}; available: {sorted(self.variables)}"
                ) from exc
        if isinstance(node, Const):
            return float(node.value)
        if not isinstance(node, Op):
            raise TypeError(f"unknown node {node!r}")

        key = canonical(node)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        spec = get_op(node.name)
        args = [self._eval(arg) for arg in node.args]
        try:
            # Overflow and invalid-value warnings are expected: a random search
            # will happily build ts_product(volume, 60).  Those results become
            # NaN in sanitize() and are reported as low coverage, not crashed on.
            with np.errstate(all="ignore"):
                result = _apply(spec, args)
        except Exception as exc:
            raise EvaluationError(f"failed evaluating {key}: {exc}") from exc

        if isinstance(result, pd.DataFrame):
            result = sanitize(result)
            self._cache_put(key, result)
        return result


class EvaluationError(RuntimeError):
    """Raised when an operator produces something unusable."""


def _apply(spec, args):
    """Dispatch on the declared signature, converting window literals to ints."""

    if spec.sig.count("W") == 1 and spec.sig.endswith("W"):
        window = int(args[-1])
        if window < 1:
            raise EvaluationError(f"{spec.name}: window must be >= 1, got {window}")
        series_args = args[:-1]
        return spec.fn(*series_args, window)
    return spec.fn(*args)


def evaluate_text(engine: Engine, text: str) -> pd.DataFrame:
    """Convenience wrapper: parse then evaluate."""

    from .parse import parse

    return engine.evaluate(parse(text))
