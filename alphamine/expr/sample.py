"""Random formula generation.

This is the baseline every smarter search has to beat (experiment P0.2).  The
sampler is type-aware: it walks the operator signatures in
:mod:`alphamine.expr.ops`, so it never emits a malformed tree and every window
argument is a legal value from :data:`~alphamine.expr.ops.WINDOWS`.
"""

from __future__ import annotations

import numpy as np

from .nodes import Const, Node, Op, Var
from .ops import CONSTANTS, OPERATORS, OpSpec, default_pool

#: Probability of stopping at a variable instead of expanding another operator.
LEAF_PROB = 0.3


def _expandable(allowed: dict[str, OpSpec]) -> list[OpSpec]:
    return [spec for spec in allowed.values()]


def _leaf(rng: np.random.Generator, variables: tuple[str, ...]) -> Var:
    return Var(variables[int(rng.integers(len(variables)))])


def _gen(
    rng: np.random.Generator,
    depth: int,
    variables: tuple[str, ...],
    specs: list[OpSpec],
    *,
    leaf_prob: float,
) -> Node:
    if depth <= 1 or (depth < 3 and rng.random() < leaf_prob):
        return _leaf(rng, variables)

    spec = specs[int(rng.integers(len(specs)))]
    args: list[Node] = []
    for kind in spec.sig:
        if kind == "S":
            args.append(
                _gen(rng, depth - 1, variables, specs, leaf_prob=leaf_prob)
            )
        elif kind == "W":
            choices = spec.windows()
            args.append(Const(float(choices[int(rng.integers(len(choices)))])))
        elif kind == "K":
            args.append(Const(float(CONSTANTS[int(rng.integers(len(CONSTANTS)))])))
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown argument kind {kind!r} for {spec.name}")
    return Op(spec.name, tuple(args))


def random_formula(
    rng: np.random.Generator | int,
    *,
    max_depth: int = 4,
    variables: tuple[str, ...] | None = None,
    allowed_ops: tuple[str, ...] | None = None,
    leaf_prob: float = LEAF_PROB,
) -> Node:
    """Sample one random formula tree.

    ``max_depth`` counts nodes on the longest path, so ``max_depth=1`` returns a
    bare variable and ``max_depth=4`` gives formulas of the shape
    ``div(ts_mean(volume, 20), ts_std(close, 60))``.
    """

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if variables is None:
        variables = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "returns",
            "vwap_proxy",
            "adv20",
        )
    names = default_pool() if allowed_ops is None else allowed_ops
    allowed = {n: OPERATORS[n] for n in names}
    specs = _expandable(allowed)
    return _gen(rng, max_depth, variables, specs, leaf_prob=leaf_prob)
