"""Expression trees.

A candidate alpha is a tree of :class:`Node`.  Its *canonical string* is used as
the constructor-cache key and as the identifier written to the run log, so two
trees that denote the same formula must produce the same string.  Commutative
operators are therefore sorted by their children's canonical strings: ``a+b`` and
``b+a`` collapse to one key, which also stops the search from spending two units
of its budget on one idea.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: Operators whose arguments may be reordered without changing the result.
COMMUTATIVE = frozenset({"add", "mul", "max2", "min2", "ts_corr", "ts_cov"})


@dataclass(frozen=True)
class Node:
    """Base class for expression nodes."""


@dataclass(frozen=True)
class Var(Node):
    """A named input series, for example ``close`` or ``adv20``."""

    name: str


@dataclass(frozen=True)
class Const(Node):
    """A numeric literal or a window length."""

    value: float


@dataclass(frozen=True)
class Op(Node):
    """An operator application."""

    name: str
    args: tuple[Node, ...]


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e9:
        return str(int(value))
    return f"{value:g}"


def canonical(node: Node) -> str:
    """Stable, human-readable string form.  Doubles as the cache key."""

    if isinstance(node, Var):
        return node.name
    if isinstance(node, Const):
        return _fmt(node.value)
    if isinstance(node, Op):
        parts = [canonical(a) for a in node.args]
        if node.name in COMMUTATIVE and len(parts) >= 2:
            # ts_corr/ts_cov carry a trailing window argument that must not move.
            n_sort = 2 if node.name in {"ts_corr", "ts_cov"} else len(parts)
            parts = sorted(parts[:n_sort]) + parts[n_sort:]
        return f"{node.name}({', '.join(parts)})"
    raise TypeError(f"unknown node type {type(node)!r}")


def depth(node: Node) -> int:
    """Length of the longest root-to-leaf path; a leaf has depth 1."""

    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, Op):
        return 1 + max(depth(a) for a in node.args)
    raise TypeError(f"unknown node type {type(node)!r}")


def size(node: Node) -> int:
    """Total number of nodes."""

    if isinstance(node, (Var, Const)):
        return 1
    if isinstance(node, Op):
        return 1 + sum(size(a) for a in node.args)
    raise TypeError(f"unknown node type {type(node)!r}")


def used_ops(node: Node) -> Counter[str]:
    """Counts of every operator name in the tree."""

    out: Counter[str] = Counter()
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Op):
            out[cur.name] += 1
            stack.extend(cur.args)
    return out


def windows_used(node: Node) -> set[float]:
    """Every window length appearing in the tree."""

    out: set[float] = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Op):
            stack.extend(cur.args)
        elif isinstance(cur, Const) and cur.value == int(cur.value):
            out.add(cur.value)
    return out
