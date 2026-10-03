"""Minimal genetic programming over the same grammar as the random baseline.

P1 in the study plan.  The point is not to find a better formula than P0.2 did -
it is to find out whether *evolution* beats *random sampling* when both are held
to the same number of evaluated formulas.  Everything here therefore counts
budget exactly the way :func:`alphamine.runner.sample_random_formulas` does: a
formula is spent when it is first proposed and evaluated, duplicates of an
already-seen canonical string cost nothing, and a generation is one batch.

Three choices worth stating, because they are what make the comparison honest:

* **Selection is on training IC only.**  Validation is read, never selected on,
  so its curve stays a clean out-of-sample measure - the same convention P0.2
  and R5-lite use.
* **Population members are re-used across generations without re-evaluation.**
  They were already paid for.  Re-evaluating them would let GP buy extra
  information for free.
* **Depth is capped at the same value the random baseline used**, so GP cannot
  win by searching a larger space.

The tree surgery is intentionally small: subtree crossover and subtree mutation
in the ``S`` slots only.  A window or constant argument has to stay a literal, so
breeding into those slots produces a tree the parser rejects - see
:func:`safe_points`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .expr import (
    OPERATORS,
    Node,
    Op,
    at,
    canonical,
    depth,
    random_formula,
    replace,
    size,
)
from .runner import evaluate_batch

#: The inputs the search may use.  Same set as the P0.2 random baseline.
VARIABLES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "vwap_proxy",
    "adv20",
)


@dataclass(frozen=True)
class GPConfig:
    """Knobs for one GP run.  Defaults are deliberately small."""

    population: int = 200
    #: Candidates drawn per tournament; the fittest wins.
    tournament: int = 3
    crossover_rate: float = 0.6
    mutation_rate: float = 0.4
    #: Best individuals that survive a generation without re-evaluation.
    elite: int = 2
    max_depth: int = 4
    leaf_prob: float = 0.3
    #: Parsimony pressure, as an IC tolerance.  Within this band the *smaller*
    #: tree wins a tournament, so bloat costs a slot on the podium without an
    #: arbitrary penalty weight to tune.  0 disables it.  The early-stopping
    #: ablation is P1's other half; this is the parsimony one.
    parsimony: float = 0.0
    variables: tuple[str, ...] = field(default=VARIABLES)
    allowed_ops: tuple[str, ...] | None = None


def safe_points(node: Node) -> list[tuple[int, ...]]:
    """Paths that may be replaced by an arbitrary series expression.

    Only the ``S`` slots of each operator, plus the root.  ``W`` (window) and
    ``K`` (constant) arguments must remain literals, so breeding into them
    yields a tree ``parse`` rejects; excluding them keeps every offspring legal
    by construction rather than by retry.
    """

    out: list[tuple[int, ...]] = [()]
    stack: list[tuple[Node, tuple[int, ...]]] = [(node, ())]
    while stack:
        current, path = stack.pop()
        if not isinstance(current, Op):
            continue
        spec = OPERATORS[current.name]
        for index, (argument, kind) in enumerate(zip(current.args, spec.sig)):
            if kind != "S":
                continue
            child = (*path, index)
            out.append(child)
            stack.append((argument, child))
    return out


def _random(rng: np.random.Generator, gp: GPConfig) -> Node:
    return random_formula(
        rng,
        max_depth=gp.max_depth,
        variables=gp.variables,
        allowed_ops=gp.allowed_ops,
        leaf_prob=gp.leaf_prob,
    )


def _score(node: Node, scores: dict[str, float]) -> float:
    return scores.get(canonical(node), -np.inf)


def _prefer(a: Node, b: Node, scores: dict[str, float], tolerance: float) -> Node:
    """The winner of a head-to-head.

    With parsimony on, a tree that is within ``tolerance`` IC of a fatter one
    wins - so bloat has to pay for itself rather than being free.  A NaN fitness
    (an unevaluated node) never wins on the tolerance path.
    """

    fa, fb = _score(a, scores), _score(b, scores)
    if tolerance > 0 and np.isfinite(fa) and np.isfinite(fb) and abs(fa - fb) <= tolerance:
        return a if size(a) <= size(b) else b
    return a if fa >= fb else b


def _tournament(
    rng, population, scores, size: int, tolerance: float = 0.0
) -> Node:
    picks = [population[int(rng.integers(len(population)))] for _ in range(size)]
    best = picks[0]
    for candidate in picks[1:]:
        best = _prefer(best, candidate, scores, tolerance)
    return best


def _breed(rng, population, scores, gp: GPConfig) -> Node:
    """One offspring: tournament selection, subtree crossover, subtree mutation."""

    parent = _tournament(rng, population, scores, gp.tournament, gp.parsimony)
    node = parent
    if len(population) > 1 and rng.random() < gp.crossover_rate:
        other = _tournament(rng, population, scores, gp.tournament, gp.parsimony)
        a, b = safe_points(parent), safe_points(other)
        node = replace(
            parent,
            a[int(rng.integers(len(a)))],
            at(other, b[int(rng.integers(len(b)))]),
        )
    if rng.random() < gp.mutation_rate:
        node = _mutate(rng, node, gp)
    if node == parent:
        # Nothing changed, so this offspring would be a duplicate and cost no
        # budget but also make no progress.  Force a mutation instead.
        node = _mutate(rng, node, gp)
    return node


def _mutate(rng, node: Node, gp: GPConfig) -> Node:
    points = safe_points(node)
    return replace(node, points[int(rng.integers(len(points)))], _random(rng, gp))


def run_gp(
    cfg: Config,
    *,
    budget: int,
    seed: int = 0,
    gp: GPConfig | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Evolve formulas until ``budget`` distinct ones have been evaluated.

    Returns the same frame shape as the random baseline - one row per formula in
    proposal order - so :func:`alphamine.runner.budget_curve` and the run
    artefact writer work on it unchanged.  The only extra column is
    ``generation``.
    """

    gp = gp or GPConfig()
    rng = np.random.default_rng(seed)

    seen: set[str] = set()
    scores: dict[str, float] = {}
    population: list[Node] = []
    frames: list[pd.DataFrame] = []
    generation = 0

    while len(seen) < budget:
        remaining = budget - len(seen)
        if population:
            target = min(max(1, gp.population - gp.elite), remaining)
        else:
            target = min(gp.population, remaining)

        fresh: list[Node] = []
        local: set[str] = set()
        attempts, limit = 0, target * 200
        while len(fresh) < target and attempts < limit:
            attempts += 1
            node = _random(rng, gp) if not population else _breed(
                rng, population, scores, gp
            )
            if depth(node) > gp.max_depth:
                continue
            text = canonical(node)
            if text in seen or text in local:
                continue
            local.add(text)
            fresh.append(node)

        if not fresh:
            break  # the reachable space is exhausted; stop rather than spin

        texts = [canonical(node) for node in fresh]
        frame = evaluate_batch(texts, cfg, workers=workers, progress=progress)
        frame["generation"] = generation
        frames.append(frame)

        seen.update(texts)
        scores.update(
            zip(texts, frame["train_rank_ic"].astype(float).to_numpy(), strict=True)
        )

        # Elites by IC, breaking exact ties towards the smaller tree.
        ranked = sorted(
            population,
            key=lambda node: (_score(node, scores), -size(node)),
            reverse=True,
        )
        population = ranked[: gp.elite] + fresh
        generation += 1

    if not frames:  # pragma: no cover - budget < 1
        raise ValueError("budget must be at least 1")

    return pd.concat(frames, ignore_index=True)


def describe(gp: GPConfig | None = None) -> str:
    """One-line summary for the run log."""

    gp = gp or GPConfig()
    return (
        f"population {gp.population}, tournament {gp.tournament}, "
        f"crossover {gp.crossover_rate:g}, mutation {gp.mutation_rate:g}, "
        f"elite {gp.elite}, depth <= {gp.max_depth}, "
        f"parsimony {gp.parsimony:g}"
    )
