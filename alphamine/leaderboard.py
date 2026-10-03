"""The equal-budget leaderboard (idea R3).

Methods are compared at the same number of *evaluated formulas*, so the number
that decides is not "best validation IC the run ever saw" - that is a selection
on validation, which no method could have made - but "among the first N formulas,
take the best **training** IC, then report its validation IC".  P1 is the worked
example of why this matters: genetic programming looks like it beats random
search by 0.0008 on the first metric and ties it exactly on the second.

The comparison is *paired* by seed.  ``sample_random_formulas(n, seed=s)`` and the
GP initialiser draw from the same generator, so GP(seed s) and random(seed s)
share their first ``population`` formulas exactly; from there on the only
difference is evolution.  A per-seed difference therefore isolates the effect of
selection rather than mixing it with which formulas happened to be drawn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .gp import GPConfig, run_gp
from .runner import evaluate_batch, sample_random_formulas

#: Budgets the leaderboard reports at.
DEFAULT_BUDGETS = (35, 100, 250, 500, 1000, 2000)

#: The metrics reported for every (method, seed, budget).
RANK_BY = "train_rank_ic"
REPORT = "valid_rank_ic"


@dataclass(frozen=True)
class Entry:
    """One method run at one seed."""

    method: str
    seed: int
    frame: pd.DataFrame

    @property
    def label(self) -> str:
        return f"{self.method}-seed{self.seed}"


def curve_at(
    frame: pd.DataFrame,
    budget: int,
    *,
    rank_by: str = RANK_BY,
    report: str = REPORT,
) -> tuple[float, float]:
    """``(best-of-N, train-selected)`` for the first ``budget`` rows.

    ``best-of-N`` is the running maximum of ``report`` - optimistic, because
    choosing the maximum on the reported column is itself a selection.
    ``train-selected`` picks the row with the best ``rank_by`` and reports its
    ``report`` value, which is the choice a method can actually make.
    """

    head = frame.head(budget)
    rank = pd.to_numeric(head[rank_by], errors="coerce")
    values = pd.to_numeric(head[report], errors="coerce")
    if not rank.notna().any():
        return float("nan"), float("nan")
    return float(values.max()), float(values.loc[rank.idxmax()])


def run_entry(
    method: str,
    seed: int,
    *,
    budget: int,
    cfg: Config,
    depth: int = 4,
    workers: int | None = None,
    gp: GPConfig | None = None,
    progress: bool = True,
) -> Entry:
    """Evaluate one method for one seed, in proposal order."""

    if method == "random":
        formulas = sample_random_formulas(budget, max_depth=depth, seed=seed)
        frame = evaluate_batch(formulas, cfg, workers=workers, progress=progress)
    elif method == "gp":
        settings = gp or GPConfig(max_depth=depth)
        frame = run_gp(
            cfg, budget=budget, seed=seed, gp=settings, workers=workers, progress=progress
        )
    else:
        raise ValueError(f"unknown method {method!r}; known: random, gp")
    return Entry(method=method, seed=seed, frame=frame)


def score_entries(
    entries: list[Entry], budgets: tuple[int, ...] = DEFAULT_BUDGETS
) -> pd.DataFrame:
    """One row per (method, seed, budget)."""

    rows = []
    for entry in entries:
        for budget in budgets:
            if budget > len(entry.frame):
                continue
            best, selected = curve_at(entry.frame, budget)
            rows.append(
                {
                    "method": entry.method,
                    "seed": entry.seed,
                    "budget": budget,
                    "best_of_n": best,
                    "selected": selected,
                }
            )
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of each metric per (method, budget)."""

    grouped = scores.groupby(["method", "budget"])
    out = grouped.agg(
        n_seeds=("seed", "nunique"),
        selected_mean=("selected", "mean"),
        selected_std=("selected", "std"),
        best_of_n_mean=("best_of_n", "mean"),
    ).reset_index()
    out["selected_sem"] = out["selected_std"] / np.sqrt(out["n_seeds"])
    return out


def render(scores: pd.DataFrame, summary: pd.DataFrame) -> str:
    """A markdown table, plus the paired per-seed differences."""

    lines = ["| budget | method | selected (mean +/- sem) | best-of-N (mean) |",
             "| --- | --- | --- | --- |"]
    for _, row in summary.sort_values(["budget", "method"]).iterrows():
        lines.append(
            f"| {int(row['budget'])} | `{row['method']}` | "
            f"{row['selected_mean']:+.4f} +/- {row['selected_sem']:.4f} "
            f"(n={int(row['n_seeds'])}) | {row['best_of_n_mean']:+.4f} |"
        )

    methods = list(dict.fromkeys(scores["method"]))  # the order they were run in
    if len(methods) == 2:
        a, b = methods
        wide = scores.pivot_table(
            index=["seed", "budget"], columns="method", values="selected"
        )
        if a in wide.columns and b in wide.columns:
            wide["difference"] = wide[b] - wide[a]
            lines += [
                "",
                f"Paired per-seed differences (`{b}` minus `{a}`) on the honest metric:",
                "",
                "| seed | budget | difference |",
                "| --- | --- | --- |",
            ]
            for (seed, budget), row in wide.iterrows():
                lines.append(f"| {seed} | {int(budget)} | {row['difference']:+.4f} |")
            mean = wide["difference"].mean()
            lines += ["", f"Mean paired difference: **{mean:+.4f}**."]
    return "\n".join(lines) + "\n"


def run_leaderboard(
    *,
    methods: tuple[str, ...] = ("random", "gp"),
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    budget: int = 2000,
    depth: int = 4,
    workers: int | None = None,
    cfg: Config | None = None,
    gp: GPConfig | None = None,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    name: str = "r3-leaderboard",
    progress: bool = True,
) -> Path:
    """Run every (method, seed) pair and write the leaderboard to a run directory."""

    cfg = cfg or Config()
    entries: list[Entry] = []
    for method in methods:
        for seed in seeds:
            print(f"--- {method} seed {seed}", flush=True)
            entry = run_entry(
                method,
                seed,
                budget=budget,
                cfg=cfg,
                depth=depth,
                workers=workers,
                gp=gp,
                progress=progress,
            )
            entries.append(entry)

    scores = score_entries(entries, budgets)
    summary = summarise(scores)

    out = Path(cfg.runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "leaderboard.csv", index=False)
    summary.to_csv(out / "aggregate.csv", index=False)
    (out / "leaderboard.md").write_text(render(scores, summary))
    (out / "summary.json").write_text(
        json.dumps(
            {
                "name": name,
                "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "methods": list(methods),
                "seeds": list(seeds),
                "budget": budget,
                "depth": depth,
                "budgets": list(budgets),
                "rank_by": RANK_BY,
                "report": REPORT,
                "aggregate": summary.to_dict(orient="records"),
            },
            indent=1,
        )
    )
    for entry in entries:
        entry.frame.to_csv(out / f"evals-{entry.label}.csv", index=False)
    return out
