#!/usr/bin/env python
"""Compare search strategies at equal evaluation budget.

The one rule of this project is that a method is judged by what it reaches after
the same number of *evaluated formulas*.  Two curves matter and they answer
different questions:

* ``best-of-N`` - the running maximum of validation IC, which is what the P0.2
  write-up plots.  It is optimistic: picking the best of N on validation is
  itself a selection, and the bigger N is the more it buys.  Fine for describing
  a search, wrong for comparing two of them.
* ``selected`` - the honest one.  Among the first N formulas take the one with
  the best **training** IC - the choice a method could actually make without
  seeing the future - and report *its* validation IC.  This is the number an
  equal-budget leaderboard (R3) has to use.

Usage::

    python scripts/compare_searches.py runs/*p02-random-depth4 runs/*p1-gp-depth4
    python scripts/compare_searches.py --budgets 35 100 250 1000 2000 runs/a runs/b
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def curve_at(frame: pd.DataFrame, budget: int) -> tuple[float, float]:
    """``(best-of-N validation IC, train-selected validation IC)`` at ``budget``."""

    head = frame.head(budget)
    train = pd.to_numeric(head["train_rank_ic"], errors="coerce")
    valid = pd.to_numeric(head["valid_rank_ic"], errors="coerce")
    if not train.notna().any():
        return float("nan"), float("nan")
    return float(valid.max()), float(valid.loc[train.idxmax()])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories, each with evals.csv")
    ap.add_argument(
        "--budgets",
        default="35,100,250,1000,2000",
        help="comma-separated evaluation budgets to compare at",
    )
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    loaded = []
    for path in args.runs:
        run = pathlib.Path(path)
        evals = run / "evals.csv"
        if not evals.is_file():
            raise SystemExit(f"{evals} not found")
        loaded.append((run.name, pd.read_csv(evals)))

    print(f"{'budget':>7s}  " + "  ".join(f"{name[:26]:>26s}" for name, _ in loaded))
    print(f"{'':>7s}  " + "  ".join(f"{'best-of-N / selected':>26s}" for _ in loaded))
    for budget in budgets:
        cells = []
        for _, frame in loaded:
            if budget > len(frame):
                cells.append(f"{'-':>26s}")
                continue
            best, selected = curve_at(frame, budget)
            cells.append(f"{best:+.4f} / {selected:+.4f}".rjust(26))
        print(f"{budget:7d}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
