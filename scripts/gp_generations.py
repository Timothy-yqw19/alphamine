#!/usr/bin/env python
"""Generation-by-generation report for a genetic-programming run.

Answers the two questions P1's second half asks:

* **Does the search actually keep improving?**  If the honest metric is flat
  after a few generations, the rest of the budget bought nothing and R3 should
  compare methods at a smaller N.
* **Does parsimony pressure shrink the trees?**  ``mean_size`` per generation is
  the bloat curve; compare two runs to see what the pressure bought.

The honest metric is the same one ``scripts/compare_searches.py`` uses: among the
formulas seen so far, take the one with the best **training** IC and report its
**validation** IC.  Choosing on validation would be a selection the search could
not have made.

Usage::

    python scripts/gp_generations.py runs/*p1-gp-depth4
    python scripts/gp_generations.py --parsimony-pass runs/*parsimony*
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from alphamine.expr import depth, parse, size  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="a GP run directory with evals.csv")
    args = ap.parse_args()

    frame = pd.read_csv(pathlib.Path(args.run) / "evals.csv")
    if "generation" not in frame.columns:
        raise SystemExit(f"{args.run}: no `generation` column - not a GP run?")

    frame["train"] = pd.to_numeric(frame["train_rank_ic"], errors="coerce")
    frame["valid"] = pd.to_numeric(frame["valid_rank_ic"], errors="coerce")
    frame["size"] = [size(parse(text)) for text in frame["formula"]]

    print(f"{args.run}: {len(frame)} formulas, {frame['generation'].nunique()} generations\n")
    print(f"{'gen':>4s} {'n':>5s} {'cum':>5s} {'best train':>11s} "
          f"{'best valid':>11s} {'selected':>9s} {'mean size':>10s}")
    best_train_seen, previous = -np.inf, -np.inf
    improved_at = []
    for generation, block in frame.groupby("generation"):
        seen = frame[frame["generation"] <= generation]
        best_train = seen["train"].max()
        chosen = seen.loc[seen["train"].idxmax()]
        if best_train > previous + 1e-12:
            improved_at.append(int(generation))
        previous = best_train
        marker = "  <- improved" if improved_at and improved_at[-1] == generation else ""
        print(
            f"{generation:>4d} {len(block):>5d} {len(seen):>5d} "
            f"{best_train:>+11.4f} {seen['valid'].max():>+11.4f} "
            f"{chosen['valid']:>+9.4f} {block['size'].mean():>10.2f}{marker}"
        )

    print()
    print(f"generations that improved the training best: {improved_at}")
    print(f"last improvement at generation {improved_at[-1] if improved_at else 'n/a'} "
          f"of {frame['generation'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
