#!/usr/bin/env python
"""P0.1 - score the 35 reproducible 101 Alphas against this panel.

The point of P0.1 is to anchor the project to the published literature: a random
search that reaches +0.0647 is only interesting if a textbook formula library
does not already reach it.

For each alpha in :mod:`alphamine.alphas101` this evaluates the factor and
records train and validation rank IC, ICIR, coverage, one-day turnover and the
decay curve at a few horizons, then writes the table to
``runs/<timestamp>-p01-alphas101/``.

Usage::

    python scripts/score_101_alphas.py
    python scripts/score_101_alphas.py --horizons 5 10 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphamine.alphas101 import formulas  # noqa: E402
from alphamine.config import Config  # noqa: E402
from alphamine.data import load_panel  # noqa: E402
from alphamine.eval import score_factor  # noqa: E402
from alphamine.eval.metrics import decay_curve  # noqa: E402
from alphamine.expr import Engine, parse  # noqa: E402

#: Published comparison point the study plan wants these numbers put against.
QUANTAALPHA_CSI300 = 0.0472


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", nargs="*", type=int, default=[5, 20])
    ap.add_argument("--name", default="p01-alphas101")
    ap.add_argument("--cache-mb", type=int, default=900)
    args = ap.parse_args()

    cfg = Config()
    panel = load_panel(cfg)
    engine = Engine(panel, cache_mb=args.cache_mb)
    print(f"panel {panel.shape[0]} dates x {panel.shape[1]} instruments")
    print(f"scoring {len(formulas())} alphas, horizons {args.horizons}\n")

    label = panel.forward_return(cfg.horizon)
    rows = []
    for number, text in formulas():
        node = parse(text)
        factor = engine.evaluate(node)
        score = score_factor(
            panel,
            factor,
            horizon=cfg.horizon,
            splits=cfg.splits,
            min_cross=cfg.min_cross,
            label=label,
            with_turnover=True,
            split_names=("train", "valid"),
        )
        row = {
            "alpha": f"alpha{number:03d}",
            "number": number,
            "status": score.status,
            "coverage": round(score.coverage, 4),
            "n_names": round(score.n_names, 1),
            "train_rank_ic": score.train_rank_ic,
            "valid_rank_ic": score.valid_rank_ic,
            "valid_icir": score.valid_icir,
            "turnover": score.turnover,
            "expression": text,
        }
        curve = decay_curve(
            panel,
            factor,
            horizons=tuple(args.horizons),
            start="2019-01-01",
            end="2020-12-31",
            min_cross=cfg.min_cross,
        )
        for horizon in args.horizons:
            row[f"decay_h{horizon}"] = float(curve[horizon])
        rows.append(row)
        print(
            f"  {row['alpha']:>9s}  valid {score.valid_rank_ic:+.4f}  "
            f"train {score.train_rank_ic:+.4f}  turnover {score.turnover:.3f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("valid_rank_ic", ascending=False)
    run = Path(cfg.runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{args.name}"
    run.mkdir(parents=True, exist_ok=True)
    frame.to_csv(run / "evals.csv", index=False)

    positive = frame[frame["valid_rank_ic"] > 0]
    summary = {
        "name": args.name,
        "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "panel": list(panel.shape),
        "n_alphas": int(len(frame)),
        "horizon": cfg.horizon,
        "decay_horizons": list(args.horizons),
        "best": {
            "alpha": frame.iloc[0]["alpha"],
            "valid_rank_ic": float(frame.iloc[0]["valid_rank_ic"]),
            "train_rank_ic": float(frame.iloc[0]["train_rank_ic"]),
        },
        "n_positive_valid": int(len(positive)),
        "median_valid_rank_ic": float(frame["valid_rank_ic"].median()),
        "mean_valid_rank_ic": float(frame["valid_rank_ic"].mean()),
        "published_comparison": {
            "quantaalpha_csi300": QUANTAALPHA_CSI300,
            "random_search_p02_best_valid": 0.0647,
        },
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{len(frame)} alphas | {len(positive)} with positive validation IC")
    print(f"median validation rank IC {frame['valid_rank_ic'].median():+.4f}, "
          f"mean {frame['valid_rank_ic'].mean():+.4f}")
    print(f"best: {frame.iloc[0]['alpha']} at {frame.iloc[0]['valid_rank_ic']:+.4f}")
    print(f"\nartefacts: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
