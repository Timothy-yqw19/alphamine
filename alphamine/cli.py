"""Command line entry points.

    alphamine check                 # what is in the data, and is the harness sane
    alphamine random --n 10000      # the random-search baseline (experiment P0.2)
    alphamine ablate --n 1000       # search-space ablations (idea R5-lite)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import field_variables, load_panel
from .expr import Engine, canonical, depth, parse
from .ablation import DEFAULT_VARIANTS, run_ablations
from .runner import (
    evaluate_batch,
    persist_run,
    plot_budget_curve,
    sample_random_formulas,
)


def _cfg_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "data", None):
        cfg.data_path = Path(args.data)
    if getattr(args, "start", None):
        cfg.start = args.start
    if getattr(args, "end", None):
        cfg.end = args.end
    cfg.workers = getattr(args, "workers", 0) or 0
    cfg.cache_mb = getattr(args, "cache_mb", cfg.cache_mb)
    cfg.shuffle_labels = bool(getattr(args, "shuffle_labels", False))
    cfg.score_test = bool(getattr(args, "score_test", False))
    cfg.score_pearson = bool(getattr(args, "pearson", False))
    return cfg


def cmd_check(args: argparse.Namespace) -> int:
    """Report the panel's shape, universe rules and the cost of one formula."""

    cfg = _cfg_from_args(args)
    panel = load_panel(cfg)
    print(f"data      : {cfg.data_path}")
    print(f"panel     : {panel!r}")
    print(f"ram/copy  : {panel.fields['close'].to_numpy().nbytes / 1e6:.0f} MB per field")

    sizes = panel.tradable.sum(axis=1)
    print(
        f"universe  : {int(sizes.min())} .. {int(sizes.max())} names/day, "
        f"mean {sizes.mean():.0f} after dropping the first {cfg.min_listing_days} "
        "traded days of each listing"
    )

    label = panel.forward_return(cfg.horizon)
    print(
        f"label     : close(t+{cfg.horizon})/close(t+1)-1, "
        f"finite {label.notna().to_numpy().mean():.1%}, "
        f"mean {np.nanmean(label.to_numpy()):+.6f}"
    )

    engine = Engine(panel, cache_mb=cfg.cache_mb)
    probes = [
        "ts_mean(volume, 20)",
        "rank(ts_std(returns, 20)) - 0.5",
        "div(sub(close, ts_mean(close, 20)), ts_std(close, 20))",
        "ts_corr(close, volume, 60)",
    ]
    import time

    for text in probes:
        node = parse(text)
        started = time.perf_counter()
        engine.evaluate(node)
        print(
            f"  probe {time.perf_counter() - started:5.2f}s  "
            f"depth={depth(node)}  {canonical(node)}"
        )
    stats = engine.stats
    print(
        f"cache     : {stats.hits} hits, {stats.misses} misses, "
        f"{stats.evictions} evictions of {cfg.cache_mb} MB"
    )
    return 0


def cmd_random(args: argparse.Namespace) -> int:
    """The random-formula baseline: best-of-N as a function of N (experiment P0.2)."""

    cfg = _cfg_from_args(args)
    formulas = sample_random_formulas(
        args.n,
        max_depth=args.depth,
        seed=args.seed,
        variables=tuple(args.variables) if args.variables else None,
    )
    print(
        f"sampled {len(formulas)} distinct formulas "
        f"(max depth {args.depth}, seed {args.seed})"
    )

    results = evaluate_batch(
        formulas, cfg, workers=args.workers, with_turnover=args.turnover
    )

    null_results = None
    if args.null:
        null_cfg = replace(cfg, shuffle_labels=True, shuffle_seed=args.seed + 777)
        print("running the shuffled-label null model on the same formulas ...")
        null_results = evaluate_batch(
            formulas, null_cfg, workers=args.workers, with_turnover=False
        )

    out_dir = persist_run(
        args.name,
        results,
        cfg,
        meta={
            "method": "random",
            "max_depth": args.depth,
            "seed": args.seed,
            "null_model": bool(args.null),
        },
    )
    if null_results is not None:
        null_results.to_csv(out_dir / "evals_null.csv", index=False)

    plot_budget_curve(
        results,
        out_dir / "budget_curve.png",
        title=f"Random search, {len(results)} formulas (depth <= {args.depth})",
        null_results=null_results,
    )

    ok = results[results["status"] == "ok"]
    print()
    print(f"artefacts      : {out_dir}")
    print(f"status         : {results['status'].value_counts().to_dict()}")
    print(f"median cost    : {pd.to_numeric(results['seconds'], errors='coerce').median():.2f}s/formula")
    for split, column in (("train", "train_rank_ic"), ("valid", "valid_rank_ic")):
        best = pd.to_numeric(ok[column], errors="coerce")
        if best.notna().any():
            idx = best.idxmax()
            print(f"best {split:5s} IC : {best.max():+.4f}   {ok.loc[idx, 'formula']}")
    if null_results is not None:
        null_best = pd.to_numeric(null_results["valid_rank_ic"], errors="coerce").max()
        print(f"null floor     : {null_best:+.4f}  (best validation IC on shuffled labels)")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """R5-lite: change one ingredient of the search space at a time."""

    cfg = _cfg_from_args(args)
    if args.variants:
        chosen = tuple(v for v in DEFAULT_VARIANTS if v.name in set(args.variants))
        missing = set(args.variants) - {v.name for v in chosen}
        if missing:
            raise SystemExit(f"unknown variants {sorted(missing)}")
    else:
        chosen = DEFAULT_VARIANTS

    out_dir = run_ablations(
        n=args.n,
        depth=args.depth,
        seed=args.seed,
        workers=cfg.resolved_workers() if not args.workers else args.workers,
        base=cfg,
        variants=chosen,
        null_variants=tuple(args.null_variants),
        name=args.name,
    )
    print(f"\nartefacts: {out_dir}")
    print((out_dir / "r5-lite.md").read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alphamine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", help="path to daily_pv.h5")
        p.add_argument("--start", help="first date to load (default 2012-01-01)")
        p.add_argument("--end", help="last date to load")
        p.add_argument("--workers", type=int, default=0, help="0 = choose automatically")
        p.add_argument(
            "--cache-mb",
            type=int,
            default=Config.cache_mb,
            help="subtree cache per worker",
        )

    check = sub.add_parser("check", help="sanity-check the panel and the harness")
    add_common(check)
    check.set_defaults(func=cmd_check)

    rnd = sub.add_parser("random", help="random-formula baseline (P0.2)")
    add_common(rnd)
    rnd.add_argument("--n", type=int, default=1000, help="evaluation budget")
    rnd.add_argument("--depth", type=int, default=4, help="maximum tree depth")
    rnd.add_argument("--seed", type=int, default=0)
    rnd.add_argument("--name", default="random")
    rnd.add_argument("--variables", nargs="*", help="restrict the input variables")
    rnd.add_argument("--turnover", action="store_true", help="also compute turnover")
    rnd.add_argument(
        "--pearson",
        action="store_true",
        help="also compute Pearson IC (rank IC is always computed)",
    )
    rnd.add_argument(
        "--score-test",
        action="store_true",
        help="also score the held-out test split (slower; normally reserved for the end)",
    )
    rnd.add_argument(
        "--null",
        action="store_true",
        help="also run the same formulas on shuffled labels (luck floor, idea R2)",
    )
    rnd.set_defaults(func=cmd_random)

    abl = sub.add_parser("ablate", help="search-space ablations (R5-lite)")
    add_common(abl)
    abl.add_argument("--n", type=int, default=1000, help="budget per variant")
    abl.add_argument("--depth", type=int, default=4)
    abl.add_argument("--seed", type=int, default=0)
    abl.add_argument("--name", default="r5-lite")
    abl.add_argument(
        "--variants",
        nargs="*",
        help="subset of variant names (default: all)",
    )
    abl.add_argument(
        "--null-variants",
        nargs="*",
        default=["baseline", "liquid300"],
        help="variants that also get a shuffled-label null run",
    )
    abl.set_defaults(func=cmd_ablate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
