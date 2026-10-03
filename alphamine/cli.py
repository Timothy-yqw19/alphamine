"""Command line entry points.

    alphamine check                 # what is in the data, and is the harness sane
    alphamine random --n 10000      # the random-search baseline (experiment P0.2)
    alphamine gp --n 10000          # minimal genetic programming (P1)
    alphamine ablate --n 1000       # search-space ablations (idea R5-lite)
    alphamine leaderboard --n 2000  # equal-budget comparison across seeds (R3)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_panel
from .expr import Engine, canonical, depth, parse
from .ablation import DEFAULT_VARIANTS, run_ablations
from .ablation import plot_turnover_profile, select_formulas, turnover_profile
from .gp import GPConfig
from .leaderboard import run_leaderboard
from .gp import VARIABLES as GP_VARIABLES
from .gp import describe as describe_gp
from .gp import run_gp
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


def cmd_gp(args: argparse.Namespace) -> int:
    """Minimal genetic programming under the same fixed budget (experiment P1).

    Selection is on training IC only, so the validation curve stays a clean
    out-of-sample measure and can be put beside the P0.2 random baseline curve
    at the same number of evaluated formulas.
    """

    cfg = _cfg_from_args(args)
    gp = GPConfig(
        population=args.population,
        tournament=args.tournament,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        elite=args.elite,
        max_depth=args.depth,
        parsimony=args.parsimony,
        variables=tuple(args.variables) if args.variables else GP_VARIABLES,
    )
    print(f"evolving {args.n} formulas ({describe_gp(gp)}, seed {args.seed})")

    results = run_gp(
        cfg, budget=args.n, seed=args.seed, gp=gp, workers=args.workers
    )

    null_results = None
    if args.null:
        null_cfg = replace(cfg, shuffle_labels=True, shuffle_seed=args.seed + 777)
        print("running the shuffled-label null model on the same formulas ...")
        null_results = evaluate_batch(
            list(results["formula"]), null_cfg, workers=args.workers,
            with_turnover=False,
        )

    out_dir = persist_run(
        args.name,
        results,
        cfg,
        meta={
            "method": "gp",
            "max_depth": args.depth,
            "seed": args.seed,
            "null_model": bool(args.null),
            "gp": describe_gp(gp),
        },
    )
    if null_results is not None:
        null_results.to_csv(out_dir / "evals_null.csv", index=False)

    plot_budget_curve(
        results,
        out_dir / "budget_curve.png",
        title=f"Genetic programming, {len(results)} formulas (depth <= {args.depth})",
        null_results=null_results,
    )

    ok = results[results["status"] == "ok"]
    print()
    print(f"artefacts      : {out_dir}")
    print(f"status         : {results['status'].value_counts().to_dict()}")
    print(
        "median cost    : "
        f"{pd.to_numeric(results['seconds'], errors='coerce').median():.2f}s/formula"
    )
    for split, column in (("train", "train_rank_ic"), ("valid", "valid_rank_ic")):
        best = pd.to_numeric(ok[column], errors="coerce")
        if best.notna().any():
            idx = best.idxmax()
            print(f"best {split:5s} IC : {best.max():+.4f}   {ok.loc[idx, 'formula']}")
    if null_results is not None:
        null_best = pd.to_numeric(null_results["valid_rank_ic"], errors="coerce").max()
        print(f"null floor     : {null_best:+.4f}  (best validation IC on shuffled labels)")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """R3: compare search methods at equal budget, across seeds.

    Reports the honest metric - pick on training, report on validation - and the
    paired per-seed difference.  Reading the best-of-N column instead would
    reward whichever method was handed the luckier sample; P1 is the worked
    example.
    """

    cfg = _cfg_from_args(args)
    budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    print(
        f"leaderboard: {list(args.methods)} x seeds {list(args.seeds)}, "
        f"budget {args.n} each"
    )
    out = run_leaderboard(
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        budget=args.n,
        depth=args.depth,
        workers=args.workers,
        cfg=cfg,
        budgets=budgets,
        name=args.name,
    )
    print(f"\nartefacts: {out}\n")
    print((out / "leaderboard.md").read_text())
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


def cmd_profile(args: argparse.Namespace) -> int:
    """Turnover profile: where in the liquidity spectrum does a signal live?

    Formulas are picked on one period (training by default) and scored bucket by
    bucket on another, so no per-bucket search inflates any bucket's IC.
    """

    run_dir = Path(args.run)
    evals = pd.read_csv(run_dir / "evals.csv")
    formulas = select_formulas(evals, rank_by=args.rank_by, top=args.top)
    print(f"{len(formulas)} formulas chosen from {run_dir.name} by {args.rank_by}")

    cfg = _cfg_from_args(args)
    panel = load_panel(cfg)
    profile = turnover_profile(
        panel,
        formulas,
        n_buckets=args.buckets,
        split=args.split,
        horizon=args.horizon,
    )

    profile.to_csv(run_dir / f"turnover-profile-{args.split}.csv")
    plot_turnover_profile(
        profile,
        run_dir / f"turnover-profile-{args.split}.png",
        title=(
            f"Rank IC by turnover decile ({args.split}), "
            f"top {len(formulas)} formulas picked on {args.rank_by}"
        ),
    )

    buckets = [f"b{index}" for index in range(1, args.buckets + 1)]
    means = profile[buckets].mean()
    errors = profile[buckets].sem()
    print()
    print("turnover decile   mean rank IC   standard error   names/day")
    sizes = profile.attrs.get("bucket_size", {})
    for index, column in enumerate(buckets, start=1):
        print(
            f"  {index:>2d} (1=liquid)   {means[column]:+11.4f}   "
            f"{errors[column]:14.4f}   {sizes.get(index, float('nan')):9.0f}"
        )
    print()
    print(f"artefacts: {run_dir}/turnover-profile-{args.split}.csv|png")
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

    gpp = sub.add_parser("gp", help="minimal genetic programming (P1)")
    add_common(gpp)
    gpp.add_argument("--n", type=int, default=1000, help="evaluation budget")
    gpp.add_argument("--depth", type=int, default=4, help="maximum tree depth")
    gpp.add_argument("--seed", type=int, default=0)
    gpp.add_argument("--name", default="gp")
    gpp.add_argument("--variables", nargs="*", help="restrict the input variables")
    gpp.add_argument("--population", type=int, default=200)
    gpp.add_argument("--tournament", type=int, default=3)
    gpp.add_argument("--crossover", type=float, default=0.6)
    gpp.add_argument("--mutation", type=float, default=0.4)
    gpp.add_argument("--elite", type=int, default=2)
    gpp.add_argument(
        "--parsimony",
        type=float,
        default=0.0,
        help="IC tolerance within which the smaller tree wins a tournament",
    )
    gpp.add_argument("--turnover", action="store_true", help="also compute turnover")
    gpp.add_argument(
        "--score-test",
        action="store_true",
        help="also score the held-out test split (slower; normally reserved for the end)",
    )
    gpp.add_argument(
        "--null",
        action="store_true",
        help="also run the same formulas on shuffled labels (luck floor, idea R2)",
    )
    gpp.set_defaults(func=cmd_gp)

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

    lb = sub.add_parser(
        "leaderboard", help="equal-budget comparison across seeds (R3)"
    )
    add_common(lb)
    lb.add_argument("--n", type=int, default=2000, help="budget per run")
    lb.add_argument("--depth", type=int, default=4)
    lb.add_argument("--methods", nargs="*", default=["random", "gp"])
    lb.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    lb.add_argument(
        "--budgets",
        default="35,100,250,500,1000,2000",
        help="comma-separated budgets to report the curves at",
    )
    lb.add_argument("--name", default="r3-leaderboard")
    lb.set_defaults(func=cmd_leaderboard)

    prof = sub.add_parser("profile", help="rank IC by turnover bucket")
    add_common(prof)
    prof.add_argument("--run", required=True, help="a run directory with evals.csv")
    prof.add_argument("--top", type=int, default=20, help="formulas to profile")
    prof.add_argument(
        "--rank-by",
        default="train_rank_ic",
        help="which column selects the formulas (default: training, so validation stays clean)",
    )
    prof.add_argument("--buckets", type=int, default=10)
    prof.add_argument("--split", default="valid", help="period to measure the IC on")
    prof.add_argument("--horizon", type=int, default=5)
    prof.set_defaults(func=cmd_profile)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
