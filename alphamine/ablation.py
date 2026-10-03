"""R5-lite: which part of the search space carries the signal?

The P0.2 result said the random baseline saturates fast and that the formulas it
finds are mostly reversal and volume/price structure.  That raises the question
the study plan files under idea R5: is the *procedure* doing anything, or is the
*grammar* a strong prior that would work under any procedure?

This module changes one ingredient of the search space at a time, holding the
evaluation budget fixed, and reports the budget curve for each variant.

Two design choices matter for the honesty of the comparison:

* the universe variants must come with their own **null model**.  A 300-name
  cross-section gives a noisier daily IC than a 3,500-name one, so the best of
  N draws is inflated by more.  Comparing +0.05 on 300 names with +0.05 on 3,500
  without matching the luck floors would be a mistake.
* restricting the universe restricts the *panel*, not just the scoring mask, so
  ``rank`` and ``zscore`` are computed inside the liquid universe.  That is what
  a study run on large caps would actually do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import liquidity_rank
from .eval import rank_ic_series
from .expr import OPERATORS, default_pool
from .runner import budget_curve, evaluate_batch, sample_random_formulas

#: Price-only inputs, used by the ``no-volume`` variant.
PRICE_VARIABLES = ("open", "high", "low", "close", "returns", "vwap_proxy")


@dataclass(frozen=True)
class Variant:
    """One search-space configuration."""

    name: str
    question: str
    universe: tuple[str, int] | None = None
    ops: tuple[str, ...] | None = None
    variables: tuple[str, ...] | None = None


def _without(categories: set[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in default_pool()
        if OPERATORS[name].category not in categories
    )


#: The ablation set.  ``baseline`` and ``liquid300`` sample the *same* formulas
#: and differ only in the universe, so that pair isolates universe from grammar.
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(
        "baseline",
        "everything: full grammar, all A-shares",
    ),
    Variant(
        "liquid300",
        "same formulas, 300 most traded names",
        universe=("top", 300),
    ),
    Variant(
        "illiquid300",
        "same formulas, 300 least traded names",
        universe=("bottom", 300),
    ),
    Variant(
        "no-cross-section",
        "drop rank/zscore/scale/demean/cs_median",
        ops=_without({"cs"}),
    ),
    Variant(
        "no-volume",
        "drop volume and adv20 inputs",
        variables=PRICE_VARIABLES,
    ),
    Variant(
        "no-time-series",
        "drop every ts_* operator",
        ops=_without({"ts"}),
    ),
)


def variant_config(base: Config, variant: Variant) -> Config:
    return replace(base, universe_slice=variant.universe)


def run_ablations(
    *,
    n: int,
    depth: int,
    seed: int,
    workers: int,
    base: Config | None = None,
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
    null_variants: tuple[str, ...] = ("baseline", "liquid300"),
    name: str = "r5-lite",
) -> Path:
    """Run every variant under the same budget and write the comparison."""

    base = base or Config()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(base.runs_dir) / f"{stamp}-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "name": name,
        "created": stamp,
        "budget_per_variant": n,
        "max_depth": depth,
        "seed": seed,
        "variants": {},
    }
    frames: dict[str, pd.DataFrame] = {}
    nulls: dict[str, pd.DataFrame] = {}

    for variant in variants:
        cfg = variant_config(base, variant)
        print(f"\n=== {variant.name}: {variant.question}")
        formulas = sample_random_formulas(
            n,
            max_depth=depth,
            seed=seed,
            variables=variant.variables,
            allowed_ops=variant.ops,
        )
        print(f"    {len(formulas)} distinct formulas, {workers} workers")
        frame = evaluate_batch(formulas, cfg, workers=workers)
        frames[variant.name] = frame
        frame.to_csv(out_dir / f"evals-{variant.name}.csv", index=False)

        entry = {
            "question": variant.question,
            "universe_slice": list(variant.universe) if variant.universe else None,
            "n_operators": len(variant.ops) if variant.ops else len(default_pool()),
            "n_variables": len(variant.variables) if variant.variables else 8,
            "status_counts": frame["status"].value_counts().to_dict(),
            "median_seconds": float(
                pd.to_numeric(frame["seconds"], errors="coerce").median()
            ),
            "curve": _budget_table(frame),
        }
        if variant.name in null_variants:
            print(f"    ... and the shuffled-label null for {variant.name}")
            null_frame = _run_null(base, variant, formulas, workers, seed)
            nulls[variant.name] = null_frame
            null_frame.to_csv(out_dir / f"evals-{variant.name}-null.csv", index=False)
            entry["null_curve"] = _budget_table(null_frame)
        summary["variants"][variant.name] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _plot(frames, nulls, out_dir / "r5-lite.png")
    (out_dir / "r5-lite.md").write_text(_markdown(summary, n))
    return out_dir


def _run_null(
    base: Config,
    variant: Variant,
    formulas: list[str],
    workers: int,
    seed: int,
) -> pd.DataFrame:
    cfg = replace(
        variant_config(base, variant), shuffle_labels=True, shuffle_seed=seed + 7919
    )
    return evaluate_batch(formulas, cfg, workers=workers, progress=False)


def _budget_table(frame: pd.DataFrame) -> dict:
    curve = budget_curve(frame, "valid_rank_ic")
    train = budget_curve(frame, "train_rank_ic")
    out = {}
    for checkpoint in (100, 250, 500, 1000):
        if len(frame) < checkpoint:
            continue
        out[str(checkpoint)] = {
            "train": _finite(train["best"].iloc[checkpoint - 1]),
            "valid": _finite(curve["best"].iloc[checkpoint - 1]),
        }
    valid = pd.to_numeric(frame["valid_rank_ic"], errors="coerce")
    out["final"] = {
        "train": _finite(pd.to_numeric(frame["train_rank_ic"], errors="coerce").max()),
        "valid": _finite(valid.max()),
        "median_valid": _finite(valid.median()),
        "best_formula": str(frame.loc[valid.idxmax(), "formula"]) if valid.notna().any() else "",
    }
    return out


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _markdown(summary: dict, n: int) -> str:
    lines = [
        f"# R5-lite: search-space ablations ({n} formulas per variant)",
        "",
        f"Seed {summary['seed']}, max depth {summary['max_depth']}, "
        f"budget {n} evaluations per variant.",
        "",
        "| variant | operators | variables | best train | best valid | null valid | valid at N=250 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in summary["variants"].items():
        final = entry["curve"]["final"]
        null = entry.get("null_curve", {}).get("final", {}).get("valid")
        at250 = entry["curve"].get("250", {}).get("valid")
        lines.append(
            f"| `{name}` | {entry['n_operators']} | {entry['n_variables']} | "
            f"{_fmt(final['train'])} | {_fmt(final['valid'])} | {_fmt(null)} | {_fmt(at250)} |"
        )
    lines += ["", "## What each variant asks", ""]
    for name, entry in summary["variants"].items():
        lines.append(f"* **{name}** - {entry['question']}")
    lines += ["", "## Best formula per variant", ""]
    for name, entry in summary["variants"].items():
        lines.append(f"* `{name}`: `{entry['curve']['final']['best_formula']}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _plot(
    frames: dict[str, pd.DataFrame],
    nulls: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    for name, frame in frames.items():
        curve = budget_curve(frame, "valid_rank_ic")
        ax.plot(curve["evaluations"], curve["best"], linewidth=1.8, label=name)
    for name, frame in nulls.items():
        curve = budget_curve(frame, "valid_rank_ic")
        ax.plot(
            curve["evaluations"],
            curve["best"],
            linestyle="--",
            linewidth=1.2,
            color="grey",
            alpha=0.8,
            label=f"{name} (shuffled labels)",
        )
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("formulas evaluated")
    ax.set_ylabel("best validation rank IC")
    ax.set_title("R5-lite: which part of the search space carries the signal?")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Turnover profile
#
# The ablation answers "which universe scores better", but comparing two
# separately-searched universes mixes the signal level with each universe's
# selection floor.  This profile removes that entirely: the formulas are chosen
# on the *training* period and then scored bucket by bucket on validation, so
# the shape across buckets is not affected by any per-bucket search.
# ---------------------------------------------------------------------------


def turnover_buckets(panel, n_buckets: int = 10, window: int = 20) -> pd.DataFrame:
    """Decile of trailing turnover, 1 = most traded."""

    ranks = liquidity_rank(panel, window)
    live = ranks.notna().sum(axis=1).replace(0, np.nan)
    fraction = ranks.sub(1).div(live, axis=0)
    buckets = np.floor(fraction * n_buckets).clip(upper=n_buckets - 1) + 1
    return buckets.where(panel.tradable)


def turnover_profile(
    panel,
    formulas: list[str],
    *,
    n_buckets: int = 10,
    window: int = 20,
    horizon: int = 5,
    min_cross: int = 50,
    split: str = "valid",
    splits: tuple | None = None,
    engine=None,
) -> pd.DataFrame:
    """Mean rank IC of each formula inside each turnover bucket.

    ``split`` selects the period the IC is measured on.  Pick formulas on the
    training period and profile them on validation; picking them on validation
    and profiling them on validation would bake the selection into every bucket.
    """

    from .eval.metrics import split_dates
    from .expr import Engine, parse

    splits = splits or Config().splits
    chosen = next(s for s in splits if s.name == split)
    engine = engine or Engine(panel, cache_mb=900)
    dates = split_dates(panel, chosen, horizon)
    window_slice = slice(dates[0], dates[-1])

    label = panel.forward_return(horizon)
    buckets = turnover_buckets(panel, n_buckets, window)
    factors = {text: engine.evaluate(parse(text)) for text in formulas}

    # Slice to the scoring window before correlating: the ranks are
    # cross-sectional, so a date slice changes nothing but the runtime.
    label_window = label.loc[window_slice]
    factors_window = {text: frame.loc[window_slice] for text, frame in factors.items()}

    out = {}
    sizes = {}
    for index in range(1, n_buckets + 1):
        mask = (panel.tradable & buckets.eq(index)).loc[window_slice]
        sizes[index] = float(mask.sum(axis=1).mean())
        for text in formulas:
            ic = rank_ic_series(
                factors_window[text], label_window, mask, min_cross=min_cross
            )
            out.setdefault(text, {})[index] = _finite(ic.mean())

    frame = pd.DataFrame(out).T
    frame.index.name = "formula"
    frame.columns = [f"b{index}" for index in frame.columns]
    frame.attrs["bucket_size"] = sizes
    frame.attrs["split"] = split
    return frame


def plot_turnover_profile(
    profile: pd.DataFrame, out_path: Path, *, title: str, top_n: int = 8
) -> None:
    """Mean rank IC with an error bar, plus the individual formulas behind it.

    The mean is the point of the chart - a single formula's bucket-to-bucket
    wiggle is mostly estimation noise - so the individual curves are drawn
    faintly and the mean carries the error bars.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(profile.columns)
    buckets = [int(column[1:]) for column in columns]
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)

    for _, row in profile.iterrows():
        ax.plot(
            buckets,
            row.to_numpy(dtype=float),
            color="grey",
            linewidth=0.8,
            alpha=0.35,
            zorder=1,
        )

    means = profile[columns].mean().to_numpy(dtype=float)
    errors = profile[columns].sem().to_numpy(dtype=float)
    ax.errorbar(
        buckets,
        means,
        yerr=errors,
        marker="o",
        linewidth=2.2,
        capsize=4,
        color="tab:blue",
        label=f"mean of {len(profile)} formulas",
        zorder=3,
    )
    for bucket, value in zip(buckets, means, strict=True):
        ax.annotate(
            f"{value:+.3f}",
            (bucket, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7.5,
            color="tab:blue",
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("turnover decile (1 = most traded)")
    ax.set_ylabel("rank IC")
    ax.set_title(title)
    ax.set_xticks(buckets)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def select_formulas(
    evals: pd.DataFrame, *, rank_by: str = "train_rank_ic", top: int = 20
) -> list[str]:
    """The best formula texts from a run, chosen on one period only."""

    ok = evals[evals["status"] == "ok"]
    scores = pd.to_numeric(ok[rank_by], errors="coerce")
    return ok.loc[scores.nlargest(top).index, "formula"].tolist()
