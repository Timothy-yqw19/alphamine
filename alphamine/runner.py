"""Running a batch of formulas under a fixed evaluation budget.

A "budget" here is a number of formulas, not a number of seconds.  The runner
keeps the order in which formulas were proposed so that
``best validation IC after n evaluations`` can be plotted - that curve, not the
final best number, is what makes two search strategies comparable.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_panel
from .eval import score_factor
from .expr import Engine, canonical, depth, parse, random_formula, size


def shuffle_within_dates(label: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute each date's forward returns across names.

    This is the null model of idea R2: the cross-sectional shape of the data is
    untouched but every real relation to a price/volume signal is destroyed, so
    whatever a search still finds is its luck floor.
    """

    arr = label.to_numpy(dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed)
    for i in range(arr.shape[0]):
        row = arr[i]
        present = np.isfinite(row)
        values = row[present]
        if values.size > 1:
            rng.shuffle(values)
            row[present] = values
    return pd.DataFrame(arr, index=label.index, columns=label.columns)


# --------------------------------------------------------------------------
# Worker plumbing.  Each process loads the panel once and then evaluates many
# formulas against it, so the per-formula cost is pure evaluation.
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(cfg: Config) -> None:
    panel = load_panel(cfg)
    label = panel.forward_return(cfg.horizon)
    if cfg.shuffle_labels:
        label = shuffle_within_dates(label, cfg.shuffle_seed)
    _WORKER.clear()
    _WORKER.update(
        cfg=cfg,
        panel=panel,
        label=label,
        engine=Engine(panel, cache_mb=cfg.cache_mb),
    )


def evaluate_one(formula: str, *, with_turnover: bool = False) -> dict:
    """Evaluate one formula in the current worker.  Returns a flat result row."""

    cfg: Config = _WORKER["cfg"]
    panel, engine, label = _WORKER["panel"], _WORKER["engine"], _WORKER["label"]

    started = time.perf_counter()
    row: dict = {"formula": formula, "status": "ok", "message": ""}
    try:
        node = parse(formula)
    except Exception as exc:  # a malformed formula still consumes budget
        row.update(status="error", message=f"parse: {exc}"[:200])
        return row

    row["depth"] = depth(node)
    row["size"] = size(node)
    try:
        factor = engine.evaluate(node)
    except Exception as exc:
        row.update(status="error", message=f"{type(exc).__name__}: {exc}"[:200])
        row["eval_seconds"] = time.perf_counter() - started
        return row
    row["eval_seconds"] = time.perf_counter() - started

    split_names = ("train", "valid", "test") if cfg.score_test else ("train", "valid")
    scoring_started = time.perf_counter()
    try:
        score = score_factor(
            panel,
            factor,
            horizon=cfg.horizon,
            splits=cfg.splits,
            min_cross=cfg.min_cross,
            label=label,
            with_turnover=with_turnover,
            with_pearson=cfg.score_pearson,
            split_names=split_names,
        )
    except Exception as exc:
        row.update(status="error", message=f"{type(exc).__name__}: {exc}"[:200])
        row["score_seconds"] = time.perf_counter() - scoring_started
        row["seconds"] = time.perf_counter() - started
        return row

    row |= score.as_dict()
    row["formula"] = formula
    row["depth"] = depth(node)
    row["size"] = size(node)
    row["score_seconds"] = time.perf_counter() - scoring_started
    row["seconds"] = time.perf_counter() - started
    return row


def evaluate_batch(
    formulas: list[str],
    cfg: Config,
    *,
    workers: int | None = None,
    with_turnover: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate every formula, preserving order."""

    workers = cfg.resolved_workers() if workers is None else max(1, workers)
    bar = _progress(total=len(formulas)) if progress else None
    try:
        if workers == 1:
            _init_worker(cfg)
            rows = []
            for formula in formulas:
                rows.append(evaluate_one(formula, with_turnover=with_turnover))
                if bar is not None:
                    bar.update(1)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                workers, initializer=_init_worker, initargs=(cfg,)
            ) as pool:
                rows = []
                iterator = pool.imap_unordered(
                    _evaluate_task,
                    ((f, with_turnover) for f in formulas),
                    chunksize=4,
                )
                for row in iterator:
                    rows.append(row)
                    if bar is not None:
                        bar.update(1)
            rows = _restore_order(rows, formulas)
    finally:
        if bar is not None:
            bar.close()
    return pd.DataFrame(rows)


def _evaluate_task(task: tuple[str, bool]) -> dict:
    formula, with_turnover = task
    row = evaluate_one(formula, with_turnover=with_turnover)
    row["_formula_key"] = formula
    return row


def _restore_order(rows: list[dict], formulas: list[str]) -> list[dict]:
    """``imap_unordered`` is faster but loses order; put it back.

    The budget curve depends on the order candidates were proposed, so this is
    correctness, not cosmetics.  Duplicate formulas are impossible because the
    sampler de-duplicates, but the index fallback keeps it safe anyway.
    """

    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.pop("_formula_key"), []).append(row)
    ordered: list[dict] = []
    for formula in formulas:
        bucket = buckets.get(formula)
        if bucket:
            ordered.append(bucket.pop())
    for bucket in buckets.values():
        ordered.extend(bucket)
    return ordered


def _progress(total: int):
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return None
    return tqdm(total=total, unit=" formula", smoothing=0.05)


def sample_random_formulas(
    n: int,
    *,
    max_depth: int = 4,
    seed: int = 0,
    variables: tuple[str, ...] | None = None,
    allowed_ops: tuple[str, ...] | None = None,
    leaf_prob: float = 0.3,
) -> list[str]:
    """``n`` distinct random formulas, in the order they were drawn."""

    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    limit = max(1000, n * 200)
    while len(out) < n and attempts < limit:
        attempts += 1
        node = random_formula(
            rng,
            max_depth=max_depth,
            variables=variables,
            allowed_ops=allowed_ops,
            leaf_prob=leaf_prob,
        )
        text = canonical(node)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def budget_curve(results: pd.DataFrame, column: str = "valid_rank_ic") -> pd.DataFrame:
    """Best value of ``column`` seen after each evaluation, in proposal order."""

    values = pd.to_numeric(results[column], errors="coerce").to_numpy(dtype="float64")
    finite = np.where(np.isfinite(values), values, np.nan)
    running = np.fmax.accumulate(np.nan_to_num(finite, nan=-np.inf))
    running[np.isneginf(running)] = np.nan
    return pd.DataFrame(
        {
            "evaluations": np.arange(1, len(values) + 1),
            "best": running,
            "current": finite,
        }
    )


def persist_run(
    name: str,
    results: pd.DataFrame,
    cfg: Config,
    *,
    meta: dict | None = None,
) -> Path:
    """Write the run's artefacts and return its directory."""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(cfg.runs_dir) / f"{stamp}-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "evals.csv", index=False)

    summary = {
        "name": name,
        "created": stamp,
        "config": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
        },
        "n_evaluations": int(len(results)),
        "status_counts": results["status"].value_counts().to_dict(),
        "median_seconds": float(pd.to_numeric(results["seconds"], errors="coerce").median()),
        "best": {},
    }
    for split, column in (
        ("train", "train_rank_ic"),
        ("valid", "valid_rank_ic"),
        ("test", "test_rank_ic"),
    ):
        series = pd.to_numeric(results[column], errors="coerce")
        if series.notna().any():
            summary["best"][split] = float(series.max())
            summary["best"][f"{split}_formula"] = results.loc[series.idxmax(), "formula"]
    if meta:
        summary["meta"] = meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return out_dir


def plot_budget_curve(
    results: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    null_results: pd.DataFrame | None = None,
) -> None:
    """Best-so-far train vs validation IC against the evaluation budget."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for column, label, style in (
        ("train_rank_ic", "best train rank IC", "-"),
        ("valid_rank_ic", "best validation rank IC", "-"),
    ):
        curve = budget_curve(results, column)
        ax.plot(curve["evaluations"], curve["best"], style, linewidth=1.8, label=label)

    if null_results is not None:
        curve = budget_curve(null_results, "valid_rank_ic")
        ax.plot(
            curve["evaluations"],
            curve["best"],
            "--",
            color="grey",
            linewidth=1.5,
            label="best validation IC, shuffled labels (luck floor)",
        )

    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("formulas evaluated")
    ax.set_ylabel("rank IC")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
