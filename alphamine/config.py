"""Paths, splits and run settings.

Values can be overridden with environment variables so that the same code runs
unchanged on a laptop and on a bigger box:

``ALPHAMINE_DATA``    path to ``daily_pv.h5`` (default: ``<repo>/data/daily_pv.h5``)
``ALPHAMINE_RUNS``    directory for run artefacts (default: ``<repo>/runs``)
``ALPHAMINE_WORKERS`` default worker-process count
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


DATA_PATH = _env_path("ALPHAMINE_DATA", PROJECT_ROOT / "data" / "daily_pv.h5")
DEBUG_DATA_PATH = _env_path("ALPHAMINE_DATA_DEBUG", PROJECT_ROOT / "data" / "daily_pv_debug.h5")
RUNS_DIR = _env_path("ALPHAMINE_RUNS", PROJECT_ROOT / "runs")


@dataclass(frozen=True)
class Split:
    """A name and an inclusive date range."""

    name: str
    start: str
    end: str | None

    def label(self) -> str:
        return f"{self.name} [{self.start}..{self.end or 'end'}]"


#: Time-ordered splits.  Never shuffle across time.
#: The last ``horizon`` rows of a split are dropped when scoring it (see
#: :func:`alphamine.eval.metrics.split_mask`) so that a label built from
#: ``close(t+h)`` can never reach into the next split.
DEFAULT_SPLITS: tuple[Split, ...] = (
    Split("train", "2012-01-01", "2018-12-31"),
    Split("valid", "2019-01-01", "2020-12-31"),
    Split("test", "2021-01-01", None),
)


@dataclass
class Config:
    """Everything a run needs to be reproducible."""

    data_path: Path = field(default_factory=lambda: DATA_PATH)
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)

    # Panel window.  Loading all of 2008-2026 is fine, but the A-share universe
    # is thin before ~2012 and the memory saving is real.
    start: str = "2012-01-01"
    end: str | None = None

    #: Forward-return horizon in trading days: label(t) = close(t+h)/close(t+1) - 1.
    #: The signal is formed at ``t`` and the trade happens at ``t+1``, so the
    #: label never uses a price that was available when the signal was computed.
    horizon: int = 5

    #: Drop this many of an instrument's first traded days.  IPO first days in
    #: China are not tradeable (no price limit on STAR/ChiNext for the first
    #: days) and would otherwise dominate every turnover and return statistic.
    min_listing_days: int = 60

    #: Include the Beijing Stock Exchange, which trades under a 30% price limit
    #: and is much thinner.  Off by default so the universe is comparable with
    #: published A-share work.
    include_bse: bool = False

    #: Keep only the ``universe_top_n`` most traded names on each date.  ``None``
    #: means the whole A-share panel.  This is a *turnover* ranking, not CSI 300
    #: membership, but it is the closest this dataset allows to the large,
    #: liquid segment that published ICs are usually quoted on.
    universe_top_n: int | None = None

    #: Minimum number of cross-sectional names before a daily IC is considered
    #: meaningful.
    min_cross: int = 100

    #: Per-process subtree cache budget in megabytes.  One full panel is ~74 MB,
    #: so this holds roughly ten subterms; measured against the alternatives it
    #: is the point where cache hits stop paying for the memory.
    cache_mb: int = 800

    #: Worker processes.  0 means "decide from ``os.cpu_count()``".
    workers: int = 0

    splits: tuple[Split, ...] = DEFAULT_SPLITS

    #: Shuffle the forward returns within each date.  This destroys any real
    #: relation while keeping the cross-sectional structure, giving a
    #: method-specific luck floor (idea R2 in the study plan).
    shuffle_labels: bool = False
    shuffle_seed: int = 12345

    #: Also score the test split during a search run.  Off by default: the study
    #: plan's own checklist says the test period is looked at once, at the end,
    #: and scoring it costs a third of the runtime for every candidate.
    score_test: bool = False

    #: Also compute Pearson IC next to rank IC.  Rank IC is the headline metric;
    #: Pearson adds a whole extra correlation pass per split, roughly a third of
    #: the per-formula cost.
    score_pearson: bool = False

    def resolved_workers(self) -> int:
        if self.workers > 0:
            return self.workers
        # Each worker holds the panel, the derived variables and its cache,
        # about 1.4 GB.  Four keeps a 16 GB laptop comfortable.
        return max(1, min(4, (os.cpu_count() or 2) - 2))

    def split_by_name(self, name: str) -> Split:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(f"unknown split {name!r}; known: {[s.name for s in self.splits]}")
