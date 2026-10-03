# Handoff

Written 2026-10-02 for whoever continues this work - human or agent - in a fresh
session. Everything below was measured in this repository, not assumed. Where a
number is quoted, the run directory that produced it is named so you can check
it.

Read this, then read `README.md`, then `docs/p02-random-baseline.md` and
`docs/r5-lite.md`. Nothing else is needed to pick the work up.

---

## 1. What the project is

**Goal.** Understand and compare the *search strategies* behind formulaic alpha
mining - random search, genetic programming, RL generation, surrogate models,
LLM loops - on a laptop, using A-share daily price/volume data.

**Not the goal.** Producing tradeable factors, matching paper benchmarks,
industrial data pipelines, live trading.

**The one rule, which everything else serves.** Hold the evaluation budget
fixed. A method is scored by the best validation rank IC it reaches after the
same number of *formulas evaluated*. Otherwise you are comparing compute, not
strategy.

That rule is why the operator set is bounded by measured wall-clock cost (see
`max_size`/`max_window` in `alphamine/expr/ops.py`) and why every run records the
evaluation index of every formula.

---

## 2. Where things are

| | |
| --- | --- |
| Repository | https://github.com/Timothy-yqw19/alphamine (public, MIT) |
| Remote | `git@github.com:Timothy-yqw19/alphamine.git`, branch `main` |
| Local path | `/Users/wangsheng/Documents/ChatGPT/New project` |
| Conda env | `alphamine`, Python 3.11.17, at `/opt/anaconda3/envs/alphamine` |
| Machine | Apple Silicon, macOS, 17.2 GB RAM, 10 cores, no CUDA |

Packages in the env: numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, tables 3.11.1,
h5py 3.16.0, matplotlib 3.11.2, pytest 9.1.1. **pandas 3.0** matters: copy-on-write
is always on and `pct_change` no longer pads by default.

Commits:

```
d179aff  feat: turnover contrast, MIT licence, and a fixed-formula decile profile
07a68c6  docs: ship the experiment charts inside the repo
1fbbf23  feat: fixed-budget harness for studying how alpha-mining systems search
```

---

## 3. Get running

```bash
cd "/Users/wangsheng/Documents/ChatGPT/New project"
conda activate alphamine

# 398 MB, not in the repo
curl -L -o data/daily_pv.h5 \
  https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/daily_pv.h5
# 1.4 MB, a 100-instrument fixture used by the tests
curl -L -o data/daily_pv_debug.h5 \
  https://huggingface.co/datasets/QuantaAlpha/qlib_csi300/resolve/main/daily_pv_debug.h5

python -m alphamine.cli check          # panel, universe, label, per-op cost
python -m pytest -q                    # 44 tests, ~0.3 s
```

`MPLCONFIGDIR=/tmp/mplcache` avoids a matplotlib cache warning; the user's home
directory is not writable from the sandbox.

---

## 4. Architecture

| Path | Responsibility |
| --- | --- |
| `alphamine/config.py` | `Config` dataclass: paths, date window, horizon, universe rules, workers, cache size, which splits to score |
| `alphamine/data.py` | Load HDF5, coerce to a date x instrument float32 panel, universe rules, forward-return labels, turnover ranking and universe restriction |
| `alphamine/expr/nodes.py` | Expression trees and the **canonical string**, which doubles as the cache key |
| `alphamine/expr/ops.py` | 38 operators in three families (`arith`, `ts`, `cs`), each with a cost-derived window cap |
| `alphamine/expr/parse.py` | Parser: `ts_mean(volume, 20)` and infix `a / b` both work; `to_rpn` for the future RL generator |
| `alphamine/expr/engine.py` | Evaluator with an LRU **byte-budgeted** subtree cache |
| `alphamine/expr/sample.py` | Type-aware random sampler |
| `alphamine/eval/metrics.py` | Cross-sectional rank IC, Pearson IC, ICIR, turnover, decay |
| `alphamine/runner.py` | Batch evaluation across worker processes, budget curves, run artefacts |
| `alphamine/ablation.py` | R5-lite variants, the turnover profile, selection of formulas by one split |
| `alphamine/cli.py` | `check`, `random`, `ablate`, `profile` |

Design decisions worth keeping:

* **Canonical string as the identity of a formula.** Commutative operators are
  sorted, so `a+b` and `b+a` are one candidate and one cache entry. This makes
  the budget count distinct ideas rather than duplicates.
* **Byte-budgeted LRU cache.** One panel is ~74 MB, so an unbounded cache dies
  within a few hundred formulas. Default 800 MB per worker.
* **Cost caps per operator.** A fixed formula budget is a fair proxy for a fixed
  compute budget only if no single operator dominates. `ts_argmax` is capped at
  window 30, `ts_rank` at 60, and `ts_median` is kept out of the default
  sampling pool entirely (measured: no fast rolling median exists in pandas or
  numpy).
* **Score rank IC, not Pearson, by default.** Pearson costs a second full
  correlation pass per split, about a third of the per-formula runtime.

---

## 5. Data facts verified by hand

Source: `QuantaAlpha/qlib_csi300` on Hugging Face, Apache-2.0.

* 14,215,449 rows, 5,982 instruments, 4,138 dates, 2008-12-29 to 2026-01-09.
* Columns: `$open $high $low $close $volume $factor`. **No vwap, no turnover in
  currency, no market cap, no industry.**
* **`$close` is already adjusted.** Large moves do not cluster in the June/July
  dividend season the way unadjusted A-share prices would, so returns come from
  `$close` directly.
* **`$close / $factor` is the raw quoted price.** Checked against known closes:
  Kweichow Moutai 1691 on 2023-06-30, Ping An 46.40, SMIC 50.52, SPD Bank 13.25
  in 2008 and 6.62 in 2023. Use it for tick-size or price-limit rules.
* `daily_pv_debug.h5` is **100 instruments x 487 dates**, not a subsample of
  5,982. Do not use it to reason about universe-scale behaviour; it is a test
  fixture.
* A naive `index.levels[1].nunique()` over-reports instruments badly (3,784 for
  the 100-instrument file) because MultiIndex levels are padded. Use
  `levels[i].unique()` or the actual values.

### Universe rules applied by `build_panel`

1. keep individual stock codes only - indices (`SH000300`, `SZ399300`),
   B-shares, and by default Beijing listings (30% price limit, far thinner);
2. keep rows where the stock actually traded (`volume > 0`, finite close), which
   removes suspension days;
3. drop each stock's first 60 traded days, so IPO first days (for example
   `SH688026 +108%` on its listing day, STAR has no price limit at first) cannot
   dominate.

Applied over 2012-01-04 to 2026-01-09 this gives **3,405 dates x 5,400
instruments, 1,239 to 5,146 names per day, mean 3,536**. The minimum is July
2015, when more than 1,400 A-share stocks were suspended at once - a useful
sanity check that the tradability filter works.

### Label and splits

`label(t) = close(t+5) / close(t+1) - 1`. Signal at `t`, trade at `t+1`, exit at
`t+5`. Splits: train 2012-2018, validation 2019-2020, test 2021 onwards. Each
split drops its last `horizon` rows so a label cannot reach into the next one.
`metrics.decay_curve` used to break that rule - it filtered by `start`/`end`
alone - and has been fixed; see bug 10 in section 7.

**The test period has not been spent, but it was computed once.** An early harness
smoke run (`runs/20261001-215104-smoke`, 60 formulas) predates `Config.score_test`
and scored the test split; nothing was selected on it, no document cites it, and
every later run leaves the test column empty. "Not spent" is the claim that
matters; "never touched" is not accurate.

Note that horizon 1 is degenerate under this convention (`close(t+1)/close(t+1) - 1`
is identically zero), so `decay_curve` starts at 2.

### Fields the search may use

`open high low close volume returns vwap_proxy adv20`

`vwap_proxy` is `(high + low + close) / 3` - a stand-in, not the variable the 101
Alphas were written for. Of the 101, **35 can be reproduced from OHLCV alone**;
47 need `vwap` (34) or an industry classification (13), and 19 are not implemented
at all. That split is reproducible rather than estimated: statically parsing the
two implementations in `yli188/WorldQuant_alpha101_code` gives 35 / 34 / 13 / 19 =
101, and the same 35 are OHLCV-only in both files. The 19 unimplemented ones are
concentrated in the high-numbered alphas - 48, 56, 58, 59, 63, 67, 69, 70, 76, 79,
80, 82, 87, 89, 90, 91, 93, 97, 100. Re-run `scripts/count_101_alphas.py` to
re-derive all of that and to print the 35 ids, which are the P0.1 worklist.

---

## 6. Results so far

### P0.2 - random baseline

Run `runs/20261001-224125-p02-random-depth4`: 2,000 distinct formulas, depth <= 4,
full panel, five workers.

| N evaluated | best train rank IC | best validation rank IC | shuffled-label validation |
| --- | --- | --- | --- |
| 100 | +0.0426 | +0.0411 | +0.0016 |
| 250 | +0.0742 | +0.0618 | +0.0035 |
| 1,000 | +0.0742 | +0.0647 | +0.0035 |
| 2,000 | +0.0742 | +0.0647 | +0.0073 |

Across all 1,852 non-degenerate formulas: median validation rank IC -0.0051,
10th percentile -0.0396, 90th percentile +0.0184. 7.4% of sampled trees were
degenerate.

**The curve saturates by N = 100.** Random search gets 63% of its final answer
from the first 5% of the budget. Compare methods at N = 100..1,000, not only at
a large N.

### R5-lite - search-space ablations

Runs `runs/20261001-225430-r5-lite` and `runs/20261002-202029-r5-lite-turnover`.
1,000 formulas per variant, same seed, same depth.

| variant | operators | inputs | best train | best valid | shuffled-label valid | net |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` (all A-shares) | 37 | 8 | +0.0742 | +0.0647 | +0.0032 | +0.0615 |
| `liquid300` (most traded) | 37 | 8 | +0.0794 | +0.0955 | +0.0112 | +0.0843 |
| `illiquid300` (least traded) | 37 | 8 | +0.0605 | +0.0757 | +0.0354 | +0.0403 |
| `no-cross-section` | 32 | 8 | +0.0772 | +0.0561 | - | - |
| `no-volume` | 37 | 6 | +0.0593 | +0.0647 | - | - |
| `no-time-series` | 20 | 8 | +0.0844 | +0.0550 | - | - |

Four findings, in order of importance:

1. **The universe is a bigger lever than any grammar change**, and it goes the
   opposite way to the obvious guess. The signal is not a small-cap illiquidity
   artifact; it is about two and a half times stronger in the most traded names.
2. **The illiquid universe is a noise trap.** Its best-of-1000 on *shuffled*
   labels is +0.0354 against +0.0112 for the liquid universe, so raw IC ranks
   the universes backwards. Its winning formula
   `ts_skew(max2(sign(low), ts_argmin(low, 15)), 20)` is finite on 49% of names,
   is `NaN` on `liquid300`, and scores **-0.0171** on the full panel.
3. **Cross-sectional normalisation is nearly worthless** (removing all five
   `rank`/`zscore`/`scale`/`demean`/`cs_median` costs 13%) and **volume inputs
   are nearly worthless** - the volume *term* is value-neutral rather than inert,
   see the note on it below.
4. **Simpler grammars overfit more.** The train-minus-validation gap is +0.0095
   for `baseline`, +0.0211 for `no-cross-section`, +0.0294 for `no-time-series`.

### The winning signal, characterised rather than read

The `baseline` winner is
`mul(ts_max(sign(volume), 60), ts_min(div(low, vwap_proxy), 30))`. `sign(volume)`
is 1 wherever volume is finite and `NaN` where it is not, so
`ts_max(sign(volume), 60)` is a 60-day "has volume been observed" gate. It is
value-neutral but not inert - the real factor is:

```
ts_min(low / ((high + low + close) / 3), 30)
```

and deleting the gate changes the validation IC by **0.00004** while lifting
coverage from 0.975 to 0.986. **Always simplify a mined formula before believing
it - but check what the part you deleted was actually doing.**

| property | value | convention |
| --- | --- | --- |
| validation rank IC, 5-day horizon | +0.0646 | `score_factor`, split tail dropped |
| validation rank IC, 20-day horizon | +0.0890 | `score_factor`, split tail dropped |
| one-day turnover of the rank-weighted book | 0.033 | `metrics.turnover` |
| correlation with 20-day realised volatility | -0.763 | mean daily cross-sectional Spearman |
| correlation with the trailing 3-day return | -0.004 | mean daily cross-sectional Spearman |
| validation rank IC on `liquid300` / all / `illiquid300` | +0.0955 / +0.0647 / +0.0298 | full stored winner, not simplified |

Earlier drafts of this table quoted +0.0642 / +0.0883 for the first two rows and a
volume-term effect of 0.0001. Those came from `metrics.decay_curve`, which did not
drop the boundary rows; it has since been fixed and now agrees with
`score_factor`, so those two values survive only as pre-fix numbers. The bottom
row is the full stored winner, because the volume gate changes coverage and
therefore the per-universe name counts.

It is a **low-volatility, price-stability characteristic**, orthogonal to
reversal, with low turnover, and it gets stronger at longer horizons. An earlier
draft of `docs/p02-random-baseline.md` guessed "short-horizon reversal" from
reading the formula; the decay and turnover measurements say otherwise. That
correction is recorded in both documents.

### Turnover decile profile

Ten formulas are not enough and searching per bucket is worse. The profile takes
the 50 formulas with the best **training** IC from the P0.2 run and scores each
one inside every turnover decile on **validation**. No search happens per bucket,
so no bucket's IC is a selected maximum.

| decile (1 = most traded) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean rank IC | +0.040 | +0.044 | +0.038 | +0.038 | +0.034 | +0.033 | +0.032 | +0.026 | +0.023 | +0.016 |
| standard error | 0.004 | 0.004 | 0.003 | 0.003 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |

Monotone from decile 2 down, ends differ by +0.024 against a combined standard
error of 0.005.

### Null models - the calibration

Three separate nulls, all measured:

| null | draws | validation max |
| --- | --- | --- |
| pure random-number factors, full panel | 200 | +0.0022 |
| shuffled labels, P0.2 grammar | 2,000 | +0.0073 |
| shuffled labels, `liquid300` | 1,000 | +0.0112 |
| shuffled labels, `illiquid300` | 1,000 | +0.0354 |

The random-number null is centred at exactly 0.0000 with a standard deviation of
0.0008, which is the evidence that the harness has no alignment or look-ahead
bias. The operational rule that falls out of it: **a validation edge smaller
than about 0.002 is not distinguishable from noise on a single seed**, and
**ICs must never be compared across universes without each universe's own
null.**

---

## 7. Bugs found and fixed - do not reintroduce them

Every one of these was a real defect that produced silently wrong numbers.
Regression tests exist for most of them.

1. **`DataFrame` divided by a row-wise `Series`.** `df / series` aligns the
   Series against the *columns*, so `zscore` and `scale` returned all-NaN.
   Needs `.div(series, axis=0)`. Same trap hit again later in the bottom
   selection: `DataFrame.gt(series)` silently selected nothing. **Any time a
   per-date Series meets a date x instrument frame, pass `axis=0`.**
2. **Listing age computed after the date window filter.** Counting a stock's
   traded days from `start` relabels every stock as newly listed and silently
   deletes the first months of the study period. Count on the full history, then
   slice.
3. **Beijing exchange codes.** `BJ(?:4|8|9)\d{4}` requires five digits after the
   prefix; BSE codes are six, like `BJ834765`. The whole exchange was being
   rejected.
4. **`decay_curve` default horizons included 1**, where the label is identically
   zero and the IC undefined.
5. **`ts_rank` via `Rolling.rank`** cost 2.6-3.4 s per evaluation. A chunked
   sliding-window comparison is 5x faster at window 20 and 3x at window 60.
6. **`ts_argmax(volume, 250)` cost 8.9 s** while `ts_mean(volume, 250)` cost
   0.15 s. A 60x spread makes a formula budget meaningless, hence the per-op
   window caps.
7. **`ts_median` has no fast implementation** in pandas or numpy (both ~3 s at
   every window). Registered, but excluded from the default sampling pool.
8. **numpy overflow warnings from `ts_product`** are expected and are suppressed
   with `np.errstate`; the results become NaN and are reported as low coverage
   rather than crashing.
9. **README image links pointed at absolute local paths**, so they were broken
   for anyone cloning the repo. Charts live in `docs/images/` now, and
   `.gitignore` has a `!docs/images/*.png` exception because `*.png` is ignored
   globally.
10. **`decay_curve` did not drop the boundary rows.** Every other scoring path
    goes through `split_dates`, which removes the last `horizon` dates of a split
    so a label cannot reach into the next one. `decay_curve` filtered its window
    by `start`/`end` alone, so at horizon 20 its default `end="2020-12-31"`
    scored labels realised in January 2021 - inside the test period. It now uses
    the same tail drop, and `test_decay_curve_agrees_with_the_split_clean_scorer`
    pins it to `score_factor`. Consequence: the P0.2 winner is +0.0646 at 5 days
    and +0.0890 at 20 days, not the +0.0642 / +0.0883 quoted in earlier drafts.

---

## 8. Performance envelope

Full panel, single core, float32:

| Operation | Cost |
| --- | --- |
| Read the HDF5 and widen to date x instrument | 3.6 s |
| One panel copy | 74 MB |
| `ts_mean(volume, 250)` | 0.15 s |
| `ts_rank(volume, 60)` | 0.97 s |
| `ts_argmax(volume, 30)` | ~1.2 s |
| One whole formula: evaluation 0.72 s + scoring 1.31 s | ~2.1 s |
| 2,000 formulas, 5 worker processes | ~18 min |

Scoring, not evaluation, is the bottleneck - four cross-sectional rank passes
per formula. Making Pearson IC opt-in already removed a third of it. The next
win would be ranking once per split instead of once per (split, metric) pair, or
moving the correlation into pure numpy. Nothing in the current results depends
on this.

---

## 9. Reproduce everything

```bash
conda activate alphamine

python -m alphamine.cli check

# P0.2 with its shuffled-label null (~50 min)
python -m alphamine.cli random --n 2000 --depth 4 --workers 5 --null \
  --name p02-random-depth4

# R5-lite, all variants (~90 min)
python -m alphamine.cli ablate --n 1000 --depth 4 --workers 5 --name r5-lite

# just the turnover contrast
python -m alphamine.cli ablate --n 1000 --depth 4 --workers 5 \
  --variants liquid300 illiquid300 --null-variants liquid300 illiquid300 \
  --name r5-lite-turnover

# the turnover decile profile (~2 min)
python -m alphamine.cli profile \
  --run runs/<p02-run-dir> --top 50 --buckets 10
```

Random search is seeded and the proposal order is recorded, so these runs are
deterministic. Each run writes `evals.csv`, `summary.json` and a chart into
`runs/<timestamp>-<name>/`.

---

## 10. Next steps, ranked

1. **P0.1 - transcribe the 35 reproducible 101 Alphas** and tabulate IC, decay
   and turnover. This is the only thing that anchors the project to the
   published literature, and it is mostly transcription rather than
   engineering.
2. **A market-cap or index-membership universe.** `liquid300` is a turnover
   ranking, not CSI 300. Without this, absolute ICs cannot be compared with
   published numbers such as QuantaAlpha's 0.0472 on CSI 300.
3. **P1 - minimal genetic programming** (~200 lines), then the parsimony and
   early-stopping ablation. The bar to beat is +0.0647 at N = 1,000 on the full
   universe and +0.0955 on `liquid300`.
4. **R2 properly** - shuffled-label runs at large N, to separate "the grammar is
   a strong prior" from "the search procedure is smart".
5. **R3 - the equal-budget leaderboard** across random / GP / RL / surrogate /
   LLM at 2,000 evaluations and five seeds. Budget for it: 60,000 evaluations is
   roughly 8 hours at five workers with the current scorer.
6. **P0.2 at 10,000 evaluations** to finish the curve as originally specified.
7. R1 (planted-alpha synthetic market), R4 (MAP-Elites), R6 (decay and
   transfer), R7 (multiple-testing correction - `factor-qc` and
   `Perception-XAlpha Lite` already implement DSR/PBO/White's Reality Check, so
   do not hand-roll them).

---

## 11. Open decisions for the user

* **When to spend the test period.** It has not been spent - though one early
  smoke run did compute it, see section 5. The study plan's
  own checklist says once, at the end; doing it before the R3 leaderboard would
  burn it.
* **Whether to add a market-cap data source.** This is the one step that needs
  data from outside the current dataset.
* **The repository history was rewritten once.** An unrelated directory
  (`秋招行测题库/`, a test-question bank that appeared in the project folder
  during the session) was committed by `git add -A` and pushed to the public
  repo as commit `ac89faf`. The branch was corrected and force-pushed; `main`
  contains only project files. **The old commit is still reachable on GitHub by
  its SHA**, which is how GitHub behaves for unreferenced commits. If that
  content should not be public, the repository needs to be made private or the
  object purged through GitHub Support. `/秋招行测题库/` is listed in
  `.git/info/exclude`, which is local-only, so it will not be re-added.

---

## 12. Traps for the next agent

* **Never run `git add -A` in this directory.** The project folder is shared
  with the user's other work; stage explicit paths.
* **Worker processes use `spawn`**, and each one loads the panel, so budget ~15 s
  of startup per worker per batch. Five workers cost about 7 GB resident; do not
  raise the count without checking memory.
* **`.git` is read-only inside the sandbox** here, so commits and config writes
  need an escalation.
* **Matplotlib needs `MPLCONFIGDIR`** set to a writable directory.
* **`runs/` and `data/` are git-ignored.** Charts meant for the documentation
  must be copied into `docs/images/`.
* **A formula that is only good inside the universe where it was found is not a
  factor.** Check every winner on at least one other universe before reporting
  it.
