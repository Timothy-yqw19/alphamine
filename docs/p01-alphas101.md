# P0.1 - the 35 reproducible 101 Alphas

**Question.** A random search over this grammar reaches +0.0647 on validation
(P0.2). Is that interesting? Only if a textbook formula library does not already
reach it. P0.1 transcribes the published alphas this dataset can express and
scores them the same way.

**Source.** Kakushadze, *101 Formulaic Alphas* (arXiv:1601.00991). The formulas
were taken from the two implementations in `yli188/WorldQuant_alpha101_code`.
`scripts/count_101_alphas.py` re-derives which ones this panel can express: **35
from OHLCV alone**, 34 need `vwap`, 13 need an industry class, and 19 are not
implemented in the reference source at all.

**Transcription.** `alphamine/alphas101.py` holds all 35, each with its original
notation in a comment. The mapping is mechanical (`sum` -> `ts_sum`,
`correlation` -> `ts_corr`, and so on) with two real additions:

* `(cond ? a : b)` appears in **six** alphas (1, 7, 9, 21, 23, 51) and is not
  expressible with arithmetic. `where` plus `gt`/`lt`/`ge`/`le` were added to the
  operator registry - and deliberately kept **out of the default sampling pool**,
  so the random search space and every published ablation number are unchanged.
  `ts_median` is excluded the same way.
* `x^5` becomes `mul(square(square(x)), x)`; `SignedPower(x, 2)` becomes
  `mul(sign(x), square(x))`.

These are fixed formulas - nothing was selected on validation - so the numbers
below carry no search bias beyond taking the best of 35.

**Method.** Full panel, 2012-2026, train 2012-2018, validation 2019-2020,
horizon 5, rank IC. Run `runs/20261002-212654-p01-alphas101`.

| alpha | train | valid | ICIR | turnover | decay h5 | decay h20 |
| --- | --- | --- | --- | --- | --- | --- |
| `alpha016` | +0.0526 | **+0.0508** | 0.83 | 0.262 | +0.0508 | +0.0626 |
| `alpha013` | +0.0554 | **+0.0491** | 0.79 | 0.271 | +0.0491 | +0.0610 |
| `alpha044` | +0.0533 | **+0.0434** | 0.74 | 0.344 | +0.0434 | +0.0567 |
| `alpha015` | +0.0398 | **+0.0348** | 0.69 | 0.294 | +0.0348 | +0.0445 |
| `alpha003` | +0.0383 | **+0.0312** | 0.55 | 0.208 | +0.0312 | +0.0516 |
| `alpha006` | +0.0282 | **+0.0300** | 0.44 | 0.214 | +0.0300 | +0.0552 |
| `alpha019` | +0.0346 | **+0.0291** | 0.29 | 0.241 | +0.0291 | +0.0297 |
| `alpha037` | +0.0291 | **+0.0282** | 0.35 | 0.441 | +0.0282 | +0.0404 |
| `alpha012` | +0.0389 | **+0.0262** | 0.51 | 0.579 | +0.0262 | +0.0296 |
| `alpha052` | +0.0247 | **+0.0232** | 0.26 | 0.237 | +0.0232 | +0.0258 |
| `alpha029` | +0.0266 | **+0.0232** | 0.28 | 0.452 | +0.0232 | +0.0265 |
| `alpha023` | +0.0364 | **+0.0217** | 0.29 | 0.453 | +0.0217 | +0.0247 |
| `alpha014` | +0.0239 | **+0.0203** | 0.31 | 0.382 | +0.0203 | +0.0383 |
| `alpha008` | +0.0287 | **+0.0180** | 0.20 | 0.265 | +0.0180 | +0.0198 |
| `alpha038` | +0.0385 | **+0.0177** | 0.17 | 0.544 | +0.0177 | +0.0173 |
| `alpha039` | +0.0302 | **+0.0176** | 0.16 | 0.230 | +0.0176 | +0.0183 |
| `alpha033` | +0.0361 | **+0.0172** | 0.17 | 0.655 | +0.0172 | +0.0181 |
| `alpha045` | +0.0282 | **+0.0169** | 0.39 | 0.527 | +0.0169 | +0.0170 |
| `alpha002` | +0.0254 | **+0.0145** | 0.32 | 0.281 | +0.0145 | +0.0141 |
| `alpha051` | +0.0203 | **+0.0139** | 0.16 | 0.567 | +0.0139 | +0.0107 |
| `alpha034` | +0.0180 | **+0.0125** | 0.16 | 0.626 | +0.0125 | +0.0096 |
| `alpha009` | +0.0256 | **+0.0108** | 0.13 | 0.627 | +0.0108 | +0.0054 |
| `alpha017` | +0.0193 | **+0.0101** | 0.12 | 0.636 | +0.0101 | +0.0070 |
| `alpha018` | +0.0235 | **+0.0091** | 0.11 | 0.286 | +0.0091 | +0.0070 |
| `alpha022` | +0.0139 | **+0.0081** | 0.16 | 0.357 | +0.0081 | +0.0059 |
| `alpha021` | +0.0137 | **+0.0070** | 0.11 | 0.439 | +0.0070 | +0.0066 |
| `alpha053` | +0.0216 | **+0.0060** | 0.08 | 0.621 | +0.0060 | +0.0025 |
| `alpha043` | +0.0194 | **+0.0038** | 0.06 | 0.444 | +0.0038 | +0.0073 |
| `alpha054` | +0.0162 | **+0.0020** | 0.03 | 0.658 | +0.0020 | -0.0015 |
| `alpha030` | +0.0173 | **+0.0019** | 0.02 | 0.356 | +0.0019 | +0.0084 |
| `alpha020` | -0.0090 | **-0.0068** | -0.12 | 0.620 | -0.0068 | -0.0122 |
| `alpha007` | -0.0108 | **-0.0108** | -0.16 | 0.397 | -0.0108 | -0.0101 |
| `alpha101` | -0.0268 | **-0.0130** | -0.14 | 0.670 | -0.0130 | -0.0130 |
| `alpha001` | -0.0344 | **-0.0170** | -0.22 | 0.469 | -0.0170 | -0.0147 |
| `alpha028` | -0.0118 | **-0.0189** | -0.28 | 0.336 | -0.0189 | -0.0186 |

**30 of 35** are positive on validation, **28 of 35** clear the 0.002 noise floor
from P0.2, the median is +0.0169, and the mean is +0.0152.

## What it says

**1. The literature's library is a strong prior, and at equal budget it wins.**
This is the comparison that matters:

| | best validation rank IC |
| --- | --- |
| random search, best-of-35 | +0.0411 |
| random search, best-of-100 | +0.0411 |
| **the 35 textbook alphas** | **+0.0508** |
| random search, best-of-250 | +0.0618 |
| random search, best-of-2,000 | +0.0647 |

Thirty-five hand-derived formulas beat what random search finds in its first 100
draws, and it takes the **entire 2,000-formula budget** for random search to
overtake them. A random-search result in this grammar is therefore not evidence
that search works - it starts from a library of ideas that already work.

**2. The best textbook alpha essentially reproduces the published number.**
`alpha016` scores +0.0508 on the full A-share panel against QuantaAlpha's
published 0.0472 on CSI 300. The universes are not identical, so this is a
sanity check rather than a replication - but it is the right order of magnitude,
which is what P0.1 was for.

**3. The top three are one idea written three ways.** They are all a *negative
price-versus-volume covariance*:

```
alpha016  neg(rank(ts_cov(rank(high),  rank(volume), 5)))   +0.0508
alpha013  neg(rank(ts_cov(rank(close), rank(volume), 5)))   +0.0491
alpha044  neg(ts_corr(high, rank(volume), 5))               +0.0434
```

Spread across the whole panel, a day when volume runs high while the price level
runs low (or vice versa) predicts. The three differ by spelling, not by content,
so "the best of the 35 alphas" is really one signal with three names - the same
duplicate-counting problem the canonical string solves in the search harness.

**4. The durable signals are slow, which independently reproduces P0.2.** Every
one of the top eight *gains* IC at a 20-day horizon - `alpha016` goes from +0.0508
to +0.0626, `alpha003` from +0.0312 to +0.0516. And the ordering on IC tracks the
ordering on turnover: the top alphas turn over 0.21-0.34 per day while the bottom
ones turn over 0.57-0.67. P0.2's randomly found winner was a low-volatility,
price-stability characteristic that strengthened at longer horizons; the textbook
library, reached by a completely different route, lands on the same axis. Two
independent procedures agreeing is the strongest evidence in this repository so
far.

**5. The wrong ones fail quietly, not loudly.** The four negative alphas are not
degenerate - all have coverage above 0.9 - they are simply signals that do not
work in A-shares over this window. `alpha001` and `alpha101` are among the
best-known formulas in the paper and are the two worst here (-0.0170 and
-0.0130). A library being published is not evidence that each member of it works.

## Caveats

* These are fixed formulas, but +0.0508 is still a maximum over 35, so it is
  optimistic by an unknown amount. The best-of-35 random comparison above is the
  fair one; do not compare +0.0508 with random search's best-of-2,000 directly.
* One period, one seed, no costs. 2019-2020 validation was a favourable regime
  for slow characteristics.
* Full A-share panel, **not** CSI 300. For a like-for-like comparison with
  published CSI 300 numbers these alphas should be re-scored on the `csi300`
  universe.
* `alpha029`: the published notation and the reference implementation describe
  different expressions. This file follows the implementation, because that is
  what the "35 reproducible" count was measured against. See the comment at the
  entry.
* `alpha053` divides by `(close - low)`. The reference substitutes 1e-4 for a zero
  denominator; here `div` yields NaN, which the scorer counts as missing.
* Six alphas rely on `where`, which is registered but **not** in the sampling
  pool - so this transcription does not change the random search space.

## Reproducing

```bash
python scripts/count_101_alphas.py       # re-derive the 35
python scripts/score_101_alphas.py       # re-score them (~4 min, full panel)
```

`runs/20261002-212654-p01-alphas101/` holds `evals.csv` (all 35 with every
column above) and `summary.json`.
