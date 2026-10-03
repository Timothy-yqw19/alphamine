# P0.2 - the random-formula baseline

**Question.** How good is the best of N random formulas, for N = 100, 1,000 and
10,000?

**Why it matters.** This curve is the bar every smarter miner has to beat. If
genetic programming cannot beat the best of 1,000 random formulas under the same
budget, its extra machinery is not buying anything.

## Method

Formulas are sampled by walking the operator signatures, so every tree is well
typed and every window comes from the legal set. Sampling is seeded and the
proposal order is recorded, because the budget curve is a function of that
order.

Scored on the full panel. Train 2012-2018, validation 2019-2020. The test split
is not touched. The budget is counted in formulas, not seconds.

## Results

Run `20261001-224125-p02-random-depth4`: 2,000 distinct formulas, maximum tree
depth 4, full panel, five worker processes.

| N evaluated | best train rank IC | best validation rank IC | shuffled-label train | shuffled-label validation |
| --- | --- | --- | --- | --- |
| 100 | +0.0426 | +0.0411 | +0.0011 | +0.0016 |
| 250 | +0.0742 | +0.0618 | +0.0029 | +0.0035 |
| 1,000 | +0.0742 | +0.0647 | +0.0029 | +0.0035 |
| 2,000 | +0.0742 | +0.0647 | +0.0029 | +0.0073 |

Across all 1,852 non-degenerate formulas: median validation rank IC -0.0051,
10th percentile -0.0396, 90th percentile +0.0184. 7.4% of sampled trees were
degenerate.

The best formula at N = 2,000 was

```
mul(ts_max(sign(volume), 60), ts_min(div(low, vwap_proxy), 30))
```

## Three findings

**1. The curve saturates almost immediately.** Best validation IC is already
+0.0411 after 100 formulas - 63% of the final value - and the remaining 1,900
evaluations buy +0.024. Any leaderboard that compares methods at a single large
N will find the differences small and possibly reversible; the interesting
region is N = 100..1,000.

**2. The signal is real, not an artefact.** The shuffled-label run, on the same
2,000 formulas, peaks at +0.0073 on validation. The real run is roughly 9x that,
and a separate control of 200 pure random-number factors peaks at +0.0022. Two
obvious A-share artefacts were checked and rejected: excluding names that were
limit-up at the entry day `t+1` changes the top formulas' IC by less than 0.003,
and restricting to the more liquid half of the universe *raises* the top IC
(+0.0642 to +0.0721).

**3. The grammar is doing the work.** The best formulas are dominated by
negation, inversion and range-position operators - `neg`, `inv`, `sub`,
`ts_argmin`, `ts_min` - and the clearest of them are textbook short-horizon
reversal: `inv(sqrt(cs_median(returns)))` is a cross-sectionally demeaned
one-day return, inverted, scoring +0.0557. `mulconst(ts_cov(abs(volume),
vwap_proxy, 5), -0.1)` is a five-day volume/price covariance with a negative
sign, scoring +0.0609.

> **Corrected by R5-lite.** Reading the formulas is not enough. Measuring the
> winning signal's decay and turnover shows it is a slow low-volatility
> characteristic (+0.0642 at 5 days, +0.0883 at 20 days, 3.3% daily turnover,
> -0.76 correlation with realised volatility, -0.004 with the 3-day return), not
> reversal. See `docs/r5-lite.md`.

That is the point of this baseline. A search procedure is not competing against
random noise; it is competing against a grammar whose *random* draws already
land on real A-share structure. It also means idea R5, search-space ablations,
deserves a higher priority than the study plan originally gave it.

## Caveats

* The universe is all A-shares, averaging 3,536 names a day. Published ICs
  (QuantaAlpha 0.0472, for instance) are quoted on CSI 300 large caps, which are
  far more efficiently priced. **The two are not comparable**, and this gap
  should be closed before any claim of "beating the literature".
* 2019-2020 validation was a strong reversal regime. The test period has not
  been looked at.
* No transaction costs, no market impact, no capacity limit.
* These are selected maxima. The luck floors above are the only guard against
  reading selection as skill.

## What to look at

* **Best train IC rises monotonically with N.** It always does, including on
  meaningless formulas. That part of the curve is not information.
* **Best validation IC is the interesting line.** Where it flattens is where
  extra random draws stop buying transferable signal.
* **The gap between the two is the overfitting signature.** For random search it
  should stay small, because random search never uses a training result to pick
  its next candidate.
* **The invalid rate.** Some random trees are degenerate - all-NaN after
  overflow in `ts_product`, or a signal covering almost no names. They still
  consume budget, so they are counted rather than hidden.
* **The null line.** The same formulas scored against shuffled labels. If the
  real curve sits well above it, the signal is in the price/volume relationship
  and not in the evaluation.
