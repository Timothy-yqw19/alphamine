# R5-lite - which part of the search space carries the signal?

**Question.** P0.2 showed the random baseline saturates fast and that its best
formulas look like ordinary price/volume quantities. Is the *procedure* doing
anything, or is the *grammar* a strong prior that would pay off under any
procedure? Idea R5 in the study plan asks which ingredients carry the signal.

**Method.** Change one ingredient at a time, hold the budget fixed at 1,000
formulas per variant, same seed, same maximum depth 4, train 2012-2018,
validation 2019-2020. Runs `20261001-225430-r5-lite`,
`20261002-202029-r5-lite-turnover` and `20261002-211321-r5-lite-csi300`.

| variant | operators | inputs | best train | best valid | shuffled-label valid |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 37 | 8 | +0.0742 | +0.0647 | +0.0032 |
| `liquid300` | 37 | 8 | +0.0794 | +0.0955 | +0.0112 |
| `illiquid300` | 37 | 8 | +0.0605 | +0.0757 | +0.0354 |
| `csi300` | 37 | 8 | +0.0460 | +0.0620 | +0.0093 |
| `no-cross-section` | 32 | 8 | +0.0772 | +0.0561 | - |
| `no-volume` | 37 | 6 | +0.0593 | +0.0647 | - |
| `no-time-series` | 20 | 8 | +0.0844 | +0.0550 | - |

![R5-lite](images/r5-lite.png)

The two universe variants were given their own shuffled-label nulls on purpose.
A 300-name cross-section produces a noisier daily IC than a 3,500-name one, so
its best-of-1000 is inflated by more: +0.0112 against +0.0032. Comparing raw ICs
across universes without those floors would have been wrong. Net of its floor,
`liquid300` still wins (+0.0843 against +0.0615).

**But `csi300` is the universe that matters, and it does not agree.** Run
`20261002-211321-r5-lite-csi300` scores the same 1,000 formulas (verified
identical in order) against true point-in-time CSI 300 membership, and reaches
only **+0.0620** on validation with a floor of +0.0093 - below `liquid300`, and
level with the whole A-share panel's +0.0647. The large `liquid300` edge is
therefore a property of *ranking by turnover* rather than of index membership.
The honest summary of the universe finding is narrower than it first looked: the
liquid *segment* ranks better, the *index* does not. `csi300` is the row to quote
against published CSI 300 work.

## The turnover contrast

`liquid300` beating the whole market is only interesting if the other end of the
market does worse. Run `20261002-202029-r5-lite-turnover` adds `illiquid300` -
the 300 *least* traded names each day - with its own null.

| variant | best train | best valid | shuffled-label valid | net of the null | liquidity, 2020 |
| --- | --- | --- | --- | --- | --- |
| `baseline` (all A-shares) | +0.0742 | +0.0647 | +0.0032 | +0.0615 | 0.0213 |
| `liquid300` (most traded) | +0.0794 | +0.0955 | +0.0112 | +0.0843 | 0.1238 |
| `illiquid300` (least traded) | +0.0605 | +0.0757 | **+0.0354** | +0.0403 | 0.0013 |

Liquidity is the 2020 mean of the 20-day average of `close * volume` in the
data's own units: the liquid universe is 5.8x the whole market, the illiquid one
is 1/16 of it.

**The illiquid universe is a noise trap.** Its best-of-1000 on *shuffled* labels
is +0.0354, three times the liquid universe's +0.0112, because a thinly traded
cross-section gives a far noisier daily IC. Read raw, `illiquid300` at +0.0757
looks like the second-best universe in the table; net of its own luck floor it is
the worst of the three. Any study quoting mined ICs on a thin universe without a
matched null will overstate them.

The winning formula makes the same point from the other side:

```
ts_skew(max2(sign(low), ts_argmin(low, 15)), 20)
```

It is finite on only 49% of the illiquid universe (147 names a day), and it does
not survive leaving that universe at all: **+0.0757** on `illiquid300`, `NaN` on
`liquid300` (96 names a day, under the 100-name floor) and **-0.0171** on the full
panel. A formula that is only good where it was found is not a factor.

### A fixed-formula profile, with no search at all

Comparing two searched universes still mixes the signal level with each
universe's selection floor. The profile removes that: take the 50 formulas with
the best *training* IC from the P0.2 run, then score each one inside every
turnover decile on the *validation* period. No search happens per bucket, so no
bucket's IC is a selected maximum.

![Turnover profile](images/turnover-profile.png)

| decile (1 = most traded) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean rank IC | +0.040 | +0.044 | +0.038 | +0.038 | +0.034 | +0.033 | +0.032 | +0.026 | +0.023 | +0.016 |
| standard error | 0.004 | 0.004 | 0.003 | 0.003 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 | 0.002 |

The gradient is monotone from decile 2 down and the two ends differ by +0.024
against a combined standard error of 0.005. Same 50 formulas, same dates, only
the cross-section changes.

The single best formula from P0.2 says the same thing on its own, at ~97%
coverage in every universe:

| `ts_min(low / vwap_proxy, 30)` | validation rank IC |
| --- | --- |
| `liquid300` (289 names/day) | +0.0955 |
| full panel (3,468 names/day) | +0.0647 |
| `illiquid300` (295 names/day) | +0.0298 |

### What this settles, and what it does not

The P0.2 signal is **not** a small-cap illiquidity artifact. The opposite: it is
about two and a half times stronger among the most traded names than among the
least traded ones, on both a fixed-formula profile and an equal-budget search,
and the winning formula is recognisably the same signal in all three universes.

It does not settle what "most traded" stands for. Turnover is attention, short
holding periods, retail participation and volatility at the same time, and this
dataset cannot separate them. It also cannot rank by market capitalisation, so a
true CSI 300 comparison is still out of reach.

## Five findings

**1. The universe is a bigger lever than any grammar change.** The same formulas
on the 300 most traded names score +0.0955 instead of +0.0647. This kills the
hypothesis that P0.2's signal was an artifact of thinly traded small caps - it
is stronger, not weaker, in the liquid segment.

Caveat that matters: this universe is selected by *trailing turnover*, so it is
a high-attention universe, not CSI 300 membership and not a market-cap ranking.
The `illiquid300` contrast below was run for exactly this reason and is what
turns the observation into a result.

**2. Cross-sectional normalisation is nearly worthless.** Dropping all five
cross-sectional operators (`rank`, `zscore`, `scale`, `demean`, `cs_median`)
costs 13% of the validation IC. The remaining arithmetic and time-series
operators still reach +0.0561 - which is below baseline's value at N = 250. The
usual habit of wrapping everything in `rank()` is doing far less work than
people assume.

**3. Volume inputs are nearly worthless, and the reason is instructive.** The
`no-volume` variant reproduces the baseline's validation IC to four decimals.
Inspect the baseline's winning formula:

```
mul(ts_max(sign(volume), 60), ts_min(div(low, vwap_proxy), 30))
```

`sign(volume)` is 1 wherever volume is finite and `NaN` where it is not, so
`ts_max(sign(volume), 60)` is a **60-day "has volume been observed" gate**. It is
value-neutral rather than inert: deleting it gives +0.0646 against +0.0647 on the
full universe (a change of 0.00004) and +0.0955 against +0.0955 on `liquid300`,
but coverage drops from 0.986 to 0.975. The mined factor is really
`ts_min(low / ((high + low + close) / 3), 30)` - pure price, no volume at all.
Any pipeline that reports mined formulas without simplifying them is reporting
decoration as structure.

**4. The two grammars found two different signals.** With time-series operators
removed, the search can only build pointwise combinations, and it landed on:

```
mul(sqrt(addconst(close, 1)), sub(returns, add(returns, volume)))   ->  +0.0550
mul(sqrt(addconst(close, 1)), neg(volume))                          ->  +0.0548
```

Those are the same formula once simplified: `sqrt(close) * (-volume)`, a
size/liquidity proxy. The full grammar never found it, because with temporal
structure available the price-stability signal pays better.

**5. Simpler grammars overfit more.** The train-minus-validation gap is +0.0095
for `baseline`, +0.0211 for `no-cross-section` and +0.0294 for
`no-time-series`. Removing structural operators raises the in-sample maximum
while lowering the out-of-sample one - the classic signature of a search that
has more freedom to fit noise per unit of budget.

## What the winning signal actually is

Not reversal, which is what the P0.2 write-up guessed. On the validation period
the simplified signal `ts_min(low / vwap_proxy, 30)`:

| property | value |
| --- | --- |
| rank IC at 5-day horizon | +0.0642 |
| rank IC at 20-day horizon | +0.0883 (rising, not decaying) |
| one-day turnover of the rank-weighted book | 0.033 |
| correlation with 20-day realised volatility | **-0.763** |
| correlation with (high - low) / close | -0.496 |
| correlation with the 3-day return | -0.004 |
| correlation with the price level | +0.049 |

The first two rows are pre-fix `decay_curve` numbers: it filtered its window by
date alone and did not drop the last `horizon` rows. That has been fixed, and it
now agrees with `score_factor`, which gives +0.0646 at 5 days and +0.0890 at 20
days. The rows above are kept as the record of what was measured at the time.

It is a **low-volatility / price-stability characteristic**, essentially
orthogonal to return reversal, with very low turnover, whose predictive power
*grows* with horizon. Rising IC and 3% daily turnover are what a slow
cross-sectional characteristic looks like, not what a fast reversal signal looks
like.

This also corrects P0.2. That write-up pointed at `inv(sqrt(cs_median(returns)))`
(+0.0557, genuinely reversal-like) and generalised from it. Both signals exist
in the grammar; the single best draw at depth 4 is the stability one, and the
characterisation should come from the decay and turnover measurements rather
than from reading the formula.

## Caveats

* `liquid300` is a turnover ranking, not CSI 300. Absolute ICs here are still not
  comparable with published CSI 300 numbers.
* No transaction costs. The stability signal turns over slowly (3.3%/day), which
  helps, but nothing here has been costed.
* 2019-2020 validation was a favourable regime: `liquid300` and `no-volume` both
  score *higher* out of sample than in sample, which is a regime statement, not
  evidence of robustness.
* All of these are selected maxima. The luck floors are the only guard.
* The test period has still not been spent, but one early smoke run did compute
  it - see `docs/HANDOFF.md` section 5.

## Next

1. Separate "liquid" from "high-attention". The `bottom300` contrast this used
   to ask for is done - it is the `illiquid300` variant above - but turnover still
   bundles attention, holding period, retail participation and volatility.
2. A market-cap or index-membership universe, if one can be sourced, so the
   numbers become comparable with the literature.
3. Only then P1 (genetic programming). The honest bar is now +0.0647 at N = 1,000
   on the full universe, or +0.0955 on `liquid300`, and a method has to beat
   both at the same budget.
