# R5-lite - which part of the search space carries the signal?

**Question.** P0.2 showed the random baseline saturates fast and that its best
formulas look like ordinary price/volume quantities. Is the *procedure* doing
anything, or is the *grammar* a strong prior that would pay off under any
procedure? Idea R5 in the study plan asks which ingredients carry the signal.

**Method.** Change one ingredient at a time, hold the budget fixed at 1,000
formulas per variant, same seed, same maximum depth 4, full panel, train
2012-2018, validation 2019-2020. Run `20261001-225430-r5-lite`.

| variant | operators | inputs | best train | best valid | shuffled-label valid |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 37 | 8 | +0.0742 | +0.0647 | +0.0032 |
| `liquid300` | 37 | 8 | +0.0794 | +0.0955 | +0.0112 |
| `no-cross-section` | 32 | 8 | +0.0772 | +0.0561 | - |
| `no-volume` | 37 | 6 | +0.0593 | +0.0647 | - |
| `no-time-series` | 20 | 8 | +0.0844 | +0.0550 | - |

![R5-lite](</Users/wangsheng/Documents/ChatGPT/New project/runs/20261001-225430-r5-lite/r5-lite.png>)

The two universe variants were given their own shuffled-label nulls on purpose.
A 300-name cross-section produces a noisier daily IC than a 3,500-name one, so
its best-of-1000 is inflated by more: +0.0112 against +0.0032. Comparing raw ICs
across universes without those floors would have been wrong. Net of its floor,
`liquid300` still wins (+0.0843 against +0.0615).

## Five findings

**1. The universe is a bigger lever than any grammar change.** The same formulas
on the 300 most traded names score +0.0955 instead of +0.0647. This kills the
hypothesis that P0.2's signal was an artifact of thinly traded small caps - it
is stronger, not weaker, in the liquid segment.

Caveat that matters: this universe is selected by *trailing turnover*, so it is
a high-attention universe, not CSI 300 membership and not a market-cap ranking.
High-turnover A-share names are exactly where short-horizon inefficiency is
usually reported to be largest. A true large-cap universe needs a market-cap
field this dataset does not have, and the bottom-300-by-turnover is the obvious
cheap contrast to run next.

**2. Cross-sectional normalisation is nearly worthless.** Dropping all five
cross-sectional operators (`rank`, `zscore`, `scale`, `demean`, `cs_median`)
costs 13% of the validation IC. The remaining arithmetic and time-series
operators still reach +0.0561 - which is below baseline's value at N = 250. The
usual habit of wrapping everything in `rank()` is doing far less work than
people assume.

**3. Volume inputs are exactly worthless, and the reason is instructive.** The
`no-volume` variant reproduces the baseline's validation IC to four decimals.
Inspect the baseline's winning formula:

```
mul(ts_max(sign(volume), 60), ts_min(div(low, vwap_proxy), 30))
```

`sign(volume)` is identically 1 everywhere volume is positive, so
`ts_max(sign(volume), 60)` is a **no-op constant**. Evaluating the formula with
that term deleted gives +0.0646 against +0.0647 on the full universe and
+0.0955 against +0.0955 on `liquid300`. The mined factor is really
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
* The test period has still not been looked at.

## Next

1. `bottom300` by turnover as a contrast, to separate "liquid" from
   "high-attention" and to price the turnover-selection effect properly.
2. A market-cap or index-membership universe, if one can be sourced, so the
   numbers become comparable with the literature.
3. Only then P1 (genetic programming). The honest bar is now +0.0647 at N = 1,000
   on the full universe, or +0.0955 on `liquid300`, and a method has to beat
   both at the same budget.
