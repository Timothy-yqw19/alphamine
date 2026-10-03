"""The 35 Formulaic Alphas this dataset can express.

The article is Kakushadze, *101 Formulaic Alphas* (arXiv:1601.00991).  Of the
101, 35 need nothing beyond the fields this panel carries; the other 66 need
`vwap`, an industry classification, or are not implemented in the reference
source at all.  ``scripts/count_101_alphas.py`` re-derives that split.

Source of the formulas: the two implementations in
``yli188/WorldQuant_alpha101_code``, one pandas-style and one Quantopian-style.
Each entry below quotes the original notation and gives the equivalent
expression in this repository's language.

Transcription rules
-------------------
* ``sum/stddev/sma``      -> ``ts_sum`` / ``ts_std`` / ``ts_mean``
* ``correlation/covariance`` -> ``ts_corr`` / ``ts_cov``
* ``product/min/max``     -> ``ts_product`` / ``ts_min`` / ``ts_max``
* ``Ts_Rank/Ts_ArgMax``   -> ``ts_rank`` / ``ts_argmax``
* ``(cond ? a : b)``      -> ``where(cond, a, b)``, with ``lt``/``gt``/``ge``
* ``x^2`` / ``x^5``       -> ``square(x)`` / ``mul(square(square(x)), x)``
* ``SignedPower(x, e)``   -> ``mul(sign(x), <power of x>)``
* ``x / const``           -> ``mulconst(x, 1/const)``, to avoid dividing by a
  scalar through ``div``

Where the two reference implementations disagree, this file follows the
executable one and says so at the entry.  ``alpha029`` is the clearest case: the
published notation and the code do not describe the same expression, and the
code is what the "35 reproducible" count was measured against.

Comparisons and ``where`` are registered in ``alphamine/expr/ops.py`` but kept
out of the default sampling pool, so transcribing these formulas does not change
the random search space or any published ablation number.
"""

from __future__ import annotations

#: alpha number -> expression in this repository's language.
ALPHAS_101: dict[int, str] = {
    1: (
        # (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        "addconst(rank(ts_argmax(mul(sign(where(lt(returns, 0), ts_std(returns, 20), close)), "
        "square(where(lt(returns, 0), ts_std(returns, 20), close))), 5)), -0.5)"
    ),
    2: (
        # (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        "neg(ts_corr(rank(delta(log(volume), 2)), rank(div(sub(close, open), open)), 6))"
    ),
    3: (
        # (-1 * correlation(rank(open), rank(volume), 10))
        "neg(ts_corr(rank(open), rank(volume), 10))"
    ),
    6: (
        # (-1 * correlation(open, volume, 10))
        "neg(ts_corr(open, volume, 10))"
    ),
    7: (
        # ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
        "where(lt(adv20, volume), neg(mul(ts_rank(abs(delta(close, 7)), 60), "
        "sign(delta(close, 7)))), -1)"
    ),
    8: (
        # (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
        "neg(rank(sub(mul(ts_sum(open, 5), ts_sum(returns, 5)), "
        "delay(mul(ts_sum(open, 5), ts_sum(returns, 5)), 10))))"
    ),
    9: (
        # ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0)
        #   ? delta(close, 1) : (-1 * delta(close, 1))))
        "where(gt(ts_min(delta(close, 1), 5), 0), delta(close, 1), "
        "where(lt(ts_max(delta(close, 1), 5), 0), delta(close, 1), neg(delta(close, 1))))"
    ),
    12: (
        # (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
        "mul(sign(delta(volume, 1)), neg(delta(close, 1)))"
    ),
    13: (
        # (-1 * rank(covariance(rank(close), rank(volume), 5)))
        "neg(rank(ts_cov(rank(close), rank(volume), 5)))"
    ),
    14: (
        # ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
        "mul(neg(rank(delta(returns, 3))), ts_corr(open, volume, 10))"
    ),
    15: (
        # (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
        "neg(ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3))"
    ),
    16: (
        # (-1 * rank(covariance(rank(high), rank(volume), 5)))
        "neg(rank(ts_cov(rank(high), rank(volume), 5)))"
    ),
    17: (
        # (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
        "mul(mul(neg(rank(ts_rank(close, 10))), rank(delta(delta(close, 1), 1))), "
        "rank(ts_rank(div(volume, adv20), 5)))"
    ),
    18: (
        # (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
        "neg(rank(add(add(ts_std(abs(sub(close, open)), 5), sub(close, open)), "
        "ts_corr(close, open, 10))))"
    ),
    19: (
        # ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
        "mul(neg(sign(add(sub(close, delay(close, 7)), delta(close, 7)))), "
        "addconst(rank(addconst(ts_sum(returns, 250), 1)), 1))"
    ),
    20: (
        # (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
        "mul(mul(neg(rank(sub(open, delay(high, 1)))), rank(sub(open, delay(close, 1)))), "
        "rank(sub(open, delay(low, 1))))"
    ),
    21: (
        # ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1)
        #  : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1
        #  : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1))))
        "where(lt(add(mulconst(ts_sum(close, 8), 0.125), ts_std(close, 8)), "
        "mulconst(ts_sum(close, 2), 0.5)), -1, "
        "where(lt(mulconst(ts_sum(close, 2), 0.5), sub(mulconst(ts_sum(close, 8), 0.125), "
        "ts_std(close, 8))), 1, where(ge(div(volume, adv20), 1), 1, -1)))"
    ),
    22: (
        # (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
        "neg(mul(delta(ts_corr(high, volume, 5), 5), rank(ts_std(close, 20))))"
    ),
    23: (
        # (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
        "where(lt(mulconst(ts_sum(high, 20), 0.05), high), neg(delta(high, 2)), 0)"
    ),
    28: (
        # scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
        "scale(sub(add(ts_corr(adv20, low, 5), mulconst(add(high, low), 0.5)), close))"
    ),
    29: (
        # The published notation and the reference code disagree.  The code is:
        # ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((close - 1), 5)))), 2))))), 5)
        #   + ts_rank(delay((-1 * returns), 6), 5)
        # The trailing `min(..., 5)` of the notation is the ts_min here, and the
        # `product(..., 1)` is a no-op that the code drops.
        "add(ts_min(rank(rank(scale(log(ts_sum(rank(rank(neg(rank(delta(addconst(close, -1), "
        "5))))), 2))))), 5), ts_rank(delay(neg(returns), 6), 5))"
    ),
    30: (
        # (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))))
        #   + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
        "div(mul(addconst(neg(rank(add(add(sign(sub(close, delay(close, 1))), "
        "sign(sub(delay(close, 1), delay(close, 2)))), sign(sub(delay(close, 2), "
        "delay(close, 3)))))), 1), ts_sum(volume, 5)), ts_sum(volume, 20))"
    ),
    33: (
        # rank((-1 * ((1 - (open / close))^1)))
        "rank(addconst(div(open, close), -1))"
    ),
    34: (
        # rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
        "rank(add(addconst(neg(rank(div(ts_std(returns, 2), ts_std(returns, 5)))), 1), "
        "addconst(neg(rank(delta(close, 1))), 1)))"
    ),
    37: (
        # (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
        "add(rank(ts_corr(delay(sub(open, close), 1), close, 200)), rank(sub(open, close)))"
    ),
    38: (
        # ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
        "mul(neg(rank(ts_rank(close, 10))), rank(div(close, open)))"
    ),
    39: (
        # ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9))))))
        #   * (1 + rank(sum(returns, 250))))
        "mul(neg(rank(mul(delta(close, 7), addconst(neg(rank(decay_linear(div(volume, adv20), "
        "9))), 1)))), addconst(rank(ts_sum(returns, 250)), 1))"
    ),
    43: (
        # (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
        "mul(ts_rank(div(volume, adv20), 20), ts_rank(neg(delta(close, 7)), 8))"
    ),
    44: (
        # (-1 * correlation(high, rank(volume), 5))
        "neg(ts_corr(high, rank(volume), 5))"
    ),
    45: (
        # (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2))
        #   * rank(correlation(sum(close, 5), sum(close, 20), 2))))
        "neg(mul(mul(rank(mulconst(ts_sum(delay(close, 5), 20), 0.05)), "
        "ts_corr(close, volume, 2)), rank(ts_corr(ts_sum(close, 5), ts_sum(close, 20), 2))))"
    ),
    51: (
        # (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05))
        #   ? 1 : ((-1 * 1) * (close - delay(close, 1))))
        "where(lt(sub(mulconst(sub(delay(close, 20), delay(close, 10)), 0.1), "
        "mulconst(sub(delay(close, 10), close), 0.1)), -0.05), 1, neg(delta(close, 1)))"
    ),
    52: (
        # ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220)))
        #   * ts_rank(volume, 5))
        "mul(mul(add(neg(ts_min(low, 5)), delay(ts_min(low, 5), 5)), "
        "rank(mulconst(sub(ts_sum(returns, 240), ts_sum(returns, 20)), 0.00454545454545455))), "
        "ts_rank(volume, 5))"
    ),
    53: (
        # (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
        # The reference implementation substitutes 1e-4 for a zero denominator;
        # here div yields NaN there, which the scorer counts as missing.
        "neg(delta(div(sub(sub(close, low), sub(high, close)), sub(close, low)), 9))"
    ),
    54: (
        # ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
        "div(neg(mul(sub(low, close), mul(square(square(open)), open))), "
        "mul(sub(low, high), mul(square(square(close)), close)))"
    ),
    101: (
        # ((close - open) / ((high - low) + .001))
        "div(sub(close, open), addconst(sub(high, low), 0.001))"
    ),
}


def formulas() -> list[tuple[int, str]]:
    """The alphas as ``(number, expression)`` pairs, ordered by number."""

    return sorted(ALPHAS_101.items())
