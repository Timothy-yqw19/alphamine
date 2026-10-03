"""alphamine - a harness for studying how alpha-mining systems search.

The whole project rests on one rule: **hold the evaluation budget fixed**.
A "method" is anything that proposes formulaic alphas; it is scored by the best
validation rank IC it reaches after the same number of formulas have been
evaluated.  Everything else in this package exists to make that comparison fair,
cheap and reproducible.

See ``README.md`` for the framing and ``docs/`` for experiment write-ups.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
