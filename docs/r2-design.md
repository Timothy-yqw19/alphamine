# R2 - is it the grammar, or the procedure?

**Status: designed, not yet run.**  It shares its real-label arm with R3, which is
in progress, so it must wait for that to finish (both need five workers).

## The question

Every method here searches the same expression grammar, and several now end up at
roughly the same place. When a method finds a formula worth +0.05 on validation,
what produced it?

* **the grammar** - random draws from this operator set already land on real
  A-share structure, so every procedure inherits it.  P0.2 concluded this and P1
  found no counter-evidence: GP's best formula came from its own random initial
  population and 1,800 further evaluations never beat it.
* **the procedure** - selection and variation navigate better than random
  sampling.  Nothing in the repository has shown this yet.

## The trap this must avoid

R5-lite established that a 300-name cross-section has a higher selection floor
than a 5,000-name one (+0.0112 against +0.0032 on shuffled labels), so **ICs must
never be compared across universes without a null per universe.**

The same argument applies to *procedures*, and it is easy to miss.  A method that
optimises harder will climb higher on shuffled labels too, because shuffled labels
are still something to optimise.  **Each method therefore needs its own floor.**
Comparing a method's net against the other method's floor is exactly the mistake
R5-lite warned about, one level up.

## Design

A 2x2, paired by seed:

| | real labels | shuffled labels |
| --- | --- | --- |
| `random` | R3 arm A | **R2 arm B** |
| `gp` | R3 arm A | **R2 arm B** |

* Same grammar, same budget (2,000 evaluations), same depth, same five seeds.
* `Config.shuffle_labels=True` permutes the forward returns **within each date**,
  destroying every real relation while preserving the cross-sectional shape and
  the missing-data pattern.
* The real-label arm is the R3 run, so R2 adds one arm rather than repeating two.
* **Metric: `selected`** - pick on training IC, report that formula's validation
  IC.  On shuffled labels the training objective is pure noise, so best-of-N would
  measure how much noise the *evaluation* saw rather than what the method did.
* **Statistic:** `net = selected(real) - selected(shuffled)`, computed per method
  per seed, then the paired difference of nets.  Pairing by seed is meaningful
  because `random(seed s)` and `gp(seed s)` share their initial population exactly.

## Pre-registered readings

This is the point of the design: each outcome has a stated meaning, decided before
the numbers arrive.

| outcome | reading |
| --- | --- |
| GP floor **>** random floor | GP *is* a working optimiser - it climbs its training objective - and on real labels the ceiling is set by the grammar. The procedure is real; the landscape is flat. |
| GP floor **=** random floor | GP is not climbing even pure noise. Selection does nothing at all here. |
| GP floor **<** random floor | GP's selection is worse than sampling: it concentrates on training noise that does not transfer. |
| **net(GP) > net(random)** | The procedure adds value beyond the grammar. **This is the only outcome that would justify GP over random**, and it is the one P1's single seed did not produce. |

A convenient consequence: the third case is not a failure of the implementation,
it is a *result about the landscape*, and the design says so in advance.

## Large-N arm

R2's original wording asks for "large N".  The floor is expected to rise with N
under best-of-N - which is precisely why the honest metric matters - but whether it
rises under `selected` is open.  One arm at N = 10,000, single method and single
seed, answers it far more cheaply than the full grid.

## Cost

| arm | evaluations | wall clock at five workers |
| --- | --- | --- |
| shuffled labels, 2 methods x 5 seeds x 2,000 | 20,000 | ~3.25 h |
| the same for large N = 10,000, 1 method x 1 seed | 10,000 | ~1.6 h |
| real-label arm | 20,000 | already running as R3 |

## Implementation

No new search code is needed.  `alphamine/leaderboard.py` already threads a
`Config` through both `evaluate_batch` and `run_gp`, and both honour
`Config.shuffle_labels`.  R2 needs a **label axis** in `run_leaderboard`:

```python
for labels in ("real", "shuffled"):
    run_cfg = cfg if labels == "real" else replace(
        cfg, shuffle_labels=True, shuffle_seed=SHUFFLE_BASE + seed
    )
```

One subtlety worth getting right: **the shuffle seed must vary with the run seed.**
Reusing a single shuffled market across all five seeds would correlate the arms and
understate the spread.

## Not in scope

A third arm running the **35 textbook alphas** would be a useful reference point -
a curated human prior with no search at all - but those are fixed formulas, and
R2 asks about procedures.  If the net effect of the procedure turns out to be zero,
that arm becomes the interesting follow-up: it would tell us whether the prior
doing the work is the grammar or the literature.
