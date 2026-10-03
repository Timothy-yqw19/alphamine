# P1 - genetic programming

**Question.** P0.2 showed random search saturates almost immediately, and P0.1
showed a 35-formula textbook library beats it at equal budget. So does *evolution*
buy anything? This is the equal-budget test of that.

**Method.** `alphamine/gp.py`, 2000 evaluated formulas, depth <= 4, seed 0, full
panel, five workers. Run `runs/20261002-220802-p1-gp-depth4`.

The design decisions that make the comparison meaningful:

* **Budget is counted exactly as the random baseline counts it** - a formula is
  spent when first proposed and evaluated, a duplicate canonical string costs
  nothing, and a generation is one batch. The initial population is 200 random
  draws from the same sampler, so the first generation *is* random search.
* **Selection is on training IC only.** Validation is read, never selected on.
* **Depth is capped at 4**, the same value the random baseline used, so GP
  cannot win by searching a larger space.
* Tournament selection (size 3), subtree crossover in the `S` slots only, subtree
  mutation, two elites carried over without re-evaluation. The tree surgery
  refuses to breed into window or constant slots, because those must stay
  literals or the parser rejects the tree.

## Result: evolution does not beat random sampling here

Two numbers per budget. `best-of-N` is the running maximum of validation IC, which
is what the P0.2 write-up plots. `selected` is the honest one: among the first N
formulas take the best **training** IC - the choice a method could actually have
made - and report *its* validation IC.

| budget | random search | genetic programming | the 35 textbook alphas |
| --- | --- | --- | --- |
| 35 | +0.0411 / +0.0411 | +0.0411 / +0.0411 | +0.0508 / +0.0491 |
| 100 | +0.0411 / +0.0411 | +0.0411 / +0.0411 | - |
| 250 | +0.0618 / +0.0618 | +0.0618 / +0.0618 | - |
| 1,000 | +0.0647 / +0.0618 | +0.0618 / +0.0618 | - |
| 2,000 | +0.0647 / +0.0618 | +0.0655 / +0.0618 | - |

*(best-of-N / selected)*

**On the honest metric the two searches are identical at every budget.** Not
"close" - identical, to four decimals: the same formula is selected, and it is the
same number.

### The tempting wrong answer

GP's `best-of-N` is **+0.0655 against random's +0.0647**, so the table can be read
as "genetic programming wins by 0.0008". That is a selection artifact and nothing
more. Both numbers are the maximum of validation IC over the run, which is a
choice neither method could have made; GP is rewarded only because its 2,000
formulas are a differently-shaped sample and one of them happens to sit higher on
the validation set. Its winner `ts_mean(sqrt(div(open, high)), 10)` scores
**+0.0107** on training - nowhere near good enough to be picked.

The honest metric picks exactly the same formula for both methods, and reports
exactly the same +0.0618.

### Why: the search never improved on its own first generation

| generation | cumulative | best train IC | best valid IC | selected |
| --- | --- | --- | --- | --- |
| 0 | 200 | **+0.0742** | +0.0618 | +0.0618 |
| 1-4 | 992 | +0.0742 | +0.0618 | +0.0618 |
| 5-9 | 1,982 | +0.0742 | +0.0655 | +0.0618 |
| 10 | 2,000 | +0.0742 | +0.0655 | +0.0618 |

**The best training IC was reached at generation 0 and never improved again.**
Ten generations of tournament selection, crossover and mutation produced a best
new training IC of **+0.0624** - well below the +0.0742 sitting in the initial
random population. Evolution had a champion in hand from the start and never
found anything to replace it with.

Two more facts point the same way:

* **Both searches found the same best-training formula**,
  `mulconst(ts_cov(abs(volume), vwap_proxy, 5), -0.1)`, at +0.074203 in both.
  Random search met it at draw 149 of 2,000.
* **220 of GP's 2,000 formulas also appear in the random run's 2,000.** The two
  searches are not exploring different regions; they overlap by 11%.

### No bloat either

Mean tree size per generation stays between 5.7 and 6.5 nodes, with no upward
drift. Parsimony pressure was off in this run, and the standard bloat problem did
not appear - presumably because the search was not deep enough for bloat to pay.

## What this means

This is the third result in the same direction and the most direct one. P0.2 found
that random draws already land on real A-share structure. P0.1 found that a
textbook library beats random sampling at equal budget. P1 now finds that adding
selection, inheritance and variation on top of random sampling changes **nothing**
in this grammar over this budget.

The likely reason is that the fitness landscape is nearly flat at this budget: the
best of 200 random draws is already at the ceiling of what the grammar reaches at
2,000 evaluations, so there is no gradient for selection to climb. The evidence is
that GP's generation 0 - which is literally a random sample of 200 - contains a
formula that nothing in 1,800 further evaluations could beat.

That is a real result about the problem, not a failure of the implementation:
selection cannot beat a flat landscape, and it will happily *look* like it wins if
the metric is chosen on the validation set.

## The parsimony ablation

`--parsimony 0.005` makes a tree that is within 0.005 IC of a fatter one win the
head-to-head in tournament selection, so bloat has to pay for itself instead of
being free. Run `runs/20261002-223227-p1-gp-parsimony` - same seed, same budget,
and its generation 0 is byte-identical to the run above, so the two differ only
in what selection did afterwards.

| budget | no parsimony (best-of-N / selected) | parsimony 0.005 |
| --- | --- | --- |
| 35 | +0.0411 / +0.0411 | +0.0411 / +0.0411 |
| 250 | +0.0618 / +0.0618 | +0.0618 / +0.0618 |
| 2,000 | +0.0655 / +0.0618 | +0.0618 / +0.0618 |

**The honest metric is unchanged**, for the same reason as before: the best
training IC is still the generation-0 champion, which both runs inherit.

**There was no bloat to suppress.** Mean tree size went from 6.16 to 5.93 nodes,
p90 from 10 to 9, and the maximum stayed at 16 in both. The unpressured run had
already shown no upward drift, so the pressure had nothing to act on - consistent
with a search that never got deep enough to bloat.

One incidental observation, and it is a good illustration of the artifact: with
parsimony on, the best-of-N column falls to +0.0618 and therefore *equals* the
honest metric. The +0.0655 in the unpressured run was not something parsimony
improved on - it is a formula that parsimony's different trajectory never sampled.
Whether a run's best-of-N exceeds its honest number is a property of the sample,
not of the method.

## Caveats

* **One seed.** The study plan's R3 calls for five. The *magnitude* of any
  GP-versus-random difference is not settled by this run; what is settled is that
  this run's apparent GP win disappears under the honest metric. R3 must use the
  train-selected metric or it will reproduce the artifact.
* One population size (200) and one set of rates. A larger population would spend
  more of the budget on generation 0 and less on evolution, which can only help
  GP, so this does not flatter random search.
* Depth 4, the same cap as P0.2. A deeper grammar might have more structure to
  exploit - but then the comparison has to re-run the baseline at the same cap.
* The parsimony ablation is run above, on one seed. It moved the honest metric by
  nothing and tree size by 0.23 nodes.

## Performance note

Each generation builds a fresh worker pool, so GP pays a spawn-and-load cost per
generation rather than once: about 11 pool startups here, roughly 3-4 minutes of
the ~21 the GP phase took. At R3's 60,000 evaluations that becomes ~300
generations and the overhead stops being negligible. A persistent worker pool is
the obvious fix and has not been done.

## Reproducing

```bash
python -m alphamine.cli gp --n 2000 --depth 4 --workers 5 --seed 0 \
  --population 200 --tournament 3 --crossover 0.6 --mutation 0.4 --elite 2 \
  --null --name p1-gp-depth4

python scripts/compare_searches.py --budgets 35,100,250,1000,2000 \
  runs/*p02-random-depth4 runs/*p1-gp-depth4 runs/*p01-alphas101

python scripts/gp_generations.py runs/*p1-gp-depth4

# the parsimony ablation: same seed, same budget, one extra knob
python -m alphamine.cli gp --n 2000 --depth 4 --workers 5 --seed 0 \
  --population 200 --parsimony 0.005 --name p1-gp-parsimony
```
