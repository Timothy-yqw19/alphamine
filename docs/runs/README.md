# Run records

`runs/` is git-ignored (it holds charts and per-formula CSVs), so a fresh clone
had **no record of the numbers quoted in the documentation** - the code could be
reproduced but the results could not be checked. This directory is that record:
for every run, the `summary.json` (config, budget curve, best formulas) *and* the
`evals.csv` that backs each number.

**One modification.** The `config` block inside each `summary.json` records
absolute paths from the machine that produced it. Those are rewritten to relative
ones (`data/daily_pv.h5`, `runs`) before committing, so the repository does not
carry a local filesystem path. Everything else, including every `evals.csv`, is
byte-identical to the original.

| run | what it is | cited by |
| --- | --- | --- |
| `20261001-215104-smoke` | first harness smoke test, 60 formulas. **The one run that scored the test split** - it predates `Config.score_test` | HANDOFF section 5 |
| `20261001-215729-timing` | operator and whole-formula timings | README, performance table |
| `20261001-224125-p02-random-depth4` | P0.2 random baseline, 2,000 formulas plus its shuffled-label null | README, `docs/p02-random-baseline.md` |
| `20261001-224411-timing2` | second timing run | README, performance table |
| `20261001-224835-smoke-ablate` | ablation smoke test, 12 formulas per variant | - |
| `20261001-225430-r5-lite` | R5-lite, five variants x 1,000 formulas | README, `docs/r5-lite.md` |
| `20261002-202029-r5-lite-turnover` | `liquid300` vs `illiquid300`, each with its own null | README, `docs/r5-lite.md` |
| `20261002-211321-r5-lite-csi300` | `csi300`: true point-in-time CSI 300 membership, same 1,000 formulas | README, `docs/HANDOFF.md`, `docs/r5-lite.md` |

## The per-formula records

`evals.csv` is the primary record behind every table - one row per evaluated
formula, 21 columns, in evaluation order, carrying its status, cost, coverage and
per-split IC. All 23 files are here, 4.2 MB in total (the P0.2 run alone has 2,001
rows plus a matching `evals_null.csv`). The `summary.json` alongside each one is
just the head of that table.

`20261001-224125-p02-random-depth4/turnover-profile-valid.csv` is included too,
because it is the direct source of the README's turnover-decile table.

## What is not here

The `budget_curve.png` charts, and every other run artefact. The two charts that
the documentation displays live in `docs/images/`; `runs/` itself stays
git-ignored, so anything else a run produced only exists on the machine that
produced it.

## Reproducing

Every run here is seeded and records its proposal order, so re-running the
commands in the README's *Quick start* regenerates the same `evals.csv`. Compare
against these summaries to check you got the same numbers; the P0.2 run's
`best.valid` of `0.06466843851522625` is a good single fingerprint.
