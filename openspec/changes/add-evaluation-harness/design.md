# Design: evaluation harness

## Everything is scored by the same code

Each method returns `(teams, mentor_assignment)` in the shape `SolveResult` uses.
Scoring goes through `solver.verify.check_solution` and `solver.stats` for every
method including the solver. No method gets its own scorer, and no method's output is
repaired before scoring — a baseline that produces a team of size 9 is recorded as a
size violation, not quietly resized.

## Baselines

| method | model of the practice it stands for |
| --- | --- |
| `random` | uniform shuffle, chunked into near-equal teams. The floor. |
| `marks_sorted` | sort by mean technical skill, chunk sequentially. The common institutional practice; produces one strong team and one weak team by construction. |
| `self_selection` | a random student founds a team and recruits the most *similar* remaining students — homophily on skill profile, extraversion and gender. Reproduces the documented failure mode: duplicated strengths and clustered demographics. |
| `direct_llm` | the whole cohort in one prompt, teams out. |

`random`, `marks_sorted` and `self_selection` are deterministic given their seed, so
their contribution to run-to-run variance is zero by construction and any variance in
the results table belongs to the cohort, not the method.

## Metrics

- **skill coverage rate** — fraction of (team, required skill) pairs with a member at
  or above the threshold. A rate, not a boolean, so partial failure is visible.
- **composition score** — the solver's own objective evaluated on whatever assignment
  the method produced (`solver.stats.composition_score`). This is the only way the
  comparison is meaningful: every method is measured on the function the solver
  optimises, so the solver has no metric of its own to hide behind.
- **demographic parity gap** — largest absolute deviation of any team's gender share
  from the cohort share.
- **violations** — count from the independent verifier, broken down by which of the
  five hard constraints was broken.
- **wall clock time** — seconds.

## Measuring LLM run-to-run variance

Adjusted Rand Index between partitions, averaged over all pairs of repeats on
*identical* input. ARI is used rather than raw member overlap because team labels are
arbitrary — a partition that is identical up to relabelling must score 1.0, and ARI
does exactly that while raw overlap does not.

ARI < 1.0 on identical input is direct evidence that the method cannot be reproduced,
which is the property the whole architecture is designed around.

## Why the CP-SAT runs use a deterministic time limit

A wall-clock limit makes the grid unreproducible: on a slower machine the solver gets
less search and returns a different incumbent. `--deterministic-time` counts solver
work units instead, so the same grid re-run on different hardware produces the same
CSV. Wall clock is still *recorded* as a metric; it is just not what bounds the run.

The solver is not run to proven optimality at these sizes — the reported
`optimality_gap` column is the honest statement of that. The claim being tested is
"zero constraint violations and a better composition score than the baselines", not
"provably optimal".

## No simulated LLM

If `GEMINI_API_KEY` is unset, `direct_llm` rows carry `status="skipped_no_api_key"`,
`violations=-1`, and are excluded from every chart aggregate; charts label the method
"no successful runs" on their face rather than drawing a zero bar. Substituting a
plausible-looking synthetic failure rate would produce a number that looks like
evidence and is not.
