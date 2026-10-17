# Proposal: add-evaluation-harness

## Why

The project's central claim is that an exact solver, not a language model, should
decide the team allocation. That claim needs evidence produced on this codebase, not
only a citation to ConstraintBench.

## What changes

Adds `eval/`, benchmarking the CP-SAT solver against four baselines on synthetic
cohorts:

| method | description |
| --- | --- |
| `cpsat` | the solver |
| `random` | uniform random partition into admissible team sizes |
| `self_selection` | simulated student self-selection: homophily on skill profile and gender |
| `marks_sorted` | sort by mean technical skill, chunk sequentially |
| `direct_llm` | whole cohort handed to Gemini, asked for the teams |

Metrics per run: skill coverage rate, composition score (the solver's own objective,
evaluated on whatever assignment the method produced), demographic parity gap, hard
constraint violation count, wall clock time.

Grid: n ∈ {50, 100, 200} × 20 seeds. Output is a CSV plus matplotlib charts comparing
the five methods on each metric.

## The direct-LLM baseline

This is the one the whole design decision rests on, so it is instrumented harder than
the others:

- Its **constraint violation count** is recorded per run, broken down by constraint.
- Its **run-to-run variance on identical input** is measured by repeating the same
  cohort `k` times and reporting pairwise assignment disagreement (Adjusted Rand
  Index between the partitions) and the spread of its metrics.

Every other method is deterministic given its seed. If the LLM is not, that shows up
directly as ARI < 1.0 against itself.

## Honesty constraints

- The LLM baseline calls the real API. There is no simulated stand-in — fabricating a
  failure rate we then cite as evidence would invalidate the result.
- If `GEMINI_API_KEY` is absent, the method is recorded with `status="skipped"` and is
  excluded from the charts, which say so on their face.
- Every method, including the solver, is scored by the same `solver.verify` and
  `solver.stats` code paths.
