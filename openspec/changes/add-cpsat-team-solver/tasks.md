# Tasks: add-cpsat-team-solver

- [x] 1. `solver/models.py` — Traits, Candidate, Cohort, ObjectiveWeights, SolverConfig,
      TeamStats, SolveResult. Type hints on all public functions.
- [x] 2. `solver/generator.py` — seeded synthetic cohort generator, 50-200 candidates,
      reproducible from `(n, seed)`.
- [x] 3. `solver/model.py` — CP-SAT build + solve: decision variables, five hard
      constraints, four objective terms, symmetry breaking, determinism parameters.
- [x] 4. `solver/verify.py` — independent re-check of all hard constraints.
- [x] 5. `solver/stats.py` — per-team composition statistics and objective breakdown.
- [x] 6. `solver/cli.py` — run a synthetic cohort from the command line, log status,
      objective, best bound, wall clock.
- [x] 7. Tests: determinism; each hard constraint verified independently; infeasible
      cohort returns INFEASIBLE; scaling smoke at n = 50, 100, 200.
- [x] 8. `ruff check` and `ruff format` clean; full `pytest` green.
