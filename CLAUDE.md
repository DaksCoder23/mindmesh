# MindMesh AI

Final year CSE project. Partitions a cohort of students into project teams using
personality, cognitive and technical assessment data.

## Core design decision

An **OR-Tools CP-SAT solver decides the team allocation.** A language model is used
for exactly two narrow jobs:

1. Converting a teacher's natural language requirements into solver constraints.
2. Writing a human readable explanation for each team **after** the solver has run.

The language model never chooses the teams.

Justification: ConstraintBench (Tso, Schmittou, Huynh & Hutchins, arXiv 2602.22465)
benchmarked LLMs solving fully specified constrained optimisation directly, with
ground truth verified by Gurobi. Feasibility ranged from 85.0% on facility location
down to **0.8% on crew assignment** — the closest analogue to team formation — with
entity hallucination and 0% optimality in several domains.

## Objective function

Derived from two meta-analyses, not from intuition.

- **Bell (2007), Journal of Applied Psychology.** Team *minimum* agreeableness and
  team *mean* conscientiousness predict team performance in field studies.
- **Peeters et al. (2006), European Journal of Personality.** Agreeableness
  elevation rho = 0.24; conscientiousness elevation rho = 0.20; agreeableness
  variability rho = -0.12; conscientiousness variability rho = -0.24.

So the objective covers: conscientiousness elevation, team minimum agreeableness
(maximise), agreeableness spread (minimise), technical skill balance across teams.

**Do not change term 1 back to a sum.** Summed team conscientiousness over a whole
cohort partition equals the cohort total whatever the partition — a constant, and the
solver will happily run while the term does nothing. The implemented term is
`max min_t (team mean conscientiousness)`. Full argument in
`openspec/changes/add-cpsat-team-solver/design.md` §1.

**Objective weights must be non-negative.** The model uses one-sided channelling for
the team min/max variables; a negative weight silently inverts what they mean.

## Hard rules

- **No LLM call may ever be on the path that produces a team assignment.** If a code
  path can reach a model provider and also reach the returned assignment, it is a bug.
- **Determinism is a hard requirement, not a nice to have.** Every solver run must be
  reproducible from the same input and seed: `num_search_workers=1`, fixed
  `random_seed`. Two runs on identical input must return byte identical assignments,
  and a test must assert this. Use `max_deterministic_time`, not
  `max_time_seconds`, whenever reproducibility matters — a wall-clock limit that fires
  mid-search makes the answer depend on machine speed.
- Infeasible cohorts return `INFEASIBLE`. Never a partial or repaired answer.
- Every run logs solver status, objective value, best bound and wall clock time.

## Data

Big Five traits in generated cohorts are sampled from parameters fitted to the Open
Psychometrics IPIP-FFM dataset (`solver/data/big_five_params.json`). Skill vectors and
gender are synthetic and must never be described as empirical. See `solver/empirical.py`.

## Conventions

- Python 3.11.
- `ruff` for lint and format. `pytest` for tests.
- Type hints on all public functions.

## Repo layout

```
solver/   CP-SAT model, data model, synthetic cohort generator
eval/     benchmark harness and baselines
api/      FastAPI backend
web/      React frontend
```

Keep changes scoped to one package. The solver package has no dependency on `api/`,
`web/`, or any LLM client.

## References

[1] S. T. Bell, "Deep-level composition variables as predictors of team performance:
    A meta-analysis," J. Applied Psychology, 92(3), 595-615, 2007.
[2] M. A. G. Peeters et al., "Personality and team performance: A meta-analysis,"
    European J. of Personality, 20(5), 377-396, 2006.
[3] J. Juarez, C. Santos, C. A. Brizuela, "A comprehensive review and a taxonomy
    proposal of team formation problems," ACM Computing Surveys, 54(7), 2021.
[4] A. Kessler et al., "A hierarchical integer linear programming approach for
    optimizing team formation in education," arXiv:2506.02756, 2025.
[5] M. Almutairi, "Teaming in the AI era," ACM UMAP, arXiv:2506.05265, 2025.
[6] J. Tso et al., "ConstraintBench," arXiv:2602.22465, 2026.
[7] L. R. Goldberg, "The development of markers for the Big-Five factor structure,"
    Psychological Assessment, 4(1), 26-42, 1992.
[8] L. Perron and F. Didier, "CP-SAT," Google OR-Tools.
