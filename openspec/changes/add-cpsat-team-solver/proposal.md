# Proposal: add-cpsat-team-solver

## Why

MindMesh AI must partition a whole cohort of students into project teams such that
hard rules (team size, skill coverage, one mentor per team, demographic parity) are
guaranteed to hold, and the resulting composition scores well on objectives taken
from the team composition literature.

ConstraintBench (arXiv 2602.22465) measured 0.8% feasibility for LLMs solving crew
assignment directly. Team formation is the same class of problem. Therefore the
allocation is produced by an exact solver, and nothing else.

## What changes

Adds the `solver/` package:

- A data model for candidates, mentors and cohorts.
- A seeded synthetic cohort generator (50-200 candidates, reproducible).
- A CP-SAT model with the hard constraints and weighted objective below.
- An independent verifier that re-checks every hard constraint on a returned
  solution without reusing solver code.
- Tests: determinism, per-constraint verification, infeasibility handling, scaling
  smoke tests at n = 50, 100, 200.

## What does not change

No `api/`, no `web/`, no LLM integration. The solver package imports no network
client and no model provider SDK.

## Hard constraints

1. Every candidate assigned to exactly one team.
2. Team size between L and U (students only; the mentor is not counted).
3. For each team and each required skill, at least one member scores at or above a
   configurable threshold on that skill.
4. Exactly one mentor per team, drawn from a separate mentor pool; a mentor serves
   at most one team.
5. Per-team count of each gender within a configurable tolerance of that gender's
   cohort proportion.

## Objective (weighted sum, weights configurable)

1. Maximise the **minimum team mean conscientiousness** across teams.
2. Maximise the **sum over teams of team minimum agreeableness**.
3. Minimise **agreeableness spread** (max − min) within each team.
4. Minimise the **spread of total technical skill** across teams.

Terms 1 and 4 differ from the first-review write-up. See `design.md` §1 for why the
originally stated "maximise summed team conscientiousness" is a constant under
partitioning and therefore cannot be optimised.

## Success criteria

- Two runs on identical input and seed return byte identical assignments.
- Infeasible cohorts return `INFEASIBLE`, never a partial answer.
- The independent verifier reports zero violations on every feasible result.
- n = 200 solves within the configured time limit.
