"""CP-SAT model for team partitioning.

This module is the only place a team assignment is produced. It imports no LLM client
and makes no network call, and it must stay that way.

See ``openspec/changes/add-cpsat-team-solver/design.md`` for the derivations, in
particular §1 (why the conscientiousness term is a minimum and not a sum) and §4 (the
range-for-variance substitution).
"""

from __future__ import annotations

import logging

from ortools.sat.python import cp_model

from solver.models import (
    Candidate,
    Cohort,
    ConscientiousnessMode,
    SolveResult,
)
from solver.stats import compute_team_stats

logger = logging.getLogger(__name__)

WEIGHT_SCALE = 1000
"""Float objective weights become integer coefficients via ``round(w * WEIGHT_SCALE)``."""

MEAN_SCALE = 10
"""Team means are stored scaled by 10, i.e. one decimal place retained."""

TRAIT_MAX = 100


class _Model:
    """Builder. Holds the CP-SAT model and the variable handles for extraction."""

    def __init__(self, cohort: Cohort) -> None:
        self.cohort = cohort
        self.num_teams = cohort.effective_num_teams()
        self.model = cp_model.CpModel()
        self.students: tuple[Candidate, ...] = cohort.candidates
        self.mentors: tuple[Candidate, ...] = cohort.mentors

        n, t = len(self.students), self.num_teams
        self.x = [[self.model.NewBoolVar(f"x_{i}_{k}") for k in range(t)] for i in range(n)]
        self.m = [
            [self.model.NewBoolVar(f"m_{j}_{k}") for k in range(t)]
            for j in range(len(self.mentors))
        ]
        self.size = [
            self.model.NewIntVar(cohort.team_size_min, cohort.team_size_max, f"size_{k}")
            for k in range(t)
        ]
        # size_is[k][s] <=> team k has exactly s members. Lets a division by team size be
        # written as one linear constraint per admissible size. design.md §2.
        self.size_is: list[dict[int, cp_model.IntVar]] = []
        for k in range(t):
            indicators = {}
            for s in range(cohort.team_size_min, cohort.team_size_max + 1):
                var = self.model.NewBoolVar(f"size_{k}_is_{s}")
                self.model.Add(self.size[k] == s).OnlyEnforceIf(var)
                self.model.Add(self.size[k] != s).OnlyEnforceIf(var.Not())
                indicators[s] = var
            self.model.AddExactlyOne(indicators.values())
            self.size_is.append(indicators)

    # -- hard constraints ---------------------------------------------------

    def add_assignment_and_size(self) -> None:
        for i in range(len(self.students)):
            self.model.AddExactlyOne(self.x[i])
        for k in range(self.num_teams):
            self.model.Add(self.size[k] == sum(self.x[i][k] for i in range(len(self.x))))

    def add_skill_coverage(self) -> None:
        threshold = self.cohort.config.skill_threshold
        for skill_index in self.cohort.required_skill_indices():
            qualified = [
                i
                for i, student in enumerate(self.students)
                if student.skills[skill_index] >= threshold
            ]
            for k in range(self.num_teams):
                self.model.Add(sum(self.x[i][k] for i in qualified) >= 1)

    def add_mentors(self) -> None:
        for k in range(self.num_teams):
            self.model.AddExactlyOne(self.m[j][k] for j in range(len(self.mentors)))
        for j in range(len(self.mentors)):
            self.model.AddAtMostOne(self.m[j])

    def add_gender_parity(self) -> None:
        tolerance = round(self.cohort.config.gender_tolerance * 1000)
        for gender in self.cohort.genders:
            proportion = round(self.cohort.gender_proportion(gender) * 1000)
            members = [i for i, s in enumerate(self.students) if s.gender == gender]
            for k in range(self.num_teams):
                count = sum(self.x[i][k] for i in members)
                self.model.Add(1000 * count >= (proportion - tolerance) * self.size[k])
                self.model.Add(1000 * count <= (proportion + tolerance) * self.size[k])

    def add_symmetry_breaking(self) -> None:
        """Teams are interchangeable labels. See design.md §8.

        Two breaks, both label-only: a student may open team ``k`` only if some
        lower-indexed student is already on team ``k-1``, and mentors are forced into
        ascending team order. Mentors appear in no objective term and in no constraint
        tying them to students, so the two are compatible.
        """
        for k in range(1, self.num_teams):
            for i in range(len(self.students)):
                if i < k:
                    # The first k students cannot fill k+1 teams.
                    self.model.Add(self.x[i][k] == 0)
                else:
                    self.model.Add(sum(self.x[j][k - 1] for j in range(i)) >= self.x[i][k])

        mentor_index = []
        for k in range(self.num_teams):
            var = self.model.NewIntVar(0, max(0, len(self.mentors) - 1), f"midx_{k}")
            self.model.Add(var == sum(j * self.m[j][k] for j in range(len(self.m))))
            mentor_index.append(var)
        for k in range(self.num_teams - 1):
            self.model.Add(mentor_index[k] < mentor_index[k + 1])

    # -- objective terms ----------------------------------------------------

    def _team_trait_sum(self, k: int, trait: str) -> cp_model.LinearExpr:
        return sum(getattr(s.traits, trait) * self.x[i][k] for i, s in enumerate(self.students))

    def _add_min_mean_bound(self, worst: cp_model.IntVar, k: int, trait: str) -> None:
        """Bound ``worst`` above by team ``k``'s mean trait score, scaled by MEAN_SCALE.

        ``worst * size_k <= MEAN_SCALE * sum_k`` is a var-by-var product, so it is split
        over the size indicators, where ``s`` is a constant and the constraint is linear.
        Only the upper bound is stated: ``worst`` is maximised, so it rises to the true
        minimum team mean on its own. design.md §2.
        """
        total = self._team_trait_sum(k, trait)
        for s, indicator in self.size_is[k].items():
            self.model.Add(s * worst <= MEAN_SCALE * total).OnlyEnforceIf(indicator)

    def _team_agreeableness_extremes(self, k: int) -> tuple[cp_model.IntVar, cp_model.IntVar]:
        """Min and max agreeableness over *assigned* members only. design.md §3.

        Channelled through the assignment booleans with ``OnlyEnforceIf``, one-sided in
        each direction. ``team_min`` carries a net positive objective coefficient and
        ``team_max`` a net negative one, so at the optimum both are tight against the
        true extremes of the assigned members.
        """
        team_min = self.model.NewIntVar(0, TRAIT_MAX, f"min_a_{k}")
        team_max = self.model.NewIntVar(0, TRAIT_MAX, f"max_a_{k}")
        for i, student in enumerate(self.students):
            agreeableness = student.traits.agreeableness
            self.model.Add(team_min <= agreeableness).OnlyEnforceIf(self.x[i][k])
            self.model.Add(team_max >= agreeableness).OnlyEnforceIf(self.x[i][k])
        return team_min, team_max

    def add_objective(self) -> dict[str, cp_model.IntVar | cp_model.LinearExpr]:
        weights = self.cohort.weights
        terms: dict[str, cp_model.IntVar | cp_model.LinearExpr] = {}

        # 1. conscientiousness
        if self.cohort.config.conscientiousness_mode is ConscientiousnessMode.MIN_MEAN:
            worst = self.model.NewIntVar(0, MEAN_SCALE * TRAIT_MAX, "min_mean_c")
            for k in range(self.num_teams):
                self._add_min_mean_bound(worst, k, "conscientiousness")
            conscientiousness_term: cp_model.LinearExpr = worst
        else:
            # Degenerate by construction: constant under a full partition. design.md §1.
            conscientiousness_term = sum(
                self._team_trait_sum(k, "conscientiousness") for k in range(self.num_teams)
            )
        terms["conscientiousness"] = conscientiousness_term

        # 2 and 3. agreeableness minimum and spread
        team_mins, spreads = [], []
        for k in range(self.num_teams):
            team_min, team_max = self._team_agreeableness_extremes(k)
            spread = self.model.NewIntVar(0, TRAIT_MAX, f"spread_a_{k}")
            self.model.Add(spread == team_max - team_min)
            team_mins.append(team_min)
            spreads.append(spread)
        terms["agreeableness_min"] = sum(team_mins)
        terms["agreeableness_spread"] = sum(spreads)

        # 4. technical skill balance across teams
        skill_totals = []
        max_total = TRAIT_MAX * self.cohort.team_size_max
        for k in range(self.num_teams):
            total = self.model.NewIntVar(0, max_total, f"skill_{k}")
            self.model.Add(
                total == sum(s.mean_skill * self.x[i][k] for i, s in enumerate(self.students))
            )
            skill_totals.append(total)
        highest = self.model.NewIntVar(0, max_total, "skill_max")
        lowest = self.model.NewIntVar(0, max_total, "skill_min")
        for total in skill_totals:
            # One-sided: the range is minimised, so both bounds tighten on their own.
            self.model.Add(highest >= total)
            self.model.Add(lowest <= total)
        skill_range = self.model.NewIntVar(0, max_total, "skill_range")
        self.model.Add(skill_range == highest - lowest)
        terms["skill_balance"] = skill_range

        self.model.Maximize(
            round(weights.conscientiousness * WEIGHT_SCALE) * terms["conscientiousness"]
            + round(weights.agreeableness_min * WEIGHT_SCALE) * terms["agreeableness_min"]
            - round(weights.agreeableness_spread * WEIGHT_SCALE) * terms["agreeableness_spread"]
            - round(weights.skill_balance * WEIGHT_SCALE) * terms["skill_balance"]
        )
        return terms


def build_model(cohort: Cohort) -> _Model:
    """Assemble the full model: hard constraints, symmetry breaking, objective."""
    builder = _Model(cohort)
    builder.add_assignment_and_size()
    builder.add_skill_coverage()
    builder.add_mentors()
    builder.add_gender_parity()
    builder.add_symmetry_breaking()
    builder.add_objective()
    return builder


def solve(cohort: Cohort) -> SolveResult:
    """Partition ``cohort`` into teams.

    Deterministic: single search worker and a fixed seed, so two runs on identical
    input return identical assignments. A wall-clock limit that actually fires breaks
    that guarantee; ``SolveResult.hit_limit`` records whether it did.
    """
    builder = build_model(cohort)
    config = cohort.config

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = config.num_search_workers
    solver.parameters.random_seed = config.random_seed
    solver.parameters.log_search_progress = config.log_search_progress
    if config.max_deterministic_time is not None:
        solver.parameters.max_deterministic_time = config.max_deterministic_time
    else:
        solver.parameters.max_time_in_seconds = config.max_time_seconds

    status = solver.Solve(builder.model)
    status_name = solver.StatusName(status)
    wall_time = solver.WallTime()

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info(
            "solve status=%s n=%d teams=%d wall=%.3fs",
            status_name,
            cohort.size,
            builder.num_teams,
            wall_time,
        )
        return SolveResult(
            status=status_name,
            teams=(),
            mentor_assignment=(),
            objective_value=None,
            best_bound=None,
            wall_time_seconds=wall_time,
            hit_limit=status_name == "UNKNOWN",
            num_teams=builder.num_teams,
            seed=config.random_seed,
            num_search_workers=config.num_search_workers,
        )

    teams: list[tuple[str, ...]] = []
    for k in range(builder.num_teams):
        members = [
            student.id
            for i, student in enumerate(builder.students)
            if solver.Value(builder.x[i][k])
        ]
        teams.append(tuple(members))

    mentor_assignment: list[str | None] = []
    for k in range(builder.num_teams):
        assigned = [
            mentor.id for j, mentor in enumerate(builder.mentors) if solver.Value(builder.m[j][k])
        ]
        mentor_assignment.append(assigned[0] if assigned else None)

    stats = compute_team_stats(cohort, tuple(teams), tuple(mentor_assignment))
    objective_terms = {
        "min_mean_conscientiousness": (
            min(s.mean_conscientiousness for s in stats) if stats else 0.0
        ),
        "sum_min_agreeableness": float(sum(s.min_agreeableness for s in stats)),
        "sum_agreeableness_spread": float(sum(s.agreeableness_spread for s in stats)),
        "skill_range": float(max(s.total_skill for s in stats) - min(s.total_skill for s in stats))
        if stats
        else 0.0,
    }

    result = SolveResult(
        status=status_name,
        teams=tuple(teams),
        mentor_assignment=tuple(mentor_assignment),
        objective_value=solver.ObjectiveValue(),
        best_bound=solver.BestObjectiveBound(),
        wall_time_seconds=wall_time,
        hit_limit=status_name == "FEASIBLE",
        num_teams=builder.num_teams,
        seed=config.random_seed,
        num_search_workers=config.num_search_workers,
        team_stats=stats,
        objective_terms=objective_terms,
    )
    logger.info(
        "solve status=%s n=%d teams=%d objective=%.1f bound=%.1f gap=%.3f wall=%.3fs",
        result.status,
        cohort.size,
        result.num_teams,
        result.objective_value or 0.0,
        result.best_bound or 0.0,
        result.optimality_gap if result.optimality_gap is not None else -1.0,
        wall_time,
    )
    return result
