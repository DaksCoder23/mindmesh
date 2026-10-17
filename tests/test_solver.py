"""Solver tests.

Every hard constraint is checked against the returned solution by
``solver.verify.check_solution``, which shares no code with the model builder.
"""

from __future__ import annotations

import time

import pytest

from solver.generator import generate_cohort
from solver.model import solve
from solver.models import Candidate, Cohort, ConscientiousnessMode, SolverConfig, Traits
from solver.stats import compute_team_stats
from solver.verify import check_solution

FAST = SolverConfig(max_deterministic_time=6.0)


def _traits(**overrides: int) -> Traits:
    base = dict(
        openness=50,
        conscientiousness=50,
        extraversion=50,
        agreeableness=50,
        neuroticism=50,
    )
    base.update(overrides)
    return Traits(**base)


# -- determinism -----------------------------------------------------------


def test_two_runs_on_the_same_input_are_identical() -> None:
    cohort = generate_cohort(60, seed=11, config=FAST)
    first = solve(cohort)
    second = solve(cohort)

    assert first.is_feasible
    assert first.teams == second.teams
    assert first.mentor_assignment == second.mentor_assignment
    assert first.objective_value == second.objective_value
    assert first.best_bound == second.best_bound


def test_generator_is_reproducible_from_n_and_seed() -> None:
    a = generate_cohort(50, seed=5)
    b = generate_cohort(50, seed=5)
    c = generate_cohort(50, seed=6)

    assert a.candidates == b.candidates
    assert a.mentors == b.mentors
    assert a.candidates != c.candidates


def test_a_different_seed_is_still_deterministic() -> None:
    cohort = generate_cohort(
        50, seed=1, config=SolverConfig(random_seed=99, max_deterministic_time=6.0)
    )
    assert solve(cohort).teams == solve(cohort).teams


# -- hard constraints ------------------------------------------------------


def test_solution_passes_independent_verification() -> None:
    cohort = generate_cohort(80, seed=3, config=FAST)
    result = solve(cohort)

    assert result.is_feasible
    assert check_solution(cohort, result.teams, result.mentor_assignment) == []


def test_every_candidate_is_assigned_exactly_once() -> None:
    cohort = generate_cohort(70, seed=4, config=FAST)
    result = solve(cohort)

    assigned = [member for team in result.teams for member in team]
    assert sorted(assigned) == sorted(c.id for c in cohort.candidates)
    assert len(assigned) == len(set(assigned))


def test_team_sizes_are_within_bounds() -> None:
    cohort = generate_cohort(70, seed=8, team_size_min=5, team_size_max=7, config=FAST)
    result = solve(cohort)

    for team in result.teams:
        assert 5 <= len(team) <= 7


def test_every_team_covers_every_required_skill() -> None:
    cohort = generate_cohort(60, seed=9, config=FAST)
    result = solve(cohort)
    lookup = {c.id: c for c in cohort.candidates}

    for team in result.teams:
        for skill in cohort.required_skills:
            position = cohort.skill_names.index(skill)
            assert any(lookup[m].skills[position] >= cohort.config.skill_threshold for m in team), (
                f"team missing {skill}"
            )


def test_each_team_has_exactly_one_distinct_mentor() -> None:
    cohort = generate_cohort(60, seed=10, config=FAST)
    result = solve(cohort)
    mentor_ids = {m.id for m in cohort.mentors}

    assert len(result.mentor_assignment) == result.num_teams
    assert all(m in mentor_ids for m in result.mentor_assignment)
    assert len(set(result.mentor_assignment)) == result.num_teams


def test_gender_counts_stay_within_tolerance() -> None:
    config = SolverConfig(gender_tolerance=0.25, max_deterministic_time=6.0)
    cohort = generate_cohort(80, seed=12, config=config)
    result = solve(cohort)
    lookup = {c.id: c for c in cohort.candidates}

    for gender in cohort.genders:
        proportion = cohort.gender_proportion(gender)
        for team in result.teams:
            count = sum(1 for m in team if lookup[m].gender == gender)
            low = (proportion - 0.25) * len(team)
            high = (proportion + 0.25) * len(team)
            assert low - 1e-9 <= count <= high + 1e-9


def test_mentors_are_never_counted_as_team_members() -> None:
    cohort = generate_cohort(50, seed=13, config=FAST)
    result = solve(cohort)
    mentor_ids = {m.id for m in cohort.mentors}

    for team in result.teams:
        assert not mentor_ids & set(team)


# -- infeasibility ---------------------------------------------------------


def _uniform_cohort(n: int, **kwargs: object) -> Cohort:
    candidates = tuple(
        Candidate(f"S{i:03d}", _traits(), (80, 80, 80), "female" if i % 2 else "male")
        for i in range(n)
    )
    mentors = tuple(
        Candidate(f"M{i:03d}", _traits(), (80, 80, 80), "male", is_mentor=True) for i in range(4)
    )
    defaults: dict[str, object] = dict(
        candidates=candidates,
        mentors=mentors,
        skill_names=("a", "b", "c"),
        required_skills=("a",),
        team_size_min=4,
        team_size_max=6,
        config=FAST,
    )
    defaults.update(kwargs)
    return Cohort(**defaults)  # type: ignore[arg-type]


def test_impossible_size_bounds_return_infeasible() -> None:
    # 10 students cannot be split into 2 teams of 6 to 6.
    cohort = _uniform_cohort(10, team_size_min=6, team_size_max=6, num_teams=2)
    result = solve(cohort)

    assert result.status == "INFEASIBLE"
    assert result.teams == ()
    assert result.objective_value is None


def test_too_few_mentors_returns_infeasible() -> None:
    # 4 mentors in the pool, 5 teams required.
    cohort = _uniform_cohort(20, team_size_min=4, team_size_max=4, num_teams=5)
    result = solve(cohort)

    assert result.status == "INFEASIBLE"


def test_uncoverable_skill_returns_infeasible() -> None:
    # Only one student clears the threshold on skill "c", but there are two teams.
    candidates = tuple(
        Candidate(
            f"S{i:03d}",
            _traits(),
            (80, 80, 90 if i == 0 else 10),
            "female" if i % 2 else "male",
        )
        for i in range(10)
    )
    mentors = tuple(
        Candidate(f"M{i:03d}", _traits(), (80, 80, 80), "male", is_mentor=True) for i in range(3)
    )
    cohort = Cohort(
        candidates=candidates,
        mentors=mentors,
        skill_names=("a", "b", "c"),
        required_skills=("c",),
        team_size_min=5,
        team_size_max=5,
        num_teams=2,
        config=FAST,
    )
    result = solve(cohort)

    assert result.status == "INFEASIBLE"


def test_impossible_gender_tolerance_returns_infeasible() -> None:
    config = SolverConfig(gender_tolerance=0.0, max_deterministic_time=6.0)
    # A 50/50 cohort with zero tolerance cannot fill odd-sized teams.
    cohort = _uniform_cohort(15, team_size_min=5, team_size_max=5, num_teams=3, config=config)
    result = solve(cohort)

    assert result.status == "INFEASIBLE"


# -- objective behaviour ---------------------------------------------------


def test_reported_terms_match_the_returned_assignment() -> None:
    cohort = generate_cohort(60, seed=14, config=FAST)
    result = solve(cohort)
    stats = compute_team_stats(cohort, result.teams, result.mentor_assignment)

    assert result.objective_terms["min_mean_conscientiousness"] == pytest.approx(
        min(s.mean_conscientiousness for s in stats)
    )
    assert result.objective_terms["sum_min_agreeableness"] == sum(
        s.min_agreeableness for s in stats
    )


def test_min_agreeableness_beats_a_naive_split() -> None:
    """The solver's worst team should not be worse than an arbitrary index split."""
    cohort = generate_cohort(60, seed=15, config=FAST)
    result = solve(cohort)

    ids = [c.id for c in cohort.candidates]
    size = len(ids) // result.num_teams
    naive = tuple(tuple(ids[k * size : (k + 1) * size]) for k in range(result.num_teams))
    naive_stats = compute_team_stats(cohort, naive)
    solver_stats = compute_team_stats(cohort, result.teams)

    assert sum(s.min_agreeableness for s in solver_stats) >= sum(
        s.min_agreeableness for s in naive_stats
    )


def test_sum_mode_is_accepted_and_degenerate() -> None:
    """The originally specified 'sum' formulation runs, and its term is a constant.

    Documented in design.md §1: every candidate is assigned exactly once, so the summed
    conscientiousness over all teams equals the cohort total whatever the partition.
    """
    config = SolverConfig(
        conscientiousness_mode=ConscientiousnessMode.SUM, max_deterministic_time=6.0
    )
    cohort = generate_cohort(50, seed=16, config=config)
    result = solve(cohort)

    assert result.is_feasible
    total = sum(c.traits.conscientiousness for c in cohort.candidates)
    lookup = {c.id: c for c in cohort.candidates}
    assigned_total = sum(lookup[m].traits.conscientiousness for team in result.teams for m in team)
    assert assigned_total == total


# -- scaling ---------------------------------------------------------------


@pytest.mark.parametrize("n", [50, 100, 200])
def test_scaling_smoke(n: int) -> None:
    config = SolverConfig(max_deterministic_time=15.0)
    cohort = generate_cohort(n, seed=n, config=config)

    started = time.monotonic()
    result = solve(cohort)
    elapsed = time.monotonic() - started

    assert result.is_feasible, f"n={n} returned {result.status}"
    assert check_solution(cohort, result.teams, result.mentor_assignment) == []
    assert elapsed < 120.0, f"n={n} took {elapsed:.1f}s"


def test_negative_objective_weights_are_rejected() -> None:
    """The one-sided objective encoding is only valid for non-negative weights."""
    from solver.models import ObjectiveWeights

    with pytest.raises(ValueError, match="must be >= 0"):
        ObjectiveWeights(agreeableness_min=-1.0)
