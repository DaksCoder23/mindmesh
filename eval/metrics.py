"""Metrics applied identically to every method."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from solver.models import Cohort
from solver.stats import composition_score, compute_team_stats
from solver.verify import check_solution


@dataclass(frozen=True)
class RunMetrics:
    """One method on one cohort."""

    skill_coverage_rate: float
    composition_score: float
    demographic_parity_gap: float
    violations: int
    violations_by_kind: dict[str, int]
    wall_time_seconds: float


def skill_coverage_rate(cohort: Cohort, teams: tuple[tuple[str, ...], ...]) -> float:
    """Fraction of (team, required skill) pairs that have a member at or above threshold."""
    stats = compute_team_stats(cohort, teams)
    required = len(cohort.required_skills)
    if not stats or required == 0:
        return 0.0
    covered = sum(len(s.covered_skills) for s in stats)
    return covered / (len(stats) * required)


def demographic_parity_gap(cohort: Cohort, teams: tuple[tuple[str, ...], ...]) -> float:
    """Largest absolute deviation of any team's gender share from the cohort share."""
    lookup = {c.id: c for c in cohort.candidates}
    worst = 0.0
    for gender in cohort.genders:
        target = cohort.gender_proportion(gender)
        for team in teams:
            if not team:
                continue
            share = sum(1 for m in team if m in lookup and lookup[m].gender == gender) / len(team)
            worst = max(worst, abs(share - target))
    return worst


def classify_violation(text: str) -> str:
    """Bucket a verifier message by which hard constraint it belongs to."""
    if "assigned to" in text or "unknown candidate" in text or "counted as a team member" in text:
        return "assignment"
    if "size" in text:
        return "team_size"
    if "at or above" in text:
        return "skill_coverage"
    if "mentor" in text:
        return "mentor"
    return "gender_parity"


def evaluate(
    cohort: Cohort,
    teams: tuple[tuple[str, ...], ...],
    mentor_assignment: tuple[str | None, ...],
    wall_time_seconds: float,
) -> RunMetrics:
    violations = check_solution(cohort, teams, mentor_assignment)
    by_kind: dict[str, int] = {}
    for violation in violations:
        kind = classify_violation(violation)
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return RunMetrics(
        skill_coverage_rate=skill_coverage_rate(cohort, teams),
        composition_score=composition_score(cohort, teams),
        demographic_parity_gap=demographic_parity_gap(cohort, teams),
        violations=len(violations),
        violations_by_kind=by_kind,
        wall_time_seconds=wall_time_seconds,
    )


def adjusted_rand_index(labels_a: dict[str, int], labels_b: dict[str, int]) -> float:
    """ARI between two partitions of the same members.

    1.0 means identical partitions up to relabelling. Used to measure whether a method
    returns the same answer twice on identical input.
    """
    shared = sorted(set(labels_a) & set(labels_b))
    if len(shared) < 2:
        return 1.0

    def pairs(count: int) -> int:
        return count * (count - 1) // 2

    contingency: dict[tuple[int, int], int] = {}
    rows: dict[int, int] = {}
    columns: dict[int, int] = {}
    for member in shared:
        cell = (labels_a[member], labels_b[member])
        contingency[cell] = contingency.get(cell, 0) + 1
        rows[cell[0]] = rows.get(cell[0], 0) + 1
        columns[cell[1]] = columns.get(cell[1], 0) + 1

    total_pairs = pairs(len(shared))
    index = sum(pairs(v) for v in contingency.values())
    row_pairs = sum(pairs(v) for v in rows.values())
    column_pairs = sum(pairs(v) for v in columns.values())
    expected = row_pairs * column_pairs / total_pairs
    maximum = (row_pairs + column_pairs) / 2
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def labels_from_teams(teams: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    return {member: index for index, team in enumerate(teams) for member in team}


def mean_pairwise_ari(runs: list[tuple[tuple[str, ...], ...]]) -> float:
    """Mean ARI over all pairs of repeats. 1.0 means perfectly reproducible."""
    if len(runs) < 2:
        return 1.0
    labels = [labels_from_teams(teams) for teams in runs]
    scores = [adjusted_rand_index(a, b) for a, b in combinations(labels, 2)]
    return sum(scores) / len(scores)
