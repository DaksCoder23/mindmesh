"""Per-team composition statistics, computed from an assignment alone.

Used by the solver to report results and by the evaluation harness to score baselines
on the same footing.
"""

from __future__ import annotations

from collections import Counter

from solver.models import Candidate, Cohort, TeamStats


def index_by_id(cohort: Cohort) -> dict[str, Candidate]:
    return {c.id: c for c in cohort.candidates + cohort.mentors}


def compute_team_stats(
    cohort: Cohort,
    teams: tuple[tuple[str, ...], ...],
    mentor_assignment: tuple[str | None, ...] = (),
) -> tuple[TeamStats, ...]:
    """Composition statistics for each team. Empty teams are reported, not skipped.

    Ids that are not in the cohort are **skipped for the statistics but not repaired**.
    This function has to survive being handed a hallucinated assignment — the direct-LLM
    baseline really does emit ids like ``'S0028" -> "S0041'`` — and crashing here would
    turn a baseline's failure into a harness failure. The unknown id is still counted as
    a violation by :func:`solver.verify.check_solution`, which sees the raw teams.
    """
    lookup = index_by_id(cohort)
    threshold = cohort.config.skill_threshold
    stats: list[TeamStats] = []

    for index, member_ids in enumerate(teams):
        members = [lookup[member_id] for member_id in member_ids if member_id in lookup]
        mentor_id = mentor_assignment[index] if index < len(mentor_assignment) else None

        if not members:
            stats.append(
                TeamStats(
                    team_index=index,
                    member_ids=(),
                    mentor_id=mentor_id,
                    size=0,
                    mean_conscientiousness=0.0,
                    min_agreeableness=0,
                    max_agreeableness=0,
                    agreeableness_spread=0,
                    mean_agreeableness=0.0,
                    total_skill=0,
                    gender_counts={},
                    covered_skills=(),
                )
            )
            continue

        agreeableness = [m.traits.agreeableness for m in members]
        conscientiousness = [m.traits.conscientiousness for m in members]
        covered = tuple(
            skill
            for skill in cohort.required_skills
            if any(m.skills[cohort.skill_names.index(skill)] >= threshold for m in members)
        )
        stats.append(
            TeamStats(
                team_index=index,
                member_ids=tuple(member_ids),
                mentor_id=mentor_id,
                size=len(members),
                mean_conscientiousness=sum(conscientiousness) / len(members),
                min_agreeableness=min(agreeableness),
                max_agreeableness=max(agreeableness),
                agreeableness_spread=max(agreeableness) - min(agreeableness),
                mean_agreeableness=sum(agreeableness) / len(members),
                total_skill=sum(m.mean_skill for m in members),
                gender_counts=dict(Counter(m.gender for m in members)),
                covered_skills=covered,
            )
        )
    return tuple(stats)


def composition_score(cohort: Cohort, teams: tuple[tuple[str, ...], ...]) -> float:
    """The solver's objective, evaluated on any assignment.

    Lets every baseline be scored on the exact function the solver optimises, which is
    the only way the comparison in ``eval/`` means anything.
    """
    stats = [s for s in compute_team_stats(cohort, teams) if s.size > 0]
    if not stats:
        return 0.0
    weights = cohort.weights
    min_mean_c = min(s.mean_conscientiousness for s in stats) * 10.0
    sum_min_a = sum(s.min_agreeableness for s in stats)
    sum_spread = sum(s.agreeableness_spread for s in stats)
    skill_range = max(s.total_skill for s in stats) - min(s.total_skill for s in stats)
    return (
        weights.conscientiousness * min_mean_c
        + weights.agreeableness_min * sum_min_a
        - weights.agreeableness_spread * sum_spread
        - weights.skill_balance * skill_range
    )
