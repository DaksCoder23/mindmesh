"""Independent verification of an assignment against the hard constraints.

Deliberately shares no code with ``solver/model.py``: a bug in the CP-SAT encoding
must not be able to hide behind a matching bug in the check. Also used by ``eval/`` to
count violations for the non-solver baselines.
"""

from __future__ import annotations

from collections import Counter

from solver.models import Cohort
from solver.stats import index_by_id


def check_solution(
    cohort: Cohort,
    teams: tuple[tuple[str, ...], ...],
    mentor_assignment: tuple[str | None, ...] = (),
    *,
    check_mentors: bool = True,
) -> list[str]:
    """Return a list of violation descriptions. Empty means the assignment is valid."""
    violations: list[str] = []
    lookup = index_by_id(cohort)
    student_ids = [c.id for c in cohort.candidates]

    # 1. every candidate assigned exactly once
    assigned = Counter(member for team in teams for member in team)
    for student_id in student_ids:
        count = assigned[student_id]
        if count != 1:
            violations.append(f"candidate {student_id} assigned to {count} teams")
    for member_id in assigned:
        if member_id not in lookup:
            violations.append(f"unknown candidate {member_id} in assignment")
        elif lookup[member_id].is_mentor:
            violations.append(f"mentor {member_id} counted as a team member")

    # 2. team size bounds
    for index, team in enumerate(teams):
        if not cohort.team_size_min <= len(team) <= cohort.team_size_max:
            violations.append(
                f"team {index} has size {len(team)}, "
                f"outside [{cohort.team_size_min}, {cohort.team_size_max}]"
            )

    # 3. required skill coverage
    threshold = cohort.config.skill_threshold
    for index, team in enumerate(teams):
        members = [lookup[m] for m in team if m in lookup]
        for skill in cohort.required_skills:
            position = cohort.skill_names.index(skill)
            if not any(m.skills[position] >= threshold for m in members):
                violations.append(f"team {index} has nobody at or above {threshold} in {skill}")

    # 4. exactly one mentor per team, each mentor used at most once
    if check_mentors:
        mentor_ids = {m.id for m in cohort.mentors}
        if len(mentor_assignment) != len(teams):
            violations.append(f"{len(mentor_assignment)} mentor slots for {len(teams)} teams")
        used = Counter(m for m in mentor_assignment if m is not None)
        for index, mentor_id in enumerate(mentor_assignment):
            if mentor_id is None:
                violations.append(f"team {index} has no mentor")
            elif mentor_id not in mentor_ids:
                violations.append(f"team {index} mentor {mentor_id} not in mentor pool")
        for mentor_id, count in used.items():
            if count > 1:
                violations.append(f"mentor {mentor_id} leads {count} teams")

    # 5. gender parity within tolerance of the cohort proportion
    tolerance = cohort.config.gender_tolerance
    for gender in cohort.genders:
        proportion = cohort.gender_proportion(gender)
        for index, team in enumerate(teams):
            if not team:
                continue
            count = sum(
                1 for member in team if member in lookup and lookup[member].gender == gender
            )
            low = (proportion - tolerance) * len(team)
            high = (proportion + tolerance) * len(team)
            # 1e-9 absorbs float error only; the solver constraint is exact in integers.
            if count < low - 1e-9 or count > high + 1e-9:
                violations.append(
                    f"team {index} has {count} {gender}, outside "
                    f"[{low:.2f}, {high:.2f}] for cohort proportion {proportion:.2f}"
                )

    return violations
