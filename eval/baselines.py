"""Non-solver baselines.

Each returns ``(teams, mentor_assignment)`` in the same shape as ``SolveResult``, so
every method is scored by the same verifier and the same statistics code. None of them
respect the hard constraints by construction — that is the point of the comparison.
"""

from __future__ import annotations

import random

from solver.models import Cohort


def admissible_sizes(cohort: Cohort, num_teams: int) -> list[int]:
    """Team sizes as equal as possible for ``num_teams`` teams.

    May fall outside ``[L, U]`` if the cohort cannot be split that way; the verifier
    then reports it as a size violation rather than the harness silently hiding it.
    """
    base, remainder = divmod(cohort.size, num_teams)
    return [base + 1] * remainder + [base] * (num_teams - remainder)


def _chunk(ids: list[str], sizes: list[int]) -> tuple[tuple[str, ...], ...]:
    teams, cursor = [], 0
    for size in sizes:
        teams.append(tuple(ids[cursor : cursor + size]))
        cursor += size
    return tuple(teams)


def _assign_mentors(cohort: Cohort, num_teams: int, rng: random.Random) -> tuple[str | None, ...]:
    """Mentors picked at random without replacement; ``None`` once the pool runs out."""
    pool = [m.id for m in cohort.mentors]
    rng.shuffle(pool)
    return tuple(pool[k] if k < len(pool) else None for k in range(num_teams))


def random_assignment(
    cohort: Cohort, num_teams: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], tuple[str | None, ...]]:
    """Uniform random partition into near-equal teams."""
    rng = random.Random(seed)
    ids = [c.id for c in cohort.candidates]
    rng.shuffle(ids)
    return _chunk(ids, admissible_sizes(cohort, num_teams)), _assign_mentors(cohort, num_teams, rng)


def marks_sorted(
    cohort: Cohort, num_teams: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], tuple[str | None, ...]]:
    """Sort by mean technical skill, chunk sequentially.

    The common institutional practice: the strongest students end up together and the
    weakest end up together.
    """
    rng = random.Random(seed)
    ordered = sorted(cohort.candidates, key=lambda c: (-c.mean_skill, c.id))
    ids = [c.id for c in ordered]
    return _chunk(ids, admissible_sizes(cohort, num_teams)), _assign_mentors(cohort, num_teams, rng)


def self_selection(
    cohort: Cohort, num_teams: int, seed: int
) -> tuple[tuple[tuple[str, ...], ...], tuple[str | None, ...]]:
    """Simulated student self-selection.

    Model: a random student starts a team, then repeatedly recruits the most *similar*
    remaining student — homophily on skill profile, extraversion and gender, which is
    what the self-selection literature describes and what produces the characteristic
    failure mode of duplicated strengths and clustered demographics.
    """
    rng = random.Random(seed)
    remaining = {c.id: c for c in cohort.candidates}
    sizes = admissible_sizes(cohort, num_teams)
    teams: list[tuple[str, ...]] = []

    for size in sizes:
        if not remaining:
            teams.append(())
            continue
        founder_id = rng.choice(sorted(remaining))
        founder = remaining.pop(founder_id)
        members = [founder_id]

        while len(members) < size and remaining:

            def similarity(other_id: str, anchor=founder) -> tuple[float, str]:
                other = remaining[other_id]
                skill_distance = sum(
                    abs(a - b) for a, b in zip(anchor.skills, other.skills, strict=True)
                ) / max(1, len(anchor.skills))
                gender_bonus = 0.0 if other.gender == anchor.gender else 12.0
                extraversion_distance = (
                    abs(anchor.traits.extraversion - other.traits.extraversion) / 4.0
                )
                return (skill_distance + gender_bonus + extraversion_distance, other_id)

            pick = min(sorted(remaining), key=similarity)
            remaining.pop(pick)
            members.append(pick)
        teams.append(tuple(members))

    return tuple(teams), _assign_mentors(cohort, num_teams, rng)


BASELINES = {
    "random": random_assignment,
    "self_selection": self_selection,
    "marks_sorted": marks_sorted,
}
