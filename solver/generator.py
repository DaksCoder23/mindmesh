"""Seeded synthetic cohort generator.

Reproducible from ``(n, seed)`` alone: uses its own random generators and never
touches global random state.

Big Five traits are drawn from empirically fitted marginals and inter-trait
correlations — see :mod:`solver.empirical`, including its caveats. **Skill vectors and
gender are synthetic**, because no public dataset pairs personality scores with
technical skill assessments for a student cohort.
"""

from __future__ import annotations

import math
import random

from solver.empirical import load_params, sample_traits
from solver.models import (
    Candidate,
    Cohort,
    ObjectiveWeights,
    SolverConfig,
    Traits,
)

DEFAULT_SKILLS: tuple[str, ...] = (
    "frontend",
    "backend",
    "database",
    "ml",
    "devops",
    "ui_design",
)

DEFAULT_GENDERS: tuple[str, ...] = ("female", "male")


def _clamped_normal(rng: random.Random, mu: float, sigma: float) -> int:
    """A trait or skill score: normal, clamped to the 0-100 scale, rounded."""
    return max(0, min(100, round(rng.gauss(mu, sigma))))


def _traits_batch(count: int, seed: int) -> list[Traits]:
    """Correlated Big Five profiles from the empirical fit, one per person."""
    order = load_params().trait_order
    rows = sample_traits(count, seed)
    return [
        Traits(**{trait: int(value) for trait, value in zip(order, row, strict=True)})
        for row in rows
    ]


def _make_skills(rng: random.Random, num_skills: int) -> tuple[int, ...]:
    """Most students are strong in one or two areas and middling elsewhere."""
    base = [_clamped_normal(rng, 40, 14) for _ in range(num_skills)]
    for index in rng.sample(range(num_skills), k=min(2, num_skills)):
        base[index] = _clamped_normal(rng, 78, 10)
    return tuple(base)


def generate_cohort(
    n: int,
    seed: int,
    *,
    skill_names: tuple[str, ...] = DEFAULT_SKILLS,
    required_skills: tuple[str, ...] | None = None,
    team_size_min: int = 4,
    team_size_max: int = 6,
    num_teams: int | None = None,
    num_mentors: int | None = None,
    female_share: float = 0.35,
    genders: tuple[str, ...] = DEFAULT_GENDERS,
    weights: ObjectiveWeights | None = None,
    config: SolverConfig | None = None,
) -> Cohort:
    """Build a reproducible synthetic cohort of ``n`` students.

    ``num_mentors`` defaults to one more than the minimum number of teams, so the
    mentor pool is a genuine pool rather than a forced one-to-one match.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(seed)

    if required_skills is None:
        required_skills = skill_names[:3]
    min_teams = num_teams if num_teams is not None else math.ceil(n / team_size_max)
    if num_mentors is None:
        num_mentors = min_teams + 1

    # One batch for the whole cohort, so the empirical inter-trait correlations hold
    # across it rather than being redrawn per person.
    traits = _traits_batch(n + num_mentors, seed)

    candidates: list[Candidate] = []
    for index in range(n):
        gender = genders[0] if rng.random() < female_share else genders[1]
        candidates.append(
            Candidate(
                id=f"S{index:04d}",
                traits=traits[index],
                skills=_make_skills(rng, len(skill_names)),
                gender=gender,
                is_mentor=False,
            )
        )

    mentors: list[Candidate] = []
    for index in range(num_mentors):
        gender = genders[0] if rng.random() < 0.5 else genders[1]
        mentors.append(
            Candidate(
                id=f"M{index:03d}",
                traits=traits[n + index],
                skills=_make_skills(rng, len(skill_names)),
                gender=gender,
                is_mentor=True,
            )
        )

    return Cohort(
        candidates=tuple(candidates),
        mentors=tuple(mentors),
        skill_names=skill_names,
        required_skills=tuple(required_skills),
        team_size_min=team_size_min,
        team_size_max=team_size_max,
        num_teams=num_teams,
        weights=weights or ObjectiveWeights(),
        config=config or SolverConfig(),
    )
