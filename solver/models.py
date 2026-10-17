"""Data model for MindMesh AI team partitioning.

Everything here is plain data. No solver, no I/O, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

TRAIT_NAMES: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)


@dataclass(frozen=True)
class Traits:
    """Big Five trait scores, each an integer 0-100."""

    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int

    def __post_init__(self) -> None:
        for name in TRAIT_NAMES:
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(f"trait {name} must be an int in 0..100, got {value!r}")


@dataclass(frozen=True)
class Candidate:
    """A student or a mentor.

    ``skills`` is positionally aligned with :attr:`Cohort.skill_names`.
    """

    id: str
    traits: Traits
    skills: tuple[int, ...]
    gender: str
    is_mentor: bool = False

    def __post_init__(self) -> None:
        for value in self.skills:
            if not isinstance(value, int) or not 0 <= value <= 100:
                raise ValueError(f"skill scores must be ints in 0..100, got {value!r}")

    @property
    def mean_skill(self) -> int:
        """Mean of the skill vector, rounded to an integer 0-100."""
        if not self.skills:
            return 0
        return round(sum(self.skills) / len(self.skills))


class ConscientiousnessMode(StrEnum):
    """How the conscientiousness objective term is formulated.

    ``MIN_MEAN`` maximises the minimum team mean conscientiousness. ``SUM`` reproduces
    the originally specified "maximise summed team conscientiousness", which is a
    constant under a whole-cohort partition and therefore has no effect on the chosen
    solution. See ``openspec/changes/add-cpsat-team-solver/design.md`` §1.
    """

    MIN_MEAN = "min_mean"
    SUM = "sum"


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights on the four objective terms. Converted to integer coefficients by
    ``round(weight * 1000)`` when the model is built."""

    conscientiousness: float = 1.0
    agreeableness_min: float = 1.0
    agreeableness_spread: float = 0.5
    skill_balance: float = 0.5

    def __post_init__(self) -> None:
        # The objective terms use one-sided channelling: each auxiliary variable is
        # pinned to its true value only because the objective pushes it there. A
        # negative weight would reverse that push and the reported term would no longer
        # mean what its name says. See design.md §3 and §7.
        for name in (
            "conscientiousness",
            "agreeableness_min",
            "agreeableness_spread",
            "skill_balance",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"objective weight {name} must be >= 0")


@dataclass(frozen=True)
class SolverConfig:
    """Solver knobs. Defaults are the deterministic configuration."""

    skill_threshold: int = 60
    gender_tolerance: float = 0.30
    random_seed: int = 42
    num_search_workers: int = 1
    max_time_seconds: float = 120.0
    max_deterministic_time: float | None = None
    conscientiousness_mode: ConscientiousnessMode = ConscientiousnessMode.MIN_MEAN
    log_search_progress: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.skill_threshold <= 100:
            raise ValueError("skill_threshold must be in 0..100")
        if not 0.0 <= self.gender_tolerance <= 1.0:
            raise ValueError("gender_tolerance must be in 0.0..1.0")
        if self.num_search_workers != 1:
            # Not fatal: determinism is only guaranteed single-threaded, and the caller
            # may knowingly trade it away. The result records the choice.
            pass


@dataclass(frozen=True)
class Cohort:
    """A cohort to partition.

    ``candidates`` are students; ``mentors`` are a separate pool. Mentors do not count
    towards team size.
    """

    candidates: tuple[Candidate, ...]
    mentors: tuple[Candidate, ...]
    skill_names: tuple[str, ...]
    required_skills: tuple[str, ...]
    team_size_min: int
    team_size_max: int
    num_teams: int | None = None
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    config: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("cohort has no candidates")
        if self.team_size_min < 1:
            raise ValueError("team_size_min must be >= 1")
        if self.team_size_max < self.team_size_min:
            raise ValueError("team_size_max must be >= team_size_min")
        for skill in self.required_skills:
            if skill not in self.skill_names:
                raise ValueError(f"required skill {skill!r} not in skill_names")
        for candidate in self.candidates + self.mentors:
            if len(candidate.skills) != len(self.skill_names):
                raise ValueError(
                    f"candidate {candidate.id} has {len(candidate.skills)} skill scores, "
                    f"expected {len(self.skill_names)}"
                )
        if any(not c.is_mentor for c in self.mentors):
            raise ValueError("every member of the mentor pool must have is_mentor=True")
        if any(c.is_mentor for c in self.candidates):
            raise ValueError("students must have is_mentor=False")

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def genders(self) -> tuple[str, ...]:
        return tuple(sorted({c.gender for c in self.candidates}))

    def gender_proportion(self, gender: str) -> float:
        return sum(1 for c in self.candidates if c.gender == gender) / self.size

    def required_skill_indices(self) -> tuple[int, ...]:
        return tuple(self.skill_names.index(s) for s in self.required_skills)

    def effective_num_teams(self) -> int:
        """Team count to build the model with.

        Uses ``num_teams`` when supplied. Otherwise the smallest count that can hold the
        cohort at the maximum team size. A count that cannot work (too few mentors, or
        size bounds that do not admit the cohort) is left to the solver, which reports
        INFEASIBLE rather than silently repairing the input.
        """
        if self.num_teams is not None:
            return self.num_teams
        return max(1, math.ceil(self.size / self.team_size_max))


@dataclass(frozen=True)
class TeamStats:
    """Composition statistics for one team."""

    team_index: int
    member_ids: tuple[str, ...]
    mentor_id: str | None
    size: int
    mean_conscientiousness: float
    min_agreeableness: int
    max_agreeableness: int
    agreeableness_spread: int
    mean_agreeableness: float
    total_skill: int
    gender_counts: dict[str, int]
    covered_skills: tuple[str, ...]


@dataclass(frozen=True)
class SolveResult:
    """Everything a run produced, including the parts you need to trust it."""

    status: str
    teams: tuple[tuple[str, ...], ...]
    mentor_assignment: tuple[str | None, ...]
    objective_value: float | None
    best_bound: float | None
    wall_time_seconds: float
    hit_limit: bool
    num_teams: int
    seed: int
    num_search_workers: int
    team_stats: tuple[TeamStats, ...] = ()
    objective_terms: dict[str, float] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE")

    @property
    def optimality_gap(self) -> float | None:
        """Relative gap between the incumbent and the best bound, or ``None``.

        0.0 means proven optimal. This is a maximisation problem, so the bound is above
        the incumbent and the gap is ``(bound - value) / |bound|``.
        """
        if self.objective_value is None or self.best_bound is None:
            return None
        if self.best_bound == 0:
            return 0.0 if self.objective_value == 0 else float("inf")
        return abs(self.best_bound - self.objective_value) / abs(self.best_bound)

    def team_of(self, candidate_id: str) -> int | None:
        for index, team in enumerate(self.teams):
            if candidate_id in team:
                return index
        return None
