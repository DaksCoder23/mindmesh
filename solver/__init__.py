"""MindMesh AI solver package.

The team assignment is produced here and nowhere else. No module in this package may
import an LLM client or make a network call.
"""

from solver.generator import generate_cohort
from solver.model import solve
from solver.models import (
    Candidate,
    Cohort,
    ConscientiousnessMode,
    ObjectiveWeights,
    SolverConfig,
    SolveResult,
    TeamStats,
    Traits,
)
from solver.stats import composition_score, compute_team_stats
from solver.verify import check_solution

__all__ = [
    "Candidate",
    "Cohort",
    "ConscientiousnessMode",
    "ObjectiveWeights",
    "SolveResult",
    "SolverConfig",
    "TeamStats",
    "Traits",
    "check_solution",
    "composition_score",
    "compute_team_stats",
    "generate_cohort",
    "solve",
]
