"""Command line entry point: solve a synthetic cohort and print the result.

python -m solver.cli --n 100 --seed 7 --deterministic-time 20
"""

from __future__ import annotations

import argparse
import logging
import sys

from solver.generator import generate_cohort
from solver.model import solve
from solver.models import ConscientiousnessMode, ObjectiveWeights, SolverConfig
from solver.verify import check_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MindMesh AI CP-SAT team solver")
    parser.add_argument("--n", type=int, default=100, help="number of students")
    parser.add_argument("--seed", type=int, default=42, help="cohort and solver seed")
    parser.add_argument("--team-size-min", type=int, default=4)
    parser.add_argument("--team-size-max", type=int, default=6)
    parser.add_argument("--skill-threshold", type=int, default=60)
    parser.add_argument("--gender-tolerance", type=float, default=0.30)
    parser.add_argument("--time-limit", type=float, default=60.0, help="wall clock seconds")
    parser.add_argument(
        "--deterministic-time",
        type=float,
        default=None,
        help="deterministic time limit; use this instead of --time-limit for reproducibility",
    )
    parser.add_argument(
        "--conscientiousness-mode",
        choices=[m.value for m in ConscientiousnessMode],
        default=ConscientiousnessMode.MIN_MEAN.value,
    )
    parser.add_argument("--w-conscientiousness", type=float, default=1.0)
    parser.add_argument("--w-agreeableness-min", type=float, default=1.0)
    parser.add_argument("--w-agreeableness-spread", type=float, default=0.5)
    parser.add_argument("--w-skill-balance", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true", help="log CP-SAT search progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = SolverConfig(
        skill_threshold=args.skill_threshold,
        gender_tolerance=args.gender_tolerance,
        random_seed=args.seed,
        max_time_seconds=args.time_limit,
        max_deterministic_time=args.deterministic_time,
        conscientiousness_mode=ConscientiousnessMode(args.conscientiousness_mode),
        log_search_progress=args.verbose,
    )
    weights = ObjectiveWeights(
        conscientiousness=args.w_conscientiousness,
        agreeableness_min=args.w_agreeableness_min,
        agreeableness_spread=args.w_agreeableness_spread,
        skill_balance=args.w_skill_balance,
    )
    cohort = generate_cohort(
        args.n,
        args.seed,
        team_size_min=args.team_size_min,
        team_size_max=args.team_size_max,
        weights=weights,
        config=config,
    )

    result = solve(cohort)

    print(f"status            {result.status}")
    print(f"teams             {result.num_teams}")
    print(f"objective         {result.objective_value}")
    print(f"best bound        {result.best_bound}")
    gap = result.optimality_gap
    print(f"optimality gap    {'n/a' if gap is None else f'{gap:.4f}'}")
    print(f"wall clock        {result.wall_time_seconds:.3f}s")
    print(f"seed / workers    {result.seed} / {result.num_search_workers}")

    if not result.is_feasible:
        print("\nNo assignment returned.")
        return 1

    print("\nobjective terms")
    for name, value in result.objective_terms.items():
        print(f"  {name:<28} {value:.2f}")

    print("\nteams")
    for stats in result.team_stats:
        genders = ", ".join(f"{g}={c}" for g, c in sorted(stats.gender_counts.items()))
        print(
            f"  team {stats.team_index:>2}  n={stats.size}  mentor={stats.mentor_id}  "
            f"meanC={stats.mean_conscientiousness:.1f}  minA={stats.min_agreeableness}  "
            f"spreadA={stats.agreeableness_spread}  skill={stats.total_skill}  {genders}"
        )
        print(f"            {', '.join(stats.member_ids)}")

    violations = check_solution(cohort, result.teams, result.mentor_assignment)
    print(f"\nindependent verification: {len(violations)} violations")
    for violation in violations:
        print(f"  {violation}")
    return 0 if not violations else 2


if __name__ == "__main__":
    sys.exit(main())
