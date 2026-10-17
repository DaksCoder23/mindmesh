"""Benchmark the CP-SAT solver against the four baselines.

    python -m eval.harness --sizes 50 100 200 --seeds 20 --out eval/results

Writes ``results.csv``, ``llm_variance.csv`` and charts.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval.baselines import BASELINES
from eval.llm_baseline import DEFAULT_PROVIDER, run_direct_llm
from eval.llm_client import DEFAULTS as LLM_PROVIDERS
from eval.llm_client import LLMConfig
from eval.metrics import RunMetrics, evaluate, mean_pairwise_ari
from solver.generator import generate_cohort
from solver.model import solve
from solver.models import Cohort, SolverConfig

logger = logging.getLogger(__name__)

METHODS = ("cpsat", "random", "self_selection", "marks_sorted", "direct_llm")


@dataclass(frozen=True)
class Row:
    method: str
    n: int
    seed: int
    status: str
    skill_coverage_rate: float
    composition_score: float
    demographic_parity_gap: float
    violations: int
    violations_assignment: int
    violations_team_size: int
    violations_skill_coverage: int
    violations_mentor: int
    violations_gender_parity: int
    wall_time_seconds: float
    optimality_gap: float | None
    model: str = ""
    """Which model produced this row. Empty for every method except direct_llm.

    Quotas are per-model and models get swapped mid-experiment, so a results file
    without this column cannot be interpreted later.
    """


def _row(
    method: str,
    n: int,
    seed: int,
    status: str,
    metrics: RunMetrics,
    gap: float | None,
    model: str = "",
) -> Row:
    kinds = metrics.violations_by_kind
    return Row(
        method=method,
        n=n,
        seed=seed,
        status=status,
        skill_coverage_rate=metrics.skill_coverage_rate,
        composition_score=metrics.composition_score,
        demographic_parity_gap=metrics.demographic_parity_gap,
        violations=metrics.violations,
        violations_assignment=kinds.get("assignment", 0),
        violations_team_size=kinds.get("team_size", 0),
        violations_skill_coverage=kinds.get("skill_coverage", 0),
        violations_mentor=kinds.get("mentor", 0),
        violations_gender_parity=kinds.get("gender_parity", 0),
        wall_time_seconds=metrics.wall_time_seconds,
        optimality_gap=gap,
        model=model,
    )


def _empty_row(
    method: str, n: int, seed: int, status: str, wall: float, model: str = ""
) -> Row:
    """A run that produced no assignment at all still occupies a row."""
    return Row(
        method=method,
        n=n,
        seed=seed,
        status=status,
        skill_coverage_rate=0.0,
        composition_score=0.0,
        demographic_parity_gap=0.0,
        violations=-1,
        violations_assignment=0,
        violations_team_size=0,
        violations_skill_coverage=0,
        violations_mentor=0,
        violations_gender_parity=0,
        wall_time_seconds=wall,
        optimality_gap=None,
        model=model,
    )


def run_cohort(cohort: Cohort, n: int, seed: int, *, llm: LLMConfig | None) -> list[Row]:
    num_teams = cohort.effective_num_teams()
    rows: list[Row] = []

    result = solve(cohort)
    if result.is_feasible:
        metrics = evaluate(cohort, result.teams, result.mentor_assignment, result.wall_time_seconds)
        rows.append(_row("cpsat", n, seed, result.status, metrics, result.optimality_gap))
    else:
        rows.append(_empty_row("cpsat", n, seed, result.status, result.wall_time_seconds))

    for name, function in BASELINES.items():
        started = time.monotonic()
        teams, mentors = function(cohort, num_teams, seed)
        elapsed = time.monotonic() - started
        rows.append(_row(name, n, seed, "ok", evaluate(cohort, teams, mentors, elapsed), None))

    if llm is not None:
        run = run_direct_llm(cohort, num_teams, config=llm, seed=seed)
        if run.status == "ok":
            metrics = evaluate(cohort, run.teams, run.mentor_assignment, run.wall_time_seconds)
            rows.append(_row("direct_llm", n, seed, run.status, metrics, None, run.model))
        else:
            logger.warning("direct_llm n=%d seed=%d: %s %s", n, seed, run.status, run.detail)
            rows.append(
                _empty_row(
                    "direct_llm", n, seed, run.status, run.wall_time_seconds, run.model
                )
            )
    return rows


def measure_llm_variance(
    cohort: Cohort, n: int, repeats: int, *, llm: LLMConfig
) -> list[dict[str, object]]:
    """Repeat the direct-LLM baseline on *identical* input and record the disagreement.

    Every other method in this harness is deterministic given its seed. If the LLM is
    not, it shows up here as ARI < 1.0 against itself.
    """
    num_teams = cohort.effective_num_teams()
    partitions: list[tuple[tuple[str, ...], ...]] = []
    records: list[dict[str, object]] = []

    for repeat in range(repeats):
        run = run_direct_llm(cohort, num_teams, config=llm, seed=0)
        if run.status != "ok":
            records.append({"n": n, "repeat": repeat, "status": run.status, "violations": -1})
            continue
        metrics = evaluate(cohort, run.teams, run.mentor_assignment, run.wall_time_seconds)
        partitions.append(run.teams)
        records.append(
            {
                "n": n,
                "repeat": repeat,
                "status": run.status,
                "violations": metrics.violations,
                "skill_coverage_rate": round(metrics.skill_coverage_rate, 4),
                "composition_score": round(metrics.composition_score, 2),
                "hallucinated_ids": len(run.hallucinated_ids),
                "missing_ids": len(run.missing_ids),
                "wall_time_seconds": round(run.wall_time_seconds, 3),
                "model": run.model,
            }
        )

    ari = mean_pairwise_ari(partitions)
    for record in records:
        record["mean_pairwise_ari"] = round(ari, 4)
        record["repeats_parsed"] = len(partitions)
    return records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write ``rows``, keeping every column any row has.

    Fields are the union across all rows, not the keys of the first one: a run that
    failed early carries fewer keys than a successful one, so taking row 0's keys
    silently drops real columns whenever the first row happens to be a failure.
    """
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d rows)", path, len(rows))


def summarise(rows: list[Row]) -> str:
    """Mean of each metric by method and n, for the console."""
    buckets: dict[tuple[str, int], list[Row]] = {}
    for row in rows:
        buckets.setdefault((row.method, row.n), []).append(row)

    lines = [
        f"{'method':<16}{'n':>5}{'coverage':>11}{'composition':>13}"
        f"{'parity gap':>12}{'violations':>12}{'seconds':>10}"
    ]
    for (method, n), group in sorted(buckets.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        scored = [r for r in group if r.violations >= 0]
        if not scored:
            lines.append(f"{method:<16}{n:>5}{'  (no successful runs)':>46}")
            continue
        count = len(scored)
        lines.append(
            f"{method:<16}{n:>5}"
            f"{sum(r.skill_coverage_rate for r in scored) / count:>11.3f}"
            f"{sum(r.composition_score for r in scored) / count:>13.1f}"
            f"{sum(r.demographic_parity_gap for r in scored) / count:>12.3f}"
            f"{sum(r.violations for r in scored) / count:>12.2f}"
            f"{sum(r.wall_time_seconds for r in scored) / count:>10.2f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MindMesh AI evaluation harness")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--seeds", type=int, default=20, help="seeds per size")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    parser.add_argument(
        "--deterministic-time",
        type=float,
        default=10.0,
        help="CP-SAT deterministic time limit per run; keeps the grid reproducible",
    )
    parser.add_argument("--team-size-min", type=int, default=4)
    parser.add_argument("--team-size-max", type=int, default=6)
    parser.add_argument(
        "--llm-provider",
        choices=sorted(LLM_PROVIDERS),
        default=DEFAULT_PROVIDER,
        help="ollama runs locally and needs no key; the rest need their provider's key",
    )
    parser.add_argument("--llm-model", default=None, help="defaults to the provider's model")
    parser.add_argument("--llm-base-url", default=None, help="override the provider endpoint")
    parser.add_argument(
        "--llm-max-wait",
        type=float,
        default=180.0,
        help="longest rate-limit sleep per call; beyond it the run is recorded as rate_limited",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="skip the direct-LLM baseline entirely"
    )
    parser.add_argument(
        "--llm-variance-repeats",
        type=int,
        default=5,
        help="repeats of the direct-LLM baseline on identical input",
    )
    parser.add_argument("--no-charts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    llm: LLMConfig | None = None
    if not args.no_llm:
        llm = LLMConfig.resolve(
            args.llm_provider,
            model=args.llm_model,
            base_url=args.llm_base_url,
            max_wait_seconds=args.llm_max_wait,
        )
        if llm.is_available:
            logger.info("direct_llm: provider=%s model=%s", llm.provider, llm.model)
        else:
            logger.warning(
                "%s: direct_llm rows will be recorded as skipped and excluded from the "
                "charts. No simulated stand-in is substituted.",
                llm.missing_key_hint,
            )

    config = SolverConfig(max_deterministic_time=args.deterministic_time)
    rows: list[Row] = []
    for n in args.sizes:
        for seed in range(args.seeds):
            cohort = generate_cohort(
                n,
                seed,
                team_size_min=args.team_size_min,
                team_size_max=args.team_size_max,
                config=config,
            )
            rows.extend(run_cohort(cohort, n, seed, llm=llm))
            logger.info("done n=%d seed=%d", n, seed)

    write_csv(args.out / "results.csv", [asdict(r) for r in rows])
    print("\n" + summarise(rows) + "\n")

    if llm is not None and args.llm_variance_repeats > 1:
        variance_rows: list[dict[str, object]] = []
        for n in args.sizes:
            cohort = generate_cohort(
                n,
                0,
                team_size_min=args.team_size_min,
                team_size_max=args.team_size_max,
                config=config,
            )
            variance_rows.extend(
                measure_llm_variance(cohort, n, args.llm_variance_repeats, llm=llm)
            )
        write_csv(args.out / "llm_variance.csv", variance_rows)

    if not args.no_charts:
        from eval.plots import make_charts

        make_charts([asdict(r) for r in rows], args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
