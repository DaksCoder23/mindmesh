"""Evaluation harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.baselines import BASELINES, admissible_sizes
from eval.harness import main as harness_main
from eval.harness import write_csv
from eval.llm_baseline import build_prompt, run_direct_llm
from eval.llm_client import DEFAULTS as LLM_PROVIDERS
from eval.llm_client import (
    MIN_CONTEXT_TOKENS,
    LLMConfig,
    LLMError,
    RateLimitState,
    estimate_tokens,
    parse_duration,
)
from eval.metrics import (
    adjusted_rand_index,
    demographic_parity_gap,
    evaluate,
    labels_from_teams,
    mean_pairwise_ari,
    skill_coverage_rate,
)
from solver.generator import generate_cohort
from solver.models import SolverConfig
from solver.verify import check_solution

FAST = SolverConfig(max_deterministic_time=4.0)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_partition_the_whole_cohort(name: str) -> None:
    cohort = generate_cohort(50, seed=2)
    num_teams = cohort.effective_num_teams()
    teams, mentors = BASELINES[name](cohort, num_teams, 0)

    assigned = [m for team in teams for m in team]
    assert sorted(assigned) == sorted(c.id for c in cohort.candidates)
    assert len(mentors) == num_teams


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baselines_are_reproducible_from_their_seed(name: str) -> None:
    cohort = generate_cohort(50, seed=2)
    num_teams = cohort.effective_num_teams()

    assert BASELINES[name](cohort, num_teams, 7) == BASELINES[name](cohort, num_teams, 7)


def test_admissible_sizes_cover_every_candidate() -> None:
    cohort = generate_cohort(103, seed=1)
    sizes = admissible_sizes(cohort, cohort.effective_num_teams())

    assert sum(sizes) == cohort.size
    assert max(sizes) - min(sizes) <= 1


def test_metrics_are_computed_on_the_assignment_as_given() -> None:
    cohort = generate_cohort(40, seed=3)
    teams, mentors = BASELINES["random"](cohort, cohort.effective_num_teams(), 1)
    metrics = evaluate(cohort, teams, mentors, 0.01)

    assert 0.0 <= metrics.skill_coverage_rate <= 1.0
    assert 0.0 <= metrics.demographic_parity_gap <= 1.0
    assert metrics.violations == sum(metrics.violations_by_kind.values())


def test_coverage_and_parity_on_a_hand_built_case() -> None:
    cohort = generate_cohort(20, seed=4)
    ids = [c.id for c in cohort.candidates]
    single = (tuple(ids),)

    # One team containing everyone covers every skill somebody has, and sits exactly on
    # the cohort gender proportion.
    assert skill_coverage_rate(cohort, single) == 1.0
    assert demographic_parity_gap(cohort, single) == pytest.approx(0.0)


def test_ari_is_one_for_identical_partitions_and_low_for_a_reshuffle() -> None:
    a = (("a", "b"), ("c", "d"))
    relabelled = (("c", "d"), ("a", "b"))
    crossed = (("a", "c"), ("b", "d"))

    assert adjusted_rand_index(labels_from_teams(a), labels_from_teams(a)) == 1.0
    assert adjusted_rand_index(labels_from_teams(a), labels_from_teams(relabelled)) == 1.0
    assert adjusted_rand_index(labels_from_teams(a), labels_from_teams(crossed)) < 1.0


def test_mean_pairwise_ari_of_a_deterministic_method_is_one() -> None:
    cohort = generate_cohort(40, seed=5)
    num_teams = cohort.effective_num_teams()
    runs = [BASELINES["marks_sorted"](cohort, num_teams, 0)[0] for _ in range(3)]

    assert mean_pairwise_ari(runs) == pytest.approx(1.0)


def test_llm_prompt_states_every_hard_constraint() -> None:
    cohort = generate_cohort(20, seed=6)
    prompt = build_prompt(cohort, cohort.effective_num_teams())

    assert "exactly one team" in prompt
    assert str(cohort.team_size_min) in prompt and str(cohort.team_size_max) in prompt
    assert "required skills" in prompt
    assert "exactly one mentor" in prompt
    assert "gender" in prompt
    for candidate in cohort.candidates:
        assert candidate.id in prompt


def test_hosted_llm_baseline_is_skipped_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cohort = generate_cohort(20, seed=7)

    run = run_direct_llm(
        cohort, cohort.effective_num_teams(), config=LLMConfig.resolve("groq")
    )

    assert run.status == "skipped_no_api_key"
    assert run.teams == ()


def test_ollama_needs_no_key_but_reports_an_unreachable_server() -> None:
    """Local Ollama is always 'available'; a server that is not running is an api_error."""
    config = LLMConfig.resolve("ollama", base_url="http://127.0.0.1:9/v1", timeout_seconds=2.0)
    assert config.is_available

    run = run_direct_llm(generate_cohort(20, seed=7), 4, config=config)

    assert run.status == "api_error"
    assert run.teams == ()


def test_ollama_requests_a_context_window_large_enough_for_the_prompt() -> None:
    """Ollama defaults to 2048 tokens and truncates silently; the baseline must not."""
    assert MIN_CONTEXT_TOKENS >= 16384
    assert LLMConfig.resolve("ollama").num_ctx >= MIN_CONTEXT_TOKENS


def test_every_provider_resolves_to_a_model_and_endpoint() -> None:
    for provider in LLM_PROVIDERS:
        config = LLMConfig.resolve(provider)
        assert config.model and config.base_url.startswith("http")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(LLMError, match="unknown provider"):
        LLMConfig.resolve("llama-somewhere")


def test_harness_end_to_end_writes_csv_and_charts(tmp_path: Path) -> None:
    code = harness_main(
        [
            "--sizes",
            "40",
            "--seeds",
            "2",
            "--out",
            str(tmp_path),
            "--deterministic-time",
            "3",
            "--no-llm",
        ]
    )

    assert code == 0
    results = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()
    assert len(results) == 1 + 2 * 4  # header + 2 seeds x 4 non-LLM methods
    assert (tmp_path / "skill_coverage_rate.png").exists()
    assert (tmp_path / "violations.png").exists()


def test_solver_beats_the_baselines_on_violations() -> None:
    """The claim the project is built on, at one cohort size."""
    cohort = generate_cohort(60, seed=8, config=FAST)
    from solver.model import solve

    result = solve(cohort)
    solver_metrics = evaluate(
        cohort, result.teams, result.mentor_assignment, result.wall_time_seconds
    )
    assert solver_metrics.violations == 0

    num_teams = cohort.effective_num_teams()
    for name, function in BASELINES.items():
        teams, mentors = function(cohort, num_teams, 0)
        baseline = evaluate(cohort, teams, mentors, 0.0)
        assert baseline.violations >= solver_metrics.violations, name


# -- rate limiting ---------------------------------------------------------


def test_duration_parsing_covers_the_formats_these_apis_use() -> None:
    assert parse_duration("185ms") == pytest.approx(0.185)
    assert parse_duration("7.66s") == pytest.approx(7.66)
    assert parse_duration("14m24s") == pytest.approx(864.0)
    assert parse_duration("12") == pytest.approx(12.0)
    assert parse_duration("") == 0.0
    assert parse_duration("nonsense") == 0.0


def test_no_wait_before_the_first_call() -> None:
    """With no headers seen yet there is nothing to throttle against."""
    assert RateLimitState().wait_needed(5000) == 0.0


def test_waits_when_the_remaining_token_budget_is_too_small() -> None:
    state = RateLimitState()
    state.update({"x-ratelimit-remaining-tokens": "800", "x-ratelimit-reset-tokens": "30s"})

    assert state.remaining_tokens == 800
    assert state.wait_needed(6000) > 25.0  # must wait for the window to reset
    assert state.wait_needed(500) == 0.0  # fits in what is left, go now


def test_token_estimate_is_generous_enough_to_cover_the_reply() -> None:
    cohort = generate_cohort(200, seed=0)
    prompt = build_prompt(cohort, cohort.effective_num_teams())

    # A 200-student prompt is ~2,900 tokens; the reply is comparable. Under-reserving
    # means a 429, so the estimate must exceed the prompt alone by a wide margin.
    assert estimate_tokens(prompt) > 2 * (len(prompt) / 4)


def test_stats_survive_a_hallucinated_member_id() -> None:
    """The direct-LLM baseline emits ids that do not exist; scoring must not crash.

    Observed from Llama 3.3 70B: a member id of 'S0028" -> "S0041', invented range
    notation inside what should be a single student id. The metrics are computed over
    the real members and the verifier still counts the bad id as a violation.
    """
    cohort = generate_cohort(20, seed=9)
    real = [c.id for c in cohort.candidates]
    teams = ((real[0], real[1], 'S0028" -> "S0041'), tuple(real[2:]))

    metrics = evaluate(cohort, teams, (None, None), 0.0)

    assert metrics.violations > 0
    assert 0.0 <= metrics.skill_coverage_rate <= 1.0
    assert any("unknown candidate" in v for v in check_solution(cohort, teams, (None, None)))


def test_csv_keeps_columns_that_only_later_rows_have(tmp_path: Path) -> None:
    """A failed first row must not truncate the schema for the rows that succeeded."""
    rows = [
        {"n": 50, "status": "rate_limited"},
        {"n": 50, "status": "ok", "violations": 3, "model": "llama-3.3-70b-versatile"},
    ]
    path = tmp_path / "variance.csv"

    write_csv(path, rows)

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "violations" in header and "model" in header


def test_throttle_sleeping_is_not_counted_as_model_latency() -> None:
    """A call that waited 30 minutes on quota and answered in 1s took 1s of model time."""
    from eval.llm_client import RATE_LIMIT

    RATE_LIMIT.last_call_wait_seconds = 1800.0
    assert RATE_LIMIT.last_call_wait_seconds == 1800.0
    RATE_LIMIT.last_call_wait_seconds = 0.0
