"""Direct-LLM baseline: hand the whole cohort to a language model and ask for the teams.

Provider-agnostic — Ollama (local, free), Groq, OpenRouter, OpenAI or Gemini, selected
with ``--llm-provider``. See :mod:`eval.llm_client`, and note in particular the context
length warning for local models.

This is the baseline the project's design decision rests on, so it is deliberately
given a *fair* run: the full constraint set in the prompt, structured JSON output, and
no post-processing. Whatever it returns is scored exactly as returned.

Nothing here is ever imported by ``solver/``. This module exists only to be beaten.

**No simulated fallback.** If the API key is missing the run is recorded as skipped.
Inventing a failure rate and then citing it as evidence would make the result
worthless.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from eval.llm_client import RATE_LIMIT, LLMConfig, LLMError, RateLimited, complete_json
from solver.models import Cohort

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "ollama"


@dataclass(frozen=True)
class LLMRun:
    """One direct-LLM attempt, including the ways it can fail short of an answer."""

    teams: tuple[tuple[str, ...], ...]
    mentor_assignment: tuple[str | None, ...]
    wall_time_seconds: float
    status: str  # "ok" | "skipped_no_api_key" | "parse_error" | "api_error" | "rate_limited"
    detail: str = ""
    hallucinated_ids: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()
    model: str = ""


def build_prompt(cohort: Cohort, num_teams: int) -> str:
    """The full problem, fully specified. No hints about how to search."""
    threshold = cohort.config.skill_threshold
    tolerance = cohort.config.gender_tolerance
    lines = [
        "You are allocating students to project teams.",
        "",
        f"Partition ALL {cohort.size} students into exactly {num_teams} teams.",
        "",
        "HARD CONSTRAINTS, all of which must hold:",
        "1. Every student appears in exactly one team. Use only the ids listed below.",
        f"2. Each team has between {cohort.team_size_min} and {cohort.team_size_max} students.",
        f"3. For each of these required skills {list(cohort.required_skills)}, every team "
        f"must contain at least one student scoring {threshold} or above on that skill.",
        "4. Each team gets exactly one mentor from the mentor pool. No mentor leads two teams.",
        f"5. For each gender, each team's share must be within {tolerance:.2f} of that "
        "gender's share of the whole cohort.",
        "",
        "OBJECTIVE, to maximise subject to the constraints:",
        "- the minimum across teams of team mean conscientiousness",
        "- the sum across teams of team minimum agreeableness",
        "- minus the within-team agreeableness spread (max minus min)",
        "- minus the spread of total technical skill across teams",
        "",
        f"SKILL ORDER: {list(cohort.skill_names)}",
        "",
        "STUDENTS (id | gender | conscientiousness | agreeableness | skills):",
    ]
    for candidate in cohort.candidates:
        lines.append(
            f"{candidate.id} | {candidate.gender} | "
            f"{candidate.traits.conscientiousness} | {candidate.traits.agreeableness} | "
            f"{list(candidate.skills)}"
        )
    lines += ["", "MENTOR POOL:", ", ".join(m.id for m in cohort.mentors), ""]
    lines.append(
        "Respond with JSON only, no prose: "
        '{"teams": [{"members": ["S0001", ...], "mentor": "M000"}, ...]}'
    )
    return "\n".join(lines)


def _parse(payload: str) -> tuple[tuple[tuple[str, ...], ...], tuple[str | None, ...]]:
    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    data = json.loads(match.group(0))
    teams, mentors = [], []
    for entry in data["teams"]:
        teams.append(tuple(str(m) for m in entry.get("members", [])))
        mentor = entry.get("mentor")
        mentors.append(str(mentor) if mentor else None)
    return tuple(teams), tuple(mentors)


def run_direct_llm(
    cohort: Cohort,
    num_teams: int,
    *,
    config: LLMConfig | None = None,
    seed: int = 0,
) -> LLMRun:
    """One direct-LLM allocation attempt. Output is scored as returned, never repaired."""
    config = config or LLMConfig.resolve("ollama")
    if not config.is_available:
        return LLMRun(
            (), (), 0.0, "skipped_no_api_key", config.missing_key_hint, model=config.model
        )

    prompt = build_prompt(cohort, num_teams)
    started = time.monotonic()
    try:
        text = complete_json(config, prompt, seed=seed)
    except RateLimited as exc:
        return LLMRun(
            (),
            (),
            time.monotonic() - started,
            "rate_limited",
            str(exc),
            model=config.model,
        )
    except LLMError as exc:
        return LLMRun(
            (), (), time.monotonic() - started, "api_error", str(exc), model=config.model
        )
    # Throttle sleeping is not model latency. See RateLimitState.last_call_wait_seconds.
    elapsed = max(0.0, time.monotonic() - started - RATE_LIMIT.last_call_wait_seconds)

    try:
        teams, mentors = _parse(text)
    except Exception as exc:
        return LLMRun((), (), elapsed, "parse_error", str(exc), model=config.model)

    known = {c.id for c in cohort.candidates}
    returned = {member for team in teams for member in team}
    return LLMRun(
        teams=teams,
        mentor_assignment=mentors,
        wall_time_seconds=elapsed,
        status="ok",
        hallucinated_ids=tuple(sorted(returned - known)),
        missing_ids=tuple(sorted(known - returned)),
        model=config.model,
    )
