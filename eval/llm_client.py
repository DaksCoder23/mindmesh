"""One chat call, against whichever provider is configured.

Ollama, Groq, OpenRouter and OpenAI all speak the same ``/v1/chat/completions`` shape,
so they share a single code path; Gemini's native API is the one exception and gets its
own branch. Uses ``urllib`` so the eval package pulls in no HTTP dependency.

Nothing here is ever imported by ``solver/``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, tuple[str, str, str | None]] = {
    # provider: (base_url, default model, env var holding the API key)
    "ollama": ("http://localhost:11434/v1", "llama3.1:8b", None),
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "meta-llama/llama-3.3-70b-instruct:free",
        "OPENROUTER_API_KEY",
    ),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    ),
}

MIN_CONTEXT_TOKENS = 16384
"""Context window requested from local models.

Ollama defaults to 2048 and **silently truncates** anything longer. Measured prompt
sizes are ~930 tokens at n=50, ~1,580 at n=100 and ~2,860 at n=200, with replies of a
comparable size — so the default loses part of the cohort at n=100 and much of the
reply at n=200, and the baseline would then be measuring truncation, not constraint
reasoning. Always send this explicitly.
"""


class LLMError(RuntimeError):
    """Any failure to obtain a response. Carries the provider's own message."""


class RateLimited(LLMError):
    """HTTP 429. ``retry_after`` is the provider's own wait instruction, in seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def parse_duration(text: str) -> float:
    """Parse the duration formats these APIs use: ``185ms``, ``7.66s``, ``14m24s``.

    Returns seconds. Unparseable input yields 0.0 rather than raising — a bad header
    must not take down a benchmark run.
    """
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)  # bare seconds, e.g. Retry-After: 12
    except ValueError:
        pass
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h)", text):
        try:
            number = float(value)
        except ValueError:
            continue
        total += number * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


@dataclass
class RateLimitState:
    """What the provider last told us about our remaining budget.

    Groq's free tier is capped on **tokens per minute**, not just requests, and the
    token budget is what actually binds here: a 200-student prompt plus its reply is
    ~6k tokens against a 12k/minute allowance. Requests alone would never trip it.
    """

    remaining_tokens: float | None = None
    tokens_reset_seconds: float = 0.0
    remaining_requests: float | None = None
    observed_at: float = 0.0
    last_call_wait_seconds: float = 0.0
    """Seconds slept for throttling during the most recent call.

    Kept separate so the recorded latency of a run is the model's, not the quota's. A
    call that waited 30 minutes for a token window and then answered in 1.2s took 1.2s
    of model time, and reporting 1800s would misrepresent the baseline badly.
    """

    def update(self, headers: object) -> None:
        get = getattr(headers, "get", lambda _k, _d=None: None)
        remaining = get("x-ratelimit-remaining-tokens")
        if remaining is not None:
            try:
                self.remaining_tokens = float(remaining)
            except ValueError:
                self.remaining_tokens = None
        requests_left = get("x-ratelimit-remaining-requests")
        if requests_left is not None:
            try:
                self.remaining_requests = float(requests_left)
            except ValueError:
                self.remaining_requests = None
        self.tokens_reset_seconds = parse_duration(get("x-ratelimit-reset-tokens") or "")
        self.observed_at = time.monotonic()

    def wait_needed(self, estimated_tokens: float) -> float:
        """Seconds to sleep before spending ``estimated_tokens``, from the last headers."""
        if self.remaining_tokens is None:
            return 0.0
        if self.remaining_tokens >= estimated_tokens:
            return 0.0
        elapsed = time.monotonic() - self.observed_at
        return max(0.0, self.tokens_reset_seconds - elapsed) + 0.5


def estimate_tokens(prompt: str) -> float:
    """Rough budget for one call: the prompt, plus room for a reply of similar size.

    Deliberately generous. Under-estimating means a 429 and a forced wait anyway, so
    the cheap error is to over-reserve.
    """
    prompt_tokens = len(prompt) / 3.5
    return prompt_tokens * 2.2 + 500


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    num_ctx: int = MIN_CONTEXT_TOKENS
    max_wait_seconds: float = 180.0
    """Longest we will sleep for a rate limit before giving up on the call.

    Groq's free tier answers a exhausted *daily* token budget with Retry-After values
    of 40+ minutes. Sleeping through those turns a benchmark into an overnight job and,
    worse, hides the reason: a run recorded as ``rate_limited`` is data, an hour of
    silent sleeping is not.
    """

    @classmethod
    def resolve(
        cls,
        provider: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
        max_wait_seconds: float = 180.0,
    ) -> LLMConfig:
        if provider not in DEFAULTS:
            raise LLMError(f"unknown provider {provider!r}; choose from {sorted(DEFAULTS)}")
        default_url, default_model, key_var = DEFAULTS[provider]
        api_key = os.environ.get(key_var) if key_var else None
        return cls(
            provider=provider,
            model=model or default_model,
            base_url=(base_url or os.environ.get("LLM_BASE_URL") or default_url).rstrip("/"),
            api_key=api_key,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            max_wait_seconds=max_wait_seconds,
        )

    @property
    def is_available(self) -> bool:
        """Ollama needs no key; every hosted provider does."""
        _, _, key_var = DEFAULTS[self.provider]
        return key_var is None or bool(self.api_key)

    @property
    def missing_key_hint(self) -> str:
        _, _, key_var = DEFAULTS[self.provider]
        return f"{key_var} not set" if key_var else ""


USER_AGENT = "mindmesh-eval/0.1"
"""Sent on every request.

``urllib`` otherwise identifies itself as ``Python-urllib/3.11``, which Cloudflare
fronts (Groq among them) reject with HTTP 403 and error code 1010 — a browser-signature
block that reads exactly like an authentication failure and is not one.
"""


def _post(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            RATE_LIMIT.update(response.headers)
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code == 429:
            wait = parse_duration(exc.headers.get("retry-after") or "")
            if not wait:
                wait = parse_duration(exc.headers.get("x-ratelimit-reset-tokens") or "")
            raise RateLimited(f"rate limited by {url}: {detail}", wait or 20.0) from exc
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"timed out after {timeout}s calling {url}") from exc


RATE_LIMIT = RateLimitState()
"""Shared across calls: the budget is per key, not per request."""

MAX_RATE_LIMIT_RETRIES = 5


def complete_json(config: LLMConfig, prompt: str, *, seed: int = 0) -> str:
    """Send ``prompt``, ask for a JSON object back, return the raw response text.

    Temperature is 0 and a seed is sent where the provider supports one — giving the
    model every chance to be reproducible. Whether it actually is, is the thing being
    measured.

    Respects the provider's published rate limits: waits proactively when the last
    response said the remaining token budget is too small for this call, and honours
    ``Retry-After`` on a 429. Local providers are not throttled.
    """
    if not config.is_available:
        raise LLMError(config.missing_key_hint)

    throttled = config.provider != "ollama"
    RATE_LIMIT.last_call_wait_seconds = 0.0
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        if throttled:
            wait = RATE_LIMIT.wait_needed(estimate_tokens(prompt))
            if wait > config.max_wait_seconds:
                raise RateLimited(
                    f"token budget needs {wait:.0f}s to reset, over the "
                    f"{config.max_wait_seconds:.0f}s limit",
                    wait,
                )
            if wait > 0:
                logger.info("rate limit: waiting %.1fs for the token budget to reset", wait)
                RATE_LIMIT.last_call_wait_seconds += wait
                time.sleep(wait)
        try:
            return _complete_once(config, prompt, seed=seed)
        except RateLimited as exc:
            if attempt == MAX_RATE_LIMIT_RETRIES or exc.retry_after > config.max_wait_seconds:
                raise
            wait = max(exc.retry_after, 1.0)
            RATE_LIMIT.last_call_wait_seconds += wait
            logger.warning(
                "rate limited (attempt %d/%d): sleeping %.1fs",
                attempt + 1,
                MAX_RATE_LIMIT_RETRIES,
                wait,
            )
            time.sleep(wait)
    raise LLMError("unreachable")


def _complete_once(config: LLMConfig, prompt: str, *, seed: int = 0) -> str:
    """One attempt, no retry handling.""" 
    if config.provider == "gemini":
        url = (
            f"{config.base_url}/models/{config.model}:generateContent"
            f"?key={config.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": config.temperature,
                "seed": seed,
            },
        }
        data = _post(url, payload, {}, config.timeout_seconds)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected Gemini response shape: {json.dumps(data)[:500]}") from exc

    # OpenAI-compatible: ollama, groq, openrouter, openai
    payload: dict[str, object] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "response_format": {"type": "json_object"},
        "seed": seed,
    }
    if config.provider == "ollama":
        # See MIN_CONTEXT_TOKENS: without this the prompt is silently truncated.
        payload["options"] = {"num_ctx": config.num_ctx, "seed": seed}

    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    data = _post(
        f"{config.base_url}/chat/completions", payload, headers, config.timeout_seconds
    )
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected response shape: {json.dumps(data)[:500]}") from exc
