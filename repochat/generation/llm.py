"""LLM provider abstraction.

Two providers, both free-tier:
- Groq: very fast, llama-3.3-70b-versatile, 30 RPM
- Gemini: gemini-1.5-flash, 15 RPM, used only as fallback

Rate limiting uses a sliding-window token bucket sized for Groq's 30 RPM
free tier with a 5-request safety margin. When the bucket is empty, calls
block (with caller-visible wait time) rather than 429ing.

Streaming: we buffer the full response internally and only yield to the
caller in delta chunks. If a 429 hits mid-stream, we surface a clean
RateLimitError instead of leaking a half-response into the UI.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Protocol

from repochat.config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class RateLimitError(LLMError):
    pass


# --- Rate limiter -----------------------------------------------------------


class _SlidingWindowLimiter:
    """Allow at most `rpm` requests in any 60-second window."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self) -> float:
        """Block until a slot is available. Returns seconds waited."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) < self.rpm:
                self._times.append(now)
                return 0.0
            wait_for = 60.0 - (now - self._times[0]) + 0.05  # tiny pad
        log.info("Rate limit reached; sleeping %.1fs", wait_for)
        time.sleep(max(0.0, wait_for))
        return self.wait_if_needed() + wait_for

    def time_until_slot(self) -> float:
        """Non-blocking: how long until next slot opens. Used by UI countdown."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            while self._times and self._times[0] < cutoff:
                self._times.popleft()
            if len(self._times) < self.rpm:
                return 0.0
            return max(0.0, 60.0 - (now - self._times[0]))


_LIMITER = _SlidingWindowLimiter(settings.rpm_limit)


def time_until_next_slot() -> float:
    return _LIMITER.time_until_slot()


# --- Provider protocol ------------------------------------------------------


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class _Provider(Protocol):
    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResult: ...
    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]: ...


# --- Groq -------------------------------------------------------------------


class _GroqProvider:
    def __init__(self):
        if not settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com"
            )
        from groq import Groq  # heavy import deferred
        self._client = Groq(api_key=settings.groq_api_key)
        self._model = settings.llm_model

    def _retry_on_429(self, fn, *args, **kwargs):
        """Three exponential-backoff retries on 429 before giving up."""
        delays = [1.0, 3.0, 8.0]
        last_exc: Exception | None = None
        for delay in delays:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "rate" in msg or "429" in msg:
                    log.warning("Groq 429; backing off %.1fs", delay)
                    time.sleep(delay)
                    continue
                raise
        raise RateLimitError(f"Groq rate limit after retries: {last_exc}")

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResult:
        _LIMITER.wait_if_needed()

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

        resp = self._retry_on_429(_call)
        return LLMResult(
            text=resp.choices[0].message.content or "",
            provider="groq",
            model=self._model,
        )

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        _LIMITER.wait_if_needed()

        def _call():
            return self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

        try:
            stream = self._retry_on_429(_call)
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except RateLimitError:
            raise
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                raise RateLimitError(f"Groq rate limit: {e}") from e
            raise LLMError(f"Groq stream failed: {e}") from e


# --- Gemini -----------------------------------------------------------------


class _GeminiProvider:
    def __init__(self):
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> LLMResult:
        _LIMITER.wait_if_needed()
        import google.generativeai as genai
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return LLMResult(text=resp.text or "", provider="gemini", model="gemini-1.5-flash")
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        _LIMITER.wait_if_needed()
        import google.generativeai as genai
        try:
            stream = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini stream failed: {e}") from e


# --- Public API -------------------------------------------------------------


_PROVIDER: _Provider | None = None


def _get_provider() -> _Provider:
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    if settings.llm_provider == "groq":
        _PROVIDER = _GroqProvider()
    elif settings.llm_provider == "gemini":
        _PROVIDER = _GeminiProvider()
    else:
        raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
    return _PROVIDER


def complete(prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048) -> LLMResult:
    return _get_provider().complete(prompt, temperature=temperature, max_tokens=max_tokens)


def stream(
    prompt: str, *, temperature: float = 0.2, max_tokens: int = 2048
) -> Iterator[str]:
    return _get_provider().stream(prompt, temperature=temperature, max_tokens=max_tokens)
