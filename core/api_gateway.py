"""Centralized API gateway: adaptive rate limiting, circuit breaking, and retry for all external APIs.

Design (v2):
  - TokenBucket: classic fixed-rate bucket (kept for direct-test compatibility).
  - AdaptiveTokenBucket: AIMD controller on top — the effective rate is cut
    multiplicatively whenever the provider signals throttling (HTTP 429 / 503
    with Retry-After) and recovers additively after sustained successes. This
    replaces static "1 req / 3s forever" pacing with self-tuning throughput.
  - CircuitBreaker: unchanged CLOSED -> OPEN -> HALF_OPEN machine.
  - APIGateway.request: adds +/-20% jitter to all backoff waits, optional
    single-flight coalescing for identical read-only calls, and
    is_available()/breaker_state() probes so callers can skip providers whose
    breaker is open instead of paying retry sleeps.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""
    def __init__(self, provider: str, cooldown_remaining: float):
        self.provider = provider
        self.cooldown_remaining = cooldown_remaining
        super().__init__(f"Circuit breaker open for {provider}, {cooldown_remaining:.0f}s remaining")


class RateLimitError(Exception):
    """Raised when a request is rate-limited by the provider (HTTP 429)."""
    def __init__(self, provider: str, retry_after: float = 5.0):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {provider}, retry after {retry_after:.1f}s")


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    Tokens are generated at a fixed rate. Threads block on acquire() until
    a token is available. No sleep-outside-lock bugs.
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens generated per second (sustained throughput).
            burst: Maximum tokens in the bucket (allows short bursts).
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """Block until a token is available. Returns True if acquired, False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self.rate
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(wait, remaining))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.rate)
        self._last_refill = now


class AdaptiveTokenBucket(TokenBucket):
    """Token bucket with AIMD-style rate adaptation.

    Starts at the provider's documented sustainable rate. When the provider
    signals throttling (429/503 with Retry-After), the effective rate is cut
    multiplicatively (floor 0.05 req/s) and held for a cooldown window.
    After 5 consecutive successes the rate recovers additively toward base.
    """

    MIN_RATE = 0.05

    def __init__(self, rate: float, burst: int = 1):
        super().__init__(rate=rate, burst=burst)
        self.base_rate = float(rate)
        self._penalized_until = 0.0
        self._success_streak = 0
        self._recover_step = max(rate * 0.1, 0.05)

    def current_rate(self) -> float:
        with self._lock:
            return self.rate

    def penalize(self, factor: float = 0.5, cooldown: float = 30.0):
        """Multiplicative decrease (the 'MD' in AIMD)."""
        with self._lock:
            factor = max(0.05, min(1.0, float(factor)))
            self.rate = max(self.MIN_RATE, self.rate * factor)
            self._penalized_until = time.monotonic() + max(0.0, cooldown)
            self._success_streak = 0

    def reward(self):
        """Additive increase toward the base rate after sustained success."""
        with self._lock:
            if self.rate >= self.base_rate:
                return
            if time.monotonic() < self._penalized_until:
                return
            self._success_streak += 1
            if self._success_streak >= 5:
                self.rate = min(self.base_rate, self.rate + self._recover_step)
                self._success_streak = 0


class CircuitBreaker:
    """Thread-safe circuit breaker with CLOSED → OPEN → HALF_OPEN state machine.

    After `threshold` consecutive failures, opens for `cooldown` seconds.
    On cooldown expiry, allows one probe request. If it succeeds, closes;
    if it fails, reopens.

    acquire() never sleeps — it raises CircuitOpen immediately so the caller
    can handle the wait externally.  This avoids blocking threads while the
    breaker is open.
    """

    def __init__(self, provider: str = "unknown", threshold: int = 5, cooldown: float = 60.0):
        self.provider = provider
        self.threshold = threshold
        self.cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() >= self._open_until:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    def acquire(self):
        """Raise CircuitOpen immediately if breaker is blocking. Never sleeps."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() >= self._open_until:
                    self._state = CircuitState.HALF_OPEN
                else:
                    remaining = self._open_until - time.monotonic()
                    raise CircuitOpen(self.provider, remaining)

    def record_success(self):
        with self._lock:
            self._consecutive_failures = 0
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker closed for %s (recovered)", self.provider)

    def record_failure(self):
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self._state = CircuitState.OPEN
                self._open_until = time.monotonic() + self.cooldown
                logger.warning(
                    "Circuit breaker OPENED for %s — %d consecutive failures, pausing for %.0fs",
                    self.provider, self._consecutive_failures, self.cooldown,
                )

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN


class SourceHealthTracker:
    """Tracks per-provider success/failure rates over a sliding window."""

    def __init__(self, window: int = 10):
        self._window = window
        self._results: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, provider: str, success: bool):
        self._results[provider].append(success)

    def success_rate(self, provider: str) -> float:
        results = self._results[provider]
        if not results:
            return 1.0
        return sum(results) / len(results)


class APIGateway:
    """Centralized rate limiting, circuit breaking, and retry for all external APIs.

    Usage:
        gateway = APIGateway()
        result = gateway.request("arxiv", search_fn, query, limit=100)
    """

    # Default rate limits based on official API documentation:
    #   arXiv:     1 req / 3 seconds (https://info.arxiv.org/help/api/user-manual.html)
    #   OpenAlex:  10 req/s polite pool, 100k/day (https://help.openalex.org/api/authentication)
    #   S2:        1000 RPS shared unauth, 1 RPS with API key
    #   LLM:       Provider-dependent; conservative for shared endpoints
    #
    # These are STARTING points: AdaptiveTokenBucket cuts the effective rate on
    # 429/503 signals and recovers after sustained success, so a provider that
    # quietly enforces a stricter limit than documented converges to it.
    DEFAULT_RATES = {
        "arxiv":     (0.33, 1),   # 1 req / 3s, no burst
        "openalex":  (5.0,  3),   # 5 req/s sustained, burst 3
        "s2":        (1.0,  2),   # 1 req/s, burst 2
        "llm":       (0.5,  1),   # 0.5 req/s, burst 1
        "embedding": (1.0,  1),   # 1 req/s, burst 1
        "s2_bulk":   (1.0,  2),   # 1 req/s, burst 2 (Semantic Scholar bulk)
        "crossref":  (1.0,  1),   # polite pool w/ mailto
        "openreview": (0.5, 1),
        "huggingface": (2.0, 2),  # 2 req/s, burst 2 (HF API polite pool)
    }

    def __init__(
        self,
        rates: Optional[dict[str, tuple[float, int]]] = None,
        circuit_threshold: int = 5,
        circuit_cooldown: float = 60.0,
    ):
        rates = rates or self.DEFAULT_RATES
        self._buckets: dict[str, AdaptiveTokenBucket] = {
            name: AdaptiveTokenBucket(rate=r, burst=b)
            for name, (r, b) in rates.items()
        }
        self._breakers = {
            name: CircuitBreaker(provider=name, threshold=circuit_threshold, cooldown=circuit_cooldown)
            for name in rates
        }
        self.health = SourceHealthTracker()
        # Single-flight coalescing: identical read-only calls issued while one
        # is already in flight share its Future instead of duplicating the HTTP
        # round trip (and the rate-limit slot).
        self._inflight: dict[str, Future] = {}
        self._inflight_lock = threading.Lock()

    def request(
        self,
        provider: str,
        fn: Callable,
        *args,
        retries: int = 3,
        backoff_base: float = 3.0,
        backoff_max: float = 180.0,
        rate_limit: bool = True,
        coalesce_key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Execute fn(*args, **kwargs) with rate limiting, circuit breaking, and retry.

        Retry strategy:
          - 429 (RateLimitError): NO fast retry. The adaptive bucket is
            penalized (effective rate cut) and we sleep max(Retry-After,
            exponential backoff) before the next attempt. A 429 means
            "stop hammering me" — the bucket learns from it.
          - Transient errors (524, 502, 503, timeout, connection reset):
            Retry with exponential backoff + jitter.
          - Other errors: Raise immediately.

        Args:
            provider: API provider name (arxiv, openalex, s2, llm, embedding).
            fn: Callable to execute.
            retries: Max retry attempts on transient errors.
            backoff_base: Base seconds for exponential backoff.
            backoff_max: Maximum backoff seconds.
            rate_limit: If False, skip token-bucket rate limiting (use when the
                        library or caller already enforces delays).
            coalesce_key: Optional key for single-flight dedup — identical
                          concurrent read-only calls share one execution.

        Returns:
            Result of fn(*args, **kwargs).

        Raises:
            CircuitOpen: If circuit breaker is open.
            RateLimitError: If rate limited (429).
            Exception: Original exception for non-retryable errors.
        """
        if coalesce_key is None:
            return self._request_inner(
                provider, fn, *args,
                retries=retries, backoff_base=backoff_base,
                backoff_max=backoff_max, rate_limit=rate_limit, **kwargs,
            )

        # Single-flight: only the leader executes; followers share its Future.
        with self._inflight_lock:
            fut = self._inflight.get(coalesce_key)
            if fut is None:
                fut = Future()
                self._inflight[coalesce_key] = fut
                leader = True
            else:
                leader = False
        if not leader:
            logger.debug("Coalesced duplicate request for %s (%s)", provider, coalesce_key[:80])
            return fut.result()
        try:
            result = self._request_inner(
                provider, fn, *args,
                retries=retries, backoff_base=backoff_base,
                backoff_max=backoff_max, rate_limit=rate_limit, **kwargs,
            )
            fut.set_result(result)
            return result
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._inflight_lock:
                self._inflight.pop(coalesce_key, None)

    def _request_inner(
        self,
        provider: str,
        fn: Callable,
        *args,
        retries: int = 3,
        backoff_base: float = 3.0,
        backoff_max: float = 180.0,
        rate_limit: bool = True,
        **kwargs,
    ) -> Any:
        bucket = self._get_bucket(provider) if rate_limit else None
        breaker = self._get_breaker(provider)
        last_error = None

        for attempt in range(retries + 1):
            # Circuit breaker check — acquire() never sleeps, raises immediately
            try:
                breaker.acquire()
            except CircuitOpen as e:
                if attempt < retries:
                    wait = self._jitter(min(backoff_max, backoff_base * (2 ** attempt)))
                    logger.warning(
                        "Circuit breaker open for %s — waiting %.1fs (attempt %d/%d)",
                        provider, wait, attempt + 1, retries + 1,
                    )
                    time.sleep(wait)
                    continue
                self.health.record(provider, False)
                raise

            # Rate limit: wait for token (skip if rate_limit=False)
            if bucket and not bucket.acquire(timeout=backoff_max):
                logger.warning("Rate limit timeout for %s after %.0fs", provider, backoff_max)
                continue

            try:
                result = fn(*args, **kwargs)
                breaker.record_success()
                self.health.record(provider, True)
                if bucket:
                    bucket.reward()
                return result
            except RateLimitError as e:
                last_error = e
                breaker.record_failure()
                self.health.record(provider, False)
                if bucket:
                    # AIMD: the provider told us we're too fast — slow down.
                    bucket.penalize(factor=0.5, cooldown=max(15.0, min(e.retry_after, 120.0)))
                sleep_time = max(e.retry_after, backoff_base * (2 ** attempt))
                sleep_time = min(backoff_max, self._jitter(sleep_time))
                logger.warning(
                    "%s rate limited (429) — effective rate now %.2f req/s, sleeping %.1fs before retry %d/%d",
                    provider, bucket.current_rate() if bucket else -1, sleep_time, attempt + 1, retries + 1,
                )
                time.sleep(sleep_time)
                if attempt < retries:
                    continue
                raise
            except Exception as e:
                last_error = e
                if self._is_transient(e):
                    if attempt < retries:
                        # Respect provider's retry_after hint (e.g., Cloudflare 524)
                        provider_hint = self._parse_retry_after(e)
                        wait = min(backoff_max, self._jitter(provider_hint or (backoff_base * (2 ** attempt))))
                        logger.warning(
                            "%s transient error (attempt %d/%d): %s — retrying in %.1fs",
                            provider, attempt + 1, retries + 1, e, wait,
                        )
                        time.sleep(wait)
                        continue
                self.health.record(provider, False)
                raise

        # All retries exhausted
        self.health.record(provider, False)
        if last_error:
            raise last_error
        return None

    @staticmethod
    def _jitter(wait: Optional[float]) -> float:
        """Add +/-20% jitter to a backoff wait to avoid synchronized retries."""
        if not wait or wait <= 0:
            return wait or 0.0
        return max(0.0, wait * random.uniform(0.8, 1.2))

    def is_available(self, provider: str) -> bool:
        """True if the circuit breaker for `provider` is not open.

        Callers (e.g. TopicHunter) probe this before building queries so an
        open breaker short-circuits to a skip instead of paying retry sleeps.
        """
        return not self._get_breaker(provider).is_open

    def breaker_state(self, provider: str) -> CircuitState:
        return self._get_breaker(provider).state

    def effective_rate(self, provider: str) -> float:
        """Current adaptive rate for a provider (requests/second)."""
        return self._get_bucket(provider).current_rate()

    def _get_bucket(self, provider: str) -> AdaptiveTokenBucket:
        if provider not in self._buckets:
            self._buckets[provider] = AdaptiveTokenBucket(rate=0.5, burst=1)
            self._breakers[provider] = CircuitBreaker(provider=provider)
        return self._buckets[provider]

    def _get_breaker(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(provider=provider)
        return self._breakers[provider]

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        """Check if exception is transient (worth retrying)."""
        msg = str(e).lower()
        return any(kw in msg for kw in (
            "524", "502", "503", "timeout", "connection", "reset",
            "eof", "broken pipe", "connectionreset",
        ))

    @staticmethod
    def _parse_retry_after(e: Exception) -> Optional[float]:
        """Extract retry_after hint from error response (e.g., Cloudflare 524)."""
        import json as _json
        msg = str(e)
        # Look for 'retry_after': N in the error dict
        for token in ("'retry_after': ", '"retry_after": '):
            idx = msg.find(token)
            if idx != -1:
                rest = msg[idx + len(token):]
                num_str = ""
                for ch in rest:
                    if ch.isdigit() or ch == '.':
                        num_str += ch
                    else:
                        break
                if num_str:
                    try:
                        return float(num_str)
                    except ValueError:
                        pass
        return None


# Module-level singleton
_gateway: Optional[APIGateway] = None
_gateway_lock = threading.Lock()


def get_gateway(**kwargs) -> APIGateway:
    global _gateway
    if _gateway is None:
        with _gateway_lock:
            if _gateway is None:
                _gateway = APIGateway(**kwargs)
    return _gateway


def reset_gateway():
    global _gateway
    with _gateway_lock:
        _gateway = None
