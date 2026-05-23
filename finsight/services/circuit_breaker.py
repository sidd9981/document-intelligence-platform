"""
Circuit breaker for MCP service calls.

Wraps any async callable. Tracks consecutive failures and opens
the circuit after the failure threshold is reached. While open,
calls fail immediately without hitting the downstream service.
After the recovery timeout a single probe call is allowed through.
Success closes the circuit. Failure restarts the timeout.

Usage:
    breaker = CircuitBreaker(name="neo4j", failure_threshold=5, recovery_timeout=30)

    result = await breaker.call(my_async_fn, arg1, arg2)

If the circuit is open, call() raises CircuitOpenError. The caller
catches this and returns a fallback result — empty graph result,
vector-only retrieval, etc.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

from finsight.telemetry.tracing import get_tracer
from finsight.services.metrics import circuit_breaker_opens_total

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""

    def __init__(self, name: str) -> None:
        super().__init__(f"circuit {name!r} is open — calls are failing fast")
        self.name = name


class CircuitBreaker:
    """Async circuit breaker.

    Thread-safe via asyncio.Lock. One instance per downstream service.
    Create at application startup and reuse across requests.

    Args:
        name: Human-readable name for logs and traces.
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait before allowing a probe call.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()
        

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, fn, *args, **kwargs):
        """Call fn(*args, **kwargs) through the circuit breaker.

        Raises:
            CircuitOpenError: If the circuit is open and the recovery
                              timeout has not elapsed.
            Exception: Any exception raised by fn, after recording
                       the failure.
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed < self.recovery_timeout:
                    raise CircuitOpenError(self.name)
                logger.info("circuit %r half-open, allowing probe", self.name)
                self._state = CircuitState.HALF_OPEN

        with tracer.start_as_current_span(f"circuit_breaker.{self.name}") as span:
            span.set_attribute("state", self._state.value)
            try:
                result = await fn(*args, **kwargs)
                await self._on_success()
                return result
            except CircuitOpenError:
                raise
            except Exception as e:
                await self._on_failure(e)
                raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("circuit %r closed after successful probe", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None

    async def _on_failure(self, exc: Exception) -> None:
        async with self._lock:
            self._failure_count += 1
            logger.warning(
                "circuit %r failure %d/%d: %s",
                self.name,
                self._failure_count,
                self.failure_threshold,
                exc,
            )
            if self._failure_count >= self.failure_threshold:
                just_opened = self._state != CircuitState.OPEN
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                if just_opened:
                    logger.error(
                        "circuit %r opened after %d failures",
                        self.name,
                        self._failure_count,
                    )
                    circuit_breaker_opens_total.labels(name=self.name).inc()

    async def reset(self) -> None:
        """Force the circuit closed. Used in tests and manual recovery."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None