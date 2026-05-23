"""
Unit tests for the circuit breaker.

No services needed. Tests state transitions directly.
"""

from __future__ import annotations

import pytest

from finsight.services.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


async def ok():
    return "ok"


async def fail():
    raise ValueError("downstream error")


@pytest.mark.asyncio
async def test_closed_by_default():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_successful_call_stays_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    result = await cb.call(ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_failures_below_threshold_stay_closed():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_threshold():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_circuit_raises_circuit_open_error():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    for _ in range(3):
        with pytest.raises(ValueError):
            await cb.call(fail)
    with pytest.raises(CircuitOpenError):
        await cb.call(ok)


@pytest.mark.asyncio
async def test_half_open_after_recovery_timeout():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.0)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb.state == CircuitState.OPEN
    result = await cb.call(ok)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.0)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)
    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_reset_closes_circuit():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=30)
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(fail)
    assert cb.state == CircuitState.OPEN
    await cb.reset()
    assert cb.state == CircuitState.CLOSED
    result = await cb.call(ok)
    assert result == "ok"


@pytest.mark.asyncio
async def test_success_resets_failure_count():
    """A success mid-stream should reset the counter so threshold restarts."""
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
    with pytest.raises(ValueError):
        await cb.call(fail)
    with pytest.raises(ValueError):
        await cb.call(fail)
    await cb.call(ok)
    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state == CircuitState.CLOSED