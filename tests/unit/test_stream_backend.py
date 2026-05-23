"""
Unit tests for the stream backend.

No running Redis needed. We mock the Redis client and verify that
publish/consume/ack/dlq call the right Redis commands with the
right arguments. The integration tests (which need a real Redis)
live in tests/integration/.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.ingestion.stream_backend import (
    MAX_RETRIES,
    STREAM_DLQ,
    STREAM_FILINGS,
    GROUP_CHUNKERS,
    RedisStreamBackend,
)


@pytest.fixture
def mock_redis() -> AsyncMock:
    client = AsyncMock()
    client.xgroup_create = AsyncMock(return_value=True)
    client.xadd = AsyncMock(return_value=b"1234567890-0")
    client.xreadgroup = AsyncMock(return_value=None)
    client.xack = AsyncMock(return_value=1)
    client.xautoclaim = AsyncMock(return_value=("0-0", [], []))
    client.xpending_range = AsyncMock(return_value=[])
    return client


@pytest.fixture
def backend(mock_redis: AsyncMock) -> RedisStreamBackend:
    return RedisStreamBackend(mock_redis)


async def test_publish_calls_xadd_with_serialized_message(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    message = {"ticker": "AAPL", "filing_type": "10-K"}
    await backend.publish(STREAM_FILINGS, message)

    mock_redis.xadd.assert_called_once()
    call_args = mock_redis.xadd.call_args
    assert call_args[0][0] == STREAM_FILINGS
    payload = json.loads(call_args[0][1]["data"])
    assert payload == message


async def test_publish_returns_message_id(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    mock_redis.xadd.return_value = b"9999-0"
    result = await backend.publish(STREAM_FILINGS, {"ticker": "TSLA"})
    assert result == "9999-0"


async def test_consume_creates_group_on_first_call(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")
    mock_redis.xgroup_create.assert_called_once_with(
        STREAM_FILINGS, GROUP_CHUNKERS, id="0", mkstream=True
    )


async def test_consume_does_not_recreate_group_on_second_call(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")
    await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")
    assert mock_redis.xgroup_create.call_count == 1


async def test_consume_returns_empty_list_when_no_messages(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    mock_redis.xreadgroup.return_value = None
    result = await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")
    assert result == []


async def test_consume_deserializes_messages(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    message = {"ticker": "MSFT", "filing_type": "10-Q"}
    mock_redis.xreadgroup.return_value = [
        (b"raw:filings", [(b"111-0", {b"data": json.dumps(message).encode()})])
    ]

    result = await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")

    assert len(result) == 1
    assert result[0]["id"] == "111-0"
    assert result[0]["data"] == message


async def test_ack_calls_xack(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    await backend.ack(STREAM_FILINGS, GROUP_CHUNKERS, "111-0")
    mock_redis.xack.assert_called_once_with(STREAM_FILINGS, GROUP_CHUNKERS, "111-0")


async def test_send_to_dlq_includes_original_stream_and_reason(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    await backend.send_to_dlq(
        stream=STREAM_FILINGS,
        message={"ticker": "AAPL"},
        reason="parse failure",
    )

    mock_redis.xadd.assert_called_once()
    call_args = mock_redis.xadd.call_args
    assert call_args[0][0] == STREAM_DLQ
    entry = json.loads(call_args[0][1]["data"])
    assert entry["original_stream"] == STREAM_FILINGS
    assert entry["reason"] == "parse failure"
    assert entry["message"] == {"ticker": "AAPL"}
    assert "failed_at" in entry


async def test_consume_busygroup_error_is_ignored(
    backend: RedisStreamBackend,
    mock_redis: AsyncMock,
) -> None:
    """BUSYGROUP means the group already exists. That's fine, not an error."""
    import redis.asyncio as aioredis
    mock_redis.xgroup_create.side_effect = aioredis.ResponseError("BUSYGROUP Consumer Group name already exists")

    result = await backend.consume(STREAM_FILINGS, GROUP_CHUNKERS, "chunker-1")
    assert result == []


async def test_stream_backend_protocol_satisfied() -> None:
    """RedisStreamBackend must satisfy the StreamBackend Protocol at runtime."""
    from finsight.ingestion.stream_backend import StreamBackend
    client = AsyncMock()
    backend = RedisStreamBackend(client)
    assert isinstance(backend, StreamBackend)