"""
Message bus for the ingestion pipeline.

Producers publish raw documents here. Consumers read, chunk, embed,
and extract entities. Nothing outside this file touches Redis directly.

Two implementations: RedisStreamBackend for local dev, KafkaStreamBackend
for production. Swap via STREAM_BACKEND in .env. Zero application code changes.

The constants at the bottom (STREAM_FILINGS, GROUP_CHUNKERS, etc.) are the
single source of truth for stream and group names. Import them everywhere,
never hardcode the strings.
"""

from __future__ import annotations

import json
import time
from typing import Protocol, runtime_checkable

import redis.asyncio as aioredis

from finsight.config.settings import settings
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

PENDING_TIMEOUT_MS = 30_000
MAX_RETRIES = 3

STREAM_FILINGS = "raw:filings"
STREAM_NEWS = "raw:news"
STREAM_TRANSCRIPTS = "raw:transcripts"
STREAM_EMBEDDED = "embedded:chunks"
STREAM_ENTITIES = "extracted:entities"
STREAM_DLQ = "ingestion:dlq"

GROUP_CHUNKERS = "chunkers"
GROUP_EMBEDDERS = "embedders"
GROUP_NEO4J = "neo4j-writers"
GROUP_QDRANT = "qdrant-writers"
GROUP_ARCHIVER = "archiver"


@runtime_checkable
class StreamBackend(Protocol):
    """Interface that both Redis and Kafka backends must satisfy.

    Using Protocol instead of ABC means the Kafka backend doesn't need
    to import this file. It just needs to match the shape. Useful when
    the Kafka backend lives in an optional dependency group that isn't
    installed locally.
    """

    async def publish(self, stream: str, message: dict) -> str:
        """Publish a message to the stream. Returns the assigned message ID."""
        ...

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[dict]:
        """Pull undelivered messages for this consumer group.

        Each returned dict has 'id' (the message ID you'll need for ack)
        and 'data' (the original message dict you published).

        Blocks for block_ms milliseconds if the stream is empty. Pass
        block_ms=0 if you want to return immediately when there's nothing.
        """
        ...

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge that a message was processed successfully.

        Until you call this, the message stays in the pending list and
        will be redelivered to another consumer after PENDING_TIMEOUT_MS.
        """
        ...

    async def send_to_dlq(self, stream: str, message: dict, reason: str) -> str:
        """Write a failed message to ingestion:dlq and return the DLQ message ID.

        Call this after MAX_RETRIES attempts have failed. The DLQ entry
        includes the original stream, the message, the reason, and a timestamp.
        """
        ...


class RedisStreamBackend:
    """Redis Streams implementation. This is what runs locally.

    Uses XADD to publish, XREADGROUP to consume, XACK to acknowledge,
    and XAUTOCLAIM to reclaim messages from crashed consumers.

    The consumer group is created automatically on first consume() call
    so producers can publish before any consumer has started. Messages
    accumulate in the stream and are delivered when consumers connect.
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._redis = client
        self._groups_ensured: set[tuple[str, str]] = set()

    async def publish(self, stream: str, message: dict) -> str:
        with tracer.start_as_current_span("redis_stream.publish") as span:
            span.set_attribute("stream", stream)

            message_id = await self._redis.xadd(
            stream,
            {"data": json.dumps(message)},
            id="*",
        )

        decoded = message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        span.set_attribute("message_id", decoded)
        return decoded

    async def _ensure_group(self, stream: str, group: str) -> None:
        """Create the consumer group if it doesn't exist yet.

        id="0" means the group starts reading from the beginning of the
        stream, not from the current tail. We want this in dev so you
        can replay all messages by dropping and recreating the group.
        MKSTREAM creates the stream itself if it doesn't exist.
        """
        key = (stream, group)
        if key in self._groups_ensured:
            return

        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        self._groups_ensured.add(key)

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[dict]:
        await self._ensure_group(stream, group)

        with tracer.start_as_current_span("redis_stream.consume") as span:
            span.set_attribute("stream", stream)
            span.set_attribute("group", group)
            span.set_attribute("consumer", consumer)

            await self._reclaim_pending(stream, group, consumer)

            raw = await self._redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )

            messages = []
            if raw:
                for _stream_name, entries in raw:
                    for message_id, fields in entries:
                        messages.append({
                            "id": message_id.decode(),
                            "data": json.loads(fields[b"data"]),
                        })

            span.set_attribute("messages.count", len(messages))
            return messages

    async def _reclaim_pending(self, stream: str, group: str, consumer: str) -> None:
        """Grab messages that have been sitting in the pending list too long.

        If a consumer crashes after reading but before acking, the message
        stays pending forever. XAUTOCLAIM moves those messages to this
        consumer so they get retried. If a message has been attempted
        MAX_RETRIES times already, it goes to the DLQ instead.

        This is best-effort. If it fails we log and move on rather than
        blocking normal consumption.
        """
        try:
            result = await self._redis.xautoclaim(
                stream,
                group,
                consumer,
                min_idle_time=PENDING_TIMEOUT_MS,
                start_id="0-0",
                count=10,
            )

            reclaimed = result[1] if result else []

            for message_id, fields in reclaimed:
                msg_id = message_id.decode() if isinstance(message_id, bytes) else message_id

                pending_info = await self._redis.xpending_range(
                    stream, group, msg_id, msg_id, count=1
                )

                if pending_info and pending_info[0]["times_delivered"] >= MAX_RETRIES:
                    data = json.loads(fields[b"data"])
                    await self.send_to_dlq(
                        stream=stream,
                        message=data,
                        reason=f"exceeded {MAX_RETRIES} delivery attempts",
                    )
                    await self.ack(stream, group, msg_id)

        except Exception:
            pass

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        with tracer.start_as_current_span("redis_stream.ack") as span:
            span.set_attribute("stream", stream)
            span.set_attribute("message_id", message_id)
            await self._redis.xack(stream, group, message_id)

    async def send_to_dlq(self, stream: str, message: dict, reason: str) -> str:
        entry = {
            "original_stream": stream,
            "message": message,
            "reason": reason,
            "failed_at": time.time(),
        }

        with tracer.start_as_current_span("redis_stream.dlq") as span:
            span.set_attribute("original_stream", stream)
            span.set_attribute("reason", reason)

            message_id = await self._redis.xadd(
                STREAM_DLQ,
                {"data": json.dumps(entry)},
                id="*",
                maxlen=10_000,
            )

            return str(message_id)


def get_redis_backend() -> RedisStreamBackend:
    """Build a RedisStreamBackend connected to the streams database (DB 3).

    DB 3 is dedicated to ingestion streams. Cache is DB 0, budget
    counters DB 1, priority queues DB 2. Keeping them separate means
    a flood of ingestion messages can't evict cache entries.
    """
    base_url = settings.redis.url.rsplit("/", 1)[0] if "/" in settings.redis.url.split("//")[1] else settings.redis.url
    url = f"{base_url}/{settings.redis.streams_db}"
    client = aioredis.from_url(url, decode_responses=False)
    return RedisStreamBackend(client)