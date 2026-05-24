"""
Qdrant writer for the ingestion pipeline.

Consumes embedded chunk events from embedded:chunks and upserts them
into the Qdrant dense collection. This is the final step in the
ingestion pipeline — after this, chunks are searchable.

Uses upsert semantics so replaying the stream never creates duplicates.
"""

from __future__ import annotations

import asyncio
import logging

from finsight.ingestion.stream_backend import (
    GROUP_QDRANT,
    STREAM_EMBEDDED,
    StreamBackend,
)
from finsight.services.vector_store import upsert_chunks
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

UPSERT_BATCH_SIZE = 64


class QdrantWriter:
    """Consumes embedded chunk events and writes them to Qdrant.

    Batches upserts for efficiency — individual upserts per chunk
    would make ingestion ~10x slower due to round-trip overhead.
    """

    def __init__(self, backend: StreamBackend, consumer_name: str) -> None:
        self._backend = backend
        self._consumer_name = consumer_name
        self._running = False

    async def run(self) -> None:
        """Start the consume loop. Runs until stop() is called."""
        self._running = True
        logger.info("qdrant writer %s starting", self._consumer_name)

        while self._running:
            try:
                messages = await self._backend.consume(
                    stream=STREAM_EMBEDDED,
                    group=GROUP_QDRANT,
                    consumer=self._consumer_name,
                    count=UPSERT_BATCH_SIZE,
                    block_ms=1000,
                )

                if messages:
                    chunks = []
                    for message in messages:
                        data = message["data"]
                        chunks.append({
                            "chunk_id": data["chunk_id"],
                            "embedding": data["embedding"],
                            "content": data["content"],
                            "metadata": data["metadata"],
                        })

                    try:
                        await upsert_chunks(chunks)
                        for message in messages:
                            await self._backend.ack(
                                STREAM_EMBEDDED, GROUP_QDRANT, message["id"]
                            )
                        logger.info("upserted %d chunks to qdrant", len(chunks))
                    except Exception as e:
                        logger.error("qdrant upsert failed: %s", e)

            except Exception as e:
                logger.error("qdrant writer consume loop error: %s", e)
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False