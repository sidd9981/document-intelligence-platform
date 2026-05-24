"""
Embedding worker for the ingestion pipeline.

Consumes raw document events from raw:filings, chunks them using the
appropriate chunker, embeds the chunks in batches, and publishes
embedded chunk events to embedded:chunks for the Qdrant writer.

Chunking and embedding happen in the same worker because chunks without
embeddings have no downstream use. The natural pipeline boundary is
raw document in, embedded chunks out.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

from finsight.config.settings import settings
from finsight.ingestion.chunkers.news_chunker import chunk_news_article
from finsight.ingestion.chunkers.sec_chunker import chunk_sec_filing
from finsight.gateway.db import get_pool
from finsight.ingestion.entity_extractor import EntityExtractor
from finsight.ingestion.stream_backend import STREAM_ENTITIES
from finsight.ingestion.chunkers.transcript_chunker import chunk_transcript
from finsight.ingestion.stream_backend import (
    GROUP_EMBEDDERS,
    STREAM_EMBEDDED,
    STREAM_FILINGS,
    StreamBackend,
)
from finsight.services.llm import embed
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

SEC_FILING_TYPES = {"10-K", "10-Q", "8-K", "DEF 14A"}
EMBEDDING_BATCH_SIZE = 32


class EmbeddingWorker:
    """Consumes raw document events and publishes embedded chunk events.

    One instance runs as a long-lived async task. Call run() to start
    the consume loop. Call stop() to shut down gracefully.
    """

    def __init__(self, backend: StreamBackend, consumer_name: str) -> None:
        self._backend = backend
        self._consumer_name = consumer_name
        self._running = False

    async def run(self) -> None:
        """Start the consume loop. Runs until stop() is called."""
        self._running = True
        logger.info("embedding worker %s starting", self._consumer_name)

        while self._running:
            try:
                messages = await self._backend.consume(
                    stream=STREAM_FILINGS,
                    group=GROUP_EMBEDDERS,
                    consumer=self._consumer_name,
                    count=5,
                    block_ms=1000,
                )

                for message in messages:
                    try:
                        await self._process_message(message)
                        await self._backend.ack(STREAM_FILINGS, GROUP_EMBEDDERS, message["id"])
                    except Exception as e:
                        logger.error(
                            "failed to process message %s: %s",
                            message["id"],
                            e,
                        )

            except Exception as e:
                logger.error("consume loop error: %s", e)
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    async def _process_message(self, message: dict) -> None:
        """Chunk, embed, and publish a single document event."""
        data = message["data"]
        doc_id = _doc_id_from_message(data)
        filing_type = data.get("filing_type", "")

        with tracer.start_as_current_span("embedder.process_message") as span:
            span.set_attribute("doc_id", doc_id)
            span.set_attribute("filing_type", filing_type)
            span.set_attribute("ticker", data.get("ticker", ""))

            chunks = _chunk_document(data, doc_id)
            span.set_attribute("chunks.count", len(chunks))

            if not chunks:
                logger.warning("no chunks produced for doc %s filing_type=%s", doc_id, filing_type)
                return

            embedded_count = 0
            for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                await self._embed_and_publish_batch(batch, data, doc_id)
                embedded_count += len(batch)

            span.set_attribute("chunks.embedded", embedded_count)
            logger.info("embedded %d chunks for doc %s", embedded_count, doc_id)

            await self._extract_and_publish_entities(data, doc_id)


    async def _extract_and_publish_entities(self, data: dict, doc_id: str) -> None:
        """Extract entities from the document and publish to extracted:entities."""
        try:
            pool = get_pool()
            extractor = EntityExtractor(backend=self._backend, db_pool=pool)
            entities = await extractor.process_document(
                doc_id=doc_id,
                text=data.get("text", ""),
                ticker=data.get("ticker", ""),
            )
            await extractor.publish_entities(
                doc_id=doc_id,
                ticker=data.get("ticker", ""),
                filing_type=data.get("filing_type", ""),
                filing_date=data.get("filing_date", ""),
                scopes=data.get("scopes", ["public"]),
                entities=entities,
            )
            logger.info("extracted %d entities for doc %s", len(entities), doc_id)
        except Exception as e:
            logger.error("entity extraction failed for doc %s: %s", doc_id, e)

    async def _embed_and_publish_batch(
        self,
        chunks: list,
        source_data: dict,
        doc_id: str,
    ) -> None:
        """Embed a batch of chunks and publish each to embedded:chunks.

        Failures on individual chunks are logged and skipped. A partial
        batch is better than losing the whole document.
        """
        for chunk in chunks:
            try:
                embedding = await embed(chunk.content)

                await self._backend.publish(STREAM_EMBEDDED, {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "embedding": embedding,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "metadata": {
                        "doc_id": doc_id,
                        "ticker": source_data.get("ticker", ""),
                        "company_name": source_data.get("company_name", ""),
                        "filing_type": source_data.get("filing_type", ""),
                        "filing_date": source_data.get("filing_date", ""),
                        "section": getattr(chunk, "section", ""),
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "embedding_model": settings.ollama.embedding_model,
                        "scopes": source_data.get("scopes", ["public"]),
                    },
                })

            except Exception as e:
                logger.error("failed to embed chunk %s: %s", chunk.chunk_id, e)
                continue


def _chunk_document(data: dict, doc_id: str) -> list:
    """Dispatch to the right chunker based on filing type.

    Returns an empty list for unknown filing types rather than raising.
    The caller logs and skips empty results.
    """
    filing_type = data.get("filing_type", "")
    text = data.get("text", "")

    if not text:
        return []

    if filing_type in SEC_FILING_TYPES:
        return chunk_sec_filing(text, doc_id)

    if filing_type == "transcript":
        filing_date_str = data.get("filing_date", "2024-01-01")
        filing_date = date.fromisoformat(filing_date_str)
        return chunk_transcript(
            text=text,
            doc_id=doc_id,
            company_name=data.get("company_name", ""),
            call_date=filing_date,
            speaker_roles=data.get("speaker_roles"),
        )

    if filing_type == "news":
        filing_date_str = data.get("filing_date", "2024-01-01")
        filing_date = date.fromisoformat(filing_date_str)
        return chunk_news_article(
            text=text,
            doc_id=doc_id,
            source=data.get("source", ""),
            headline=data.get("headline", ""),
            published_date=filing_date,
        )

    logger.warning("unknown filing type %s for doc %s — skipping", filing_type, doc_id)
    return []


def _doc_id_from_message(data: dict) -> str:
    """Generate a stable doc ID from the message fields.

    Uses ticker + filing_type + filing_date so the same filing
    published twice produces the same doc ID and Qdrant upserts
    overwrite rather than duplicate.
    """
    import hashlib
    raw = f"{data.get('ticker', '')}:{data.get('filing_type', '')}:{data.get('filing_date', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]