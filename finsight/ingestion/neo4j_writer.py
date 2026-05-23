"""
Neo4j writer for the ingestion pipeline.

Consumes entity events from extracted:entities and writes company nodes,
filing nodes, and MENTIONED_IN relationships to the knowledge graph.

All writes use MERGE so the writer is safe to run multiple times on
the same events. Replay and retry never produce duplicate nodes.

Phase 2 scope: Company and Filing nodes, MENTIONED_IN relationships.
Richer relationships (SUPPLIES_TO, COMPETITOR_OF) come in Phase 3.
"""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncGraphDatabase, AsyncDriver

from finsight.config.settings import settings
from finsight.ingestion.stream_backend import (
    GROUP_NEO4J,
    STREAM_ENTITIES,
    StreamBackend,
)
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class Neo4jWriter:
    """Consumes entity events and writes them to Neo4j.

    Pass the driver in so this is testable without a real Neo4j instance.
    Call run() to start the consume loop. Call stop() to shut down.
    """

    def __init__(self, backend: StreamBackend, driver: AsyncDriver, consumer_name: str) -> None:
        self._backend = backend
        self._driver = driver
        self._consumer_name = consumer_name
        self._running = False

    async def run(self) -> None:
        """Start the consume loop. Runs until stop() is called."""
        self._running = True
        logger.info("neo4j writer %s starting", self._consumer_name)

        while self._running:
            try:
                messages = await self._backend.consume(
                    stream=STREAM_ENTITIES,
                    group=GROUP_NEO4J,
                    consumer=self._consumer_name,
                    count=20,
                    block_ms=1000,
                )

                for message in messages:
                    try:
                        await self._process_message(message)
                        await self._backend.ack(STREAM_ENTITIES, GROUP_NEO4J, message["id"])
                    except Exception as e:
                        logger.error(
                            "failed to process entity message %s: %s",
                            message["id"],
                            e,
                        )

            except Exception as e:
                logger.error("neo4j consume loop error: %s", e)
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    async def _process_message(self, message: dict) -> None:
        """Write a single entity event to Neo4j."""
        data = message["data"]
        entity = data.get("entity", {})

        if not entity.get("canonical_cik"):
            return

        with tracer.start_as_current_span("neo4j_writer.process_message") as span:
            span.set_attribute("cik", entity["canonical_cik"])
            span.set_attribute("doc_id", data.get("doc_id", ""))

            async with self._driver.session() as session:
                await session.execute_write(
                    _merge_company_node,
                    cik=entity["canonical_cik"],
                    name=entity["canonical_name"],
                    ticker=data.get("ticker", ""),
                    scopes=data.get("scopes", ["public"]),
                )

                await session.execute_write(
                    _merge_filing_node,
                    doc_id=data["doc_id"],
                    ticker=data.get("ticker", ""),
                    filing_type=data.get("filing_type", ""),
                    filing_date=data.get("filing_date", ""),
                    scopes=data.get("scopes", ["public"]),
                )

                await session.execute_write(
                    _merge_mentioned_in,
                    cik=entity["canonical_cik"],
                    doc_id=data["doc_id"],
                )


async def _merge_company_node(
    tx,
    cik: str,
    name: str,
    ticker: str,
    scopes: list[str],
) -> None:
    """MERGE a Company node by CIK.

    CIK is the canonical identifier from EDGAR. Using it as the merge
    key means 'Apple Inc.' and 'Apple' and '$AAPL' all resolve to the
    same node as long as entity resolution did its job upstream.
    """
    await tx.run(
        """
        MERGE (c:Company {cik: $cik})
        ON CREATE SET
            c.name = $name,
            c.ticker = $ticker,
            c.scopes = $scopes,
            c.created_at = datetime()
        ON MATCH SET
            c.name = $name,
            c.ticker = $ticker,
            c.scopes = $scopes,
            c.updated_at = datetime()
        """,
        cik=cik,
        name=name,
        ticker=ticker,
        scopes=scopes,
    )


async def _merge_filing_node(
    tx,
    doc_id: str,
    ticker: str,
    filing_type: str,
    filing_date: str,
    scopes: list[str],
) -> None:
    """MERGE a Filing node by doc_id."""
    await tx.run(
        """
        MERGE (f:Filing {doc_id: $doc_id})
        ON CREATE SET
            f.ticker = $ticker,
            f.type = $filing_type,
            f.date = date($filing_date),
            f.scopes = $scopes,
            f.created_at = datetime()
        ON MATCH SET
            f.scopes = $scopes,
            f.updated_at = datetime()
        """,
        doc_id=doc_id,
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        scopes=scopes,
    )


async def _merge_mentioned_in(tx, cik: str, doc_id: str) -> None:
    """MERGE a MENTIONED_IN relationship between a Company and a Filing.

    Both nodes must already exist — this runs after the company and
    filing merges in the same transaction sequence.
    """
    await tx.run(
        """
        MATCH (c:Company {cik: $cik})
        MATCH (f:Filing {doc_id: $doc_id})
        MERGE (c)-[:MENTIONED_IN]->(f)
        """,
        cik=cik,
        doc_id=doc_id,
    )


def build_writer(backend: StreamBackend, consumer_name: str = "neo4j-writer-1") -> Neo4jWriter:
    """Build a Neo4jWriter connected to the configured Neo4j instance."""
    driver = AsyncGraphDatabase.driver(
        settings.neo4j.uri if hasattr(settings, "neo4j") else "bolt://localhost:7687",
        auth=("neo4j", "changeme"),
    )
    return Neo4jWriter(backend=backend, driver=driver, consumer_name=consumer_name)