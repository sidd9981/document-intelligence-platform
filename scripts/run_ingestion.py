"""
Full ingestion pipeline runner.

Starts the EDGAR producer and all consumers concurrently.
Run this to populate Qdrant and Neo4j with real SEC filings.

Usage:
    python scripts/run_ingestion.py

Tickers and years are hardcoded for the initial run. Edit TICKERS
and START_YEAR to ingest more data.

Requires all core + graph services running:
    docker compose -f infra/docker-compose.yml --profile core --profile graph up --wait
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from finsight.gateway.db import close_pool, init_pool
from finsight.ingestion.embedder import EmbeddingWorker
from finsight.ingestion.entity_extractor import EntityExtractor
from finsight.ingestion.neo4j_writer import Neo4jWriter
from finsight.ingestion.producers.edgar_producer import build_producer
from finsight.ingestion.stream_backend import get_redis_backend
from finsight.services.llm import close_client, init_client
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
)
from finsight.telemetry.tracing import setup_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "TSLA"]
FILING_TYPE = "10-K"
START_YEAR = 2022
SCOPES = ["public", "analysis", "risk"]

WORKER_STOP_AFTER_IDLE_SECONDS = 30


async def run_producer(backend, tickers: list[str]) -> None:
    """Produce filings for all tickers into the stream."""
    import httpx

    async with httpx.AsyncClient(
        headers={"User-Agent": "FinSight research@example.com"},
        timeout=30.0,
        follow_redirects=True,
    ) as http_client:
        from finsight.ingestion.producers.edgar_producer import EdgarProducer
        producer = EdgarProducer(backend=backend, http_client=http_client)

        total = 0
        for ticker in tickers:
            logger.info("producing %s %s %d+", ticker, FILING_TYPE, START_YEAR)
            try:
                count = await producer.produce_filings(
                    ticker=ticker,
                    filing_type=FILING_TYPE,
                    start_year=START_YEAR,
                    scopes=SCOPES,
                )
                logger.info("produced %d filings for %s", count, ticker)
                total += count
            except Exception as e:
                logger.error("failed to produce %s: %s", ticker, e)

        logger.info("producer done: %d total filings published", total)


async def run_embedding_worker(backend) -> None:
    """Consume raw:filings, chunk, embed, publish to embedded:chunks."""
    worker = EmbeddingWorker(backend=backend, consumer_name="embedder-1")
    logger.info("embedding worker started")
    await worker.run(idle_stop_seconds=WORKER_STOP_AFTER_IDLE_SECONDS)
    logger.info("embedding worker done")


async def run_entity_extractor(backend) -> None:
    """Consume raw:filings, extract entities, publish to extracted:entities."""
    extractor = EntityExtractor(backend=backend, consumer_name="extractor-1")
    logger.info("entity extractor started")
    await extractor.run(idle_stop_seconds=WORKER_STOP_AFTER_IDLE_SECONDS)
    logger.info("entity extractor done")


async def run_neo4j_writer(backend) -> None:
    """Consume extracted:entities and write to Neo4j."""
    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "changeme"),
    )
    writer = Neo4jWriter(backend=backend, driver=driver, consumer_name="neo4j-1")
    logger.info("neo4j writer started")
    await writer.run(idle_stop_seconds=WORKER_STOP_AFTER_IDLE_SECONDS)
    await driver.close()
    logger.info("neo4j writer done")


async def main() -> None:
    setup_tracing()
    await init_pool()
    await init_qdrant()
    await ensure_collections_exist()
    await init_client()

    backend = get_redis_backend()

    logger.info("starting ingestion pipeline for tickers: %s", TICKERS)

    # await run_producer(backend, TICKERS)
    # logger.info("producer finished, starting workers to drain the stream")
    logger.info("draining existing stream — skipping producer")

    from finsight.ingestion.embedder import EmbeddingWorker
    from finsight.ingestion.neo4j_writer import Neo4jWriter
    from neo4j import AsyncGraphDatabase

    embedder = EmbeddingWorker(backend=backend, consumer_name="embedder-1")

    neo4j_driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "changeme"),
    )
    neo4j_writer = Neo4jWriter(
        backend=backend,
        driver=neo4j_driver,
        consumer_name="neo4j-1",
    )

    async def stop_after(worker, seconds: int) -> None:
        await asyncio.sleep(seconds)
        worker.stop()

    idle_seconds = 300

    await asyncio.gather(
        embedder.run(),
        neo4j_writer.run(),
        stop_after(embedder, idle_seconds),
        stop_after(neo4j_writer, idle_seconds),
    )

    await neo4j_driver.close()
    logger.info("ingestion complete")

    await close_client()
    await close_qdrant()
    await close_pool()
if __name__ == "__main__":
    asyncio.run(main())