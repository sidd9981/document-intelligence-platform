"""
Seeds the canonical_entities table from the SEC EDGAR company tickers JSON.

Downloads the full company registry (~10k companies) and inserts into
Postgres. Entity resolution during ingestion matches against this table.

Run once before ingestion:
    python scripts/seed_canonical_entities.py

Safe to run multiple times — uses INSERT ... ON CONFLICT DO UPDATE.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from finsight.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "FinSight research@example.com"


async def seed() -> None:
    logger.info("fetching EDGAR company registry")

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
    ) as http:
        response = await http.get(EDGAR_TICKERS_URL)
        response.raise_for_status()
        data = response.json()

    logger.info("fetched %d companies", len(data))

    conn = await asyncpg.connect(settings.postgres.dsn)

    try:
        inserted = 0
        for entry in data.values():
            cik = str(entry["cik_str"]).zfill(10)
            name = entry["title"]
            ticker = entry.get("ticker", "").upper()

            await conn.execute(
                """
                INSERT INTO canonical_entities (cik, official_name, tickers)
                VALUES ($1, $2, $3)
                ON CONFLICT (cik) DO UPDATE SET
                    official_name = EXCLUDED.official_name,
                    tickers = EXCLUDED.tickers,
                    updated_at = NOW()
                """,
                cik,
                name,
                [ticker] if ticker else [],
            )
            inserted += 1

        logger.info("seeded %d canonical entities", inserted)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())