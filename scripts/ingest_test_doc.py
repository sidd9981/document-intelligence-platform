"""
Ingests a single test document for Phase 1 validation.

This script inserts a small excerpt of Apple's 2023 10-K risk factors
section so we have real content to query against. It is used only for
development and testing — Phase 2 replaces this with the full EDGAR
ingestion pipeline.

Usage:
    python scripts/ingest_test_doc.py
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from finsight.gateway.db import close_pool, init_pool
from finsight.ingestion.ingest import ingest_document
from finsight.services.llm import close_client, init_client
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
)
from finsight.telemetry.tracing import setup_tracing

TEST_DOCUMENT = """
Apple Inc. faces significant risks related to its dependence on third-party
manufacturers and suppliers. The Company relies on sole-source suppliers for
certain components, including semiconductors and display components. The
Company depends on the performance of its suppliers, including Taiwan
Semiconductor Manufacturing Company Limited, which manufactures substantially
all of the Company's custom silicon chips.

The Company's business is subject to the risks of international operations.
The Company's manufacturing is concentrated in China, and a significant
portion of the Company's revenue is generated in international markets.
Changes in trade policies, tariffs, and export controls could adversely
affect the Company's business, financial condition, and results of operations.

The Company faces intense competition in all markets in which it operates.
The markets for the Company's products and services are highly competitive.
The Company competes with a broad range of companies, including Samsung
Electronics Co., Ltd., Alphabet Inc., Microsoft Corporation, and other
global and domestic companies. The Company's competitors have significant
resources and capabilities.

The Company's success depends on its ability to attract and retain key
personnel. The Company depends on the continued services of its executive
officers and other key employees. The loss of the services of these
individuals could adversely affect the Company's business. Competition for
qualified personnel is intense in the technology industry.

The Company is subject to complex and changing laws and regulations
worldwide. The Company is subject to a variety of laws and regulations in
the United States and internationally, including laws and regulations
relating to privacy, data protection, consumer protection, and competition.
Compliance with these laws and regulations is complex and may impose
significant costs on the Company.
"""


async def main() -> None:
    """Initialize services and ingest the test document."""
    setup_tracing()

    await init_pool()
    await init_qdrant()
    await ensure_collections_exist()
    await init_client()

    try:
        doc_id = await ingest_document(
            text=TEST_DOCUMENT,
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=date(2023, 10, 27),
            section="Risk Factors",
            scopes=["public", "analysis", "risk"],
            source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        )
        print(f"ingested document: {doc_id}")
    finally:
        await close_client()
        await close_qdrant()
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())