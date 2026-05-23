"""
Unit tests for the entity extractor.

No running services needed. We mock the DB pool and stream backend
to verify extraction, resolution, and routing logic in isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.ingestion.entity_extractor import (
    CONFIDENCE_THRESHOLD,
    CanonicalEntity,
    EntityExtractor,
    ExtractedEntity,
    _extract_raw_entities,
    _is_valid_uuid,
)
from finsight.ingestion.stream_backend import STREAM_ENTITIES


@pytest.fixture
def mock_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.publish = AsyncMock(return_value="111-0")
    return backend


@pytest.fixture
def mock_pool() -> MagicMock:
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    pool.acquire = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=conn), __aexit__=AsyncMock(return_value=False)))
    return pool


@pytest.fixture
def extractor(mock_backend: AsyncMock, mock_pool: MagicMock) -> EntityExtractor:
    e = EntityExtractor(backend=mock_backend, db_pool=mock_pool)
    e._ticker_to_canonical = {
        "AAPL": CanonicalEntity(cik="0000320193", official_name="Apple Inc.", tickers=["AAPL"]),
        "TSMC": CanonicalEntity(cik="0001046179", official_name="Taiwan Semiconductor Manufacturing Co", tickers=["TSMC"]),
        "MSFT": CanonicalEntity(cik="0000789019", official_name="Microsoft Corporation", tickers=["MSFT"]),
    }
    e._cik_to_canonical = {v.cik: v for v in e._ticker_to_canonical.values()}
    e._registry_loaded = True
    return e


def test_extract_raw_entities_finds_dollar_tickers():
    text = "The company relies on $AAPL and $TSMC for its supply chain."
    entities = _extract_raw_entities(text)
    raw_texts = {e.raw_text for e in entities}
    assert "AAPL" in raw_texts
    assert "TSMC" in raw_texts


def test_extract_raw_entities_finds_cik_references():
    text = "As referenced in CIK No. 0000320193 the company disclosed the following."
    entities = _extract_raw_entities(text)
    raw_texts = {e.raw_text for e in entities}
    assert "0000320193" in raw_texts


def test_extract_raw_entities_deduplicates():
    text = "$AAPL reported earnings. $AAPL beat estimates. $AAPL guidance was strong."
    entities = _extract_raw_entities(text)
    aapl_count = sum(1 for e in entities if e.raw_text == "AAPL")
    assert aapl_count == 1


async def test_process_document_resolves_known_ticker(
    extractor: EntityExtractor,
) -> None:
    text = "Apple relies on $TSMC for chip manufacturing."
    resolved = await extractor.process_document("doc-001", text, "AAPL")
    canonical_names = {e.canonical_name for e in resolved}
    assert "Taiwan Semiconductor Manufacturing Co" in canonical_names


async def test_process_document_includes_filing_company(
    extractor: EntityExtractor,
) -> None:
    text = "The company had a strong quarter."
    resolved = await extractor.process_document("doc-001", text, "AAPL")
    ciks = {e.canonical_cik for e in resolved}
    assert "0000320193" in ciks


async def test_process_document_resolved_entities_above_threshold(
    extractor: EntityExtractor,
) -> None:
    text = "$AAPL and $MSFT are competitors."
    resolved = await extractor.process_document("doc-001", text, "AAPL")
    for entity in resolved:
        assert entity.confidence >= CONFIDENCE_THRESHOLD


async def test_process_document_unknown_entity_goes_to_provisional(
    extractor: EntityExtractor,
    mock_pool: MagicMock,
) -> None:
    text = "$AAPL competes with $ZZZZ in the market."
    await extractor.process_document("doc-001", text, "AAPL")

    conn = mock_pool.acquire.return_value.__aenter__.return_value
    assert conn.execute.called


async def test_publish_entities_publishes_to_correct_stream(
    extractor: EntityExtractor,
    mock_backend: AsyncMock,
) -> None:
    entities = [
        ExtractedEntity(
            raw_text="AAPL",
            entity_type="company",
            canonical_cik="0000320193",
            canonical_name="Apple Inc.",
            confidence=1.0,
        )
    ]
    await extractor.publish_entities(
        doc_id="doc-001",
        ticker="AAPL",
        filing_type="10-K",
        filing_date="2023-10-27",
        scopes=["public"],
        entities=entities,
    )

    mock_backend.publish.assert_called_once()
    assert mock_backend.publish.call_args[0][0] == STREAM_ENTITIES


async def test_publish_entities_event_contains_canonical_fields(
    extractor: EntityExtractor,
    mock_backend: AsyncMock,
) -> None:
    entities = [
        ExtractedEntity(
            raw_text="AAPL",
            entity_type="company",
            canonical_cik="0000320193",
            canonical_name="Apple Inc.",
            confidence=1.0,
        )
    ]
    await extractor.publish_entities(
        doc_id="doc-001",
        ticker="AAPL",
        filing_type="10-K",
        filing_date="2023-10-27",
        scopes=["public"],
        entities=entities,
    )

    event = mock_backend.publish.call_args[0][1]
    assert event["entity"]["canonical_cik"] == "0000320193"
    assert event["entity"]["canonical_name"] == "Apple Inc."


def test_is_valid_uuid_accepts_valid_uuid():
    assert _is_valid_uuid("550e8400-e29b-41d4-a716-446655440000")


def test_is_valid_uuid_rejects_non_uuid():
    assert not _is_valid_uuid("doc-001")
    assert not _is_valid_uuid("0000320193")